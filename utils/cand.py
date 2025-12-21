import math
import random

import torch


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Save the training model
    """
    torch.save(state, filename)


def cand2tuple(cand):
    return tuple(cand.int().flatten().tolist())


def tuple2cand(cand_tuple):
    return torch.tensor(cand_tuple).reshape(-1)


class EvolutionCand:
    def __init__(
        self,
        args,
        device=None,
        stage_num=None,
        channels=None,
        channels_extra=None,
        channel_shape_0=None,
        channel_shape_1=None,
    ):
        self.device = device
        self.stage_num = stage_num
        self.channels = channels
        self.channels_extra = channels_extra
        self.channel_shape_0 = channel_shape_0
        self.channel_shape_1 = channel_shape_1
        self.args = args
        self.first_float = args.first_float
        self.layer_len_start = args.layer_len_start
        self.layer_len_end = args.layer_len_end

        base_cand = self.channels[:, math.floor(self.channel_shape_1 / 2)].cpu().tolist()

        base_cand_all = []
        # Generate a smallest architecture candidate
        for ii in range(len(self.stage_num)):
            for jj in range(self.stage_num[ii][0]):
                base_cand_all.append(base_cand[ii])

        if args.baseline_arch == "mobilenet":
            self.layer_len_min = len(base_cand_all)
        elif args.baseline_arch == "resnet18" or args.baseline_arch == "resnet20":
            self.layer_len_min = len(base_cand_all)

        smallest_cand_channel = base_cand_all + (self.layer_len_end - self.layer_len_min) * [-1]
        smallest_cand_channel = torch.tensor(smallest_cand_channel).to(device)
        smallest_cand_stride = self.generate_stride_combinations(
            smallest_cand_channel, self.layer_len_min, self.layer_len_end
        )
        smallest_cand_conv_type = [0] * self.layer_len_min + (self.layer_len_end - self.layer_len_min) * [-1]
        smallest_cand_conv_type = torch.tensor(smallest_cand_conv_type).to(device)
        self.smallest_cand = torch.cat([smallest_cand_channel, smallest_cand_stride, smallest_cand_conv_type], dim=0)
        # print(f"smallest_cand_channel:{smallest_cand_channel}")
        # print(f"smallest_cand_stride:{smallest_cand_stride}")
        # print(f"smallest_cand_conv_type:{smallest_cand_conv_type}")

        # Generate a biggest architecture candidate
        biggest_cand_channel = base_cand_all + (self.layer_len_end - self.layer_len_min) * [base_cand[-1]]
        biggest_cand_channel = torch.tensor(biggest_cand_channel).to(device)
        # print(biggest_cand_channel)
        biggest_cand_stride = self.generate_stride_combinations(
            biggest_cand_channel, self.layer_len_end, self.layer_len_end
        )
        biggest_cand_conv_type = [0] * self.layer_len_end
        biggest_cand_conv_type = torch.tensor(biggest_cand_conv_type).to(device)

        self.biggest_cand = torch.cat([biggest_cand_channel, biggest_cand_stride, biggest_cand_conv_type], dim=0)
        # print(f"self.smallest_cand:{self.smallest_cand}")
        # print(f"self.biggest_cand:{self.biggest_cand}")
        # exit()

    def generate_stride_combinations(self, channels, channels_len, max_length):
        stride_ = torch.full((1, max_length), -1).reshape(-1).to(self.device)
        if self.args.baseline_arch == "mobilenet":
            stride_[0] = 2
            for ii in range(2, channels_len, 1):
                if channels[ii - 1] > channels[0] and channels[ii] > channels[0] and channels[ii] != channels[ii - 1]:
                    stride_[ii - 1] = 2
                else:
                    stride_[ii - 1] = 1
            stride_[channels_len - 1] = 1
        elif self.args.baseline_arch == "resnet18" or self.args.baseline_arch == "resnet20":
            if self.args.dataset in ["cifar10", "cifar100"]:
                stride_[0] = 1
            elif self.args.dataset in ["whu-19", "ucm"]:
                stride_[0] = 2
            for ii in range(1, channels_len, 1):
                if channels[ii] > channels[0] and channels[ii] != channels[ii - 1]:
                    stride_[ii] = 2
                else:
                    stride_[ii] = 1

        count1 = (stride_ == 2).sum().item()
        count2 = (stride_ != -1).sum().item()
        if count1 == 5 and count2 == channels_len and self.args.baseline_arch == "mobilenet":
            return stride_
        elif (
            count1 == 3
            and count2 == channels_len
            and self.args.baseline_arch == "resnet18"
            and self.args.dataset in ["cifar10", "cifar100"]
        ):
            return stride_
        elif (
            count1 == 4
            and count2 == channels_len
            and self.args.baseline_arch == "resnet18"
            and self.args.dataset in ["whu-19", "ucm"]
        ):
            return stride_

        elif (
            count1 == 3
            and count2 == channels_len
            and self.args.baseline_arch == "resnet20"
            and self.args.dataset in ["cifar10", "cifar100"]
        ):
            return stride_
        elif (
            count1 == 2
            and count2 == channels_len
            and self.args.baseline_arch == "resnet20"
            and self.args.dataset in ["whu-19", "ucm"]
        ):
            return stride_
        else:
            print(f"Error channels is:{channels}")
            print(f"Error stride is:{stride_}")
            return False

    # op是只计算二值的op，暂时没加float的，之后加上
    def get_parm_ops(self, cand, search_width):
        cand_shape = int(cand.shape[0] / 3)
        channel_1 = cand[:cand_shape]
        stride_1 = cand[cand_shape : cand_shape * 2]
        conv_type_1 = cand[cand_shape * 2 :]
        # print(f"channel_1:{channel_1}")
        # print(f"stride_1:{stride_1}")
        # print(f"conv_type_1:{conv_type_1}")

        count_1 = (channel_1 != -1).sum().item()
        stride_2 = torch.ones_like(stride_1)
        stride_2[0] = stride_1[0]
        output_w = torch.ones_like(stride_1)
        output_w[0] = search_width / stride_2[0]

        # cal op
        # cal parm
        kernels = torch.where(conv_type_1 == 0, 3, 1)
        parm_singles = torch.zeros_like(channel_1)
        op_singles = torch.zeros_like(channel_1)
        if self.first_float == 1:
            parm_singles[0] = kernels[0] * kernels[0] * 3 * channel_1[0] * 4  # byte
            op_singles[0] = kernels[0] * kernels[0] * 3 * channel_1[0] * output_w[0] * output_w[0]
        elif self.first_float == 0:
            parm_singles[0] = kernels[0] * kernels[0] * 3 * channel_1[0]  # byte
            op_singles[0] = kernels[0] * kernels[0] * 3 * channel_1[0] * output_w[0] * output_w[0] / 8

        if self.args.dataset in ["imagenet100", "ImageNet1000", "cifar10", "cifar100"]:
            pass
        elif self.args.dataset in ["whu-19", "ucm"]:
            if self.first_float == 1:
                parm_singles[0] = parm_singles[0] + 3 * 3 * channel_1[0] * channel_1[0] * 4
                op_singles[0] = op_singles[0] + 3 * 3 * channel_1[0] * channel_1[0] * output_w[0] * output_w[0]
            elif self.first_float == 0:
                parm_singles[0] = parm_singles[0] + 3 * 3 * channel_1[0] * channel_1[0]
                op_singles[0] = op_singles[0] + 3 * 3 * channel_1[0] * channel_1[0] * output_w[0] * output_w[0] / 8

        if self.args.dataset in ["imagenet100", "ImageNet1000", "cifar10", "cifar100"]:
            pass
        elif self.args.dataset in ["whu-19", "ucm"]:
            stride_2[0] = stride_2[0] * 2

        for ii in range(1, count_1):
            stride_2[ii] = stride_2[ii - 1] * stride_1[ii]
            output_w[ii] = search_width / stride_2[ii]

            # move11_ops = output_w[ii-1]*output_w[ii-1]*channel_1[ii-1]
            # move11_parm = channel_1[ii] * 4 #byte

            conv_ops = kernels[ii] * kernels[ii] * channel_1[ii - 1] * channel_1[ii] * output_w[ii] * output_w[ii] / 64
            conv_parm = kernels[ii] * kernels[ii] * channel_1[ii - 1] * channel_1[ii] / 8  # byte

            if channel_1[ii - 1] != channel_1[ii] and self.args.dataset in ["cifar10", "cifar100", "whu-19", "ucm"]:
                conv_ops = conv_ops + channel_1[ii - 1] * channel_1[ii] * output_w[ii] * output_w[ii]
                conv_parm = conv_parm + channel_1[ii - 1] * channel_1[ii] * 4

            op_singles[ii] = conv_ops
            parm_singles[ii] = conv_parm

        if self.args.dataset == "imagenet100" or self.args.dataset == "cifar100":
            class_num = 100
        elif self.args.dataset == "ImageNet1000":
            class_num = 1000
        elif self.args.dataset == "cifar10":
            class_num = 10
        elif self.args.dataset == "whu-19":
            class_num = 19
        elif self.args.dataset == "ucm":
            class_num = 21

        fc_ops = channel_1[count_1 - 1] * class_num
        fc_parm = channel_1[count_1 - 1] * class_num * 4
        # print(f"search_width:{search_width}")
        # print(f"op_singles:{op_singles}")
        # print(parm_singles)
        # print(op_singles)
        # print(output_w)
        parm_total = parm_singles.sum() + fc_parm
        op_total = op_singles.sum() + fc_ops

        return parm_total, op_total

    def get_random_cand(self):

        found_1 = False
        while not found_1:
            stage_num_select = []
            for ii in range(len(self.stage_num)):
                rand_indices = torch.randint(0, len(self.stage_num[ii]), (1, 1)).to(self.device)
                stage_num_select.append(self.stage_num[ii][rand_indices.cpu().item()])
            stage_num_select = torch.tensor(stage_num_select).to(self.device).reshape(1, -1)
            channels_num = stage_num_select.sum()
            # print(f"get_random_cand stage_num_select:{stage_num_select}")
            # print(f"get_random_cand channels_num:{channels_num}")
            # exit()
            if channels_num <= self.layer_len_end and channels_num >= self.layer_len_start:
                found_1 = True

        channel_1 = self.generate_channel_combinations(stage_num_select, self.channels, self.layer_len_end)

        stride_1 = self.generate_stride_combinations(channel_1, channels_num, self.layer_len_end).to(self.device)

        if self.args.dataset == "imagenet100":
            conv_type_combinations = self.generate_conv_combinations(
                torch.tensor([0, 1]).to(self.device), channels_num, self.layer_len_end
            ).to(self.device)
        elif self.args.dataset in ["cifar10", "cifar100", "whu-19", "ucm"]:
            conv_type_combinations = self.generate_conv_combinations_cifar(
                torch.tensor([0, 1]).to(self.device), channels_num, self.layer_len_end
            ).to(self.device)
        # print(f"get_random_cand conv_type_combinations.shape:{conv_type_combinations.shape}")

        found_2 = False
        while not found_2:
            rand_indices_2 = torch.randint(0, conv_type_combinations.shape[0], (1, 1)).to(self.device)
            conv_type_1 = conv_type_combinations[rand_indices_2, :].view(-1)

            if_condition_1 = self.check_the_feature(channel_1, conv_type_1, stride_1, channels_num=channels_num)
            if_condition_2 = self.check_the_conv_type_array(
                channel_1, conv_type_1, stride_1, channels_num=channels_num
            )
            # print(f"if_condition_1:{if_condition_1}")
            # print(f"if_condition_2:{if_condition_2}")
            if if_condition_1 and if_condition_2:
                found_2 = True

        res = torch.cat([channel_1, stride_1, conv_type_1], dim=0)
        return res

    def get_random_range_cand(self, ops_min, ops_max, parm_min, parm_max):
        found = False
        while not found:
            res = self.get_random_cand()
            parm, ops = self.get_parm_ops(res, self.args.search_width)
            # print(f"res:{res}")
            # print(f"parm:{parm}, ops:{ops}")
            if (ops_min <= ops <= ops_max) == 0:
                continue
            if (parm_min <= parm <= parm_max) == 0:
                continue
            found = True
        return res

    def get_random_max_cand(self, combinations_array):
        found_1 = False
        while not found_1:
            rand_indices = torch.randint(0, len(combinations_array), (1, 1)).to(self.device)
            stage_num_select = combinations_array[rand_indices]
            stage_num_select = torch.tensor(stage_num_select).to(self.device).reshape(1, -1)
            channels_num = stage_num_select.sum()
            if channels_num <= self.args.layer_len_end and channels_num >= self.args.layer_len_start:
                found_1 = True

        channel_1 = self.generate_channel_combinations(stage_num_select, self.channels, self.args.layer_len_end)

        stride_1 = self.generate_stride_combinations(channel_1, channels_num, self.args.layer_len_end).to(self.device)

        if self.args.dataset == "imagenet100":
            conv_type_combinations = self.generate_conv_combinations(
                torch.tensor([0, 1]).to(self.device), channels_num, self.args.layer_len_end
            ).to(self.device)
        elif self.args.dataset in ["cifar10", "cifar100", "whu-19", "ucm"]:
            conv_type_combinations = self.generate_conv_combinations_cifar(
                torch.tensor([0, 1]).to(self.device), channels_num, self.args.layer_len_end
            ).to(self.device)

        found_2 = False
        while not found_2:
            rand_indices_2 = torch.randint(0, conv_type_combinations.shape[0], (1, 1)).to(self.device)
            conv_type_1 = conv_type_combinations[rand_indices_2, :].view(-1)

            if_condition_1 = self.check_the_feature(channel_1, conv_type_1, stride_1, channels_num=channels_num)
            if_condition_2 = self.check_the_conv_type_array(
                channel_1, conv_type_1, stride_1, channels_num=channels_num
            )
            if if_condition_1 and if_condition_2:
                found_2 = True

        res = torch.cat([channel_1, stride_1, conv_type_1], dim=0)

        return res

    def generate_channel_combinations(self, combinations_array, channels, max_length):
        # print(f"generate_channel_combinations combinations_array.shape:{combinations_array.shape}")
        found = False
        while not found:
            rand_indices = torch.randint(0, self.channels_extra.shape[1], (self.channels_extra.shape[0], 1)).to(
                self.device
            )
            # print(f"rand_indices:{rand_indices}")
            # print(f"rand_indices:{rand_indices.shape}")
            # print(f"self.channels_extra:{self.channels_extra.shape}")

            channels_select = self.channels_extra[torch.arange(self.channels_extra.shape[0]).view(-1, 1), rand_indices]
            # print(f"channels_select:{channels_select}")
            # print(f"channels_select:{channels_select.shape}")

            for ii in range(1, channels.shape[0]):
                if channels_select[ii] <= channels_select[ii - 1]:
                    break

                if ii == channels.shape[0] - 1:
                    found = True

        original_shape = combinations_array.shape
        new_shape = (original_shape[0], original_shape[1] + 1)
        new_combinations_array = torch.zeros(new_shape, dtype=combinations_array.dtype).to(self.device)
        new_combinations_array[:, : original_shape[1]] = combinations_array
        sum_upto_index_6 = torch.sum(combinations_array, dim=1)
        new_combinations_array[:, -1] = max_length - sum_upto_index_6

        channel_array_flat = torch.tensor([]).view(1, -1).to(self.device)
        for i, combination in enumerate(new_combinations_array):
            repeated_indices = torch.repeat_interleave(channels_select, combination).view(1, -1)
            if i == 0:
                channel_array_flat = repeated_indices
            else:
                channel_array_flat = torch.cat((channel_array_flat, repeated_indices), dim=0)
        # print(f"generate_channel_combinations channel_array_flat.shape:{channel_array_flat}")
        return channel_array_flat.view(-1)

    def get_value_indices(self, channels_num_all):
        value_indices = torch.arange(len(channels_num_all))

        return value_indices

    def generate_conv_combinations(self, channels, channels_len, max_length):
        combinations_tensor = self.combinations_with_replacement_torch(channels, repeat=channels_len)
        # print(combinations_tensor)
        # print(f"combinations_tensor' shape: {combinations_tensor.shape}")
        zero_matrix = torch.full((combinations_tensor.shape[0], max_length - combinations_tensor.shape[1]), -1).to(
            self.device
        )
        combinations_tensor = torch.cat((combinations_tensor, zero_matrix), dim=1)
        combinations_tensor = combinations_tensor.view(-1, max_length)
        return combinations_tensor

    def generate_conv_combinations_cifar(self, channels, channels_len, max_length):
        zero_matrix = torch.zeros((1, channels_len)).to(self.device).int()
        combinations_tensor = torch.full((1, max_length - channels_len), -1).to(self.device).int()
        combinations_tensor = torch.cat((zero_matrix, combinations_tensor), dim=1)
        combinations_tensor = combinations_tensor.view(1, max_length)
        return combinations_tensor

    def combinations_with_replacement_torch(self, iterable, repeat=None):
        pool = iterable.clone().detach()
        n = len(pool)
        if not n and repeat:
            return
        indices = torch.zeros(repeat, dtype=torch.long).to(self.device)
        combinations_tensor = torch.empty((0, repeat), dtype=pool.dtype).to(self.device)
        while True:
            combination = pool[indices]
            combinations_tensor = torch.cat((combinations_tensor, combination.unsqueeze(0)), dim=0)
            for i in reversed(range(repeat)):
                if indices[i] != n - 1:
                    break
            else:
                return combinations_tensor
            indices[i:] = indices[i] + 1
        return combinations_tensor

    def check_the_feature(self, channel_, conv_, stride_, channels_num=None):
        width_origin = self.args.search_width

        # Use cumulative products to calculate the scaling factor at each stage
        cumulative_stride_product = torch.cumprod(stride_[:channels_num], dim=0)
        width_array = width_origin / cumulative_stride_product

        # Calculate the number of features at each stage
        feature_array = width_array * width_array * channel_[:channels_num]

        # print(f"check_the_feature feature_array{feature_array}")
        # Check if the number of features at any stage exceeds the threshold
        return int((feature_array <= width_origin * width_origin * 128).all())

    def check_the_conv_type_array(self, channel_, conv_, stride_, channels_num=None):
        """
        conv_ == 0: 3x3Conv,
        conv_ == 1: 1x1Conv,
        1x1Conv and stride==2 cannot exist at the same time
        """
        condition_mask = (conv_[:channels_num] == 1) & (stride_[:channels_num] == 2)
        condition_mask_2 = conv_[:channels_num] > 1
        if torch.any(condition_mask):
            return 0
        if torch.any(condition_mask_2):
            print(f"conv_type error:{conv_}")
            exit()
        return int(conv_[0] == 0)

    def change_the_conv_type_array(self, channel_, conv_, stride_, channels_num=None):
        """
        conv_ == 0: 3x3Conv,
        conv_ == 1: 1x1Conv,
        1x1Conv and stride==2 cannot exist at the same time
        """
        condition_mask = (conv_[:channels_num] == 1) & (stride_[:channels_num] == 2)
        mask_len = len(conv_) - len(condition_mask)

        padding = torch.ones(mask_len, dtype=condition_mask.dtype, device=condition_mask.device)
        condition_mask = torch.cat((condition_mask, padding))

        conv_[condition_mask == 0] = 0
        conv_[0] == 0
        return conv_
