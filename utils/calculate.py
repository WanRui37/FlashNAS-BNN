import math
import os
import random
import sys
import time

import torch

from .config import *
from .dataset import *

sys.path.append("../")
from net_bw.dataset_bw import *
from net_bw.Predict_BW import *
from net_predict.net.Predict_RNN import *

output_path = "./net_predict_output/"
model_config = Config()
layer_len_max = model_config.layer_len_max
embed_dim = model_config.embed_dim
hidden_size = model_config.hidden_size
dropout = model_config.dropout

LANE_NUM = 16
VEC_SIZE = 32
PE_NUM_X = 4
PE_NUM_Y = 7


def get_cand_acc(cand, args, device):
    """
    get_cand_acc(): 得到acc
    """
    cand_shape = int(cand.shape[0] / 3)
    channel_1 = cand[:cand_shape]
    stride_1 = cand[cand_shape : cand_shape * 2]
    conv_type_1 = cand[cand_shape * 2 :]
    count_1 = (channel_1 != -1).sum().item()
    channel_1 = channel_1.cpu().numpy().tolist()
    stride_1 = stride_1.cpu().numpy().tolist()
    conv_type_1 = conv_type_1.cpu().numpy().tolist()

    flops_total, bops_total, flparm_total, bparm_total, flops_single, bops_single, flparm_single, bparm_single = (
        cal_total_flops_parm(
            channel_=channel_1, conv_=conv_type_1, stride_=stride_1, count_1=count_1, acc_width=args.acc_width
        )
    )

    feature_single_vec = feature2vec(
        cand_shape, channel_1, conv_type_1, stride_1, flops_single, bops_single, flparm_single, bparm_single, count_1
    )
    # print(f"channel_1:{channel_1}")
    # print(f"count_1:{count_1}")
    unique_channels = find_channel_different_count(channel_1, count_1, args.dataset)

    feature_total_vec = [flops_total] + [bops_total] + [flparm_total] + [bparm_total] + [count_1] + unique_channels
    best_accuracy_array = [0]
    best_accuracy_np = np.array(best_accuracy_array, dtype=np.float32)
    feature_single_vec_np = np.array(feature_single_vec, dtype=np.float32)
    feature_total_vec_np = np.array(feature_total_vec, dtype=np.float32)

    best_accuracy_torch = torch.from_numpy(best_accuracy_np)
    feature_single_vec_torch = torch.from_numpy(feature_single_vec_np).unsqueeze(0)
    feature_total_vec_torch = torch.from_numpy(feature_total_vec_np).unsqueeze(0)

    predict_set = TextDataset(
        best_accuracy_torch, feature_single_vec_torch, feature_total_vec_torch, layer_len_max=args.layer_len_end
    )
    predict_data = DataLoader(predict_set, 1, shuffle=False)

    if "imagenet" in args.dataset:
        model = Predict_harsbnn(embed_dim, hidden_size, dropout, device).to(device)
    elif args.dataset in ["cifar10", "cifar100"]:
        model = Predict_harsbnn_cifar(embed_dim, hidden_size, dropout, device).to(device)

    state_dict = torch.load(getattr(args, "model_path", output_path + "/" + args.dataset + "/CharRNN_OUR_best.pth"), map_location=device)
    # state_dict = torch.load(output_path + '/CharRNN_lstm1.pth')
    # state_dict = torch.load(output_path + '/CharRNN_lstm7.pth')
    # state_dict = torch.load(output_path + '/CharRNN_lstm13.pth')
    # state_dict = torch.load(output_path + '/CharRNN_lstm25.pth')
    # state_dict = torch.load(output_path + '/CharRNN_lstm37.pth')
    # state_dict = torch.load(output_path + '/CharRNN_lstm49.pth')
    # state_dict = torch.load(output_path + '/CharRNN_lstm61.pth')
    # state_dict = torch.load(output_path + '/CharRNN_lstm85.pth')
    # state_dict = torch.load(output_path + '/CharRNN_lstm127.pth')
    model.load_state_dict(state_dict)
    model.eval()

    for data in predict_data:
        x, y = data
        x = x.to(device)
        y = y.to(device, dtype=torch.float32)
        with torch.no_grad():
            output = model(x)
        return output


def get_cand_acc_model(model, cand, args, device):
    """
    get_cand_acc_model(): 得到acc
    """
    cand_shape = int(cand.shape[0] / 3)
    channel_1 = cand[:cand_shape]
    stride_1 = cand[cand_shape : cand_shape * 2]
    conv_type_1 = cand[cand_shape * 2 :]
    count_1 = (channel_1 != -1).sum().item()
    channel_1 = channel_1.cpu().numpy().tolist()
    stride_1 = stride_1.cpu().numpy().tolist()
    conv_type_1 = conv_type_1.cpu().numpy().tolist()

    flops_total, bops_total, flparm_total, bparm_total, flops_single, bops_single, flparm_single, bparm_single = (
        cal_total_flops_parm(
            channel_=channel_1, conv_=conv_type_1, stride_=stride_1, count_1=count_1, acc_width=args.acc_width
        )
    )

    print(f"channel_1:{channel_1}")
    print(f"conv_type_1:{conv_type_1}")
    print(f"stride_1:{stride_1}")
    feature_single_vec = feature2vec(
        cand_shape, channel_1, conv_type_1, stride_1, flops_single, bops_single, flparm_single, bparm_single, count_1
    )
    unique_channels = find_channel_different_count(channel_1, count_1, args.dataset)

    feature_total_vec = [flops_total] + [bops_total] + [flparm_total] + [bparm_total] + [count_1] + unique_channels
    best_accuracy_array = [0]
    best_accuracy_np = np.array(best_accuracy_array, dtype=np.float32)
    feature_single_vec_np = np.array(feature_single_vec, dtype=np.float32)
    feature_total_vec_np = np.array(feature_total_vec, dtype=np.float32)

    best_accuracy_torch = torch.from_numpy(best_accuracy_np)
    feature_single_vec_torch = torch.from_numpy(feature_single_vec_np).unsqueeze(0)
    feature_total_vec_torch = torch.from_numpy(feature_total_vec_np).unsqueeze(0)

    if args.dataset == "imagenet100":
        predict_set = TextDataset(
            best_accuracy_torch, feature_single_vec_torch, feature_total_vec_torch, layer_len_max=args.layer_len_end
        )
    elif args.dataset in ["cifar10", "cifar100"]:
        predict_set = TextDataset_cifar(
            best_accuracy_torch, feature_single_vec_torch, feature_total_vec_torch, layer_len_max=args.layer_len_end
        )

    predict_data = DataLoader(predict_set, 1, shuffle=False)

    for data in predict_data:
        x, y = data
        x = x.to(device)
        y = y.to(device, dtype=torch.float32)
        with torch.no_grad():
            output = model(x)
        return output


def cal_throught(x, alpha=0.8, bw=12.8, max_perform=5734.4):
    """
    cal_throught(): 计算效率
    """
    max_perform = torch.tensor(max_perform).to(x.device).float()
    throught = alpha * torch.min(bw * x, max_perform)
    # print(f"throught:{throught}")

    return throught


def cal_act_lantency(conv_complexity, bw_complexity, alpha=0.8, bw=12.8, max_perform=5734.4):
    """
    cal_act_lantency(): 计算单个kernel延迟
    """
    # print(f"conv_complexity:{conv_complexity}")
    # print(f"bw_complexity:{bw_complexity}")
    throught = cal_throught(
        conv_complexity.float() / bw_complexity.float(),
        alpha=alpha,
        bw=bw,
        max_perform=max_perform,
    )

    latency_singles = conv_complexity / (throught)

    # print(f"conv_complexity:{conv_complexity}")
    # print(f"throught:{throught}")
    # print(f"latency_singles:{latency_singles}")
    # exit()
    return latency_singles


def get_bw_feature(list, type_tensor, embed_dim_1, embed_dim_2, device=None):
    """
    W_in, c_in, W_out, c_out,
    compute, parm, input, output, tpye
    """
    device = torch.device(device) if device is not None else model_config.device
    concat_tensor = torch.tensor([float(data) for data in list], device=device, dtype=torch.float32)
    type_tensor = torch.tensor(to_one_hot(type_tensor, embed_dim_2, 0), device=device)
    concat_tensor = torch.cat((concat_tensor, type_tensor), dim=0).view(1, -1)
    return normalization_simple(concat_tensor, embed_dim_1)


def get_latency(
    args, cand, alpha, bw, max_perform, platform, baseline_arch, search_width, model, embed_dim_1, embed_dim_2
):
    """
    get_latency(): 计算总延迟
    """
    if isinstance(cand, list) or isinstance(cand, tuple):
        cand = torch.tensor(cand).to(args.device)
    else:
        pass
    cand_shape = int(cand.shape[0] / 3)
    channel_1 = cand[:cand_shape]
    stride_1 = cand[cand_shape : cand_shape * 2]
    conv_type_1 = cand[cand_shape * 2 :]
    count_1 = (channel_1 != -1).sum().item()

    stride_2 = torch.ones_like(stride_1)
    stride_2[0] = stride_1[0]
    output_w = torch.ones_like(stride_1)
    output_w[0] = search_width / 2

    # cal op
    kernels = torch.where(conv_type_1 == 0, 3, 1)
    op_singles = torch.zeros_like(channel_1)
    latency_singles = torch.zeros_like(channel_1).float()
    op_singles[0] = kernels[0] * kernels[0] * 3 * channel_1[0] * output_w[0] * output_w[0] * 2

    if platform == "fpga":
        input_c = torch.tensor([48], dtype=float).to(args.device)
        output_size = output_w[0] * output_w[0] * channel_1[0]
        conv_complexity = (
            input_c
            * torch.ceil(output_w[0].float() / PE_NUM_X)
            * PE_NUM_X
            * torch.ceil(output_w[0].float() / PE_NUM_Y)
            * PE_NUM_Y
            * channel_1[0]
            * 2
            / (10**9)
        )
        conv_complexity = kernels[0] * kernels[0] * conv_complexity
        bw_complexity = (
            kernels[0] * kernels[0] * input_c * channel_1[0] * torch.ceil((output_w[0].float() / PE_NUM_Y)) / 8
            + output_size * 2
            + channel_1[0] * 4
        ) / (10**9)
        latency_singles[0] = cal_act_lantency(
            conv_complexity, bw_complexity, alpha=alpha, bw=bw, max_perform=max_perform
        )

    elif platform == "cpu":
        conv_complexity = op_singles[0]
        parm = kernels[0] * kernels[0] * 3 * channel_1[0]
        input_size = search_width * search_width * 3
        output_size = output_w[0] * output_w[0] * channel_1[0]
        x = get_bw_feature(
            [search_width, 3, output_w[0], channel_1[0], conv_complexity, parm, input_size, output_size],
            0,
            embed_dim_1,
            embed_dim_2,
            device=args.device,
        )
        conv_bw = torch.pow(2, model(x))
        latency_conv = cal_act_lantency(conv_complexity, conv_bw, alpha=alpha, bw=bw, max_perform=max_perform)

        latency_singles[0] = latency_conv

        if baseline_arch == "resnet18" or baseline_arch == "resnet20":
            pass
        elif baseline_arch == "mobilenet":
            pass

    elif platform == "gpu":
        pass

    for ii in range(1, count_1):
        stride_2[ii] = stride_2[ii - 1] * stride_1[ii]
        output_w[ii] = search_width / stride_2[ii]
        op_singles[ii] = (
            kernels[ii] * kernels[ii] * channel_1[ii - 1] * channel_1[ii] * output_w[ii] * output_w[ii] * 2
        )

        if platform == "fpga":
            input_c = channel_1[ii - 1]
            output_size = output_w[ii] * output_w[ii] * channel_1[ii]

            conv_complexity = (
                input_c
                * torch.ceil((output_w[ii].float() / PE_NUM_X))
                * PE_NUM_X
                * torch.ceil(output_w[ii].float() / PE_NUM_Y)
                * PE_NUM_Y
                * channel_1[ii]
                * 2
                / 1000000000
            )
            conv_complexity = kernels[ii] * kernels[ii] * conv_complexity
            conv_bw = (
                kernels[ii] * kernels[ii] * input_c * channel_1[ii] * torch.ceil((output_w[ii].float() / PE_NUM_Y)) / 8
                + output_size * 2
                + channel_1[ii] * 4
            ) / 1000000000

            latency_singles[ii] = cal_act_lantency(
                conv_complexity, conv_bw, alpha=alpha, bw=bw, max_perform=max_perform
            )

        elif platform == "cpu":
            conv_complexity = op_singles[ii]
            parm = (kernels[ii] ** 2) * channel_1[ii - 1] * channel_1[ii]
            input_size = (output_w[ii - 1] ** 2) * channel_1[ii - 1]
            output_size = (output_w[ii] ** 2) * channel_1[ii]
            if kernels[ii] == 3:
                type = 3
            elif kernels[ii] == 1:
                type = 2
            x = get_bw_feature(
                [
                    output_w[ii - 1],
                    channel_1[ii - 1],
                    output_w[ii],
                    channel_1[ii],
                    conv_complexity,
                    parm,
                    input_size,
                    output_size,
                ],
                type,
                embed_dim_1,
                embed_dim_2,
                device=args.device,
            )
            conv_bw = torch.pow(2, model(x))
            latency_conv = cal_act_lantency(conv_complexity, conv_bw, alpha=alpha, bw=bw, max_perform=max_perform)

            latency_singles[ii] = latency_conv

        elif platform == "gpu":
            pass

    if args.dataset == "cifar100" or args.dataset == "imagenet100":
        class_num = 100
    elif args.dataset == "cifar10":
        class_num = 10
    elif args.dataset == "whu-19":
        class_num = 19
    elif args.dataset == "ucm":
        class_num = 21

    conv_complexity = channel_1[count_1 - 1] * class_num * 2
    parm = channel_1[count_1 - 1] * class_num
    input_size = channel_1[count_1 - 1]
    output_size = class_num

    # print(f"channel_1[count_1-1]:{channel_1[count_1-1]}")
    # print(f"x:{x}")
    if platform != "fpga":
        x = get_bw_feature(
            [1, channel_1[count_1 - 1], 1, class_num, conv_complexity, parm, input_size, output_size],
            1,
            embed_dim_1,
            embed_dim_2,
            device=args.device,
        )
        conv_bw = torch.pow(2, model(x))
        # print(f"conv_bw:{conv_bw}")
        latency_conv = cal_act_lantency(conv_complexity, conv_bw, alpha=alpha, bw=bw, max_perform=max_perform)
        latency_total = (latency_singles.sum() + latency_conv) * (10**3)
    else:
        latency_total = (latency_singles.sum()) * (10**3)

    # print(f"latency_total:{latency_total}")
    if math.isnan(latency_total.cpu().item()):
        print(f"latency_singles:{latency_singles}")
        print(f"latency_fc:{latency_conv}")
    return latency_total


def get_latency_real_time(args, cand):
    all_layer_latency = []

    if args.latency_mode == "onnx":
        num_list = get_index(args.latency_txt_path)
        sorted_num_list = sorted(num_list)
        total_items = len(sorted_num_list)

        # 计算进度节点
        progress_nodes = [int(total_items * i / 10) for i in range(1, 11)]

        print(f"开始处理ONNX模式延迟测试，共{total_items}个架构")

        for ii, single_num in enumerate(sorted_num_list):
            single_latency_path = args.latency_txt_path + "reactnet_" + str(single_num) + ".txt"
            all_layer_latency.append(get_final_latency(single_latency_path))

            # 检查是否达到进度节点
            if ii + 1 in progress_nodes:
                progress = int((ii + 1) / total_items * 100)
                node_num = progress_nodes.index(ii + 1) + 1
                print(f"进度节点 {node_num}/10: 已完成 {ii + 1}/{total_items} ({progress}%)")
    else:
        total_items = len(cand)

        # 计算进度节点
        progress_nodes = [int(total_items * i / 10) for i in range(1, 11)]

        print(f"开始处理PyTorch模式延迟测试，共{total_items}个架构，设备: {args.latency_device}")

        for ii, single_num in enumerate(cand):
            all_layer_latency.append(measure_latency_pytorch(cand[ii], args, args.latency_device))

            # 检查是否达到进度节点
            if ii + 1 in progress_nodes:
                progress = int((ii + 1) / total_items * 100)
                node_num = progress_nodes.index(ii + 1) + 1
                print(f"进度节点 {node_num}/10: 已完成 {ii + 1}/{total_items} ({progress}%)")

    print("延迟测试完成！")
    return all_layer_latency


def measure_latency_pytorch(cand, args, device):
    """Time the actual candidate model in PyTorch; milliseconds per batch."""
    from flashnas.models import build_model

    if args.dataset != "imagenet100":
        raise ValueError("This release provides ImageNet-100 candidate networks only")
    if args.latency_iterations < 1 or args.latency_batch_size < 1 or args.latency_warmup < 0:
        raise ValueError("Invalid latency measurement counts")
    device_obj = torch.device(device)
    model = build_model(cand.detach().cpu().tolist(), num_classes=100, stage=2).to(device_obj).eval()
    input_tensor = torch.randn(args.latency_batch_size, 3, args.search_width, args.search_width, device=device_obj)
    with torch.inference_mode():
        for _ in range(args.latency_warmup):
            model(input_tensor)
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
        start = time.perf_counter()
        for _ in range(args.latency_iterations):
            model(input_tensor)
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
    latency_ms = (time.perf_counter() - start) * 1000 / args.latency_iterations
    return torch.tensor([latency_ms], device=args.device)


def get_index(folder_path):
    num_list = []
    for filename in os.listdir(folder_path):
        # print(f"filename:{filename}")
        if filename.startswith("reactnet_") and filename.endswith(".txt"):
            try:
                index_str = filename.split("_")[1]
                index_str = index_str.split(".")[0]
                print(f"index_str:{index_str}")
                index = int(index_str)
                num_list.append(index)
            except (IndexError, ValueError):
                continue
    return num_list


def get_final_latency(file):
    num = 0
    with open(file, "r") as file:
        for line in file:
            if "run avg_time:" in line:
                start_index = line.find("run avg_time:") + len("run avg_time:")
                end_index = line.find("ms")
                num_str = line[start_index:end_index].strip()
                num = float(num_str)
        if num != 0:
            return num
        else:
            print("Can't get_final_latency.")
            exit()


def to_one_hot(value, num_classes, offset):
    """
    to_one_hot(): 一个特征做one hot编码
    """
    one_hot_vector = [0] * num_classes
    one_hot_vector[value + offset] = 1
    return one_hot_vector


def feature2vec(
    layer_len_max, channel_, conv_, stride_, flops_single, bops_single, flparm_single, bparm_single, count_1
):
    """
    feature2vec(): 特征转向量
    """
    all_list = []
    for ii in range(layer_len_max):
        if ii < count_1:
            conv_one_hot = to_one_hot(conv_[ii], 3, 0)
            single_list = (
                [channel_[ii]]
                + conv_one_hot
                + [flops_single[ii]]
                + [bops_single[ii]]
                + [flparm_single[ii]]
                + [bparm_single[ii]]
            )
        else:
            conv_one_hot = to_one_hot(2, 3, 0)
            single_list = [0] + conv_one_hot + [0] + [0] + [0] + [0]
        all_list.append(single_list)
    return all_list


def cal_n_multiply(n, lst):
    """
    cal_n_multiply(): 求前n个的乘积
    """
    result = 1
    for num in lst[:n]:
        result *= num
    return result


def cal_conv_flops_parm(channel_out, channel_in, width_out, kernel):
    flops = channel_out * channel_in * width_out * width_out * kernel * kernel
    parm = kernel * kernel * channel_in * channel_out + channel_out
    return flops, parm


def cal_bn_flops_parm(channel_out, width_out):
    flops = 4 * channel_out * width_out * width_out
    parm = 4 * channel_out
    return flops, parm


def cal_total_flops_parm(channel_, conv_, stride_, count_1, acc_width):
    flops_total = 0
    bops_total = 0
    flparm_total = 0
    bparm_total = 0
    width_origin = acc_width
    flops_single_array = []
    bops_single_array = []
    flparm_single_array = []
    bparm_single_array = []
    for ii in range(count_1):
        width_single = width_origin / cal_n_multiply(ii + 1, stride_)
        # print(f"width_single:{width_single}")
        if conv_[ii] == 0:
            kernel_single = 3
        else:
            kernel_single = 1
        if ii == 0:
            conv_flops, conv_parm = cal_conv_flops_parm(channel_[ii], 3, width_single, kernel_single)
            bn_flops, bn_parm = cal_bn_flops_parm(channel_[ii], width_single)
            other_flops = 0
            flops_single = conv_flops + bn_flops + other_flops
            bops_single = 0
            flparm_single = conv_parm + bn_parm
            bparm_single = 0
        else:
            conv_flops, conv_parm = cal_conv_flops_parm(channel_[ii], 3, width_single, kernel_single)
            bn_flops, bn_parm = cal_bn_flops_parm(channel_[ii], width_single)
            if stride_[ii] == 2:
                pool_flops = (
                    width_single * width_single * stride_[ii] * stride_[ii] * channel_[ii]
                    + width_single * width_single * channel_[ii]
                )
            else:
                pool_flops = 0
            other_flops = (
                width_single * width_single * stride_[ii] * stride_[ii] * channel_[ii]
                + width_single * width_single * channel_[ii] * 4
                + pool_flops
            )
            other_parm = channel_[ii] * 4
            flops_single = bn_flops + other_flops
            bops_single = conv_flops
            flparm_single = bn_parm + other_parm
            bparm_single = conv_parm
            # print(f"No:{ii}, flops_single:{flops_single:2f}, bops_single:{bops_single:2f}, width_single:{width_single}",
            #     f"width_origin:{width_origin}, cal_n_multiply:{cal_n_multiply(ii+1, stride_)}")

        flops_single_array.append(flops_single)
        bops_single_array.append(bops_single)
        flparm_single_array.append(flparm_single)
        bparm_single_array.append(bparm_single)

        flops_total = flops_total + flops_single
        bops_total = bops_total + bops_single
        flparm_total = flparm_total + flparm_single
        bparm_total = bparm_total + bparm_single
    flops_total = flops_total + width_single * width_single * channel_[-1] + channel_[-1] * 100
    return (
        flops_total,
        bops_total,
        flparm_total,
        bparm_total,
        flops_single_array,
        bops_single_array,
        flparm_single_array,
        bparm_single_array,
    )


def find_channel_different_count(stage_out_channels, count, dataset=None):
    """
    find_channel_different_count():
        找到channel当中有多少不同的元素
        并输出相应的个数的字典
    """

    if dataset == "imagenet100":
        unique_channels_num = 6
    elif dataset in ["cifar10", "cifar100"]:
        unique_channels_num = 4
    else:
        print("Please input the correct dataset!")
        exit()

    count_dict = {}
    # print(f"stage_out_channels 1:{stage_out_channels}")
    stage_out_channels = stage_out_channels[:count]
    # print(f"stage_out_channels 2:{stage_out_channels}")
    if type(stage_out_channels) == list:
        pass
    else:
        stage_out_channels = stage_out_channels.cpu().numpy().tolist()
    for channel in stage_out_channels:
        if channel in count_dict:
            count_dict[channel] += 1
        else:
            count_dict[channel] = 1

    # 找出所有出现过的不同元素
    unique_channels = list(count_dict.keys())

    result_list = []
    for channel in unique_channels:
        result_list.append(count_dict[channel])

    # print(result_list)

    # 确保列表长度为6，如果不足6个，用0填充
    while len(result_list) < unique_channels_num:
        result_list.append(0)

    return result_list
