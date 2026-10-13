# Support both module execution and direct script invocation.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
import os
import os.path as osp
import random
import time
import warnings

import torch
import torch.nn as nn

import net_predict.net as models
from net_predict.net.Predict_RNN import *

# import models as models
from utils import cand2tuple, get_logger, tuple2cand
from utils.calculate import *
from utils.cand import *
from utils.config import Config
from utils.pseudo import *

model_config = Config()
layer_len_max = model_config.layer_len_max
embed_dim = model_config.embed_dim
hidden_size = model_config.hidden_size
dropout = model_config.dropout

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str, required=True)
parser.add_argument("--mode", type=str, default="Search", help="Only-Init | Search", choices=["Only-Init", "Search"])
parser.add_argument("--arch", metavar="ARCH", default="Predict_harsbnn")
parser.add_argument("--model_path", type=str, default="./net_predict_output/")

parser.add_argument("--max-epochs", type=int, default=20)
parser.add_argument("--population-num", type=int, default=512)
parser.add_argument("--m-prob", type=float, default=0.2)
parser.add_argument("--crossover-num", type=int, default=128)
parser.add_argument("--mutation-num", type=int, default=128)
parser.add_argument("--layer_len_start", type=int, default=6)
parser.add_argument("--layer_len_end", type=int, default=32)
parser.add_argument("--baseline_arch", type=str, default="mobilenet", choices=["mobilenet", "resnet18", "resnet20"])
parser.add_argument(
    "--dataset", type=str, default="imagenet100", choices=["imagenet100", "cifar10", "cifar100", "whu-19", "ucm"]
)
parser.add_argument("--acc_width", type=int, default=224, choices=[32, 224, 416])
parser.add_argument("--search_width", type=int, default=224, choices=[32, 224, 416])
parser.add_argument("--ops-min", type=float, default=40)
parser.add_argument("--ops-max", type=float, default=250)
parser.add_argument("--parm-min", type=float, default=50)
parser.add_argument("--parm-max", type=float, default=100)
parser.add_argument("--latency-min", type=float, default=8)
parser.add_argument("--latency-max", type=float, default=15)
parser.add_argument("--interval", type=float, default=10)
parser.add_argument("--gpu", default="5", type=str, help="GPU id to use.")
parser.add_argument("--seed", default=0, type=int, help="seed for initializing training.")
parser.add_argument("--first_float", default=1, type=int)
parser.add_argument("--print_log", default=0, type=int)
parser.add_argument("--init_output_path", default="./1-output.txt", type=str)
parser.add_argument(
    "--save_epochs", type=str, default="", help='Comma-separated list of epochs to save checkpoint (e.g., "5,10,15")'
)
parser.add_argument(
    "--nsga2", action="store_true", default=False, help="Enable NSGA-II algorithm for Pareto front search"
)
parser.add_argument(
    "--time_mode",
    type=str,
    default="None",
    choices=["None", "real_measure", "predict", "None_and_real"],
    help="Time measurement mode: None (use OPs), real_measure (PyTorch measurement), predict (prediction model)",
)
parser.add_argument("--latency_batch_size", type=int, default=1, help="Batch size for latency measurement")
parser.add_argument("--latency_warmup", type=int, default=20, help="Warmup iterations for latency measurement")
parser.add_argument("--latency_iterations", type=int, default=30, help="Iterations for latency measurement")
parser.add_argument("--latency_device", type=str, default="cuda:0", choices=["cpu", "cuda:0"])
parser.add_argument("--fpga", action="store_true", default=False, help="Enable FPGA search")


def seed(seed=0):
    import os
    import random
    import sys

    import numpy as np
    import torch

    sys.setrecursionlimit(100000)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class EvolutionSearcher:
    def __init__(self, model, logger, args, device):
        self.model = model
        self.device = device
        self.args = args
        self.max_epochs = args.max_epochs
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.ops_min = args.ops_min
        self.ops_max = args.ops_max
        self.parm_min = args.parm_min
        self.parm_max = args.parm_max
        self.latency_min = args.latency_min
        self.latency_max = args.latency_max
        if args.time_mode in ["None", "None_and_real"]:
            self.interval_step = int((self.ops_max - self.ops_min) // args.interval)
        else:
            self.interval_step = (self.latency_max - self.latency_min) / args.interval
        self.mode = args.mode
        self.nsga2 = args.nsga2  # 新增：NSGA-II控制参数
        self.time_mode = args.time_mode
        if self.args.dataset == "imagenet100":
            self.channels = torch.tensor(
                [
                    [16, 24, 32, 48, 64],
                    [32, 48, 64, 96, 128],
                    [64, 96, 128, 160, 192],
                    [160, 192, 256, 320, 384],
                    [320, 384, 512, 640, 768],
                    [640, 768, 1024, 1280, 1536],
                ]
            ).to(self.device)
            self.stage_num = [
                range(2, 10 + 1),
                range(2, 10 + 1),
                range(2, 10 + 1),
                range(4, 20 + 1),
                range(6, 20 + 1),
                range(4, 20 + 1),
            ]
        elif self.args.dataset in ["cifar10", "cifar100", "whu-19", "ucm"]:
            if self.args.baseline_arch == "resnet18":
                # self.channels = torch.tensor([[32, 48, 64, 80, 96],
                #                                 [96, 112, 128, 144, 160],
                #                                 [192, 224, 256, 288, 320],
                #                                 [384, 448, 512, 576, 640]]).to(device)
                self.channels = torch.tensor(
                    [
                        [32, 48, 56, 64, 72, 80, 96, 104, 108, 112, 116],
                        [96, 112, 120, 128, 136, 144, 160, 176, 184, 188, 192],
                        [192, 224, 240, 256, 272, 288, 320, 336, 344, 352, 384],
                        [368, 376, 384, 400, 416, 448, 480, 512, 544, 576, 640],
                    ]
                ).to(device)
                self.stage_num = [range(3, 7 + 1), range(2, 9 + 1), range(2, 9 + 1), range(2, 7 + 1)]
            elif self.args.baseline_arch == "resnet20":
                self.channels = torch.tensor([[8, 12, 16, 20, 24], [24, 28, 32, 36, 40], [48, 56, 64, 72, 80]]).to(
                    device
                )
                self.stage_num = [range(3, 10 + 1), range(3, 10 + 1), range(3, 10 + 1)]

        self.channel_shape_0 = self.channels.shape[0]
        self.channel_shape_1 = self.channels.shape[1]

        self.channels_extra = torch.cat(
            (self.channels, torch.tensor([-1] * self.channel_shape_1).view(-1, self.channel_shape_1).to(self.device)),
            dim=0,
        )

        if not osp.exists(args.logdir):
            os.makedirs(args.logdir)

        self.logger = logger

        self.checkpoint_name = os.path.join(args.logdir, "info.pth.tar")
        self.m = EvolutionCand(
            args,
            device=self.device,
            stage_num=self.stage_num,
            channels=self.channels,
            channels_extra=self.channels_extra,
            channel_shape_0=self.channel_shape_0,
            channel_shape_1=self.channel_shape_1,
        )

        self.final_candidates = []
        self.final_candidates_ops = []
        self.final_candidates_parm = []
        self.final_candidates_acc = []
        self.final_candidates_latency = []
        self.vis_dict = {}
        self.epoch = 0
        self.candidates = []
        self.pareto_global = {}

        # 新增：解析保存的epoch列表
        if args.save_epochs:
            self.save_epochs = [int(x.strip()) for x in args.save_epochs.split(",")]
        else:
            self.save_epochs = []
        # 新增：存储指定epoch的信息
        self.epoch_candidates = {}  # 存储每个指定epoch的candidates
        self.epoch_ops = {}  # 存储每个指定epoch的ops
        self.epoch_parm = {}  # 存储每个指定epoch的parm
        self.epoch_acc = {}  # 存储每个指定epoch的acc
        self.epoch_latency = {}  # 存储每个指定epoch的latency

    def save_cand_checkpoint(self):
        save_checkpoint(
            {
                "final_candidates": self.final_candidates,
                "final_candidates_acc": self.final_candidates_acc,
                "final_candidates_ops": self.final_candidates_ops,
                "final_candidates_parm": self.final_candidates_parm,
                "final_candidates_latency": self.final_candidates_latency,
                # 新增：保存指定epoch的信息
                "save_epochs": self.save_epochs,
                "epoch_candidates": self.epoch_candidates,
                "epoch_ops": self.epoch_ops,
                "epoch_parm": self.epoch_parm,
                "epoch_acc": self.epoch_acc,
                "epoch_latency": self.epoch_latency,
            },
            filename=os.path.join(self.args.logdir, "info.pth"),
        )

    def get_stage_num(self, channel_1, channels_select):
        count = torch.zeros_like(channels_select).to(self.device)
        for ii in range(channels_select.shape[0]):
            count[ii] = torch.sum(channel_1 == channels_select[ii]).item()
        return count

    def get_channel_index(self, channel_1):
        unique_channel = torch.unique(channel_1, sorted=True)
        indices = torch.zeros((self.channels.shape[0])).int().to(self.device)

        for ii in range(self.channels.shape[0]):
            target_row = self.channels[ii, :]
            indices[ii] = (target_row == unique_channel[ii]).nonzero(as_tuple=True)[0]

        return indices

    def is_legal(self, cand):
        cand_tuple = cand2tuple(cand)
        if cand_tuple not in self.vis_dict:
            self.vis_dict[cand_tuple] = {}
        info = self.vis_dict[cand_tuple]

        if "visited" in info:
            return False
        if "ops" not in info:
            info["parm"], info["ops"] = self.m.get_parm_ops(cand, self.args.search_width)

        if self.time_mode != "None" and "latency" not in info:
            if self.time_mode == "real_measure" or self.time_mode == "None_and_real":
                info["latency"] = measure_latency_pytorch(cand, self.args, self.device)
            elif self.time_mode == "predict":
                info["latency"] = measure_latency_predict(cand, self.args, self.device)
            if info["latency"] < self.latency_min or info["latency"] > self.latency_max:
                info["visited"] = True
                return False
        else:
            info["latency"] = torch.tensor([0]).to(self.device)

        if self.mode == "Only-Init":
            info["acc"] = torch.tensor([0]).to(self.device)
        else:
            try:
                info["acc"] = get_cand_acc_model(self.model, cand, self.args, self.device)
            except:
                print("Try acc is error.")
                info["acc"] = torch.tensor([0]).to(self.device)
            if info["acc"] >= 1 and self.args.dataset in ["cifar10", "cifar100"]:
                info["acc"] = torch.tensor([0]).to(self.device)
            elif info["acc"] >= 1:
                info["acc"] = torch.tensor([0]).to(self.device)

        info["visited"] = True
        return True

    def check_random(self, cand):

        cand_shape = int(cand.shape[0] / 3)
        channel_1 = cand[:cand_shape]
        stride_1 = cand[cand_shape : cand_shape * 2]
        conv_type_1 = cand[cand_shape * 2 :]
        count_1 = (channel_1 != -1).sum().item()
        count_2 = (stride_1 != -1).sum().item()
        count_3 = (conv_type_1 != -1).sum().item()
        if count_1 != count_2 or count_2 != count_3 or count_1 != count_3:
            print("Error cand:")
            print(cand)
            exit()

        chanel_count_list = find_channel_different_count(channel_1, count_1, dataset=self.args.dataset)
        # print(chanel_count_list)
        found_1 = 0
        for ii in range(len(self.stage_num)):
            if chanel_count_list[ii] < self.stage_num[ii][0]:
                continue
            if ii == len(self.stage_num) - 1:
                found_1 = 1

        if not self.is_legal(cand) or found_1 == 0:
            return False
        else:
            return True

    def get_random(self, num):
        cnt = 0
        while len(self.candidates) < num:
            if cnt == 0:
                cnt += 1
                cand = self.m.smallest_cand
            elif cnt == 1:
                cnt += 1
                cand = self.m.biggest_cand
            else:
                cand = self.m.get_random_range_cand(self.ops_min, self.ops_max, self.parm_min, self.parm_max)

            if self.check_random(cand) == False:
                continue
            cand_tuple = cand2tuple(cand)
            self.candidates.append(cand_tuple)

            # 修改日志输出以显示延迟信息
            info = self.vis_dict[cand_tuple]
            self.logger.info(
                "random {}/{}: {}, parm:{}, ops:{}, latency:{:.4f}ms, acc:{}".format(
                    len(self.candidates),
                    num,
                    cand_tuple,
                    info["parm"],
                    info["ops"],
                    info["latency"].cpu().item(),
                    info["acc"],
                )
            )

    def get_max_cand(self, max_cand_num=50):
        if self.args.dataset == "imagenet100":
            combinations_array = torch.tensor(
                [
                    [10, 4, 4, 4, 5, 5],
                    [4, 10, 4, 4, 5, 5],
                    [4, 4, 10, 4, 5, 5],
                    [2, 2, 2, 20, 3, 3],
                    [2, 2, 2, 3, 20, 3],
                    [2, 2, 2, 3, 3, 20],
                    [3, 3, 3, 15, 4, 4],
                    [3, 3, 3, 4, 15, 4],
                    [3, 3, 3, 4, 4, 15],
                ]
            ).to(self.device)
        elif self.args.dataset in ["cifar10", "cifar100", "whu-19", "ucm"] and self.args.baseline_arch == "resnet18":
            combinations_array = torch.tensor(
                [
                    [self.args.layer_len_end - 6, 2, 2, 2],
                    [3, self.args.layer_len_end - 7, 2, 2],
                    [3, 2, self.args.layer_len_end - 7, 2],
                    [3, 2, 2, self.args.layer_len_end - 7],
                    [self.args.layer_len_end - 9, 3, 3, 3],
                    [4, self.args.layer_len_end - 10, 3, 3],
                    [4, 3, self.args.layer_len_end - 10, 3],
                    [4, 3, 3, self.args.layer_len_end - 10],
                ]
            ).to(self.device)
        cnt = 0
        while cnt < max_cand_num:
            cand = self.m.get_random_max_cand(combinations_array)
            if self.check_random(cand) == False:
                continue
            else:
                cnt += 1
            cand_tuple = cand2tuple(cand)
            self.candidates.append(cand_tuple)
            self.logger.info(
                "random_max {}/{}: {}, parm:{}, ops:{}, acc:{}".format(
                    len(self.candidates),
                    max_cand_num,
                    cand_tuple,
                    self.vis_dict[cand_tuple]["parm"],
                    self.vis_dict[cand_tuple]["ops"],
                    self.vis_dict[cand_tuple]["acc"],
                )
            )

    def get_mutation(self):
        res = []
        while len(res) < self.mutation_num:
            ori_cand = tuple2cand(random.choice(list(self.pareto_global.values())))
            cand = ori_cand.clone()
            cand = cand.view(3, -1).to(self.device)
            channel_1 = cand[0]
            stride_1 = cand[1]
            conv_type_1 = cand[2]
            conv_type_2 = torch.zeros_like(conv_type_1).int().to(self.device)
            channel_index_1 = self.get_channel_index(channel_1[channel_1 != -1])
            channel_index_2 = torch.cat((channel_index_1, torch.tensor([0]).int().to(self.device)), dim=0)
            stage_num_1 = self.get_stage_num(
                channel_1, self.channels[torch.arange(self.channels.shape[0]).view(-1), channel_index_1]
            )
            stage_num_2 = torch.cat((stage_num_1, torch.tensor([0]).int().to(self.device)), dim=0)

            found_1 = 0
            while True:
                for stage_num in range(len(self.stage_num)):
                    if random.random() < self.m_prob:
                        d = random.choice(self.stage_num[stage_num])
                        stage_num_2[stage_num] = d

                count_1 = stage_num_2.sum().item()
                if self.args.layer_len_start <= count_1 and count_1 <= self.args.layer_len_end:
                    break
                else:
                    found_1 += 1
                    if found_1 >= self.mutation_num * 10:
                        stage_num_2[: len(self.stage_num)] = stage_num_1
                        print("found_1 >= self.mutation_num*10!!!!")
                        break
                # print(f"found_1:{found_1}")

            found_3 = False
            while not found_3:
                for stage_num in range(len(self.stage_num)):
                    if random.random() < self.m_prob:
                        d = torch.randint(0, self.channels_extra.shape[1], (1, 1)).to(self.device)
                        channel_index_2[stage_num] = d
                    else:
                        channel_index_2[stage_num] = channel_index_1[stage_num]
                channels_select = self.channels_extra[
                    torch.arange(self.channels_extra.shape[0]).view(-1), channel_index_2
                ]

                for ii in range(1, len(self.stage_num)):
                    if channels_select[ii] <= channels_select[ii - 1]:
                        break
                    if ii == len(self.stage_num) - 1:
                        found_3 = True
                # print(f"found_3:{found_3}")

            count_2 = stage_num_2[: len(self.stage_num)].sum().item()
            # print(count_2)
            stage_num_2[-1] = self.args.layer_len_end - count_2
            # print(stage_num_2)
            channel_2 = torch.repeat_interleave(channels_select, stage_num_2)
            stride_2 = self.m.generate_stride_combinations(channel_2, count_2, self.args.layer_len_end).to(self.device)

            accumulator_2 = 0
            accumulator_1 = 0
            for ii in range(len(self.stage_num)):
                accumulator_min = torch.min(stage_num_2[ii], stage_num_1[ii])
                conv_type_2[accumulator_2 : accumulator_2 + accumulator_min] = conv_type_1[
                    accumulator_1 : accumulator_1 + accumulator_min
                ]
                accumulator_2 += stage_num_2[ii]
                accumulator_1 += stage_num_1[ii]
            conv_type_2[count_2 : self.args.layer_len_end] = torch.tensor(
                [-1] * (self.args.layer_len_end - count_2), dtype=int
            )

            found_2 = False
            while not found_2:
                for ii in range(count_2):
                    if random.random() < self.m_prob:
                        if self.args.dataset == "imagenet100":
                            d = random.choice([0, 1])
                        elif self.args.dataset in ["cifar10", "cifar100", "whu-19", "ucm"]:
                            d = random.choice([0])
                        conv_type_2[ii] = d
                check_feature = self.m.check_the_feature(channel_2, conv_type_2, stride_2, channels_num=count_2)
                check_conv_type = self.m.check_the_conv_type_array(
                    channel_2, conv_type_2, stride_2, channels_num=count_2
                )
                if check_conv_type == 0:
                    conv_type_2 = self.m.change_the_conv_type_array(
                        channel_2, conv_type_2, stride_2, channels_num=count_2
                    )
                if check_feature:
                    found_2 = True
            cand = torch.cat([channel_2, stride_2, conv_type_2], dim=0)
            cand_shape = int(cand.shape[0] / 3)
            channel_1 = cand[:cand_shape]
            stride_1 = cand[cand_shape : cand_shape * 2]
            conv_type_1 = cand[cand_shape * 2 :]
            count_1 = (channel_1 != -1).sum().item()
            count_2 = (stride_1 != -1).sum().item()
            count_3 = (conv_type_1 != -1).sum().item()
            if count_1 != count_2 or count_2 != count_3 or count_1 != count_3:
                print(cand)
                exit()

            parm, ops = self.m.get_parm_ops(cand, self.args.search_width)
            if not (self.ops_min <= ops <= self.ops_max):
                continue
            if not (self.parm_min <= parm <= self.parm_max):
                continue
            if not self.is_legal(cand):
                print("get_crossover is_legal")
                continue
            cand_tuple = cand2tuple(cand)
            res.append(cand_tuple)
            if self.args.print_log == 1:
                self.logger.info("crossover {}/{}, {}".format(len(res), self.crossover_num, cand_tuple))
        return res

    def get_mask_channel(
        self, mask, cand1, cand2, channel_index_1, channel_index_2, channels_select_1, channels_select_2
    ):
        found = False
        mask_channel_1 = mask * cand1[0]
        mask_channel_2 = (1 - mask) * cand2[0]
        new_mask_channel_1 = mask_channel_1.to(self.device)
        new_mask_channel_2 = mask_channel_2.to(self.device)
        channel_index_new = torch.zeros_like(channels_select_1).int().to(self.device)

        for ii in range(len(self.stage_num)):
            count1 = torch.sum(torch.eq(mask_channel_1, channels_select_1[ii])).item()
            count2 = torch.sum(torch.eq(mask_channel_2, channels_select_2[ii])).item()
            if count1 >= count2:
                channel_index_new[ii] = channel_index_1[ii]
                new_mask_channel_2[new_mask_channel_2 == channels_select_2[ii]] = channels_select_1[ii]
            else:
                channel_index_new[ii] = channel_index_2[ii]
                new_mask_channel_1[new_mask_channel_1 == channels_select_1[ii]] = channels_select_2[ii]

        channels_select = self.channels[torch.arange(self.channels.shape[0]).view(-1), channel_index_new]
        for ii in range(1, len(self.stage_num)):
            if channels_select[ii] <= channels_select[ii - 1]:
                break
            if ii == len(self.stage_num) - 1:
                found = True

        new_mask_channel_3 = new_mask_channel_1 + new_mask_channel_2

        return new_mask_channel_3, channel_index_new, found

    def get_crossover(self):
        res = []
        while len(res) < self.crossover_num:
            cand1 = tuple2cand(random.choice(list(self.pareto_global.values())))
            cand1 = cand1.view(3, -1).to(self.device)
            channel_1 = cand1[0]
            channel_index_1 = self.get_channel_index(channel_1[channel_1 != -1])
            channels_select_1 = self.channels[torch.arange(self.channels.shape[0]).view(-1), channel_index_1]
            stage_num_1 = self.get_stage_num(channel_1, channels_select_1)

            cand2 = tuple2cand(random.choice(list(self.pareto_global.values())))
            cand2 = cand2.view(3, -1).to(self.device)
            channel_2 = cand2[0]
            channel_index_2 = self.get_channel_index(channel_2[channel_2 != -1])
            channels_select_2 = self.channels[torch.arange(self.channels.shape[0]).view(-1), channel_index_2]
            stage_num_2 = self.get_stage_num(channel_2, channels_select_2)

            d_list = []
            for i in range(len(self.stage_num)):
                d1 = stage_num_1[i]
                d2 = stage_num_2[i]
                d_list.append(random.choice([d1, d2]))
            mask = torch.rand_like(cand1[0].float()).round().int()
            cand = mask * cand1 + (1 - mask) * cand2
            channel_3, channel_index_3, found_4 = self.get_mask_channel(
                mask, cand1, cand2, channel_index_1, channel_index_2, channels_select_1, channels_select_2
            )
            if found_4 == False:
                continue

            conv_type_3 = cand[2]

            channels_select_3 = self.channels[torch.arange(self.channels.shape[0]).view(-1), channel_index_3]
            stage_num_3 = self.get_stage_num(channel_3, channels_select_3)

            channel_3_tmp = channel_3[channel_3 != -1]
            channel_3_final, channel_3_indices = torch.sort(channel_3_tmp)
            conv_type_3 = conv_type_3[channel_3 != -1]

            if torch.any(stage_num_3 == 0) == True:
                continue

            count_3 = channel_3_tmp.shape[0]
            if count_3 < self.args.layer_len_start or count_3 > self.args.layer_len_end:
                print(f"get_crossover found_4:{found_4}")
                continue

            channel_3_final = torch.concat(
                [channel_3_final, torch.tensor([-1] * (self.args.layer_len_end - count_3), dtype=int).to(self.device)],
                dim=0,
            )
            stride_3 = self.m.generate_stride_combinations(channel_3_final, count_3, self.args.layer_len_end).to(
                self.device
            )
            conv_type_3 = torch.concat(
                [
                    conv_type_3[channel_3_indices],
                    torch.tensor([-1] * (self.args.layer_len_end - count_3), dtype=int).to(self.device),
                ],
                dim=0,
            )
            if self.m.check_the_feature(channel_3_final, conv_type_3, stride_3, channels_num=count_3) == 0:
                # print(f"get_crossover check_the_feature:{self.m.check_the_feature(channel_3_final, conv_type_3, stride_3, channels_num=count_3)}")
                continue
            if self.m.check_the_conv_type_array(channel_3_final, conv_type_3, stride_3, channels_num=count_3) == 0:
                condition_mask = (conv_type_3[:count_3] == 1) & (stride_3[:count_3] == 2)
                condition_index = condition_mask.nonzero()

                for ii in condition_index:
                    conv_type_3[ii] = 0
                conv_type_3[0] = 0
            cand = torch.cat([channel_3_final, stride_3, conv_type_3], dim=0)
            cand_shape = int(cand.shape[0] / 3)
            channel_1 = cand[:cand_shape]
            stride_1 = cand[cand_shape : cand_shape * 2]
            conv_type_1 = cand[cand_shape * 2 :]
            count_1 = (channel_1 != -1).sum().item()
            count_2 = (stride_1 != -1).sum().item()
            count_3 = (conv_type_1 != -1).sum().item()
            if count_1 != count_2 or count_2 != count_3 or count_1 != count_3:
                print(cand)
                exit()

            parm, ops = self.m.get_parm_ops(cand, self.args.search_width)
            if not (self.ops_min <= ops <= self.ops_max):
                continue
            if not (self.parm_min <= parm <= self.parm_max):
                continue
            if not self.is_legal(cand):
                print("get_crossover is_legal")
                continue
            cand_tuple = cand2tuple(cand)
            res.append(cand_tuple)
            if self.args.print_log == 1:
                self.logger.info("crossover {}/{}, {}".format(len(res), self.crossover_num, cand_tuple))
        return res

    def update_frontier(self):
        # 创建一个反向映射字典，用于快速查找候选架构是否存在
        existing_cand_map = {}
        for existing_f, existing_cand in self.pareto_global.items():
            existing_cand_map[existing_cand] = existing_f

        for cand_tuple in self.candidates:
            acc = self.vis_dict[cand_tuple]["acc"]

            if self.time_mode not in ["None", "None_and_real"]:
                # 使用延迟作为优化目标
                latency = self.vis_dict[cand_tuple]["latency"]

                # 直接使用浮点数运算，避免整数截断问题
                # 先将latency转换为Python浮点数，然后进行运算
                latency_value = latency.cpu().item()
                f = round(latency_value / float(self.interval_step)) * float(self.interval_step)
                # f = torch.tensor([f_value]).to(latency.device)

                # print(f"self.interval_step:{self.interval_step}")
                # print(f"f:{f}")
            else:
                # 使用OPs作为优化目标
                ops = self.vis_dict[cand_tuple]["ops"]
                f = int(torch.round(ops / self.interval_step) * self.interval_step)

            # 使用字典快速查找，时间复杂度从O(n)降到O(1)
            if cand_tuple in existing_cand_map:
                continue

            if f not in self.pareto_global or self.vis_dict[self.pareto_global[f]]["acc"] < acc:
                self.pareto_global[f] = cand_tuple

    def nsga2_selection(self):
        """NSGA-II算法：非支配排序 + 拥挤距离"""
        if len(self.candidates) == 0:
            return []

        objectives = []
        for cand_tuple in self.candidates:
            info = self.vis_dict[cand_tuple]
            if self.time_mode in ["None", "None_and_real"]:
                objectives.append([-info["acc"].cpu().item(), info["latency"].cpu().item()])
            else:
                objectives.append([-info["acc"].cpu().item(), info["ops"].cpu().item()])

        objectives = torch.tensor(objectives)

        # 非支配排序
        fronts = self._fast_non_dominated_sort(objectives)

        # 计算拥挤距离
        crowding_distances = self._crowding_distance_assignment(fronts, objectives)

        # 选择下一代种群
        selected_indices = []
        remaining = self.population_num

        for front in fronts:
            if len(selected_indices) + len(front) <= remaining:
                selected_indices.extend(front)
            else:
                # 按拥挤距离排序并选择
                front_with_distances = [(i, crowding_distances[i]) for i in front]
                front_with_distances.sort(key=lambda x: x[1], reverse=True)
                selected_from_front = [i for i, _ in front_with_distances[: remaining - len(selected_indices)]]
                selected_indices.extend(selected_from_front)
                break

        # 返回选中的候选解
        selected_candidates = [self.candidates[i] for i in selected_indices]
        return selected_candidates

    def _fast_non_dominated_sort(self, objectives):
        """快速非支配排序算法"""
        num_solutions = len(objectives)
        dominates = [[] for _ in range(num_solutions)]
        dominated_by = [0] * num_solutions
        fronts = [[]]

        for i in range(num_solutions):
            for j in range(num_solutions):
                if i == j:
                    continue
                if self._dominates(objectives[i], objectives[j]):
                    dominates[i].append(j)
                elif self._dominates(objectives[j], objectives[i]):
                    dominated_by[i] += 1

            if dominated_by[i] == 0:
                fronts[0].append(i)

        # 如果初始front为空，直接返回空列表
        if not fronts[0]:
            return fronts

        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominates[i]:
                    dominated_by[j] -= 1
                    if dominated_by[j] == 0:
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def _dominates(self, obj1, obj2):
        """判断obj1是否支配obj2"""
        # obj1在所有目标上都不比obj2差，且至少在一个目标上更好
        not_worse = all(obj1[i] <= obj2[i] for i in range(len(obj1)))
        better = any(obj1[i] < obj2[i] for i in range(len(obj1)))
        return not_worse and better

    def _crowding_distance_assignment(self, fronts, objectives):
        """计算拥挤距离"""
        num_solutions = len(objectives)
        crowding_distances = [0.0] * num_solutions

        for front in fronts:
            if len(front) == 0:
                continue

            # 对每个目标函数计算拥挤距离
            for m in range(objectives.shape[1]):
                # 按第m个目标排序
                sorted_front = sorted(front, key=lambda i: objectives[i, m])

                # 边界点的拥挤距离设为无穷大
                crowding_distances[sorted_front[0]] = float("inf")
                crowding_distances[sorted_front[-1]] = float("inf")

                # 计算中间点的拥挤距离
                min_obj = objectives[sorted_front[0], m]
                max_obj = objectives[sorted_front[-1], m]

                if max_obj - min_obj < 1e-10:  # 避免除零
                    continue

                for idx in range(1, len(sorted_front) - 1):
                    i = sorted_front[idx]
                    prev_obj = objectives[sorted_front[idx - 1], m]
                    next_obj = objectives[sorted_front[idx + 1], m]
                    crowding_distances[i] += (next_obj - prev_obj) / (max_obj - min_obj)

        return crowding_distances

    def search(self):
        self.logger.info(
            "population_num = {} mutation_num = {} "
            "crossover_num = {} max_epochs = {}".format(
                self.population_num, self.mutation_num, self.crossover_num, self.max_epochs
            )
        )
        self.logger.info("Init")
        self.get_random(self.population_num)
        # exit()

        if self.mode == "Only-Init":
            self.get_max_cand()
            if os.path.exists(self.args.init_output_path):
                os.remove(self.args.init_output_path)

            for cand_tuple in self.candidates:
                self.logger.info(
                    "OPs Stage.{}, Top-1 acc: {:.4f}, OPs: {:.2f}, Parm: {:.2f}".format(
                        cand_tuple,
                        self.vis_dict[cand_tuple]["acc"].cpu().item(),
                        self.vis_dict[cand_tuple]["ops"].cpu().item(),
                        self.vis_dict[cand_tuple]["parm"].cpu().item(),
                    )
                )
                cand_tensor = tuple2cand(cand_tuple)
                channel_tensor = cand_tensor[: cand_tensor.shape[0] // 3]
                stride_tensor = cand_tensor[cand_tensor.shape[0] // 3 : cand_tensor.shape[0] // 3 * 2]
                conv_type_tensor = cand_tensor[cand_tensor.shape[0] // 3 * 2 :]

                channel_tensor = channel_tensor[channel_tensor != -1]
                stride_tensor = stride_tensor[stride_tensor != -1]
                conv_type_tensor = conv_type_tensor[conv_type_tensor != -1]

                with open(self.args.init_output_path, "a+") as f:
                    f.write(str(channel_tensor.tolist()))
                    f.write("\n")
                    f.write(str(conv_type_tensor.tolist()))
                    f.write("\n")
                    f.write(str(stride_tensor.tolist()))
                    f.write("\n")
        else:
            self.update_frontier()

            while self.epoch < self.max_epochs:
                self.logger.info("Epoch: {}/{}".format(self.epoch, self.max_epochs))

                mutation = self.get_mutation()
                crossover = self.get_crossover()

                if self.nsga2:
                    all_candidates = self.candidates + mutation + crossover
                    self.candidates = all_candidates
                    selected_candidates = self.nsga2_selection()
                    self.candidates = selected_candidates
                    self.logger.info(
                        "NSGA-II selected {} candidates from {} total".format(
                            len(selected_candidates), len(all_candidates)
                        )
                    )
                else:
                    self.candidates = mutation + crossover

                self.update_frontier()

                ops_stages = sorted(list(self.pareto_global.keys()))

                # 新增：如果当前epoch在保存列表中，记录当前epoch的信息
                if (self.epoch + 1) in self.save_epochs:
                    current_epoch = self.epoch + 1
                    self.epoch_candidates[current_epoch] = []
                    self.epoch_ops[current_epoch] = []
                    self.epoch_parm[current_epoch] = []
                    self.epoch_acc[current_epoch] = []
                    self.epoch_latency[current_epoch] = []

                    for s in ops_stages:
                        cand_tuple = self.pareto_global[s]
                        self.epoch_candidates[current_epoch].append(cand_tuple)
                        self.epoch_acc[current_epoch].append(self.vis_dict[cand_tuple]["acc"])
                        self.epoch_ops[current_epoch].append(self.vis_dict[cand_tuple]["ops"])
                        self.epoch_parm[current_epoch].append(self.vis_dict[cand_tuple]["parm"])
                        self.epoch_latency[current_epoch].append(self.vis_dict[cand_tuple]["latency"])

                    self.logger.info(
                        "Saved checkpoint for epoch {} with {} candidates".format(
                            current_epoch, len(self.epoch_candidates[current_epoch])
                        )
                    )

                for s in ops_stages:
                    cand_tuple = self.pareto_global[s]
                    if self.epoch == self.max_epochs - 1:
                        self.final_candidates.append(cand_tuple)
                        self.final_candidates_acc.append(self.vis_dict[cand_tuple]["acc"])
                        self.final_candidates_ops.append(self.vis_dict[cand_tuple]["ops"])
                        self.final_candidates_parm.append(self.vis_dict[cand_tuple]["parm"])
                        self.final_candidates_latency.append(self.vis_dict[cand_tuple]["latency"])

                    stage_prefix = "OPs" if self.args.time_mode in ["None", "None_and_real"] else "Latency"
                    self.logger.info(
                        "{} Stage.{}, {}, Top-1 acc: {:.4f}, OPs: {:.2f}, Latency: {:.4f}ms, Parm: {:.2f}".format(
                            stage_prefix,
                            s,
                            cand_tuple,
                            self.vis_dict[cand_tuple]["acc"].cpu().item(),
                            self.vis_dict[cand_tuple]["ops"].cpu().item(),
                            self.vis_dict[cand_tuple]["latency"].cpu().item(),
                            self.vis_dict[cand_tuple]["parm"].cpu().item(),
                        )
                    )
                self.epoch += 1

            self.del_cand()

            self.save_cand_checkpoint()

    def del_cand(self, Sigma=3):
        # 计算均值和标准差
        acc_values = torch.tensor([acc.cpu().item() for acc in self.final_candidates_acc])
        mean_acc = torch.mean(acc_values)
        std_acc = torch.std(acc_values)

        # 计算3西格玛范围
        lower_bound = mean_acc - Sigma * std_acc
        upper_bound = mean_acc + Sigma * std_acc

        # 筛选出在3西格玛范围内的acc
        valid_indices = []
        for i, acc in enumerate(acc_values):
            if lower_bound <= acc <= upper_bound:
                valid_indices.append(i)

        # 保留符合条件的数据
        self.final_candidates = [self.final_candidates[i] for i in valid_indices]
        self.final_candidates_acc = [self.final_candidates_acc[i] for i in valid_indices]
        self.final_candidates_ops = [self.final_candidates_ops[i] for i in valid_indices]
        self.final_candidates_parm = [self.final_candidates_parm[i] for i in valid_indices]
        self.final_candidates_latency = [self.final_candidates_latency[i] for i in valid_indices]

        # 按acc从小到大排序
        sorted_indices = sorted(range(len(self.final_candidates_acc)), key=lambda k: self.final_candidates_acc[k])
        self.final_candidates = [self.final_candidates[i] for i in sorted_indices]
        self.final_candidates_acc = [self.final_candidates_acc[i] for i in sorted_indices]
        self.final_candidates_ops = [self.final_candidates_ops[i] for i in sorted_indices]
        self.final_candidates_parm = [self.final_candidates_parm[i] for i in sorted_indices]
        self.final_candidates_latency = [self.final_candidates_latency[i] for i in sorted_indices]

        # 修改：根据time_mode参数调整最终输出格式
        for s, cand_tuple in enumerate(self.final_candidates):
            stage_prefix = "OPs" if self.args.time_mode in ["None", "None_and_real"] else "Latency"
            self.logger.info(
                "{} Stage.{}, {}, Top-1 acc: {:.4f}, OPs: {:.2f}, Latency: {:.2f}ms, Parm: {:.2f}".format(
                    stage_prefix,
                    s,
                    cand_tuple,
                    self.final_candidates_acc[s].cpu().item(),
                    self.final_candidates_ops[s].cpu().item(),
                    self.vis_dict[cand_tuple]["latency"].cpu().item(),
                    self.final_candidates_parm[s].cpu().item(),
                )
            )


def measure_latency_predict(cand, args, device):
    if args.fpga:
        platform = "fpga"
        model_bw = None
    else:
        platform = "cpu"
        model_bw = Predict_BW(embed_dim_1=8, embed_dim_2=4, hidden_size=256).to(args.device)
        state_dict = torch.load(args.bw_output_path + "/model_best.pth")
        model_bw.load_state_dict(state_dict)
        model_bw.eval()

    latency = get_latency(
        args,
        cand,
        alpha=0.9,
        bw=12.8,
        max_perform=5160,
        platform=platform,
        baseline_arch=args.baseline_arch,
        search_width=args.search_width,
        model=model_bw,
        embed_dim_1=8,
        embed_dim_2=4,
    )

    return torch.tensor([latency]).to(args.device)


def main():
    args = parser.parse_args()
    seed(args.seed)
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    print(f"Use GPU: {args.gpu} for training")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(f"Device selected for computation: {device}")

    if not osp.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = get_logger(name="Search", log_file=osp.join(args.logdir, "search_first.log"), log_level=logging.INFO)

    t = time.time()
    # print(models.__dict__)
    model = models.__dict__[args.arch](
        embed_dim=model_config.embed_dim,
        hidden_size=model_config.hidden_size,
        dropout=model_config.dropout,
        device=device,
    ).to(device)

    # args.model_path = args.model_path + "/" + args.dataset + "/" + "CharRNN_OUR_best.pth"
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    searcher = EvolutionSearcher(model, logger, args, device=device)
    searcher.search()
    logger.info("total searching time = {:.2f} hours".format((time.time() - t) / 3600))


if __name__ == "__main__":
    main()
