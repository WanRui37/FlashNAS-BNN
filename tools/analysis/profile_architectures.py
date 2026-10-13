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
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from net_predict.net.Predict_RNN import *
from utils.calculate import *
from utils.cand import *
from utils.config import Config
from utils.dataset import *

model_config = Config()
layer_len_max = model_config.layer_len_max
embed_dim = model_config.embed_dim
hidden_size = model_config.hidden_size
dropout = model_config.dropout

parser = argparse.ArgumentParser()

parser.add_argument("--alpha", type=float, default=0.8)
parser.add_argument("--bw", type=float, default=12.8)
parser.add_argument("--max_perform", type=float, default=5734.4)
parser.add_argument("--platform", type=str, default="fpga", choices=["fpga", "cpu", "gpu"])
parser.add_argument("--precision", type=int, default="8", choices=[32, 16, 8])

parser.add_argument("--layer_len_start", type=int, default=6, choices=[3, 4, 6])
parser.add_argument("--layer_len_end", type=int, default=32, choices=[25, 28, 32])
parser.add_argument("--baseline_arch", type=str, default="resnet18", choices=["mobilenet", "resnet18", "resnet20"])
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    choices=["imagenet100", "ImageNet1000", "cifar10", "cifar100", "whu-19", "ucm"],
)
parser.add_argument("--acc_width", type=int, default=224, choices=[32, 224, 416])
parser.add_argument("--search_width", type=int, default=224, choices=[32, 224, 416])

parser.add_argument("--read_txt", type=int, default=0, choices=[0, 1])
parser.add_argument("--architecture_file", default="./1-output.txt")
parser.add_argument("--first_float", type=int, default=1, choices=[0, 1])

args = parser.parse_args()


def extract_text_from_file(files="./1-output.txt"):
    STAGE_OUT_CHANNEL = []
    CONV_TYPE_ARRAY = []
    STRIDE_ARRAY = []
    with open(files, "r") as file:
        lines = file.readlines()

    for index, line in enumerate(lines):
        line_str = line.strip()  # 去除行首行尾空白字符并转化为str
        if index % 3 == 0:
            STAGE_OUT_CHANNEL.append(line_str)
        elif index % 3 == 1:
            CONV_TYPE_ARRAY.append(line_str)
        elif index % 3 == 2:
            STRIDE_ARRAY.append(line_str)

    return STAGE_OUT_CHANNEL, CONV_TYPE_ARRAY, STRIDE_ARRAY


if __name__ == "__main__":
    if args.dataset == "imagenet100" or args.dataset == "ImageNet1000":
        channels = torch.tensor(
            [
                [16, 24, 32, 48, 64],
                [32, 48, 64, 96, 128],
                [64, 96, 128, 160, 192],
                [160, 192, 256, 320, 384],
                [320, 384, 512, 640, 768],
                [640, 768, 1024, 1280, 1536],
            ]
        ).to(model_config.device)
        stage_num = [
            range(2, 10 + 1),
            range(2, 10 + 1),
            range(2, 10 + 1),
            range(2, 20 + 1),
            range(2, 20 + 1),
            range(2, 20 + 1),
        ]
    elif args.dataset in ["cifar10", "cifar100", "whu-19", "ucm"]:
        if args.baseline_arch == "resnet18":
            channels = torch.tensor(
                [[32, 48, 64, 80, 96], [96, 112, 128, 144, 160], [192, 224, 256, 288, 320], [384, 448, 512, 576, 640]]
            ).to(model_config.device)
            stage_num = [range(3, 7 + 1), range(2, 9 + 1), range(2, 9 + 1), range(2, 7 + 1)]
        elif args.baseline_arch == "resnet20":
            channels = torch.tensor([[8, 12, 16, 20, 24], [24, 28, 32, 36, 40], [48, 56, 64, 72, 80]]).to(
                model_config.device
            )
            stage_num = [range(3, 10 + 1), range(3, 10 + 1), range(3, 10 + 1)]
    channel_shape_0 = channels.shape[0]
    channel_shape_1 = channels.shape[1]
    m = EvolutionCand(
        args,
        device=model_config.device,
        stage_num=stage_num,
        channels=channels,
        channel_shape_0=channel_shape_0,
        channel_shape_1=channel_shape_1,
    )

    channel_1 = []
    stride_1 = []
    conv_type_1 = []
    Name = ["ReActNet", "AdaBin-20", "AdaBin-18"]
    search_width = 0
    #### mobilenet
    channel_1.append([32] * 2 + [64] * 2 + [128] * 4 + [256] * 4 + [512] * 12 + [1024] * 3)
    stride_1.append([2] + [1] * 2 + [2] + [1] * 3 + [2] + [1] * 3 + [2] + [1] * 11 + [2] + [1] * 3)
    conv_type_1.append([0] * 2 + [1] + [0, 1] * 12)

    # #### resnet20
    # channel_1.append([16] + [16]*6 + [32]*6  + [64]*6)
    # stride_1.append([1] + [1]*6 + [2] + [1]*5 + [2] + [1]*5)
    # conv_type_1.append([0]*19)

    # #### resnet18
    # channel_1.append([64] + [64]*4 + [128]*4  + [256]*4 + [512]*4)
    # stride_1.append([1] + [1]*4 + [2] + [1]*3 + [2] + [1]*3 + [2] + [1]*3)
    # conv_type_1.append([0]*17)
    for ii in range(len(channel_1)):
        if ii == 0:
            search_width = 224
            baseline_arch = "mobilenet"
        elif ii == 1:
            search_width = 32
            baseline_arch = "resnet20"
        elif ii == 2:
            search_width = 32
            baseline_arch = "resnet18"

        print(f"channel:{channel_1[ii]}")
        print(f"stride:{stride_1[ii]}")
        print(f"conv_type:{conv_type_1[ii]}")
        channel_1[ii].extend([-1] * (layer_len_max - len(channel_1[ii])))
        stride_1[ii].extend([-1] * (layer_len_max - len(stride_1[ii])))
        conv_type_1[ii].extend([-1] * (layer_len_max - len(conv_type_1[ii])))
        cand_1 = channel_1[ii] + stride_1[ii] + conv_type_1[ii]
        cand_1 = torch.Tensor(cand_1).int().to(model_config.device)
        try:
            accuracy_1 = get_cand_acc(cand_1, args, model_config.device)
        except:
            accuracy_1 = torch.tensor([0]).to(model_config.device)
        print(f"search_width:{search_width}")
        # latency_1 = get_latency(args, cand_1, args.alpha, args.bw,
        #                             args.max_perform, args.platform,
        #                             baseline_arch, search_width=search_width)
        latency_1 = 0

        parm_1, ops_1 = m.get_parm_ops(cand_1, search_width=search_width)
        print(
            f"### {Name[ii]} accuracy:{accuracy_1.cpu().item():4f} "
            f"| latency:{latency_1:4f}"
            f"| parm:{parm_1.cpu().item():4f} | ops:{ops_1.cpu().item():4f}"
        )

    channel_2 = []
    conv_type_2 = []
    stride_2 = []

    if args.read_txt == 0:
        # 224尺寸
        # channel_2 = [32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 1024, 1024, 1024, 1024, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        # stride_2 = [2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        # conv_type_2 = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

        # 416尺寸
        # channel_2 = [32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, -1, -1, -1, -1, -1, -1]
        # stride_2 = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
        # conv_type_2 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1]

        # 16个epochs是56.29%，128个epochs是76.45%
        # channel_2 = [32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024, -1, -1, -1, -1, -1, -1]
        # stride_2 = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
        # conv_type_2 = [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, -1, -1, -1, -1, -1, -1]

        # 16个epochs是59.42%，128个epochs是76.64%
        # channel_2 = [32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, -1, -1, -1, -1, -1, -1]
        # stride_2 = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
        # conv_type_2 = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, -1, -1, -1, -1, -1, -1]

        # 16个epochs是52.35%，128个epochs是75.43%
        # channel_2 = [32, 32, 64, 64, 64, 64, 128, 256, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, -1, -1, -1]
        # stride_2 = [2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
        # conv_type_2 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, -1, -1, -1]

        # 16个epochs是%，128个epochs是%
        # channel_2 = [32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024, -1]
        # stride_2 = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, -1]
        # conv_type_2 = [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, -1]

        # 16个epochs是63.265%，128个epochs是%
        # channel_2 = [32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 128, 128, 256, 256, 256, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024] + [-1] * 6
        # conv_type_2 = [0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1] + [-1] * 6
        # stride_2 = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1] + [-1] * 6

        # 16个epochs是64.33%
        # channel_2 = [32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 128, 256, 256, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, -1, -1, -1]
        # stride_2 = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
        # conv_type_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, -1, -1, -1]

        # 16个epochs是67.395%
        # channel_2 = [32, 32, 32, 64, 64, 64, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        # stride_2 = [2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        # conv_type_2 = [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

        # 128 个epochs是80.04%, 速度差不多，精度更高(imagenet)
        # channel_2 = [16, 16, 32, 32, 32, 32, 96, 96, 96, 96, 96, 96, 96, 160, 160, 160, 160, 160, 160, 320, 320, 320, 320, 320, 1024, 1024, 1024, 1024, 1024, 1024, 1024, -1]
        # conv_type_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, -1]
        # stride_2 = [2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, -1]

        # 128 个epochs是, 速度更快，精度差不多(imagenet)
        # channel_2 = [16, 16, 16, 16, 16, 16, 64, 64, 64, 96, 96, 96, 96, 160, 160, 160, 384, 384, 384, 384, 384, 384, 640, 640, 640, 640, 640, 640, 640, 640, 640, 640]
        # stride_2 = [2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # conv_type_2 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0]

        # 600 个epochs是, 速度更快，精度差不多(cifar10)
        # channel_2 = [80, 112, 112, 288, 288, 288, 288, 288, 288, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576]
        # conv_type_2 = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0]
        # stride_2 = [1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        """cifar10"""
        # channel_2.append([96, 112, 288, 288, 288, 288, 288, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576])
        # conv_type_2.append([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # stride_2.append([1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # channel_2.append([80, 112, 112, 288, 288, 288, 288, 288, 288, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576])
        # conv_type_2.append([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0])
        # stride_2.append([1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        # channel_2.append([96, 112, 112, 112, 112, 112, 112, 112, 112, 192, 640, 640, 640, 640, 640, 640, 640, 640, 640, 640])
        # conv_type_2.append([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0])
        # stride_2.append([1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        # channel_2.append([96, 112, 112, 112, 112, 112, 256, 256, 256, 256, 640, 640, 640, 640, 640, 640, 640, 640, 640, 640])
        # conv_type_2.append([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        # stride_2.append([1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        # channel_2.append([96, 112, 112, 112, 112, 224, 224, 224, 224, 224, 224, 576, 576, 576, 576, 576, 576, 576])
        # conv_type_2.append([0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # stride_2.append([1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1])
        """cifar10"""

        # 128 个epochs是80.04%, 速度差不多，精度更高(imagenet)
        # channel_2.append([16, 16, 32, 32, 32, 32, 96, 96, 96, 96, 96, 96, 96, 160, 160, 160, 160, 160, 160, 320, 320, 320, 320, 320, 1024, 1024, 1024, 1024, 1024, 1024, 1024])
        # conv_type_2.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0])
        # stride_2.append([2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1])

        # 128 个epochs是81.53%,
        # channel_2.append([24, 24, 24, 24, 24, 24, 24, 24, 48, 48, 48, 48, 48, 48, 128, 128, 128, 128, 192, 192, 192, 384, 384, 640, 640, 640, 640, 640, 640, 640, 640, 640])
        # conv_type_2.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        # stride_2.append([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        # channel_2.append([32, 32, 48, 48, 48, 96, 96, 256, 256, 256, 256, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280])
        # stride_2.append([2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # conv_type_2.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # channel_2.append([48, 48, 64, 64, 96, 96, 192, 192, 192, 192, 192, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 1536, 1536, 1536, 1536, 1536])
        # stride_2.append([2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1])
        # conv_type_2.append([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1])

        # HARS-BNN-A
        # channel_2.append([24, 24, 24, 24, 24, 24, 24, 24, 48, 48, 48, 48, 48, 48, 128, 128, 128, 128, 192, 192, 192, 384, 384, 640, 640, 640, 640, 640, 640, 640, 640, 640])
        # conv_type_2.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        # stride_2.append([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        # HARS-BNN-B
        # channel_2.append([16, 16, 16, 48, 48, 48, 48, 48, 48, 48, 128, 128, 128, 128, 192, 192, 192, 512, 512, 640, 640, 640, 640, 640, 640, 640, 640, 640, 640])
        # stride_2.append([2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # conv_type_2.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1])

        # HARS-BNN-C
        # channel_2.append([32, 32, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 1024, 1024, 1024])
        # stride_2.append([2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1])
        # conv_type_2.append([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1])

        # HARS-BNN-C
        # channel_2.append([32, 32, 32, 32, 32, 32, 32, 32, 64, 96, 96, 96, 96, 192, 192, 192, 192, 320, 320, 320, 320, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280])
        # stride_2.append([2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # conv_type_2.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1])

        # HARS-BNN-18-D-1
        # channel_2.append([48, 48, 48, 160, 160, 160, 160, 160, 160, 160, 320, 320, 320, 320, 320, 320, 576, 576, 576, 576, 576])
        # stride_2.append([2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1])
        # conv_type_2.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # HARS-BNN-18-D-2
        # channel_2.append([64, 64, 64, 64, 64, 128, 128, 128, 128, 196, 196, 196, 196, 384, 384, 384, 384, 384, 384])
        # stride_2.append([1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1])
        # conv_type_2.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # harsbnn_v20
        channel_2.append(
            [104, 104, 104, 104, 104, 104, 192, 192, 192, 192, 192, 192, 192, 224, 224, 224, 224, 224, 640, 640]
        )
        stride_2.append([1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1])
        conv_type_2.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    else:
        channel_2_str, conv_type_2_str, stride_2_str = extract_text_from_file(
            files=args.architecture_file
        )
        for ii in range(len(channel_2_str)):
            channel_2.append([int(item.strip()) for item in channel_2_str[ii][1:-1].split(",")])
            conv_type_2.append([int(item.strip()) for item in conv_type_2_str[ii][1:-1].split(",")])
            stride_2.append([int(item.strip()) for item in stride_2_str[ii][1:-1].split(",")])

    for ii in range(len(channel_2)):
        print(f"channel:{channel_2[ii]}")
        print(f"stride:{stride_2[ii]}")
        print(f"conv_type:{conv_type_2[ii]}")

        channel_2[ii].extend([-1] * (layer_len_max - len(channel_2[ii])))
        stride_2[ii].extend([-1] * (layer_len_max - len(stride_2[ii])))
        conv_type_2[ii].extend([-1] * (layer_len_max - len(conv_type_2[ii])))
        channel_2_tensor = torch.Tensor(channel_2[ii]).int().to(model_config.device)
        stride_2_tensor = torch.Tensor(stride_2[ii]).int().to(model_config.device)
        conv_type_2_tensor = torch.Tensor(conv_type_2[ii]).int().to(model_config.device)
        channels_num = (channel_2_tensor != -1).sum().item()

        if args.dataset == "imagenet100":
            search_width = 224
        elif args.dataset == "cifar10" or args.dataset == "cifar100":
            search_width = 32

        print(f"search_width:{search_width}")
        cand_2 = torch.concat((channel_2_tensor, stride_2_tensor, conv_type_2_tensor), dim=0)
        try:
            accuracy_2 = get_cand_acc(cand_2, args, model_config.device)
        except:
            accuracy_2 = torch.tensor([0]).to(model_config.device)

        # latency_2 = get_latency(args, cand_2, args.alpha, args.bw,
        #                             args.max_perform, args.platform,
        #                             args.baseline_arch, search_width=search_width)
        latency_2 = 0

        parm_2, ops_2 = m.get_parm_ops(cand_2, search_width=search_width)
        print(
            f"### No.{ii+1:3d} accuracy:{accuracy_2.cpu().item():4f} "
            f"| latency:{latency_2:4f}"
            f"| parm:{parm_2.cpu().item():4f} | ops:{ops_2.cpu().item():4f}"
        )
