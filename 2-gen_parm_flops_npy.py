import argparse
import itertools
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
from torch.cuda.amp import autocast as autocast

parser = argparse.ArgumentParser(description="PyTorch Get npy dataset!")
parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet100"])
parser.add_argument("--epochs", default=32, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--seed", default=100, type=int)
parser.add_argument(
    "--np_save_path", type=str, default="/mnt/wr/Train_Quant_Imagenet100_184_channel_v8_v1_改1个x/2-npy/"
)
args = parser.parse_args()


def cal_n_multiply(n, lst):
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


def cal_total_flops_parm(channel_, conv_, stride_, width_origin=224):
    flops_total = 0
    bops_total = 0
    flparm_total = 0
    bparm_total = 0
    width_origin = width_origin
    flops_single_array = []
    bops_single_array = []
    flparm_single_array = []
    bparm_single_array = []
    for ii in range(len(channel_)):
        width_single = width_origin / cal_n_multiply(ii + 1, stride_)
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


def get_epoch_prec(txt_file, epoch_max):
    accuracy_array = []
    epoch_array = []
    epoch = 1
    best_accuracy = 0
    with open(txt_file, "r") as file:
        content = file.read()
        matches = re.findall(r" * Prec@1\s+(\d+\.\d+)", content)
        if matches:
            for precision in matches:
                epoch_array.append(int(epoch))
                accuracy_array.append(float(precision))
                epoch = epoch + 1

            matches_best = re.findall(r"Best Acc:(\d+\.\d+)%", content)
            if len(matches_best) >= epoch_max:
                # best_accuracy = float(matches_best[-1])
                best_accuracy = float(matches_best[epoch_max - 1])
            else:
                print("未找到匹配内容 Best Acc")
                best_accuracy = 0
        else:
            print("未找到匹配内容 * Prec@1.")
            best_accuracy = 0

    return accuracy_array, epoch_array, best_accuracy


def get_epoch_prec_cifar(txt_file, epoch_max):
    accuracy_array = []
    epoch_array = []
    epoch = 1
    best_accuracy = 0
    with open(txt_file, "r") as file:
        content = file.read()
        matches = re.findall(r"Best_acc:(\d+\.\d+)", content)
        if matches:
            for precision in matches:
                epoch_array.append(int(epoch))
                accuracy_array.append(float(precision))
                epoch = epoch + 1

            matches_best = re.findall(r"Best_acc:(\d+\.\d+)", content)
            if len(matches_best) >= epoch_max:
                best_accuracy = float(matches_best[epoch_max - 1])
            else:
                print("未找到匹配内容 Best Acc")
                best_accuracy = 0
        else:
            print("未找到匹配内容 * Prec@1.")
            best_accuracy = 0

    return accuracy_array, epoch_array, best_accuracy


def to_one_hot(value, num_classes, offset):
    one_hot_vector = [0] * num_classes
    one_hot_vector[value + offset] = 1
    return one_hot_vector


def feature2vec(layer_len_max, channel_, conv_, stride_, flops_single, bops_single, flparm_single, bparm_single):
    all_list = []
    for ii in range(layer_len_max):
        if ii < len(channel_):
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


def find_channel_different_count(stage_out_channels, args):
    if args.dataset == "imagenet100":
        unique_channels_num = 6
    elif args.dataset in ["cifar10", "cifar100"]:
        unique_channels_num = 4
    else:
        print("Please input the correct dataset!")
        exit()
    count_dict = {}
    count = []
    for channel in stage_out_channels:
        if channel in count_dict:
            count_dict[channel] += 1
        else:
            count_dict[channel] = 1

    # 找出所有出现过的不同元素
    unique_channels = list(count_dict.keys())
    # 确保列表长度为6，如果不足6个，用0填充
    while len(unique_channels) < unique_channels_num:
        unique_channels.append(0)

    result_list = [count_dict.get(channel, 0) for channel in unique_channels]

    return result_list


if __name__ == "__main__":
    stride_array = []
    conv_type_array = []
    stage_out_channels = []
    concat_type_array = []

    if args.dataset == "imagenet100":
        epoch_max = 16
        layer_len_max = 32
        width_origin = 224
        output_directory = "./outputs_all"
        input_directory = "./1-output_all.txt"
        file_part_name = "BNN"
        file_part_index = 3
    elif args.dataset in ["cifar10", "cifar100"]:
        epoch_max = 32
        layer_len_max = 25
        width_origin = 32
        output_directory = "/mnt/wr/2_Remote/cifar_sousuo/checkpoint/"
        input_directory = "/mnt/wr/2_Remote/cifar_sousuo/1-output.txt"
        file_part_name = "HARSBNN"
        file_part_index = 9

    # Traverse the architecture obtained from the search
    with open(input_directory, "r") as f:
        content = f.read()

        parts = content.split("\n")
        result_lists = []
        for part in parts:
            if part:
                cleaned_part = part.strip("[]()").replace(",", "")
                sub_list = [int(item) for item in cleaned_part.split(" ") if item]
                result_lists.append(sub_list)
        print(f"There are a total of {int(len(result_lists)/3)} architectures")

    flops_total_array, bops_total_array, flparm_total_array, bparm_total_array = [[] for _ in range(4)]
    flops_single_array, bops_single_array, flparm_single_array, bparm_single_array = [[] for _ in range(4)]
    feature_single_vec_array, feature_total_vec_array = [], []
    unique_channels_array = []

    # Calculation of flops and parm
    for ii in range(int(len(result_lists) / 3) + 1):
        if ii < int(len(result_lists) / 3):
            stage_out_channels = result_lists[0 + ii * 3]
            conv_type_array = result_lists[1 + ii * 3]
            stride_array = result_lists[2 + ii * 3]
        elif ii == int(len(result_lists) / 3):
            if args.dataset in ["cifar10", "cifar100"]:
                break
            elif args.dataset == "imagenet100":
                stage_out_channels = [
                    32,
                    32,
                    64,
                    64,
                    128,
                    128,
                    128,
                    128,
                    256,
                    256,
                    256,
                    256,
                    512,
                    512,
                    512,
                    512,
                    512,
                    512,
                    512,
                    512,
                    512,
                    512,
                    512,
                    512,
                    1024,
                    1024,
                    1024,
                ]
                conv_type_array = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                stride_array = [2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]

        flops_total, bops_total, flparm_total, bparm_total, flops_single, bops_single, flparm_single, bparm_single = (
            cal_total_flops_parm(stage_out_channels, conv_type_array, stride_array, width_origin=width_origin)
        )
        print(
            f"The {ii+1:<4d} arch | flops:{flops_total/10**6:>8.2f}M | bops:{bops_total/10**6:>8.2f}M | "
            f"flparm:{flparm_total/10**6:>8.2f}M | bparm:{bparm_total/10**6:>8.2f}M |"
        )
        flops_total_array.append(flops_total)
        bops_total_array.append(bops_total)
        flparm_total_array.append(flparm_total)
        bparm_total_array.append(bparm_total)
        flops_single_array.append(flops_single)
        bops_single_array.append(bops_single)
        flparm_single_array.append(flparm_single)
        bparm_single_array.append(bparm_single)

        feature_single_vec = feature2vec(
            layer_len_max,
            stage_out_channels,
            conv_type_array,
            stride_array,
            flops_single,
            bops_single,
            flparm_single,
            bparm_single,
        )
        unique_channels = find_channel_different_count(stage_out_channels, args)

        feature_total_vec = (
            [flops_total] + [bops_total] + [flparm_total] + [bparm_total] + [len(stage_out_channels)] + unique_channels
        )
        feature_single_vec_array.append(feature_single_vec)
        feature_total_vec_array.append(feature_total_vec)
        unique_channels_array.append(unique_channels)
        # print(f"{feature_single_vec}")

    # 去输出的log文件找到每个epoch的accuracy
    bnn_dirs = []
    num_index = []
    for item in os.listdir(output_directory):
        item_path = os.path.join(output_directory, item)
        if os.path.isdir(item_path) and item.startswith(file_part_name):
            try:
                # 提取数字部分并转换为整数
                num = int(item[file_part_index:])
                bnn_dirs.append((num, item_path))
                num_index.append(num - 1)
            except ValueError:
                print(f"Cannot extract files:{item_path}")
                # exit(0)
                continue

    best_accuracy_array = []
    bnn_dirs.sort(key=lambda x: x[0])
    for num, path in bnn_dirs:
        if args.dataset in ["cifar10", "cifar100"]:
            accuracy_array, epoch_array, best_accuracy = get_epoch_prec_cifar(path + "/log.txt", epoch_max=epoch_max)
        elif args.dataset == "imagenet100":
            accuracy_array, epoch_array, best_accuracy = get_epoch_prec(path + "/log.txt", epoch_max=epoch_max)
        if len(accuracy_array) >= epoch_max:
            print(
                f"{path}:{accuracy_array}, {epoch_array}, {best_accuracy}, "
                f"{flops_total_array[num-1]}, {bops_total_array[num-1]}, {flparm_total_array[num-1]}, {bparm_total_array[num-1]}"
            )
            best_accuracy_array.append(best_accuracy)
        else:
            best_accuracy_array.append(best_accuracy)
            print(f"{path} is incomplete")

    # Add ReActNet
    if args.dataset in ["cifar10", "cifar100"]:
        pass
    elif args.dataset == "imagenet100":
        best_accuracy_array.append(65.835)

    best_accuracy_np = np.array(best_accuracy_array, dtype=np.float32)
    feature_single_vec_np = np.array(feature_single_vec_array, dtype=np.float32)
    feature_total_vec_np = np.array(feature_total_vec_array, dtype=np.float32)
    unique_channels_np = np.array(unique_channels_array, dtype=np.float32)

    print(f"best_accuracy_np:{best_accuracy_np.shape}")
    print(f"feature_single_vec_np:{feature_single_vec_np.shape}")
    print(f"feature_total_vec_np:{feature_total_vec_np.shape}")
    print(f"unique_channels_np:{unique_channels_np.shape}")
    # exit()

    nonzero_indices = np.nonzero(best_accuracy_np)[0]
    best_accuracy_np = best_accuracy_np[nonzero_indices]
    feature_single_vec_np = feature_single_vec_np[num_index, :, :][nonzero_indices]
    feature_total_vec_np = feature_total_vec_np[num_index, :][nonzero_indices]
    unique_channels_np = unique_channels_np[num_index, :][nonzero_indices]

    if not os.path.exists(args.np_save_path):
        os.makedirs(args.np_save_path)
    np_save_path = args.np_save_path + "/" + args.dataset + "/"
    np.save(np_save_path + "2-np_best_accuracy.npy", best_accuracy_np)
    np.save(np_save_path + "2-np_feature_single_vec.npy", feature_single_vec_np)
    np.save(np_save_path + "2-np_feature_total_vec.npy", feature_total_vec_np)
    np.save(np_save_path + "2-np_unique_channels.npy", unique_channels_np)
