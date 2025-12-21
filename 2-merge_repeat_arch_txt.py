import os
import os.path as osp
import random
import time
import warnings

import torch

from utils.config import Config

model_config = Config()


def extract_text_from_file(files=None):
    channel_list = []
    conv_type_list = []
    stride_list = []
    with open(files, "r") as file:
        lines = file.readlines()

    for index, line in enumerate(lines):
        line_str = line.strip()  # 去除行首行尾空白字符并转化为str
        result_num_list = [int(item.strip()) for item in line_str[1:-1].split(",")]
        if index % 3 == 0:
            channel_list.append(result_num_list)
        elif index % 3 == 1:
            conv_type_list.append(result_num_list)
        elif index % 3 == 2:
            stride_list.append(result_num_list)

    return channel_list, conv_type_list, stride_list


def list_convert_tensor(channel_list, conv_type_list, stride_list, layer_len_end, device):
    tensor_all = torch.tensor([]).to(device)

    for ii in range(len(channel_list)):
        channel_list_tensor = channel_list[ii] + [-1] * (layer_len_end - len(channel_list[ii]))
        conv_type_list_tensor = conv_type_list[ii] + [-1] * (layer_len_end - len(channel_list[ii]))
        stride_list_tensor = stride_list[ii] + [-1] * (layer_len_end - len(channel_list[ii]))

        channel_list_tensor = torch.tensor(channel_list_tensor).int().to(device).view(-1)
        conv_type_list_tensor = torch.tensor(conv_type_list_tensor).int().to(device).view(-1)
        stride_list_tensor = torch.tensor(stride_list_tensor).int().to(device).view(-1)

        if ii == 0:
            tensor_single = torch.concat((channel_list_tensor, conv_type_list_tensor, stride_list_tensor), dim=0)
            tensor_all = tensor_single.view(1, -1)
        else:
            tensor_single = torch.concat((channel_list_tensor, conv_type_list_tensor, stride_list_tensor), dim=0)
            tensor_all = torch.concat((tensor_all, tensor_single.view(1, -1)), dim=0)
    return tensor_all


def remove_same_vectors(tensor_all_main, tensor_all_other):
    """
    检查tensor_all_other中是否有和tensor_all_main相同的向量，并去掉这些相同向量

    Args:
        tensor_all_main (torch.Tensor): 主张量
        tensor_all_other (torch.Tensor): 要检查并处理的张量
    """
    unique_tensors = []
    for ii in range(tensor_all_other.shape[0]):
        is_same = torch.any(torch.all(torch.eq(tensor_all_other[ii], tensor_all_main), dim=1))
        if is_same:
            pass
        else:
            unique_tensors.append(tensor_all_other[ii])

    return (
        torch.stack(unique_tensors).to(tensor_all_main.device)
        if unique_tensors
        else torch.empty(0).to(tensor_all_main.device)
    )


if __name__ == "__main__":
    output_path_main = "./1-output_test_1.txt"
    output_path_other = "./1-output_test_2.txt"
    output_path_all = "./1-output_test_3.txt"
    layer_len_max = model_config.layer_len_max

    channel_list, conv_type_list, stride_list = extract_text_from_file(output_path_main)
    # print(channel_list)

    tensor_all_main = list_convert_tensor(
        channel_list, conv_type_list, stride_list, layer_len_max, model_config.device
    )
    # print(tensor_all_main)

    channel_list, conv_type_list, stride_list = extract_text_from_file(output_path_other)

    tensor_all_other = list_convert_tensor(
        channel_list, conv_type_list, stride_list, layer_len_max, model_config.device
    )

    tensor_all_other = remove_same_vectors(tensor_all_main, tensor_all_other)

    # tensor_all = torch.concat((tensor_all_main, tensor_all_other), dim=0)
    tensor_all = tensor_all_other

    if os.path.exists(output_path_all):
        os.remove(output_path_all)

    for ii in range(tensor_all.shape[0]):
        channel_tensor = tensor_all[ii, : tensor_all.shape[1] // 3]
        conv_type_tensor = tensor_all[ii, tensor_all.shape[1] // 3 : tensor_all.shape[1] // 3 * 2]
        stride_tensor = tensor_all[ii, tensor_all.shape[1] // 3 * 2 :]

        channel_tensor = channel_tensor[channel_tensor != -1]
        stride_tensor = stride_tensor[stride_tensor != -1]
        conv_type_tensor = conv_type_tensor[conv_type_tensor != -1]
        with open(output_path_all, "a+") as f:
            f.write(str(channel_tensor.tolist()))
            f.write("\n")
            f.write(str(conv_type_tensor.tolist()))
            f.write("\n")
            f.write(str(stride_tensor.tolist()))
            f.write("\n")
