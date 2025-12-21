import os
import os.path as osp
import random
import shutil
import time
import warnings

import torch

from utils.config import Config

model_config = Config()


def find_max_bnn_folder(outputs_dir):
    bnn_folders = []
    for root, dirs, _ in os.walk(outputs_dir):
        for dir_name in dirs:
            if dir_name.startswith("BNN"):
                bnn_folders.append(dir_name)

    numbers = []
    for folder in bnn_folders:
        try:
            number_str = folder[3:]
            number = int(number_str)
            numbers.append(number)
        except ValueError:
            # 如果无法转换为整数，可能是格式不符合预期，跳过该文件夹
            continue

    if not numbers:
        return None

    max_number = max(numbers)
    return max_number


def copy_and_rename_folders(source_dir, destination_dir, max_num=None):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for root, dirs, _ in os.walk(source_dir):
        for dir_name in dirs:
            if dir_name.startswith("BNN"):
                new_dir_name = "BNN" + str(int(dir_name[3:]) + max_num)
                source_path = os.path.join(root, dir_name)
                destination_path = os.path.join(destination_dir, new_dir_name)
                shutil.copytree(source_path, destination_path)


if __name__ == "__main__":
    output_path_main = "./outputs_old_汇聚了v5到v7的所有_epoch16"
    output_path_other = "./outputs"
    output_path_all = "./outputs_all"

    # if os.path.exists(output_path_all):
    #     os.remove(output_path_all)

    max_num = find_max_bnn_folder(output_path_main)
    print(f"max_num:{max_num}")

    copy_and_rename_folders(output_path_main, output_path_all, max_num=0)
    copy_and_rename_folders(output_path_other, output_path_all, max_num=max_num)
