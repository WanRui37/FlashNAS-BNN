"""Generate aligned predictor features from architecture records and measured logs."""
import argparse
from pathlib import Path
import re
import numpy as np
from tools.data.architectures import read_architectures

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['imagenet100', 'cifar10', 'cifar100'], required=True)
    parser.add_argument('--architectures', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True, help='Directory containing numbered training runs')
    parser.add_argument('--epochs', type=int, required=True, help='Recorded labeling budget')
    parser.add_argument('--prefix', default='BNN', help='Log directory prefix; e.g. BNN or HARSBNN')
    parser.add_argument('--output', type=Path, required=True, help='New output directory for this dataset')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    records = read_architectures(args.architectures)
    layer_max, width = (32, 224) if args.dataset == 'imagenet100' else (25, 32)
    labels, single, total, unique = [], [], [], []
    for index, (channels, kernels, strides) in enumerate(records, 1):
        if len(channels) > layer_max:
            raise ValueError(f'Architecture {index} exceeds {layer_max} layers')
        log = args.logs / f'{args.prefix}{index}' / 'log.txt'
        parse = get_epoch_prec if args.dataset == 'imagenet100' else get_epoch_prec_cifar
        accuracy, _, best = parse(str(log), args.epochs)
        if len(accuracy) < args.epochs or best == 0:
            raise ValueError(f'Incomplete measured labels: {log}')
        features = cal_total_flops_parm(channels, kernels, strides, width)
        fpops, bops, fpparm, bparm, *per_layer = features
        counts = find_channel_different_count(channels, args)
        labels.append(best)
        single.append(feature2vec(layer_max, channels, kernels, strides, *per_layer))
        total.append([fpops, bops, fpparm, bparm, len(channels)] + counts)
        unique.append(counts)
    if not labels:
        raise ValueError('No architecture records')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, values in [('best_accuracy', labels), ('feature_single_vec', single),
                         ('feature_total_vec', total), ('unique_channels', unique)]:
        np.save(args.output / f'2-np_{name}.npy', np.asarray(values, dtype=np.float32))
    print(f'Wrote {len(labels)} measured, aligned samples to {args.output}')


if __name__ == '__main__':
    main()
