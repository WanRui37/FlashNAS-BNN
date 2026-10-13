import argparse
import logging
import os
import os.path as osp
import random
import time
import warnings

import numpy
import torch
import torch.nn as nn

from net_bw.dataset_bw import *
from net_bw.Predict_BW import *
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
parser.add_argument("--model_path", type=str, default="./net_predict_output/imagenet100/CharRNN_OUR_best.pth")
parser.add_argument("--mode", type=str, default="real_time", choices=["real_time", "predict", "use_pth"])
parser.add_argument("--use_pth_name", type=str, default="info.pth", help="使用的pth文件名")

# 添加延迟测试模式选择参数
parser.add_argument(
    "--latency_mode", type=str, default="onnx", choices=["pytorch", "onnx"], help="选择延迟测试方式: pytorch或onnx"
)
parser.add_argument("--latency_batch_size", type=int, default=1, help="Batch size for latency measurement")
parser.add_argument("--latency_warmup", type=int, default=20, help="Warmup iterations for latency measurement")
parser.add_argument("--latency_iterations", type=int, default=30, help="Iterations for latency measurement")
parser.add_argument("--latency_device", type=str, default="cuda:0")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--save_pth", action="store_true", help="Save the final candidates")

parser.add_argument("--alpha", type=float, default=0.8)
parser.add_argument("--bw", type=float, default=12.8)
parser.add_argument("--max_perform", type=float, default=5734.4)
parser.add_argument("--platform", type=str, default="fpga", choices=["fpga", "cpu", "gpu"])
parser.add_argument("--precision", type=int, default="8", choices=[32, 16, 8])
parser.add_argument("--w", type=float, nargs="+", default=[0.5], help="权重列表，用于权衡精度和延迟")

parser.add_argument("--layer_len_start", type=int, default=6)
parser.add_argument("--layer_len_end", type=int, default=32)
parser.add_argument("--baseline_arch", type=str, default="resnet20", choices=["mobilenet", "resnet18", "resnet20"])
parser.add_argument("--dataset", type=str, default="cifar10", choices=["imagenet100", "cifar10", "cifar100"])
parser.add_argument("--acc_width", type=int, default=224)
parser.add_argument("--search_width", type=int, default=224)
parser.add_argument("--T_min", type=float, default=6.0)
parser.add_argument("--T_max", type=float, default=12.0)
parser.add_argument("--acc_max_k", type=float, default=0.5)

parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--embed_dim_1", type=int, default=8)
parser.add_argument("--embed_dim_2", type=int, default=4)
parser.add_argument("--bw_output_path", type=str, default="./net_bw_output/")
parser.add_argument("--first_float", default=1, type=int)
parser.add_argument(
    "--latency_txt_path",
    type=str,
    default="/mnt/wr/Train_Quant_Imagenet100_official/stage2/outputs/BNN1_different/2-run_result/",
)


def main():
    args = parser.parse_args()
    main_worker(args)


def get_scalar_value(data):
    """
    通用标量值处理函数，处理各种类型的数据转换为标量
    支持: list, tuple, torch.Tensor, numpy.ndarray, 标量数值
    """
    if data is None:
        return 0.0

    if isinstance(data, (list, tuple)):
        # 如果是列表或元组，取第一个有效元素
        if data:
            return get_scalar_value(data[0])
        else:
            return 0.0
    elif hasattr(data, "cpu") and hasattr(data, "item"):
        # 如果是PyTorch tensor，转换为数值
        return data.cpu().item()
    elif isinstance(data, torch.Tensor):
        # 如果是tensor但可能没有cpu()方法
        if data.numel() == 1:
            return data.item()
        else:
            return float(data)
    elif isinstance(data, numpy.ndarray):
        # 如果是numpy数组
        if data.size == 1:
            return data.item()
        else:
            return float(data)
    else:
        # 其他情况直接转换为float
        try:
            return float(data)
        except (ValueError, TypeError):
            return 0.0


def main_worker(args):
    if args.dataset == "imagenet100":
        channels = torch.tensor(
            [
                [16, 24, 32, 48, 64],
                [32, 48, 64, 96, 128],
                [64, 96, 128, 160, 192],
                [160, 192, 256, 320, 384],
                [320, 384, 512, 640, 768],
                [640, 768, 1024, 1280, 1536],
            ]
        ).to(args.device)
        stage_num = [
            range(2, 10 + 1),
            range(2, 10 + 1),
            range(2, 10 + 1),
            range(2, 20 + 1),
            range(2, 20 + 1),
            range(2, 20 + 1),
        ]
    elif args.dataset == "cifar10" or args.dataset == "cifar100":
        if args.baseline_arch == "resnet18":
            channels = torch.tensor(
                [[32, 48, 64, 80, 96], [96, 112, 128, 144, 160], [192, 224, 256, 288, 320], [384, 448, 512, 576, 640]]
            ).to(args.device)
            stage_num = [range(3, 6 + 1), range(3, 6 + 1), range(3, 6 + 1), range(3, 6 + 1)]
        elif args.baseline_arch == "resnet20":
            channels = torch.tensor([[8, 12, 16, 20, 24], [24, 28, 32, 36, 40], [48, 56, 64, 72, 80]]).to(args.device)
            stage_num = [range(3, 10 + 1), range(3, 10 + 1), range(3, 10 + 1)]
    channel_shape_0 = channels.shape[0]
    channel_shape_1 = channels.shape[1]
    m = EvolutionCand(
        args,
        device=args.device,
        stage_num=stage_num,
        channels=channels,
        channel_shape_0=channel_shape_0,
        channel_shape_1=channel_shape_1,
    )

    if not osp.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = get_logger(name="Search", log_file=osp.join(args.logdir, "search_second.log"), log_level=logging.INFO)

    checkpoint = torch.load(args.use_pth_name, map_location=args.device)

    t = time.time()
    try:
        final_candidates = checkpoint["final_candidates"]
        final_candidates_acc = checkpoint["final_candidates_acc"]
        final_candidates_ops = checkpoint["final_candidates_ops"]
        final_candidates_parm = checkpoint["final_candidates_parm"]
    except:
        final_candidates = checkpoint["epoch_candidates"][20]
        final_candidates_acc = checkpoint["epoch_acc"][20]
        final_candidates_ops = checkpoint["epoch_ops"][20]
        final_candidates_parm = checkpoint["epoch_parm"][20]

    if args.mode == "use_pth":
        try:

            final_candidates_latency = checkpoint["final_candidates_latency"]
        except:
            final_candidates_latency = checkpoint["epoch_latency"][20]

    final_candidates = torch.tensor(final_candidates).to(args.device)
    final_candidates_acc = torch.tensor(final_candidates_acc).to(args.device)
    final_candidates_ops = torch.tensor(final_candidates_ops).to(args.device)
    final_candidates_parm = torch.tensor(final_candidates_parm).to(args.device)

    if args.mode == "predict":
        if args.platform == "gpu":
            raise ValueError("GPU CLE predictor is not included in this release")
        model_config.device = torch.device(args.device)
        model_bw = None
        if args.platform == "cpu":
            model_bw = Predict_BW(
                embed_dim_1=args.embed_dim_1, embed_dim_2=args.embed_dim_2, hidden_size=args.hidden_size
            ).to(args.device)
            state_dict = torch.load(args.bw_output_path + "/model_best.pth", map_location=args.device)
            model_bw.load_state_dict(state_dict)
            model_bw.eval()

        final_candidates_latency = [
            get_latency(
                args,
                cand,
                args.alpha,
                args.bw,
                args.max_perform,
                args.platform,
                args.baseline_arch,
                args.search_width,
                model_bw,
                args.embed_dim_1,
                args.embed_dim_2,
            )
            for cand in final_candidates
        ]
    elif args.mode == "real_time":
        final_candidates_ops, id_sort = torch.sort(final_candidates_ops.flatten(), descending=True)

        if args.save_pth:
            indices = id_sort[:]
        else:
            indices = id_sort[:]  # Keep every candidate aligned with its measurements

        final_candidates = final_candidates[indices]
        final_candidates_ops = final_candidates_ops[indices]
        final_candidates_acc = final_candidates_acc[indices]
        final_candidates_parm = final_candidates_parm[indices]

        mode_name = "Onnx" if args.latency_mode == "onnx" else "PyTorch"
        print(f"Using {mode_name} to test latency on device: {args.latency_device}")
        final_candidates_latency = get_latency_real_time(args, final_candidates)

        # 确保所有延迟值都是标量数值
        processed_latency = []
        for latency in final_candidates_latency:
            if isinstance(latency, (list, tuple)) and latency:
                latency = latency[0]  # 如果是列表，取第一个元素
            if hasattr(latency, "cpu") and hasattr(latency, "item"):
                latency = latency.cpu().item()  # 如果是tensor，转换为数值
            elif isinstance(latency, torch.Tensor):
                latency = latency.item() if latency.numel() == 1 else float(latency)
            processed_latency.append(float(latency))

        final_candidates_latency = processed_latency

        if args.save_pth:
            save_checkpoint(
                {
                    "final_candidates": final_candidates,
                    "final_candidates_acc": final_candidates_acc,
                    "final_candidates_ops": final_candidates_ops,
                    "final_candidates_parm": final_candidates_parm,
                    "final_candidates_latency": final_candidates_latency,
                },
                filename=os.path.join(args.logdir, "info_second.pth"),
            )
            exit()

    final_candidates_latency = torch.tensor(final_candidates_latency).to(args.device)
    mean_latency = torch.mean(final_candidates_latency)
    std_latency = torch.std(final_candidates_latency)
    print(f"mean_latency:{mean_latency}")
    print(f"std_latency:{std_latency}")

    mask = (final_candidates_latency >= args.T_min) & (final_candidates_latency <= args.T_max).to(args.device)
    final_candidates = final_candidates[mask, :]
    final_candidates_acc = final_candidates_acc[mask]
    final_candidates_ops = final_candidates_ops[mask]
    final_candidates_parm = final_candidates_parm[mask]
    final_candidates_latency = final_candidates_latency[mask]

    if args.baseline_arch == "mobilenet":
        channel_1 = [32] * 2 + [64] * 2 + [128] * 4 + [256] * 4 + [512] * 12 + [1024] * 3 + [-1] * (32 - 27)
        stride_1 = [2] + [1] * 2 + [2] + [1] * 3 + [2] + [1] * 3 + [2] + [1] * 11 + [2] + [1] * 3 + [-1] * (32 - 27)
        conv_type_1 = [0] * 2 + [1] + [0, 1] * 12 + [-1] * (32 - 27)
        Name = "ReactNet"
    elif args.baseline_arch == "resnet20":
        channel_1 = [16] + [16] * 6 + [32] * 6 + [64] * 6 + [-1] * (args.layer_len_end - 19)
        stride_1 = [1] + [1] * 6 + [2] + [1] * 5 + [2] + [1] * 5 + [-1] * (args.layer_len_end - 19)
        conv_type_1 = [0] * 19 + [-1] * (args.layer_len_end - 19)
        Name = "AdaBin(resnet20)"
    elif args.baseline_arch == "resnet18":
        channel_1 = [64] + [64] * 4 + [128] * 4 + [256] * 4 + [512] * 4 + [-1] * (args.layer_len_end - 17)
        stride_1 = [1] + [1] * 4 + [2] + [1] * 3 + [2] + [1] * 3 + [2] + [1] * 3 + [-1] * (args.layer_len_end - 17)
        conv_type_1 = [0] * 17 + [-1] * (args.layer_len_end - 17)
        Name = "AdaBin(resnet18)"

    cand_1 = channel_1 + stride_1 + conv_type_1
    cand_1 = torch.Tensor(cand_1).int().to(args.device)

    reacetnet_array = cand_1.cpu().numpy().tolist()
    accuracy_1 = get_cand_acc(cand_1, args, args.device)

    parm_1, ops_1 = m.get_parm_ops(cand_1, search_width=args.search_width)

    if args.mode == "predict":
        latency_1 = get_latency(
            args,
            reacetnet_array,
            args.alpha,
            args.bw,
            args.max_perform,
            args.platform,
            args.baseline_arch,
            args.search_width,
            model_bw,
            args.embed_dim_1,
            args.embed_dim_2,
        )
    else:
        if args.latency_mode == "onnx":
            latency_1 = get_latency_real_time(args, [cand_1])
        else:
            latency_1 = get_latency_real_time(args, [cand_1])

    logger.info(
        "{} Arch.{}, Top-1 acc: {:.4f}, Latency: {:.4f}ms, OPs: {:f}, Parm: {:f}".format(
            Name,
            reacetnet_array,
            get_scalar_value(accuracy_1),
            get_scalar_value(latency_1),
            get_scalar_value(ops_1),
            get_scalar_value(parm_1),
        )
    )

    if torch.all(mask == 0):
        logger.info("T set is not correct !!!")
        exit()

    mean_acc = torch.mean(final_candidates_acc)
    std_acc = torch.std(final_candidates_acc)
    logger.info(f"mean_acc:{mean_acc}")
    logger.info(f"std_acc:{std_acc}")

    mask = final_candidates_acc - mean_acc <= args.acc_max_k * std_acc
    final_candidates_latency = final_candidates_latency[mask]
    final_candidates_acc = final_candidates_acc[mask]
    final_candidates_ops = final_candidates_ops[mask]
    final_candidates_parm = final_candidates_parm[mask]
    final_candidates = final_candidates[mask, :]

    for ii in range(final_candidates.shape[0]):
        logger.info(
            "{:4d} Top-1 acc: {:.4f}, Latency: {:.4f}ms, OPs: {:f}, Parm: {:f} Arch.{}".format(
                ii,
                get_scalar_value(final_candidates_acc[ii]),
                get_scalar_value(final_candidates_latency[ii]),
                get_scalar_value(final_candidates_ops[ii]),
                get_scalar_value(final_candidates_parm[ii]),
                final_candidates[ii, :].cpu().numpy().tolist(),
            )
        )
    max_latency = torch.max(final_candidates_latency)
    min_latency = torch.min(final_candidates_latency)
    logger.info(f"max_latency:{max_latency}")
    logger.info(f"min_latency:{min_latency}")
    final_candidates_latency_norm = (final_candidates_latency - min_latency) / (max_latency - min_latency)
    max_acc = torch.max(final_candidates_acc)
    min_acc = torch.min(final_candidates_acc)
    logger.info(f"max_acc:{max_acc}")
    logger.info(f"min_acc:{min_acc}")
    final_candidates_acc_norm = (max_acc - final_candidates_acc) / (max_acc - min_acc)

    # 处理权重参数，确保是列表
    if isinstance(args.w, (int, float)):
        w_list = [args.w]
    else:
        w_list = args.w

    logger.info(f"开始搜索，权重列表: {w_list}")

    # 保存所有权重下的最优结果
    all_best_results = []

    for w_value in w_list:
        logger.info(f"\n=== 当前权重 w = {w_value} ===")

        # 计算目标函数 J
        J = w_value * final_candidates_acc_norm + (1 - w_value) * final_candidates_latency_norm
        min_index = torch.argmin(J.to(args.device), dim=0, keepdim=False)

        # 记录当前权重下的最优结果
        best_cand = final_candidates[min_index, :]
        best_acc = get_scalar_value(final_candidates_acc[min_index])
        best_latency = get_scalar_value(final_candidates_latency[min_index])
        best_ops = get_scalar_value(final_candidates_ops[min_index])
        best_parm = get_scalar_value(final_candidates_parm[min_index])

        # 解析架构信息
        cand_shape = int(best_cand.shape[0] / 3)
        mask = best_cand[:cand_shape] != -1
        true_count = mask.sum().item()
        channel_1 = best_cand[:cand_shape][mask]
        stride_1 = best_cand[cand_shape : cand_shape * 2][mask]
        conv_type_1 = best_cand[cand_shape * 2 :][mask]

        # 保存结果
        result = {
            "w": w_value,
            "candidate": best_cand.cpu().numpy().tolist(),
            "accuracy": best_acc,
            "latency": best_latency,
            "ops": best_ops,
            "params": best_parm,
            "channels": channel_1.cpu().numpy().tolist(),
            "strides": stride_1.cpu().numpy().tolist(),
            "conv_types": conv_type_1.cpu().numpy().tolist(),
            "layer_count": true_count,
        }
        all_best_results.append(result)

        # 记录当前权重下的最优结果
        logger.info(f"权重 w = {w_value} 下的最优架构:")
        logger.info(
            "Best Arch.{}, Top-1 acc: {:.4f}, Latency: {:.4f}ms, OPs: {:d}, Parm: {:d}".format(
                result["candidate"],
                result["accuracy"],
                result["latency"],
                int(result["ops"]),
                int(result["params"]),
            )
        )
        logger.info(
            "Best channel.{}\n, Best stride.{}\n, Best conv.{}\n, Best layer_num.{}".format(
                result["channels"],
                result["strides"],
                result["conv_types"],
                result["layer_count"],
            )
        )

    # 输出所有权重下的结果汇总
    logger.info("\n=== 所有权重下的最优架构汇总 ===")
    logger.info("权重\t精度\t延迟\t层数")
    logger.info("-" * 50)
    for result in all_best_results:
        logger.info(f"{result['w']:.2f}\t{result['accuracy']:.4f}\t{result['latency']:.2f}ms\t{result['layer_count']}")

    # 保存所有结果到文件
    if args.save_pth:
        save_checkpoint(
            {
                "all_best_results": all_best_results,
                "final_candidates": final_candidates.cpu().numpy().tolist(),
                "final_candidates_acc": final_candidates_acc.cpu().numpy().tolist(),
                "final_candidates_ops": final_candidates_ops.cpu().numpy().tolist(),
                "final_candidates_parm": final_candidates_parm.cpu().numpy().tolist(),
                "final_candidates_latency": final_candidates_latency.cpu().numpy().tolist(),
            },
            filename=os.path.join(args.logdir, "info_all_weights.pth"),
        )

    logger.info("total searching time = {:.4f} hours".format((time.time() - t) / 3600))

    logger.info(
        "Best Arch.{}, Top-1 acc: {:.4f}, Latency: {:.4f}ms, OPs: {:d}, Parm: {:d}".format(
            final_candidates[min_index, :].cpu().numpy().tolist(),
            get_scalar_value(final_candidates_acc[min_index]),
            get_scalar_value(final_candidates_latency[min_index]),
            get_scalar_value(final_candidates_ops[min_index]),
            get_scalar_value(final_candidates_parm[min_index]),
        )
    )

    cand = final_candidates[min_index, :]
    cand_shape = int(cand.shape[0] / 3)
    mask = cand[:cand_shape] != -1
    true_count = mask.sum().item()
    channel_1 = cand[:cand_shape][mask]
    stride_1 = cand[cand_shape : cand_shape * 2][mask]
    conv_type_1 = cand[cand_shape * 2 :][mask]
    logger.info(
        "Best channel.{}\n, Best stride.{}\n, Best conv.{}\n, Best layer_num.{}".format(
            channel_1.cpu().numpy().tolist(),
            stride_1.cpu().numpy().tolist(),
            conv_type_1.cpu().numpy().tolist(),
            true_count,
        )
    )


if __name__ == "__main__":
    main()
