# Support both module execution and direct script invocation.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import os
from typing import Dict, List, Optional, Tuple

# 设置matplotlib非交互式后端，提高批量生成图片的性能
import matplotlib

# 先导入numpy和torch，再设置matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # 使用Agg后端，适合批量生成图片
import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["font.serif"] = ["Times New Roman"]
# 其他性能优化参数
plt.rcParams["savefig.dpi"] = 300  # 设置保存图片的分辨率
plt.rcParams["figure.dpi"] = 100  # 设置显示图片的分辨率（不影响保存质量）
plt.rcParams["agg.path.chunksize"] = 10000  # 提高大图片的渲染性能


class ParetoAnalyzer:
    """帕累托分析器类，支持多种目标模式"""

    def __init__(self, mode: str = "ops"):
        """
        初始化分析器

        Args:
            mode: 分析模式，支持 "ops" 或 "latency"
        """
        self.mode = mode
        self.target_name = "Ops" if mode == "ops" else "Latency"
        self.target_key = "epoch_ops" if mode == "ops" else "epoch_latency"

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """加载检查点文件"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return checkpoint

    def normalize_data(self, data: np.ndarray, min_val: float = None, max_val: float = None) -> np.ndarray:
        """归一化函数，使用公式: (x - min) / (max - min)"""
        if min_val is None:
            min_val = np.min(data)
        if max_val is None:
            max_val = np.max(data)

        if max_val == min_val:
            return np.ones_like(data)

        return (data - min_val) / (max_val - min_val)

    def normalize_target_acc(
        self,
        acc: np.ndarray,
        target: np.ndarray,
        global_min_acc: float = None,
        global_max_acc: float = None,
        global_min_target: float = None,
        global_max_target: float = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用指定的归一化公式:
        Acc归一化: (Acc_max - Acc(a)) / (Acc_max - Acc_min)
        Target归一化: (Target(a) - Target_min) / (Target_max - Target_min)
        """
        if global_min_acc is None:
            global_min_acc = np.min(acc)
        if global_max_acc is None:
            global_max_acc = np.max(acc)
        if global_min_target is None:
            global_min_target = np.min(target)
        if global_max_target is None:
            global_max_target = np.max(target)

        # Acc归一化: (Acc_max - Acc(a)) / (Acc_max - Acc_min)
        acc_normalized = (global_max_acc - acc) / (global_max_acc - global_min_acc)

        # Target归一化: (Target(a) - Target_min) / (Target_max - Target_min)
        target_normalized = (target - global_min_target) / (global_max_target - global_min_target)

        return acc_normalized, target_normalized, global_min_acc, global_max_acc, global_min_target, global_max_target

    def non_dominated_sort(self, points: np.ndarray, maximize: List[bool] = None) -> List[List[int]]:
        """
        非支配排序算法
        points: (n_points, n_objectives) 形状的数组
        maximize: 每个目标是否要最大化，True表示最大化，False表示最小化
        """
        n_points = points.shape[0]

        if maximize is None:
            maximize = [False] * points.shape[1]  # 默认都是最小化

        # 调整目标方向：统一为最小化
        adjusted_points = points.copy()
        for i, max_flag in enumerate(maximize):
            if max_flag:
                adjusted_points[:, i] = -adjusted_points[:, i]  # 最大化转为最小化

        # 计算支配关系
        dominated_by = [[] for _ in range(n_points)]
        domination_count = np.zeros(n_points, dtype=int)

        for i in range(n_points):
            for j in range(i + 1, n_points):
                # 检查i是否支配j
                i_dominates_j = np.all(adjusted_points[i] <= adjusted_points[j]) and np.any(
                    adjusted_points[i] < adjusted_points[j]
                )
                # 检查j是否支配i
                j_dominates_i = np.all(adjusted_points[j] <= adjusted_points[i]) and np.any(
                    adjusted_points[j] < adjusted_points[i]
                )

                if i_dominates_j:
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif j_dominates_i:
                    dominated_by[j].append(i)
                    domination_count[i] += 1

        # 分层排序
        fronts = []
        current_front = np.where(domination_count == 0)[0].tolist()

        while current_front:
            fronts.append(current_front)
            next_front = []

            for i in current_front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)

            current_front = next_front

        return fronts

    def get_pareto_front(self, points: np.ndarray, maximize: List[bool] = None) -> Tuple[np.ndarray, List[int]]:
        """获取帕累托前沿点"""
        fronts = self.non_dominated_sort(points, maximize)
        pareto_indices = fronts[0] if fronts else []
        return points[pareto_indices], pareto_indices

    def filter_outliers_statistical(self, target: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """基于统计方法剔除异常值"""
        if len(target) < 3:  # 数据太少时直接返回所有
            return np.arange(len(target))

        # 计算IQR（四分位距）
        q1_target, q3_target = np.percentile(target, [25, 75])
        iqr_target = q3_target - q1_target
        lower_target = q1_target - 1.5 * iqr_target
        upper_target = q3_target + 1.5 * iqr_target

        q1_acc, q3_acc = np.percentile(acc, [25, 75])
        iqr_acc = q3_acc - q1_acc
        lower_acc = q1_acc - 1.5 * iqr_acc
        upper_acc = q3_acc + 1.5 * iqr_acc

        valid_indices = []
        for i in range(len(target)):
            if (
                lower_target <= target[i] <= upper_target
                and lower_acc <= acc[i] <= upper_acc
                and target[i] > 0
                and 0 <= acc[i] <= 1.0
            ):
                valid_indices.append(i)

        return np.array(valid_indices)

    def collect_all_epochs_data(self, checkpoint: Dict, epochs: List[int]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """收集所有epoch的数据"""
        all_target = []
        all_acc = []
        epoch_labels = []

        for epoch in epochs:
            if epoch in checkpoint["epoch_candidates"]:
                target_data = checkpoint[self.target_key][epoch]
                acc = checkpoint["epoch_acc"][epoch]

                # 转换为numpy数组
                target_array = np.array([t.item() if isinstance(t, torch.Tensor) else t for t in target_data])
                acc_array = np.array([a.item() if isinstance(a, torch.Tensor) else a for a in acc])

                # 剔除异常值
                valid_indices = self.filter_outliers_statistical(target_array, acc_array)
                target_array = target_array[valid_indices]
                acc_array = acc_array[valid_indices]

                all_target.extend(target_array)
                all_acc.extend(acc_array)
                epoch_labels.extend([epoch] * len(target_array))

        return np.array(all_target), np.array(all_acc), epoch_labels

    def analyze_epoch_data_global(
        self,
        checkpoint: Dict,
        epoch: int,
        global_min_acc: float,
        global_max_acc: float,
        global_min_target: float,
        global_max_target: float,
    ) -> Dict:
        """分析单个epoch的数据，使用全局归一化参数"""
        if epoch not in checkpoint["epoch_candidates"]:
            raise ValueError(f"Epoch {epoch} 的数据不存在")

        # 获取该epoch的数据
        candidates = checkpoint["epoch_candidates"][epoch]
        target_data = checkpoint[self.target_key][epoch]
        acc = checkpoint["epoch_acc"][epoch]

        # 转换为numpy数组
        target_array = np.array([t.item() if isinstance(t, torch.Tensor) else t for t in target_data])
        acc_array = np.array([a.item() if isinstance(a, torch.Tensor) else a for a in acc])

        # 剔除异常值
        valid_indices = self.filter_outliers_statistical(target_array, acc_array)
        target_array = target_array[valid_indices]
        acc_array = acc_array[valid_indices]
        candidates = [candidates[i] for i in valid_indices]  # 同时更新候选架构列表

        # 使用全局参数归一化
        acc_normalized, target_normalized, _, _, _, _ = self.normalize_target_acc(
            acc_array, target_array, global_min_acc, global_max_acc, global_min_target, global_max_target
        )

        # 组合目标函数 (target越小越好，acc越大越好)
        # 由于acc已经按照(Acc_max - Acc(a))归一化，所以acc_normalized越小越好
        # target_normalized也是越小越好
        objectives = np.column_stack([target_normalized, acc_normalized])

        # 非支配排序
        fronts = self.non_dominated_sort(objectives, maximize=[False, False])  # 两个目标都是最小化

        # 获取帕累托前沿
        pareto_points, pareto_indices = self.get_pareto_front(objectives, maximize=[False, False])

        return {
            "epoch": epoch,
            "candidates": candidates,
            "target": target_array,
            "acc": acc_array,
            "target_normalized": target_normalized,
            "acc_normalized": acc_normalized,
            "objectives": objectives,
            "fronts": fronts,
            "pareto_points": pareto_points,
            "pareto_indices": pareto_indices,
            "num_candidates": len(candidates),
            "num_pareto": len(pareto_indices),
        }

    def plot_pareto_front_global(self, analysis_results: Dict, save_path: str = None):
        """绘制帕累托前沿图（使用全局归一化）- 显示所有点，但突出显示帕累托前沿点"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6))

        epoch = analysis_results["epoch"]
        target = analysis_results["target"]
        acc = analysis_results["acc"]
        pareto_indices = analysis_results["pareto_indices"]
        target_normalized = analysis_results["target_normalized"]
        acc_normalized = analysis_results["acc_normalized"]

        # 优化：减少点的数量或调整散点图参数
        # 如果数据点过多，可以考虑采样或使用更高效的绘制方式
        if len(target) > 10000:
            # 当数据点超过10000时，使用plot替代scatter以提高性能
            ax1.plot(target, acc, "o", alpha=0.3, color="gray", markersize=3, label="All candidates")
            ax1.plot(
                target[pareto_indices],
                acc[pareto_indices],
                "*",
                alpha=0.9,
                label="Pareto front",
                color="red",
                markersize=8,
            )

            ax2.plot(
                target_normalized, acc_normalized, "o", alpha=0.3, color="gray", markersize=3, label="All candidates"
            )
            ax2.plot(
                target_normalized[pareto_indices],
                acc_normalized[pareto_indices],
                "*",
                alpha=0.9,
                label="Pareto front",
                color="red",
                markersize=8,
            )
        else:
            # 绘制所有点（灰色，较小）
            ax1.scatter(target, acc, alpha=0.3, color="gray", s=20, label="All candidates")
            # 绘制帕累托前沿点（红色星形，更大）
            ax1.scatter(
                target[pareto_indices],
                acc[pareto_indices],
                alpha=0.9,
                label="Pareto front",
                color="red",
                s=150,
                marker="*",
            )

            # 绘制所有点（灰色，较小）
            ax2.scatter(target_normalized, acc_normalized, alpha=0.3, color="gray", s=20, label="All candidates")
            # 绘制帕累托前沿点（红色星形，更大）
            ax2.scatter(
                target_normalized[pareto_indices],
                acc_normalized[pareto_indices],
                alpha=0.9,
                label="Pareto front",
                color="red",
                s=150,
                marker="*",
            )

        ax1.set_xlabel(f"{self.target_name} - Lower is better")
        ax1.set_ylabel("Accuracy (Acc) - Higher is better")
        ax1.set_title(f"Epoch {epoch} - Pareto Front (Original Data)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel(f"Normalized {self.target_name}")
        ax2.set_ylabel("Normalized Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Pareto front plot saved to: {save_path}")

        # 关闭图形，释放内存
        plt.close(fig)

    def compare_epochs_global(
        self,
        checkpoint: Dict,
        epochs: List[int],
        global_min_acc: float,
        global_max_acc: float,
        global_min_target: float,
        global_max_target: float,
        save_path: str = None,
    ):
        """比较多个epoch的帕累托前沿（使用全局归一化）- 显示所有点，但突出显示帕累托前沿点"""
        plt.figure(figsize=(6, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "8"]

        for i, epoch in enumerate(epochs):
            try:
                analysis = self.analyze_epoch_data_global(
                    checkpoint, epoch, global_min_acc, global_max_acc, global_min_target, global_max_target
                )
                target_norm = analysis["target_normalized"]
                acc_norm = analysis["acc_normalized"]
                pareto_indices = analysis["pareto_indices"]

                # 优化：减少点的数量或调整散点图参数
                if len(target_norm) > 10000:
                    # 当数据点超过10000时，使用plot替代scatter以提高性能
                    plt.plot(
                        target_norm,
                        acc_norm,
                        "o",
                        color=colors[i],
                        markersize=3,
                        alpha=0.2,
                        label=f"Epoch {epoch} (all)",
                    )
                    plt.plot(
                        target_norm[pareto_indices],
                        acc_norm[pareto_indices],
                        "*",
                        color=colors[i],
                        markersize=8,
                        alpha=0.8,
                        label=f"Epoch {epoch} (Pareto)",
                    )
                else:
                    # 绘制所有点（较小，较浅）
                    plt.scatter(
                        target_norm,
                        acc_norm,
                        color=colors[i],
                        s=20,
                        alpha=0.2,
                        label=f"Epoch {epoch} (all)",
                        marker=markers[i % len(markers)],
                    )

                    # 绘制帕累托前沿点（较大，较深，星形）
                    pareto_target_norm = target_norm[pareto_indices]
                    pareto_acc_norm = acc_norm[pareto_indices]
                    plt.scatter(
                        pareto_target_norm,
                        pareto_acc_norm,
                        color=colors[i],
                        s=100,
                        alpha=0.8,
                        label=f"Epoch {epoch} (Pareto)",
                        marker="*",
                    )

            except ValueError as e:
                print(f"Warning: {e}")
                continue

        plt.xlabel(f"Normalized {self.target_name})")
        plt.ylabel("Normalized Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Multi-epoch comparison plot saved to: {save_path}")

        # 关闭图形，释放内存
        plt.close()

    def save_analysis_results(self, analysis_results: Dict, output_dir: str, global_params: Dict):
        """保存分析结果到文件"""
        epoch = analysis_results["epoch"]
        result_file = os.path.join(output_dir, f"epoch_{epoch}_analysis_{self.mode}.txt")

        with open(result_file, "w") as f:
            f.write(f"Epoch {epoch} Analysis Results - Mode: {self.mode}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Number of candidate architectures: {analysis_results['num_candidates']}\n")
            f.write(f"Number of Pareto front points: {analysis_results['num_pareto']}\n")
            f.write(
                f"{self.target_name} range: {analysis_results['target'].min():.2f} - {analysis_results['target'].max():.2f}\n"
            )
            f.write(f"Acc range: {analysis_results['acc'].min():.4f} - {analysis_results['acc'].max():.4f}\n")
            f.write("Global normalization parameters:\n")
            f.write(f"  Acc_min: {global_params['min_acc']:.4f}, Acc_max: {global_params['max_acc']:.4f}\n")
            f.write(
                f"  {self.target_name}_min: {global_params['min_target']:.2f}, {self.target_name}_max: {global_params['max_target']:.2f}\n\n"
            )

            f.write("Pareto front points:\n")
            f.write(f"Index\t{self.target_name}\t\tAcc\t\tNormalized {self.target_name}\tNormalized Acc\n")
            for idx in analysis_results["pareto_indices"]:
                f.write(
                    f"{idx}\t\t{analysis_results['target'][idx]:.2f}\t{analysis_results['acc'][idx]:.4f}\t"
                    f"{analysis_results['target_normalized'][idx]:.4f}\t{analysis_results['acc_normalized'][idx]:.4f}\n"
                )

        return result_file

    def check_data_availability(self, checkpoint: Dict) -> bool:
        """检查检查点中是否包含所需的数据"""
        if self.mode == "latency" and "epoch_latency" not in checkpoint:
            print(f"Warning: Latency data not found in checkpoint. Available keys: {list(checkpoint.keys())}")
            return False
        return True


def main():
    parser = argparse.ArgumentParser(description="Analyze Pareto front during search process")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path")
    parser.add_argument(
        "--epoch", type=int, default=None, help="Specific epoch to analyze, if None analyze all saved epochs"
    )
    parser.add_argument("--output_dir", type=str, default="./pareto_analysis", help="Output directory")
    parser.add_argument(
        "--mode",
        type=str,
        default="ops",
        choices=["ops", "latency"],
        help="Pareto analysis mode: ops (operations) or latency",
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化分析器
    analyzer = ParetoAnalyzer(mode=args.mode)

    # 加载检查点
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = analyzer.load_checkpoint(args.checkpoint)

    # 检查数据可用性
    if not analyzer.check_data_availability(checkpoint):
        print(f"Data not available for mode: {args.mode}")
        return

    # 获取保存的epoch列表
    save_epochs = checkpoint.get("save_epochs", [])
    print(f"Saved epochs: {save_epochs}")
    print(f"Analysis mode: {args.mode}")

    if not save_epochs:
        print("No saved epoch data found")
        return

    # 分析指定epoch或所有epoch
    epochs_to_analyze = [args.epoch] if args.epoch is not None else save_epochs

    # 先收集所有epoch的数据，计算全局归一化参数
    print("\nCollecting all epoch data to calculate global normalization parameters...")
    all_target, all_acc, _ = analyzer.collect_all_epochs_data(checkpoint, epochs_to_analyze)

    # 计算全局最小最大值
    global_min_acc = np.min(all_acc)
    global_max_acc = np.max(all_acc)
    global_min_target = np.min(all_target)
    global_max_target = np.max(all_target)

    global_params = {
        "min_acc": global_min_acc,
        "max_acc": global_max_acc,
        "min_target": global_min_target,
        "max_target": global_max_target,
    }

    print(f"Global Acc range: {global_min_acc:.4f} - {global_max_acc:.4f}")
    print(f"Global {analyzer.target_name} range: {global_min_target:.2f} - {global_max_target:.2f}")

    # 对每个epoch进行分析
    for epoch in epochs_to_analyze:
        try:
            print(f"\nAnalyzing Epoch {epoch} (Mode: {args.mode}):")
            analysis = analyzer.analyze_epoch_data_global(
                checkpoint, epoch, global_min_acc, global_max_acc, global_min_target, global_max_target
            )

            print(f"  Number of candidate architectures: {analysis['num_candidates']}")
            print(f"  Number of Pareto front points: {analysis['num_pareto']}")
            print(f"  {analyzer.target_name} range: {analysis['target'].min():.2f} - {analysis['target'].max():.2f}")
            print(f"  Acc range: {analysis['acc'].min():.4f} - {analysis['acc'].max():.4f}")

            # 保存分析结果
            result_file = analyzer.save_analysis_results(analysis, args.output_dir, global_params)
            print(f"  Analysis results saved to: {result_file}")

            # 绘制帕累托前沿图
            plot_file = os.path.join(args.output_dir, f"epoch_{epoch}_pareto_{args.mode}.pdf")
            analyzer.plot_pareto_front_global(analysis, plot_file)

        except ValueError as e:
            print(f"  Error: {e}")
            continue

    # 如果分析多个epoch，绘制比较图
    if len(epochs_to_analyze) > 1:
        compare_file = os.path.join(args.output_dir, f"epochs_comparison_global_{args.mode}.pdf")
        analyzer.compare_epochs_global(
            checkpoint,
            epochs_to_analyze,
            global_min_acc,
            global_max_acc,
            global_min_target,
            global_max_target,
            compare_file,
        )


if __name__ == "__main__":
    main()
