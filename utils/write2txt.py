import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def write2txt_list(path, key_value):
    f = open(path, "w")
    for i in list(key_value):
        print(i, file=f)
    f.close()


def write2txt_all(path, key_value):
    f = open(path, "w")
    print(key_value, file=f)
    f.close()


def Draw_Linear_Graph(*args, xlabel="epoch", ylabel="loss", figsize=(10, 7), output_file_path=None):
    plt.figure(figsize=figsize, dpi=300)
    for i in args:
        x = np.linspace(0, len(i[0]), len(i[0]))
        plt.plot(x, i[0], label=i[1], linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(output_file_path)
    plt.close()


import pandas as pd
import seaborn as sns


def draw_scatter(actual_data, predicted_data, name="./scatter_plot.png"):
    # 设置seaborn样式
    sns.set_style("whitegrid")
    plt.rc("font", family="Times New Roman")

    # 设置图片清晰度和背景
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    fig, ax = plt.subplots(figsize=(8, 6))

    # 将数据转换为DataFrame格式，便于seaborn处理
    data = pd.DataFrame({"Actual": actual_data.flatten(), "Predicted": predicted_data.flatten()})

    # 使用seaborn绘制带置信区间的散点图
    # regplot会自动计算并绘制回归线和置信区间
    sns.regplot(
        data=data,
        x="Actual",
        y="Predicted",
        scatter_kws={"alpha": 0.7, "s": 50, "color": "#1f77b4", "edgecolors": "white", "linewidths": 0.8},
        line_kws={"color": "red", "linewidth": 1.5},
        ci=95,  # 95%置信区间
        ax=ax,
    )

    # 计算R²值
    slope, intercept, r_value, p_value, std_err = stats.linregress(actual_data.flatten(), predicted_data.flatten())

    # 添加标题和坐标轴标签
    ax.set_xlabel("Actual Values", fontsize=16, fontweight="bold")
    ax.set_ylabel("Predicted Values", fontsize=16, fontweight="bold")

    # 设置坐标轴范围和刻度
    min_val = min(min(actual_data), min(predicted_data))
    max_val = max(max(actual_data), max(predicted_data))
    margin = 0.02 * (max_val - min_val)
    ax.set_xlim([min_val - margin, max_val + margin])
    ax.set_ylim([min_val - margin, max_val + margin])

    # 添加网格线
    ax.grid(True, linestyle="-", alpha=0.4, color="#d3d3d3", linewidth=0.8)

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis="both", which="major", labelsize=13)

    # 添加图例（包括R²值）
    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], color="red", lw=2.5, label=f"R² = {r_value**2:.3f}")]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=13, frameon=True, fancybox=True, shadow=True)

    # 设置图框样式，使外框线更好地包裹图表
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_linewidth(1.2)
    ax.spines["right"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # 可选：设置边框颜色
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # 保存图片并关闭图形
    plt.tight_layout()
    plt.savefig(name, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
