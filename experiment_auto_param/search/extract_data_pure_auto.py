import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import random

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
f_size = 20

dataset_s = [
    "gist-960-euclidean",
    "sift-128-euclidean",
    "glove-100-angular"
]

baseline_s = {
    "glove-100-angular": ["search_glove-100-angular_C32_2.0_300_S16_1.6_-1_auto.txt",
                          "search_glove-100-angular_C32_2.0_300_S32_2.0_-1_auto.txt"],

    "sift-128-euclidean": ["search_sift-128-euclidean_C32_2.0_300_S32_2.0_-1_auto.txt",
                           "search_sift-128-euclidean_C32_2.0_300_S16_1.2_-1_auto.txt"],

    "gist-960-euclidean": ["search_gist-960-euclidean_C32_2.0_300_S16_1.0_-1_auto.txt",
                           "search_gist-960-euclidean_C32_2.0_300_S16_2.0_-1_auto.txt"]

}

NAME = {
    "glove-100-angular": "GLOVE-100",

    "sift-128-euclidean": "SIFT1M",

    "gist-960-euclidean": "GIST1M"
}

dataset = dataset_s[0]
def extract_data():
    # 定义文件路径
    base_dir = "/tbase-project/vsag/experiment_auto_param/search"

    # 正则表达式用于提取 recall 和 QPS
    pattern = r"recall: ([\d.]+), QPS: ([\d.]+)"

    # 初始化存储数据的字典
    data_manual = []
    data_auto = []

    # 遍历目录下的所有文件
    for mode in ["manual", "auto"]:
        mode_dir = os.path.join(base_dir, mode)

        # 检查模式目录是否存在
        if not os.path.exists(mode_dir):
            print(f"Directory for mode '{mode}' does not exist. Skipping...")
            continue

        for file_name in os.listdir(mode_dir):
            if not dataset in file_name:
                continue
            file_path = os.path.join(mode_dir, file_name)

            # 检查文件是否存在
            if not os.path.isfile(file_path):
                print(f"File '{file_path}' does not exist. Skipping...")
                continue

            # 打开文件并读取内容
            try:
                with open(file_path, "r") as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading file '{file_path}': {e}. Skipping...")
                continue

            # 使用正则表达式提取 recall 和 QPS
            matches = re.findall(pattern, content)
            if not matches:
                print(f"No valid data found in file '{file_path}'. Skipping...")
                continue

            for match in matches:
                recall, qps = float(match[0]), float(match[1])

                # 根据模式将数据归类
                if mode == "manual":
                    data_manual.append({"file": file_name, "recall": recall, "QPS": qps})
                else:
                    data_auto.append({"file": file_name, "recall": recall, "QPS": qps})

    # 转换为 Pandas DataFrame
    df_manual = pd.DataFrame(data_manual)
    df_auto = pd.DataFrame(data_auto)

    # 打印结果
    print("Manual Mode Data:")
    print(df_manual)

    print("\nAuto Mode Data:")
    print(df_auto)

    # 如果需要保存为 CSV 文件
    if not df_manual.empty:
        df_manual.to_csv("manual_data.csv", index=False)
    if not df_auto.empty:
        df_auto.to_csv("auto_data.csv", index=False)

    return df_manual, df_auto







import matplotlib.pyplot as plt
import os
import pandas as pd

def extract_pareto_front(df):
    """
    提取 Pareto Front，基于 Pareto 定义。
    """
    # 按 recall 排序
    df = df.sort_values(by="recall").reset_index(drop=True)

    pareto_front = []
    max_qps = float('-inf')

    # 从后向前遍历，确保 recall 升序
    for i in range(len(df) - 1, -1, -1):
        current_qps = df.loc[i, "QPS"]
        if current_qps > max_qps:  # 如果当前点的 QPS 更大，则加入 Pareto Front
            pareto_front.append(df.loc[i])
            max_qps = current_qps

    # 转换为 DataFrame 并按 recall 排序
    pareto_df = pd.DataFrame(pareto_front).sort_values(by="recall").reset_index(drop=True)
    return pareto_df

def plot_auto_with_manual_baselines(df_auto, baseline_files, dataset, filter_recall=False, recall_threshold=0.975):
    """
    从 auto 的 DataFrame 中提取 Pareto Front，并绘制在同一张图中。
    包括 auto 的总体 Pareto Front，以及手动选择的两个 baseline 文件的 Pareto Front。
    可选地过滤 recall >= recall_threshold 的部分。

    参数:
        df_auto (pd.DataFrame): 包含 'file', 'recall' 和 'QPS' 列的 auto 模式数据。
        baseline_files (list): 手动选择的两个 baseline 文件名。
        dataset (str): 数据集名称，用于保存图片文件名。
        filter_recall (bool): 是否过滤 recall >= recall_threshold 的部分，默认为 False。
        recall_threshold (float): recall 的过滤阈值，默认为 0.975。
    """
    # 提取 auto 的总体 Pareto Front
    pareto_auto_total = extract_pareto_front(df_auto)

    # 获取手动选择的两个 baseline 文件
    if len(baseline_files) != 2:
        raise ValueError("Exactly two baseline files must be provided.")

    df_baseline_1 = df_auto[df_auto["file"] == baseline_files[0]]
    df_baseline_2 = df_auto[df_auto["file"] == baseline_files[1]]

    # 提取两个 baseline 文件的 Pareto Front
    pareto_baseline_1 = extract_pareto_front(df_baseline_1)
    pareto_baseline_2 = extract_pareto_front(df_baseline_2)

    # 绘制散点图和 Pareto Front 曲线
    plt.figure(figsize=(5, 4), dpi=600)  # 6:4 比例，更高清晰度

    # 根据 filter_recall 决定是否过滤数据
    if filter_recall:
        df_auto_filtered = df_auto[df_auto["recall"] >= recall_threshold]
        pareto_auto_total_filtered = pareto_auto_total[pareto_auto_total["recall"] >= recall_threshold]
        pareto_baseline_1_filtered = pareto_baseline_1[pareto_baseline_1["recall"] >= recall_threshold]
        pareto_baseline_2_filtered = pareto_baseline_2[pareto_baseline_2["recall"] >= recall_threshold]
    else:
        df_auto_filtered = df_auto
        pareto_auto_total_filtered = pareto_auto_total
        pareto_baseline_1_filtered = pareto_baseline_1
        pareto_baseline_2_filtered = pareto_baseline_2

    # 所有散点
    plt.scatter(df_auto_filtered["recall"], df_auto_filtered["QPS"],
                label="All Running Cases", alpha=0.5, color="lightblue", s=10)

    # 总体 Pareto Front 曲线
    if not pareto_auto_total_filtered.empty:
        # 灰色虚线背景
        plt.plot(pareto_auto_total_filtered["recall"], pareto_auto_total_filtered["QPS"],
                 linestyle="--", color="gray", linewidth=1, zorder=1)
        # 主曲线
        plt.plot(pareto_auto_total_filtered["recall"], pareto_auto_total_filtered["QPS"],
                 label="Auto Tuned Result",
                 marker='s', color="#4CAF50", markersize=8, markeredgecolor="black", linewidth=2, zorder=2)

    # 手动选择的两个 baseline 文件的 Pareto Front
    if not pareto_baseline_1_filtered.empty:
        # 灰色虚线背景
        plt.plot(pareto_baseline_1_filtered["recall"], pareto_baseline_1_filtered["QPS"],
                 linestyle="--", color="gray", linewidth=1, zorder=1)
        # 主曲线
        plt.plot(pareto_baseline_1_filtered["recall"], pareto_baseline_1_filtered["QPS"],
                 label=f"Random Picked Config A",
                 marker='^', color="#FFC107", markersize=8, markeredgecolor="black", linewidth=2, zorder=2)
    if not pareto_baseline_2_filtered.empty:
        # 灰色虚线背景
        plt.plot(pareto_baseline_2_filtered["recall"], pareto_baseline_2_filtered["QPS"],
                 linestyle="--", color="gray", linewidth=1, zorder=1)
        # 主曲线
        plt.plot(pareto_baseline_2_filtered["recall"], pareto_baseline_2_filtered["QPS"],
                 label=f"Random Picked Config B",
                 marker='*', color="#03A9F4", markersize=10, markeredgecolor="black", linewidth=2, zorder=2)

    # 添加标题和轴标签
    plt.xlabel("Recall@10", fontsize=f_size)
    plt.ylabel("QPS", fontsize=f_size)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # 设置 y 轴为对数刻度，并固定刻度为 10^1, 10^2, 10^3, 10^4
    plt.yscale("log")
    plt.yticks([10**4], ["$10^4$"], fontsize=f_size)
    def format_func(value, tick_number):
        return f"{value:.2f}"  # 保留两位小数
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.xticks(fontsize=f_size)

    # 如果启用了过滤，则限定 x 轴范围
    if filter_recall:
        plt.xlim(recall_threshold, 1.0)

    # 确保目录存在
    output_dir = "/tbase-project/vsag/experiment_auto_param/search/fig_pure_auto"
    os.makedirs(output_dir, exist_ok=True)

    # 保存图片（不包含图例）
    # plt.savefig(f"{output_dir}/{dataset}.jpg", bbox_inches="tight")
    plt.savefig(f"{output_dir}/{dataset}.png", format="png", bbox_inches="tight")
    plt.savefig(f"{output_dir}/{dataset}.pdf", format="pdf", bbox_inches="tight")
    plt.close()  # 关闭图形以释放内存

def save_legend_as_image(dataset):
    """
    单独绘制图例并保存为图片。

    参数:
        dataset (str): 数据集名称，用于保存图片文件名。
    """
    # 创建一个空白图形
    fig = plt.figure(figsize=(4, 0.5),dpi=600)  # 宽度较大，高度较小
    ax = fig.add_subplot(111)

    # 定义图例内容
    handles = [
        plt.Line2D([0], [0], marker='s', color="#4CAF50", markersize=8, markeredgecolor="black", linewidth=2, linestyle="-", label="Auto Tuned Result"),
        plt.Line2D([0], [0], color="lightblue", linestyle="", marker='o', markersize=5, alpha=0.5, label="All Running Cases"),
        plt.Line2D([0], [0], marker='^', color="#FFC107", markersize=8, markeredgecolor="black", linewidth=2, linestyle="-", label="Random Picked Config A"),
        plt.Line2D([0], [0], marker='*', color="#03A9F4", markersize=10, markeredgecolor="black", linewidth=2, linestyle="-", label="Random Picked Config B"),
    ]

    # 绘制图例
    legend = ax.legend(handles=handles, loc="center", ncol=2, fontsize=f_size, frameon=False)

    # 隐藏坐标轴
    ax.axis("off")

    # 确保目录存在
    output_dir = "/tbase-project/vsag/experiment_auto_param/search/fig_pure_auto"
    os.makedirs(output_dir, exist_ok=True)

    # 保存图例图片
    # plt.savefig(f"{output_dir}/legend.jpg", bbox_inches="tight")
    plt.savefig(f"{output_dir}/legend.png", format="png", bbox_inches="tight")
    plt.savefig(f"{output_dir}/legend.pdf", format="pdf", bbox_inches="tight")
    plt.close()  # 关闭图形以释放内存




save_legend_as_image(dataset_s[0])
for dataset in dataset_s:
    # dataset = dataset_s[0]
    df_manual, df_auto = extract_data()
    baseline_files = baseline_s[dataset]

    # 不过滤 recall
    plot_auto_with_manual_baselines(df_auto, baseline_files, dataset)
