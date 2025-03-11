import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import random

dataset_s = [
    "sift-128-euclidean",
    "fashion-mnist-784-euclidean",
    "glove-100-angular"]

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
import random

def plot_combined_pareto_with_random_files(df_manual, df_auto):
    """
    从 manual 和 auto 的 DataFrame 中提取 Pareto Front，并绘制在同一张图中。
    包括 manual 和 auto 的总体 Pareto Front，以及随机选择的两个 manual 文件的 Pareto Front。

    参数:
        df_manual (pd.DataFrame): 包含 'file', 'recall' 和 'QPS' 列的 manual 模式数据。
        df_auto (pd.DataFrame): 包含 'recall' 和 'QPS' 列的 auto 模式数据。
    """
    def extract_pareto_front(df):
        """辅助函数：提取 Pareto Front"""
        df = df.sort_values(by="recall").reset_index(drop=True)
        pareto_front = []
        max_qps = float('-inf')
        for i in range(len(df) - 1, -1, -1):  # 从后向前遍历
            if df.loc[i, "QPS"] > max_qps:
                pareto_front.append(df.loc[i])
                max_qps = df.loc[i, "QPS"]
        return pd.DataFrame(pareto_front).sort_values(by="recall").reset_index(drop=True)

    # 提取 manual 和 auto 的总体 Pareto Front
    pareto_manual_total = extract_pareto_front(df_manual)
    pareto_auto_total = extract_pareto_front(df_auto)

    # 获取 manual 模式的所有文件名
    manual_files = df_manual["file"].unique()
    if len(manual_files) < 2:
        raise ValueError("Not enough files in manual mode to pick two Pareto Fronts.")

    # 随机选择两个 manual 文件
    random_files = random.sample(list(manual_files), 2)
    df_manual_1 = df_manual[df_manual["file"] == random_files[0]]
    df_manual_2 = df_manual[df_manual["file"] == random_files[1]]

    # 提取两个 manual 文件的 Pareto Front
    pareto_manual_1 = extract_pareto_front(df_manual_1)
    pareto_manual_2 = extract_pareto_front(df_manual_2)

    # 绘制散点图和 Pareto Front 曲线
    plt.figure(figsize=(10, 6))

    # 所有散点
    plt.scatter(df_manual["recall"], df_manual["QPS"], label="Manual Mode (All Points)", alpha=0.5, color="gray")
    plt.scatter(df_auto["recall"], df_auto["QPS"], label="Auto Mode (All Points)", alpha=0.5, color="lightblue")

    # 总体 Pareto Front 曲线
    plt.plot(pareto_manual_total["recall"], pareto_manual_total["QPS"],
             label="Manual Mode (Overall Pareto Front)",
             marker='o', color="red", linewidth=2, linestyle="--")
    plt.plot(pareto_auto_total["recall"], pareto_auto_total["QPS"],
             label="Auto Mode (Overall Pareto Front)",
             marker='s', color="blue", linewidth=2)

    # 随机选择的两个 manual 文件的 Pareto Front
    plt.plot(pareto_manual_1["recall"], pareto_manual_1["QPS"],
             label=f"Random Manual File 1: {random_files[0]} (Pareto Front)",
             marker='^', color="green", linewidth=2)
    plt.plot(pareto_manual_2["recall"], pareto_manual_2["QPS"],
             label=f"Random Manual File 2: {random_files[1]} (Pareto Front)",
             marker='*', color="purple", linewidth=2)

    # 添加图例、标题和轴标签
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("QPS", fontsize=12)
    plt.title("Comparison of Overall and Random Manual Pareto Fronts with Auto Pareto Front", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"/tbase-project/vsag/experiment_auto_param/search/fig/{dataset}.jpg")


dataset = dataset_s[0]
df_manual, df_auto = extract_data()
plot_combined_pareto_with_random_files(df_manual, df_auto)
