import os
import re
from collections import defaultdict

# 定义日志文件夹路径
LOG_DIR = "/tbase-project/vsag/experiment_auto_param/auto_build_log"

# 正则表达式匹配所需信息
GRAPH_COST_PATTERN = re.compile(r"hnsw graph cost (\d+\.\d+)s")
SPACE_COST_PATTERN = re.compile(r"serialize with space cost: (\d+\.\d+) MB")

# 初始化统计结果
stats = defaultdict(lambda: {"total_time": 0, "total_space": 0})

# 遍历文件夹中的所有文件
for filename in os.listdir(LOG_DIR):
    if filename.startswith("build_"):
        # 提取数据集名称
        dataset_match = re.match(r"build_(.+?)_", filename)
        if not dataset_match:
            continue
        dataset = dataset_match.group(1)

        # 打开日志文件并解析内容
        file_path = os.path.join(LOG_DIR, filename)
        with open(file_path, "r") as f:
            content = f.read()

        # 提取 hnsw graph cost
        graph_cost_match = GRAPH_COST_PATTERN.search(content)
        if graph_cost_match:
            graph_cost = float(graph_cost_match.group(1))
            stats[dataset]["total_time"] += graph_cost

        # 提取 serialize space cost
        space_cost_match = SPACE_COST_PATTERN.search(content)
        if space_cost_match:
            space_cost = float(space_cost_match.group(1))
            stats[dataset]["total_space"] += space_cost

# 输出统计报告
print("Dataset Statistics Report:")
print("==========================")
for dataset, values in stats.items():
    total_hours = values["total_time"] / 3600  # 转换为小时
    total_gb = values["total_space"] / 1024    # 转换为 GB
    print(f"Dataset: {dataset}")
    print(f"  Total Build Time: {total_hours:.2f} hours")
    print(f"  Total Memory Space: {total_gb:.2f} GB")
