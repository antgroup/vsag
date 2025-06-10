import matplotlib.pyplot as plt
import json
import os
from matplotlib.lines import Line2D

data_hnswlib = """
{"QPS":1290.3226318359375,"RT":0.0007749999640509486,"Recall":0.9235000014305115,"ef_search":30}
{"QPS":814.3322143554688,"RT":0.0012280000373721123,"Recall":0.9577000141143799,"ef_search":60}
{"QPS":723.5889892578125,"RT":0.0013819999294355512,"Recall":0.9627000093460083,"ef_search":70}
{"QPS":643.0868530273438,"RT":0.001554999966174364,"Recall":0.9678000211715698,"ef_search":80}
{"QPS":584.4535522460938,"RT":0.0017109999898821115,"Recall":0.9714999794960022,"ef_search":90}
{"QPS":537.0569458007813,"RT":0.0018619999755173922,"Recall":0.9722999930381775,"ef_search":100}
{"QPS":354.8616027832031,"RT":0.0028180000372231007,"Recall":0.9836999773979187,"ef_search":150}
{"QPS":271.444091796875,"RT":0.003684000112116337,"Recall":0.9860000014305115,"ef_search":200}
{"QPS":216.30975341796875,"RT":0.004623000044375658,"Recall":0.9886999726295471,"ef_search":250}
{"QPS":180.40771484375,"RT":0.005543000064790249,"Recall":0.9908999800682068,"ef_search":300}
"""

data2 = """
{"QPS":2392.344482421875,"RT":0.0004180000105407089,"Recall":0.895799994468689,"ef_search":30}
{"QPS":1686.3406982421875,"RT":0.0005929999751970172,"Recall":0.9395999908447266,"ef_search":60}
{"QPS":1529.052001953125,"RT":0.0006539999740198255,"Recall":0.9503999948501587,"ef_search":70}
{"QPS":1390.8206787109375,"RT":0.0007190000033006072,"Recall":0.9571999907493591,"ef_search":80}
{"QPS":1287.001220703125,"RT":0.0007770000374875963,"Recall":0.9617000222206116,"ef_search":90}
{"QPS":1206.2725830078125,"RT":0.0008289999677799642,"Recall":0.964900016784668,"ef_search":100}
{"QPS":753.0120849609375,"RT":0.0013279999839141965,"Recall":0.9769999980926514,"ef_search":150}
{"QPS":625.0,"RT":0.001600000075995922,"Recall":0.9829999804496765,"ef_search":200}
{"QPS":534.4735107421875,"RT":0.0018710000440478325,"Recall":0.9883999824523926,"ef_search":250}
{"QPS":467.9457092285156,"RT":0.002136999974027276,"Recall":0.9908000230789185,"ef_search":300}
"""

# 将数据解析为列表
data_list1 = [json.loads(line) for line in data_hnswlib.strip().split("\n")]
data_list2 = [json.loads(line) for line in data2.strip().split("\n")]

# 提取数据集 1 的字段
qps_values1 = [item["QPS"] for item in data_list1]
recall_values1 = [item["Recall"] for item in data_list1]

# 提取数据集 2 的字段
qps_values2 = [item["QPS"] for item in data_list2]
recall_values2 = [item["Recall"] for item in data_list2]

# 定义样式
LINE_STYLES = {
    "vsag": {"color": "#03A9F4", "marker": "o"},  # 清新蓝色 + 圆形
    "hnswlib": {"color": "#FF5722", "marker": "s"},  # 清新橙色 + 方形
    "faiss-ivfpqfs": {"color": "red", "marker": ">"},  # 红色 + 三角形
}

# 绘制曲线
plt.figure(figsize=(5, 4), dpi=600)

# 数据集 1 的 QPS vs Recall 曲线
style1 = LINE_STYLES["hnswlib"]
plt.plot(recall_values1, qps_values1,
         label="hnswlib", marker=style1["marker"], color=style1["color"],
         markersize=8, markeredgecolor="black", linewidth=2, zorder=2)

# 数据集 2 的 QPS vs Recall 曲线
style2 = LINE_STYLES["vsag"]
plt.plot(recall_values2, qps_values2,
         label="vsag", marker=style2["marker"], color=style2["color"],
         markersize=8, markeredgecolor="black", linewidth=2, zorder=2)

# 设置图形属性
f_size = 16
plt.xlabel("Recall@10", fontsize=f_size)
plt.ylabel("QPS", fontsize=f_size)
plt.grid(True, linestyle="--", alpha=0.6)  # 添加网格线，使用浅灰色虚线
plt.tight_layout()

# 设置 y 轴为对数刻度，并调整范围使曲线差距更明显
plt.yscale("log")
plt.ylim(10**2, 10**4)  # 调整 QPS 轴范围
plt.yticks([10**2, 10**3, 10**4], ["$10^2$", "$10^3$", "$10^4$"], fontsize=f_size)
plt.xticks(fontsize=f_size)

# 保存图表为 PNG
output_dir = "./fig"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "100m_recall@10.pdf"), format="pdf")

# 显示图形
plt.show()

# 单独绘制图例
def draw_legend(output_dir):
    """
    绘制单独的图例并保存为 PDF 文件。
    """
    # 创建一个新的画布用于绘制图例
    fig, ax = plt.subplots(figsize=(3, 0.1), dpi=600)  # 图例画布大小
    ax.axis("off")  # 关闭坐标轴

    # 创建图例项
    legend_elements = [
        Line2D([0], [0], marker=LINE_STYLES["hnswlib"]["marker"], color="w",
               label="hnswlib", markerfacecolor=LINE_STYLES["hnswlib"]["color"],
               markersize=10, markeredgecolor="black"),
        Line2D([0], [0], marker=LINE_STYLES["vsag"]["marker"], color="w",
               label="vsag", markerfacecolor=LINE_STYLES["vsag"]["color"],
               markersize=10, markeredgecolor="black"),
        Line2D([0], [0], marker=LINE_STYLES["faiss-ivfpqfs"]["marker"], color="w",
               label="faiss-ivfpqfs", markerfacecolor=LINE_STYLES["faiss-ivfpqfs"]["color"],
               markersize=10, markeredgecolor="black"),
    ]

    # 绘制图例
    legend = ax.legend(
        handles=legend_elements,
        loc="center",  # 图例位置
        ncol=3,  # 每排两个图例项
        fontsize=16,  # 字体大小
        frameon=False,  # 不显示边框
        handlelength=1.0,  # 标记长度
        handletextpad=0.3,  # 标记与文本之间的间距
        columnspacing=0.5,  # 列与列之间的间距
        borderpad=0.2  # 内边距（如果有边框）
    )

    # 保存图例为 PDF 文件
    plt.savefig(os.path.join(output_dir, "legend.pdf"), format="pdf", bbox_inches="tight")
    plt.close()  # 关闭图形以释放内存

# 调用绘制图例函数
draw_legend(output_dir)
