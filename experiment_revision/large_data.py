import matplotlib.pyplot as plt
import json
import os
from matplotlib.lines import Line2D


data_hnswlib = """
{"QPS":2123.456787109375,"RT":0.00047093024477362633,"Recall":0.7749999761581421,"ef_search":10}
{"QPS":1421.4876708984375,"RT":0.0007034883601590991,"Recall":0.8947674632072449,"ef_search":20}
{"QPS":1095.5413818359375,"RT":0.0009127907105721533,"Recall":0.929651141166687,"ef_search":30}
{"QPS":900.5235595703125,"RT":0.0011104651493951678,"Recall":0.9441860318183899,"ef_search":40}
{"QPS":761.0619506835938,"RT":0.0013139534275978804,"Recall":0.9511628150939941,"ef_search":50}
{"QPS":659.0037841796875,"RT":0.0015174419386312366,"Recall":0.9593023061752319,"ef_search":60}
{"QPS":587.0307006835938,"RT":0.0017034884076565504,"Recall":0.9622092843055725,"ef_search":70}
{"QPS":525.993896484375,"RT":0.001901162788271904,"Recall":0.9645348787307739,"ef_search":80}
{"QPS":475.13812255859375,"RT":0.0021046511828899384,"Recall":0.9656976461410522,"ef_search":90}
{"QPS":435.4430236816406,"RT":0.002296511782333255,"Recall":0.9709302186965942,"ef_search":100}
{"QPS":305.5061950683594,"RT":0.003273255890235305,"Recall":0.9860464930534363,"ef_search":150}
{"QPS":239.2211456298828,"RT":0.0041802325285971165,"Recall":0.9854651093482971,"ef_search":200}
{"QPS":195.89976501464844,"RT":0.0051046512089669704,"Recall":0.9866279363632202,"ef_search":250}
{"QPS":167.64132690429688,"RT":0.005965116433799267,"Recall":0.9872093200683594,"ef_search":300}
"""

data2 = """
{"QPS":5212.12109375,"RT":0.0001918604684760794,"Recall":0.6226744055747986,"ef_search":10}
{"QPS":4300.0,"RT":0.000232558129937388,"Recall":0.6889534592628479,"ef_search":20}
{"QPS":3245.282958984375,"RT":0.000308139540720731,"Recall":0.8761627674102783,"ef_search":30}
{"QPS":2356.164306640625,"RT":0.0004244185984134674,"Recall":0.9040697813034058,"ef_search":50}
{"QPS":2150.0,"RT":0.000465116259874776,"Recall":0.9238371849060059,"ef_search":60}
{"QPS":1890.1099853515625,"RT":0.0005290697445161641,"Recall":0.9395349025726318,"ef_search":70}
{"QPS":1720.0,"RT":0.0005813953466713428,"Recall":0.949999988079071,"ef_search":80}
{"QPS":1563.6363525390625,"RT":0.0006395349046215415,"Recall":0.9563953280448914,"ef_search":90}
{"QPS":1421.4876708984375,"RT":0.0007034883601590991,"Recall":0.960465133190155,"ef_search":100}
{"QPS":939.8907470703125,"RT":0.001063953503035009,"Recall":0.9691860675811768,"ef_search":150}
{"QPS":744.5887451171875,"RT":0.0013430232647806406,"Recall":0.9738371968269348,"ef_search":200}
{"QPS":625.4545288085938,"RT":0.0015988372033461928,"Recall":0.9767441749572754,"ef_search":250}
{"QPS":539.1849365234375,"RT":0.001854651141911745,"Recall":0.9790697693824768,"ef_search":300}
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
plt.savefig(os.path.join(output_dir, "10m_recall@10.pdf"), format="pdf")

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
