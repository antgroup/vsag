import matplotlib.pyplot as plt
import json
import os
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['fontsize'] = 'stix'
f_size = 20
is_10m = True

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

data_ivf = """
100 20.0
time cost:  0.011262768507003784 s
recall:  0.8907
100 25.0
time cost:  0.01119942831993103 s
recall:  0.8907
200 20.0
time cost:  0.02227343511581421 s
recall:  0.9242
200 25.0
time cost:  0.020970146656036377 s
recall:  0.9244
300 20.0
time cost:  0.02960456919670105 s
recall:  0.9412
300 25.0
time cost:  0.029481457948684692 s
recall:  0.9414
350 20.0
time cost:  0.033858831882476806 s
recall:  0.9476
350 25.0
time cost:  0.03385238480567932 s
recall:  0.9478
400 20.0
time cost:  0.03911312174797058 s
recall:  0.952
400 25.0
time cost:  0.03940741944313049 s
recall:  0.9522
500 20.0
time cost:  0.046947184085845944 s
recall:  0.9579
500 25.0
time cost:  0.046777939558029176 s
recall:  0.9581000000000001
1000 20.0
time cost:  0.09032109689712524 s
recall:  0.9718
1000 25.0
time cost:  0.08879923391342164 s
recall:  0.9722000000000001
"""

data_scann = """
QPS: 43.46, Recall: 0.8788
QPS: 40.52, Recall: 0.8805
QPS: 36.54, Recall: 0.8817
QPS: 33.37, Recall: 0.8951
QPS: 30.68, Recall: 0.9062
QPS: 28.37, Recall: 0.9067
QPS: 24.41, Recall: 0.9084
QPS: 20.61, Recall: 0.9100
QPS: 16.88, Recall: 0.9172
QPS: 14.27, Recall: 0.9315
QPS: 11.07, Recall: 0.9464
"""

data_nndescent = """"""

if is_10m:

    data_scann = """
    QPS: 530.45, Recall: 0.8233
    QPS: 475.03, Recall: 0.8250
    QPS: 420.10, Recall: 0.8291
    QPS: 396.73, Recall: 0.8541
    QPS: 364.64, Recall: 0.8663
    QPS: 290.72, Recall: 0.8669
    QPS: 302.42, Recall: 0.8698
    QPS: 255.82, Recall: 0.8709
    QPS: 211.29, Recall: 0.8802
    QPS: 180.04, Recall: 0.9035
    QPS: 156.39, Recall: 0.9209
    QPS: 140.17, Recall: 0.9227
    QPS: 116.93, Recall: 0.9453
    """

    data_nndescent = """
    QPS: 384.30, Recall: 0.8767
    QPS: 282.47, Recall: 0.9157
    QPS: 229.71, Recall: 0.9494
    QPS: 151.41, Recall: 0.9773
    QPS: 89.04, Recall: 0.9837
    QPS: 45.05, Recall: 0.9872
    QPS: 11.85, Recall: 0.9895
    """

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

    data_ivf = """
    100 10.0
    time cost:  0.00769732580628506 s
    recall:  0.7883720930232557
    100 20.0
    time cost:  0.007815811523171358 s
    recall:  0.788953488372093
    100 25.0
    time cost:  0.007906980292741643 s
    recall:  0.7895348837209303
    200 10.0
    time cost:  0.013993063638376635 s
    recall:  0.8424418604651163
    200 20.0
    time cost:  0.014225567496100138 s
    recall:  0.8441860465116279
    200 25.0
    time cost:  0.014196278050888416 s
    recall:  0.8441860465116279
    300 10.0
    time cost:  0.019959397094194278 s
    recall:  0.8738372093023257
    300 20.0
    time cost:  0.02006053370098735 s
    recall:  0.8755813953488372
    300 25.0
    time cost:  0.020705310411231463 s
    recall:  0.8761627906976743
    350 10.0
    time cost:  0.02336385638214821 s
    recall:  0.8837209302325582
    350 20.0
    time cost:  0.022855005985082583 s
    recall:  0.8854651162790699
    350 25.0
    time cost:  0.022953425728997518 s
    recall:  0.8854651162790699
    400 10.0
    time cost:  0.025594920613044917 s
    recall:  0.8918604651162791
    400 20.0
    time cost:  0.025726498559463854 s
    recall:  0.8930232558139535
    400 25.0
    time cost:  0.02693087139794993 s
    recall:  0.8924418604651163
    500 10.0
    time cost:  0.03122556209564209 s
    recall:  0.9029069767441861
    500 20.0
    time cost:  0.03125909594602363 s
    recall:  0.9040697674418605
    500 25.0
    time cost:  0.0313526613767757 s
    recall:  0.9040697674418605
    1000 10.0
    time cost:  0.059341328088627306 s
    recall:  0.9377906976744187
    1000 20.0
    time cost:  0.05869978666305542 s
    recall:  0.9383720930232559
    1000 25.0
    time cost:  0.05818388073943382 s
    recall:  0.9395348837209302
    """

def process_data(data):
    qps_values5 = []
    recall_values5 = []
    for line in data.strip().split('\n'):
        # 分割每一行的键值对
        parts = line.split(',')
        qps_part = parts[0].strip()  # QPS部分
        recall_part = parts[1].strip()  # Recall部分

        # 提取数值
        qps_value = float(qps_part.split(':')[1].strip())
        recall_value = float(recall_part.split(':')[1].strip())

        # 添加到对应的列表中
        qps_values5.append(qps_value)
        recall_values5.append(recall_value)
    return qps_values5, recall_values5

# 将数据解析为列表
data_list1 = [json.loads(line) for line in data_hnswlib.strip().split("\n")]
data_list2 = [json.loads(line) for line in data2.strip().split("\n")]

# 提取数据集 1 的字段
qps_values1 = [item["QPS"] for item in data_list1]
recall_values1 = [item["Recall"] for item in data_list1]

# 提取数据集 2 的字段
qps_values2 = [item["QPS"] for item in data_list2]
recall_values2 = [item["Recall"] for item in data_list2]

qps_values3 = []
recall_values3 = []
lines = data_ivf.strip().split("\n")

# 遍历每一行，解析所需数据
for i in range(0, len(lines), 3):  # 每3行为一组
    # 提取 time cost 行
    time_cost_line = lines[i + 1].strip()
    time_cost_value = float(time_cost_line.split(":")[1].split("s")[0].strip())

    # 提取 recall 行
    recall_line = lines[i + 2].strip()
    recall_value = float(recall_line.split(":")[1].strip())

    # 计算 QPS
    qps = 1.0 / time_cost_value

    # 存储结果
    qps_values3.append(qps)
    recall_values3.append(recall_value)

qps_values4, recall_values4 = process_data(data_scann)

qps_values5 = []
recall_values5 = []
if is_10m:
    qps_values5, recall_values5 = process_data(data_nndescent)


# 定义样式

LINE_STYLES = {
    "full_redundant": {"color": "#4CAF50", "marker": "o"},  # 清新绿色 + 圆形
    "partial_redundant": {"color": "#03A9F4", "marker": "s"},  # 清新蓝色 + 方形
    "no_redundant": {"color": "#FFC107", "marker": "^"},  # 清新黄色 + 三角形
    "no_opt": {"color": "#E91E63", "marker": "D"},  # 清新粉色 + 菱形
    "no_bdc": {"color": "#9C27B0", "marker": "*"},  # 清新紫色 + 星形
    "no_bdc_opt": {"color": "#FF5722", "marker": "p"},  # 清新橙色 + 五边形
    "no_prefetch": {"color": "#00BCD4", "marker": "h"},  # 清新青色 + 六边形
    "hnswlib": {"color": "#795548", "marker": "X"}  # 清新棕色 + 十字
}


LINE_STYLES = {
    "vsag": {"color": "#03A9F4", "marker": "o"},  # 清新蓝色 + 圆形
    "hnswlib": {"color": "#FF5722", "marker": "s"},  # 清新橙色 + 方形
    "faiss-ivfpqfs": {"color": "#E91E63", "marker": ">"},  # 红色 + 三角形
    "scann": {"color": "#4CAF50", "marker": "v"},
    "nndescent": {"color": "#9C27B0", "marker": "D"},
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


# 数据集 3 的 QPS vs Recall 曲线
style3 = LINE_STYLES["faiss-ivfpqfs"]
plt.plot(recall_values3, qps_values3,
         label="faiss-ivfpqfs", marker=style3["marker"], color=style3["color"],
         markersize=8, markeredgecolor="black", linewidth=2, zorder=2)


# 数据集 3 的 QPS vs Recall 曲线
style4 = LINE_STYLES["scann"]
plt.plot(recall_values4, qps_values4,
         label="scann", marker=style4["marker"], color=style4["color"],
         markersize=8, markeredgecolor="black", linewidth=2, zorder=2)


# 数据集 3 的 QPS vs Recall 曲线
style5 = LINE_STYLES["nndescent"]
plt.plot(recall_values5, qps_values5,
         label="nndescent", marker=style5["marker"], color=style5["color"],
         markersize=8, markeredgecolor="black", linewidth=2, zorder=2)

# 设置图形属性
plt.xlabel("Recall@10", fontsize=f_size)
plt.ylabel("QPS", fontsize=f_size)
plt.grid(True, linestyle="--", alpha=0.6)  # 添加网格线，使用浅灰色虚线
plt.tight_layout()

# 设置 y 轴为对数刻度，并调整范围使曲线差距更明显
plt.yscale("log")
plt.ylim(10**1, 10**4)  # 调整 QPS 轴范围
plt.yticks([10**2, 10**3, 10**4], ["$10^2$", "$10^3$", "$10^4$"], fontsize=f_size)
def format_func(value, tick_number):
    return f"{value:.2f}"  # 保留两位小数

# 应用格式化器
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
plt.xticks(fontsize=f_size)

# 保存图表为 PNG
output_dir = "./fig"
os.makedirs(output_dir, exist_ok=True)
if is_10m:
    plt.savefig(os.path.join(output_dir, "10m_recall@10.png"), format="png")
    plt.savefig(os.path.join(output_dir, "10m_recall@10.pdf"), format="pdf")
else:
    plt.savefig(os.path.join(output_dir, "100m_recall@10.png"), format="png")
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
        Line2D([0], [0], marker=LINE_STYLES["vsag"]["marker"], color="w",
               label="vsag", markerfacecolor=LINE_STYLES["vsag"]["color"],
               markersize=10, markeredgecolor="black"),
        Line2D([0], [0], marker=LINE_STYLES["hnswlib"]["marker"], color="w",
               label="hnswlib", markerfacecolor=LINE_STYLES["hnswlib"]["color"],
               markersize=10, markeredgecolor="black"),
        Line2D([0], [0], marker=LINE_STYLES["nndescent"]["marker"], color="w",
               label="nndescent", markerfacecolor=LINE_STYLES["nndescent"]["color"],
               markersize=10, markeredgecolor="black"),
        Line2D([0], [0], marker=LINE_STYLES["faiss-ivfpqfs"]["marker"], color="w",
               label="faiss-ivfpqfs", markerfacecolor=LINE_STYLES["faiss-ivfpqfs"]["color"],
               markersize=10, markeredgecolor="black"),
        Line2D([0], [0], marker=LINE_STYLES["scann"]["marker"], color="w",
               label="scann", markerfacecolor=LINE_STYLES["scann"]["color"],
               markersize=10, markeredgecolor="black"),
    ]

    # 绘制图例
    legend = ax.legend(
        handles=legend_elements,
        loc="center",  # 图例位置
        ncol=5,  # 每排两个图例项
        fontsize=16,  # 字体大小
        frameon=False,  # 不显示边框
        handlelength=1.0,  # 标记长度
        handletextpad=0.3,  # 标记与文本之间的间距
        columnspacing=0.5,  # 列与列之间的间距
        borderpad=0.2  # 内边距（如果有边框）
    )

    # 保存图例为 PDF 文件
    plt.savefig(os.path.join(output_dir, "legend.png"), format="png", bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "legend.pdf"), format="pdf", bbox_inches="tight")
    plt.close()  # 关闭图形以释放内存

# 调用绘制图例函数
draw_legend(output_dir)
