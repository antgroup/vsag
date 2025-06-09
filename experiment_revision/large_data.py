import matplotlib.pyplot as plt
import json

data_hnswlib = """
{"QPS":1594.8963623046875,"RT":0.0006270000012591481,"Recall":0.8076000213623047,"ef_search":10}
{"QPS":1592.356689453125,"RT":0.0006280000088736415,"Recall":0.8978999853134155,"ef_search":20}
{"QPS":1290.3226318359375,"RT":0.0007749999640509486,"Recall":0.9235000014305115,"ef_search":30}
{"QPS":1077.586181640625,"RT":0.000927999964915216,"Recall":0.9384999871253967,"ef_search":40}
{"QPS":931.0986938476563,"RT":0.0010740000288933516,"Recall":0.9483000040054321,"ef_search":50}
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

# 绘制曲线
plt.figure(figsize=(12, 8))

# 数据集 1 的 QPS vs Recall 曲线
plt.plot(recall_values1, qps_values1, label="hnswlib", marker='o', color='blue')

# 数据集 2 的 QPS vs Recall 曲线
plt.plot(recall_values2, qps_values2, label="VSAG", marker='s', color='red', linestyle='--')

# 添加标题和标签
plt.title("QPS vs Recall (Two Datasets)", fontsize=16)
plt.xlabel("Recall", fontsize=14)
plt.ylabel("QPS", fontsize=14)

# 添加图例
plt.legend(fontsize=12)

# 显示网格
plt.grid(True)

# 显示图形
plt.show()

plt.savefig("recall@100.png")
