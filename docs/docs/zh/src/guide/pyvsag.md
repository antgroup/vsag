# pyvsag

`pyvsag` 是 VSAG 的 Python 绑定包，接口封装基于 [pybind11](https://github.com/pybind/pybind11) 实现，源代码位于仓库 [`python_bindings/`](https://github.com/antgroup/vsag/tree/main/python_bindings) 目录，打包脚本位于 [`python/`](https://github.com/antgroup/vsag/tree/main/python)。

## 安装

从 PyPI 安装最新发布版本：

```bash
pip install pyvsag
```

需要在 Linux 环境下使用（`manylinux2014` wheel）。如果希望构建本地 wheel，可以运行：

```bash
# 构建特定 Python 版本的 wheel
make pyvsag PY_VERSION=3.11

# 或一次构建所有受支持版本
make pyvsag-all
```

## 快速开始

`pyvsag` 暴露一个与 C++ `Index` 对象对应的 `Index` 类，构建与搜索参数使用 JSON 字符串传递：

```python
import json
import numpy as np
import pyvsag

dim = 128
num_elements = 1000

ids = np.arange(num_elements, dtype=np.int64)
data = np.float32(np.random.random((num_elements, dim)))

index_params = json.dumps({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": dim,
    "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 16,
        "ef_construction": 100,
    },
})

index = pyvsag.Index("hgraph", index_params)
failed_ids = index.build(vectors=data, ids=ids, num_elements=num_elements, dim=dim)
if failed_ids:
    print("未插入的 ID:", failed_ids)

query = np.float32(np.random.random(dim))
search_params = json.dumps({"hgraph": {"ef_search": 100}})
result_ids, result_dists = index.knn_search(
    vector=query, k=10, parameters=search_params,
)
for rid, rdist in zip(result_ids, result_dists):
    print(f"{rid}: {rdist}")
```

完整示例请查阅仓库中的 [`examples/python/`](https://github.com/antgroup/vsag/tree/main/examples/python) 目录，建议从 `103_index_hgraph.py` 开始。

## 保存与加载

```python
index.save("index.bin")

new_index = pyvsag.Index("hgraph", index_params)
new_index.load("index.bin")
```

`save` 与 `load` 保留普通 C++ 二进制流格式。底层序列化/反序列化失败抛出
`pyvsag.VsagError`；文件打开或最终流状态失败抛出 `OSError`。
这些检查不使保存成为事务操作：报错前已有文件可能被截断，加载失败也不保证回滚。
绑定不新增“原生持久化往返保留已删除 ID 状态”的保证；该原生持久化问题仍待处理。
请对重要索引执行 round-trip 校验，并验证应用所依赖的行为。

## 索引操作

除构建、搜索、保存与加载外，Python 绑定还提供以下索引管理方法：

| 方法 | 返回值 | 说明 |
| --- | --- | --- |
| `get_num_elements()` | `int` | 索引当前元素数量 |
| `get_memory_usage()` | `int` | 索引占用的内存字节数 |
| `check_id_exist(id)` | `bool` | 指定 ID 当前是否存在 |
| `get_min_max_id()` | `(min_id, max_id)` | 底层失败抛出 `VsagError`；支持情况和空索引行为取决于后端 |
| `add(vectors, ids, num_elements, dim)` | `list[int]` | 返回插入失败的 ID；整体失败抛出 `VsagError`；稠密 dtype 必须与索引一致 |
| `remove(ids)` | `int` | 返回实际删除数量；`0` 表示成功但未删除任何 ID，底层失败抛出 `VsagError` |
| `calc_distances_by_id(query, ids)` / `cal_distance_by_id(query, ids)`（旧别名） | `numpy.ndarray` | 以一维 float32 数组按 ID 顺序返回距离；缺失 ID 为 `-1`，底层参数无效或操作不受支持时抛出包含 C++ 错误码和信息的 `VsagError` |

```python
# 添加两条稠密向量
new_vectors = np.random.random((2, dim)).astype(np.float32)
new_ids = np.array([10_001, 10_002], dtype=np.int64)
failed_ids = index.add(new_vectors, new_ids, len(new_ids), dim)

assert index.check_id_exist(10_001)
print(index.get_num_elements(), index.get_memory_usage())
print(index.get_min_max_id())

# 计算一个 query 到候选 ID 列表的距离
candidate_ids = np.array([10_001, 10_002], dtype=np.int64)
distances = index.cal_distance_by_id(query, candidate_ids)

# 删除 ID；返回值为实际删除数量
removed = index.remove(candidate_ids)
```

`add` 接受展开的连续数组或 row-major 二维矩阵。`dtype: "float16"` 时传入
`numpy.float16`；`dtype: "bfloat16"` 时传入包含 BF16 位模式的 `numpy.uint16` 数组。
具体操作是否受支持仍取决于索引类型。

### 返回值、错误与迁移

| API / 输入 | 契约 |
| --- | --- |
| `build`（稠密与 CSR 重载）、`add` | 返回原生后端拒绝插入的 ID 列表 `list[int]`，包括部分失败。`[]` 仅表示未报告拒绝 ID，不是跨后端的去重保证；例如 IVF 与 HGraph/BruteForce 的重复 ID 行为不同。整体失败抛出异常，不转换为失败 ID 列表。 |
| 稠密 `knn_search`、`range_search` 及其 `_with_statistics` 版本 | 返回实际结果数量的一维 ID 和距离数组，数量可以小于 `k` 或为零；不新增填充或 sentinel。统计版本在其后附加原生 JSON 统计字符串。 |
| `knn_search`、`knn_search_with_statistics` 的 CSR 重载 | 保留现有 `(num_queries, k)` 矩形契约；每行先是有效结果前缀，之后以 ID `-1`、距离 `+inf` 填充。统计版本附加每个 query 行对应的 JSON 字符串列表。 |
| 底层操作失败 | 抛出 `pyvsag.VsagError`（`RuntimeError` 子类），保留原生错误信息、整数 `.code`（`vsag::ErrorType`）和字符串 `.operation`。适用于构造、普通/统计搜索、构建/添加、删除、最小最大 ID、距离计算及持久化。 |
| 文件打开 / 最终流状态失败 | `save` 或 `load` 抛出 `OSError`；底层序列化错误仍使用 `VsagError`。 |
| 绑定输入校验失败 | 形状、稠密 dtype/布局、CSR 一致性或大小溢出错误抛出 `ValueError`；Python 参数/类型转换也可能抛出 `TypeError`。 |

迁移时，原先依赖 build/add 返回 `None` 的调用方应改为检查失败 ID 列表；固定遍历 `k` 个稠密结果的代码
应改用 `len(result_ids)` 或 `zip(result_ids, result_dists)`。不要以空结果、`remove == 0` 或
`(-1, -1)` 替代捕获操作失败。现有 `except RuntimeError` 仍能捕获 `VsagError`；需要结构化处理时
使用其属性，不要解析错误字符串。

稀疏批查询应根据距离识别填充，**不要只检查 ID**：`-1` 可能是合法的原生 ID。
原生距离为有限值时，可逐行使用以下掩码：

```python
valid = ~np.isposinf(result_dists[row])
row_ids = result_ids[row][valid]
row_dists = result_dists[row][valid]
```

填充契约不归一化原生非有限距离。空索引、空 query/批次、`k == 0`、不支持的操作及重复 ID
继续遵循公共 C++ API/后端语义；不新增统一拒绝规则，也不消除后端差异。
尤其是原生空结果提前返回可能发生在后端参数校验之前。

### 稠密与 CSR 输入布局

稠密 build/add 接受含 `num_elements * dim` 个值的一维 C 连续数组，或形状严格等于
`(num_elements, dim)` 的 C 连续矩阵；单条稠密搜索 query 必须是一维 C 连续数组。
这些路径要求 dtype 与配置严格一致（`float32`、`float16` 或存储 BF16 位模式的 `uint16`），
不会静默转换 dtype 或复制非连续稠密输入。空 `(0, dim)` build/add 矩阵是零行批次，而非单条 query。
在原生指针读取前检查形状乘积、字节大小以及 int64/Python 大小类型收窄。

ID 和 CSR 数组则保留可转换输入行为：按需转换为连续的 `int64` ID、`uint32` 行指针/索引和
`float32` 值，正负步长视图均保持逻辑顺序。CSR 行指针必须从零开始、单调不减，末项等于
indices/values 的共同长度；行数和数组大小必须在支持的整数范围内。`index_pointers=[0]`
配合空 indices/values 表示零行，`[0, 0]` 则表示一条空稀疏向量。包括空输出在内的结果数组
均显式设置 NumPy 字节步长。

### 距离绑定输入安全

`calc_distances_by_id` 及其旧别名接受一个长度等于索引维度的一维稠密 query 和一维 ID 数组。过短、过长及空 query 由 C++ Dataset 校验拒绝，不会被当作缺失 ID。NumPy 输入按需转为连续的 float32 / int64 数组，正负步长的 query 和 ID 视图均保持逻辑顺序。这不为绑定新增稀疏、多向量或多 query 支持。

Node.js 对应的 `calcDistancesById(query, ids)` 和旧别名 `calDistanceById(query, ids)` 仍返回一维 `Float32Array`。query 必须为 `Float32Array`，ID 必须为 `BigInt64Array`，其他数组类型抛出 `TypeError`；实际 query 长度通过 C++ Dataset API 校验，底层操作失败时抛出包含 C++ 信息的 `Error`。缺失 ID 仍返回 `-1`，不属于操作失败。

## 与 C++ 库的关系

`pyvsag` 绑定的是同一份核心 C++ 实现，行为和性能特征与 C++ 版本保持一致。因此：

- 大多数 C++ 参数在 Python 中以相同的 JSON 字段传递；
- C++ 版本新增的索引类型、量化方式、距离度量会随 `pyvsag` 的下一个 release 一同发布；
- 构建 wheel 时所使用的依赖项与发布版 C++ 库相同（OpenBLAS、libaio 等）。

关于可用参数和索引类型，请参考 [创建索引](create_index.md) 和 [索引参数](../resources/index_parameters.md)。
