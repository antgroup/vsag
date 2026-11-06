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
index.build(vectors=data, ids=ids, num_elements=num_elements, dim=dim)

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

当前 `save` 与 `load` 包装不会传播底层序列化和反序列化操作返回的错误。请确认输出文件
已经生成，并对重要索引执行保存后重新加载的 round-trip 校验，再依赖持久化结果。

## 索引操作

除构建、搜索、保存与加载外，Python 绑定还提供以下索引管理方法：

| 方法 | 返回值 | 说明 |
| --- | --- | --- |
| `get_num_elements()` | `int` | 索引当前元素数量 |
| `get_memory_usage()` | `int` | 索引占用的内存字节数 |
| `check_id_exist(id)` | `bool` | 指定 ID 当前是否存在 |
| `get_min_max_id()` | `(min_id, max_id)` | 操作失败时返回 `(-1, -1)` |
| `add(vectors, ids, num_elements, dim)` | `None` | 添加连续的稠密向量矩阵；dtype 必须与索引一致 |
| `remove(ids)` | `int` | 删除一维 `int64` ID 数组；没有 ID 被删除和底层操作失败时都会返回 `0` |
| `calc_distances_by_id(query, ids)` / `cal_distance_by_id(query, ids)`（旧别名） | `numpy.ndarray` | 以一维 float32 数组按 ID 顺序返回距离；缺失 ID 为 `-1`，底层参数无效或操作不受支持时抛出包含 C++ 错误信息的 `RuntimeError` |

```python
# 添加两条稠密向量
new_vectors = np.random.random((2, dim)).astype(np.float32)
new_ids = np.array([10_001, 10_002], dtype=np.int64)
index.add(new_vectors, new_ids, len(new_ids), dim)

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
具体操作是否受支持仍取决于索引类型。输入校验以及 `add`、`calc_distances_by_id` 等操作会抛出 Python 异常；
`remove`、`save` 与 `load` 则使用上述 sentinel 或未检查行为，
不会统一传播底层操作错误。

### 距离绑定输入安全

`calc_distances_by_id` 及其旧别名接受一个长度等于索引维度的一维稠密 query 和一维 ID 数组。过短、过长及空 query 由 C++ Dataset 校验拒绝，不会被当作缺失 ID。NumPy 输入按需转为连续的 float32 / int64 数组，正负步长的 query 和 ID 视图均保持逻辑顺序。这不为绑定新增稀疏、多向量或多 query 支持。

Node.js 对应的 `calcDistancesById(query, ids)` 和旧别名 `calDistanceById(query, ids)` 仍返回一维 `Float32Array`。query 必须为 `Float32Array`，ID 必须为 `BigInt64Array`，其他数组类型抛出 `TypeError`；实际 query 长度通过 C++ Dataset API 校验，底层操作失败时抛出包含 C++ 信息的 `Error`。缺失 ID 仍返回 `-1`，不属于操作失败。

## 稀疏词权重辅助函数

`pyvsag.sparse` 提供了把人类可读的 `{词: 权重}` 映射转换为 SINDI 等稀疏索引所需的 CSR 数组的辅助函数。它们只依赖 NumPy，不引入新的运行时依赖。

| 辅助函数 | 返回值 | 说明 |
| --- | --- | --- |
| `build_term_vocabulary(documents)` | `dict[str, int]` | 对所有唯一词条按字典序排序并从 `0` 开始编号，因此词表不受字典插入顺序或文档顺序影响，ID 始终落在 `uint32` 范围内。空映射会引发 `ValueError`（SINDI 无法存储空行）。 |
| `TermVocabulary(vocabulary)` | 可复用的映射 | 包装 `build_term_vocabulary` 的结果并只校验一次 ID。同一份词表需要转换大量查询时传入它而不是原始字典，查询转换的耗时便只与查询词条数相关，而不再随词表大小增长。 |
| `term_weights_to_csr(documents, vocabulary)` | `(index_pointers, indices, values)` | 返回 `uint32` / `uint32` / `float32` 数组，可直接传给 `index.build(...)` 和 `index.knn_search(...)`。每行按词条 ID 排序。**空映射被拒绝**（引发 `ValueError`），因为 SINDI::add 在添加时会静默丢弃空文档，而 SINDI::KnnSearch 排斥空查询。**词表中的 ID 必须唯一**——重复 ID 会引发 `ValueError`。 |

两个函数都接受任意可迭代的映射序列，因此文档和查询可以复用同一份词表。查询中出现词表之外的词条会抛出 `KeyError`；非字符串词条、非整数 ID、超出 `uint32` 范围的 ID、非有限权重以及无法用 `float32` 表示的权重都会被显式拒绝，不会被静默截断。

```python
import json
import numpy as np
import pyvsag

from pyvsag.sparse import TermVocabulary, build_term_vocabulary, term_weights_to_csr

documents = [
    {"vector": 1.0, "search": 0.8, "guide": 0.6},
    {"sparse": 1.0, "retrieval": 0.9, "sindi": 0.8},
    {"vector": 1.0, "database": 0.7, "index": 0.9},
]
ids = np.arange(len(documents), dtype=np.int64)

vocabulary = build_term_vocabulary(documents)
index_pointers, indices, values = term_weights_to_csr(documents, vocabulary)

index_params = json.dumps({
    "dtype": "sparse",
    "dim": max(len(document) for document in documents),
    "metric_type": "ip",
    "index_param": {"term_id_limit": len(vocabulary), "window_size": 10000},
})
index = pyvsag.Index("sindi", index_params)
index.build(
    index_pointers=index_pointers,
    indices=indices,
    values=values,
    ids=ids,
)

# 复用词表只校验一次；之后每次查询转换只处理查询本身的词条，
# 不再扫描整个词表。
query_vocabulary = TermVocabulary(vocabulary)

query = {"vector": 1.0, "search": 0.8}
query_pointers, query_indices, query_values = term_weights_to_csr(
    [query], query_vocabulary
)
result_ids, result_distances = index.knn_search(
    query_pointers,
    query_indices,
    query_values,
    2,
    json.dumps({"sindi": {"n_candidate": 3}}),
)
```

注意 `dim` 表示单个向量中非零词条数的最大值，而不是词表大小；`term_id_limit` 必须覆盖最大词条 ID 加一。完整可运行示例见
[`examples/python/109_index_sindi.py`](https://github.com/antgroup/vsag/blob/main/examples/python/109_index_sindi.py)。

## 与 C++ 库的关系

`pyvsag` 绑定的是同一份核心 C++ 实现，行为和性能特征与 C++ 版本保持一致。因此：

- 大多数 C++ 参数在 Python 中以相同的 JSON 字段传递；
- C++ 版本新增的索引类型、量化方式、距离度量会随 `pyvsag` 的下一个 release 一同发布；
- 构建 wheel 时所使用的依赖项与发布版 C++ 库相同（OpenBLAS、libaio 等）。

关于可用参数和索引类型，请参考 [创建索引](create_index.md) 和 [索引参数](../resources/index_parameters.md)。
