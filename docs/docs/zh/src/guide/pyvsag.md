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
| `cal_distance_by_id(query, ids)` | `numpy.ndarray` | 每个 ID 返回一个 float32 距离；无效 ID 为 `-1`，底层操作失败时整个结果保持为 `-1` |

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
具体操作是否受支持仍取决于索引类型。输入校验和 `add` 等操作会抛出 Python 异常；
`remove`、`cal_distance_by_id`、`save` 与 `load` 则使用上述 sentinel 或未检查行为，
不会统一传播底层操作错误。

## 与 C++ 库的关系

`pyvsag` 绑定的是同一份核心 C++ 实现，行为和性能特征与 C++ 版本保持一致。因此：

- 大多数 C++ 参数在 Python 中以相同的 JSON 字段传递；
- C++ 版本新增的索引类型、量化方式、距离度量会随 `pyvsag` 的下一个 release 一同发布；
- 构建 wheel 时所使用的依赖项与发布版 C++ 库相同（OpenBLAS、libaio 等）。

关于可用参数和索引类型，请参考 [创建索引](create_index.md) 和 [索引参数](../resources/index_parameters.md)。
