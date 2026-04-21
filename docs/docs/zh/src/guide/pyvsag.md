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
    "hnsw": {"max_degree": 16, "ef_construction": 100},
})

index = pyvsag.Index("hnsw", index_params)
index.build(vectors=data, ids=ids, num_elements=num_elements, dim=dim)

query = np.float32(np.random.random(dim))
search_params = json.dumps({"hnsw": {"ef_search": 100}})
result_ids, result_dists = index.knn_search(
    vector=query, k=10, parameters=search_params,
)
for rid, rdist in zip(result_ids, result_dists):
    print(f"{rid}: {rdist}")
```

完整示例请查阅仓库中的 [`examples/python/`](https://github.com/antgroup/vsag/tree/main/examples/python) 目录，建议从 `example_hnsw.py` 开始。

## 与 C++ 库的关系

`pyvsag` 绑定的是同一份核心 C++ 实现，行为和性能特征与 C++ 版本保持一致。因此：

- 大多数 C++ 参数在 Python 中以相同的 JSON 字段传递；
- C++ 版本新增的索引类型、量化方式、距离度量会随 `pyvsag` 的下一个 release 一同发布；
- 构建 wheel 时所使用的依赖项与发布版 C++ 库相同（OpenBLAS、libaio 等）。

关于可用参数和索引类型，请参考 [创建索引](create_index.md) 和 [索引参数](../resources/index_parameters.md)。
