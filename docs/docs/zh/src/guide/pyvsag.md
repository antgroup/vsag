# pyvsag

`pyvsag` 是 VSAG 官方的 Python 绑定，是一个纯 Python 层，通过 [cffi](https://cffi.readthedocs.io/) 调用 [C API](c_api.md)（`libvsag_c`）。每个 CPU 架构只需一个 `py3-none-manylinux_2_28` wheel，即可适用于所有 CPython 3.10+ 版本。数据仅以指针、形状和 dtype 的形式跨越边界，因此不依赖 NumPy 的 ABI。源代码位于 [`python/`](https://github.com/antgroup/vsag/tree/main/python)。

## 安装

```bash
pip install pyvsag
```

要求：

- **平台：** Linux `x86_64` 或 `aarch64`，glibc 2.28 及以上（`manylinux_2_28`，例如 Ubuntu 20.04+、
  Debian 10+、RHEL/Alma/Rocky 8+）。macOS、Windows 以及 Alpine 等基于 musl 的发行版没有 wheel。
- **Python：** CPython 3.10 及以上。支持 3.10 到 3.14；wheel 不依赖 Python ABI，更新的版本无需新的 wheel。
- **依赖：** `numpy` 和 `cffi`，会自动安装。wheel 自带 OpenMP 运行时，不依赖 libaio、libcurl 和 OpenBLAS。

### 每晚构建（TestPyPI）

`pyvsag TestPyPI` 工作流会在每晚和手动触发时，用 `main` 构建 wheel 并发布到 TestPyPI（上传可暂停）。
版本号带 `.devN`，需要加 `--pre`；依赖仍从正式 PyPI 安装：

```bash
pip install --pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pyvsag
```

### 从源码构建

```bash
make pyvsag
pip install wheelhouse/pyvsag-*.whl
```

`make pyvsag` 在仓库根目录运行 `scripts/python/local_build_wheel.sh`，并把 wheel 写入 `wheelhouse/`。
有 Docker 时在 `manylinux_2_28` 镜像中构建，并在镜像内运行 `tests/python` 的一部分用例；没有 Docker 时退回普通的
`python -m build`，把 wheel 安装到当前虚拟环境并针对它运行 `tests/python`，得到的 wheel 带 `linux_<arch>` 标签，
而不是 `manylinux_2_28`。如果当前不在虚拟环境中，脚本会在仓库内创建 `.venv/`。

如果想直接使用源码树中的构建结果，而不打包 wheel：

```bash
make debug VSAG_ENABLE_PYBINDS=ON              # 生成 build/src/libvsag_c.so
bash scripts/python/prepare_python_build.sh    # 生成 python/pyvsag/_version.py
export PYVSAG_LIBRARY="$PWD/build/src/libvsag_c.so"
export PYTHONPATH="$PWD/python"
```

`PYVSAG_LIBRARY` 指定要加载的 `libvsag_c`；未设置时，`pyvsag` 加载与 Python 文件放在一起的那一份。

## 快速开始

`pyvsag.Index(name, parameters)` 接受索引名称和 JSON 字符串形式的参数，与 C++ 的
`vsag::Factory::CreateIndex` 签名对应：

```python
import json
import numpy as np
import pyvsag

dim = 128
num_elements = 10_000

data = np.random.random((num_elements, dim)).astype(np.float32)
ids = np.arange(num_elements, dtype=np.int64)

index_params = json.dumps({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": dim,
    "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 32,
        "ef_construction": 300,
    },
})

index = pyvsag.Index("hgraph", index_params)
index.build(vectors=data, ids=ids, num_elements=num_elements, dim=dim)

query = np.random.random(dim).astype(np.float32)
search_params = json.dumps({"hgraph": {"ef_search": 100}})
result_ids, result_dists = index.knn_search(
    vector=query, k=10, parameters=search_params,
)
for rid, rdist in zip(result_ids, result_dists):
    print(f"{rid}: {rdist}")
```

可用的索引类型和参数请参考 [创建索引](create_index.md) 和 [索引参数](../resources/index_parameters.md)。完整示例位于仓库的
[`examples/python/`](https://github.com/antgroup/vsag/tree/main/examples/python) 目录，建议从
`103_index_hgraph.py` 开始。

## 数据类型与输入规则

| 索引 `dtype` | 传给 `build`、`add` 和各搜索方法的向量 |
| --- | --- |
| `float32` | `numpy.float32` |
| `float16` | `numpy.float16` |
| `bfloat16` | 包含 bfloat16 原始位模式的 `numpy.uint16` 数组 |
| `sparse` | CSR 数组：`index_pointers`（`uint32`）、`indices`（`uint32`）、`values`（`float32`） |

`pyvsag` 不会转换或复制稠密向量，以保持零拷贝；不匹配的输入会以 `ValueError` 拒绝：

- 稠密数组的 dtype 必须与上表完全一致，使用本机字节序，并且是 C 连续的（二维矩阵必须是行优先，且每行长度等于 `dim`）；
- ID 数组以及稀疏的 `index_pointers`、`indices` 必须是整数数组，数值要落在目标类型（`int64`、`uint32`）范围内；
  浮点 ID 和越界数值会被拒绝，而不是被截断；
- `build` 和 `add` 的 `ids` 必须是一维数组，每个向量对应一个元素。

## 搜索

| 方法 | 返回值 |
| --- | --- |
| `knn_search(vector, k, parameters)` | `(ids, distances)`：形状为 `(k,)` 的数组（`int64`、`float32`）。搜索未能填满的位置为 `-1`。 |
| `knn_search(index_pointers, indices, values, k, parameters)` | 稀疏查询批次的同样结果，形状为 `(num_queries, k)`。 |
| `range_search(point, threshold, parameters)` | `threshold` 范围内所有结果的 `(ids, distances)`；没有结果时为空数组。 |
| `knn_search_with_statistics(...)` / `range_search_with_statistics(...)` | 上述数组，外加 JSON 字符串形式的搜索统计信息（稀疏批次为字符串列表）。见 [搜索统计](search_statistics.md)。 |

参数可以按位置或按关键字传递（`vector=`、`k=`、`parameters=`、`point=`、`threshold=`）。

## 稀疏向量

稀疏向量使用 CSR 布局：第 `i` 行拥有 `indices[index_pointers[i]:index_pointers[i + 1]]` 及对应的
`values`；`index_pointers` 从 `0` 开始且不递减。

```python
index = pyvsag.Index("sindi", json.dumps({
    "dtype": "sparse",
    "dim": 128,
    "metric_type": "ip",
    "index_param": {"use_reorder": True, "doc_prune_ratio": 0.0, "window_size": 50000},
}))
index.build(index_pointers=ptrs, indices=cols, values=vals, ids=ids)
index.add(ptrs, cols, vals, new_ids)
result_ids, result_dists = index.knn_search(
    q_ptrs, q_cols, q_vals, 10, json.dumps({"sindi": {"n_candidate": 100}}),
)
```

`pyvsag.sparse` 还提供稀疏 HDF5 基准数据布局的解码，见 [数据集格式](../resources/dataset_format.md)。

## 保存与加载

```python
index.save("index.bin")

new_index = pyvsag.Index("hgraph", index_params)
new_index.load("index.bin")
```

底层序列化或反序列化失败（包括文件无法打开）时，`save` 与 `load` 会抛出 `RuntimeError`。

## 索引操作

| 方法 | 返回值 | 说明 |
| --- | --- | --- |
| `build(...)` | `None` | 稠密 `build(vectors, ids, num_elements, dim)` 或稀疏 `build(index_pointers, indices, values, ids)`。 |
| `add(...)` | `None` | 与 `build` 相同的两种形式。 |
| `remove(ids)` | `int` | 删除一维整数 ID 数组，返回实际删除的数量。没有 ID 被删除和操作失败时都返回 `0`。 |
| `get_num_elements()` | `int` | 索引当前元素数量。 |
| `get_memory_usage()` | `int` | 索引占用的内存字节数。 |
| `check_id_exist(id)` | `bool` | 指定 ID 当前是否存在。 |
| `get_min_max_id()` | `(min_id, max_id)` | 操作失败时返回 `(-1, -1)`。 |
| `calc_distances_by_id(query, ids)`（旧别名 `cal_distance_by_id`） | `numpy.ndarray` | 每个 ID 一个 float32 距离，缺失 ID 为 `-1`。 |

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
distances = index.calc_distances_by_id(query, candidate_ids)

# 删除 ID；返回值为实际删除数量
removed = index.remove(candidate_ids)
```

`add` 接受展开的连续数组或行优先的二维矩阵。具体操作是否受支持仍取决于索引类型。

`calc_distances_by_id` 接受一个一维 float32 query（长度等于索引维度）和一个一维 ID 数组。长度不对的 query
和空 query 会以 `RuntimeError` 拒绝，不会被当作缺失 ID。其他 NumPy 输入会按需转为连续的 float32 / int64
数组，正负步长的视图都保持逻辑顺序。它不支持稀疏、多向量或多 query 输入。

## 错误处理

| 失败情形 | 行为 |
| --- | --- |
| `pyvsag.Index(...)` 无法创建索引 | `RuntimeError`（`error type: UNSUPPORTED_INDEX`、`error type: invalid_parameter` 或 `error type: unexpectedError`） |
| `build`、`add`、`save`、`load`、`get_num_elements`、`get_memory_usage`、`check_id_exist`、`calc_distances_by_id`、`*_with_statistics` | `RuntimeError`，消息以操作名开头并附带 C++ 错误信息 |
| `knn_search` | 不抛异常：ids 和 distances 都填 `-1` |
| `range_search` | 不抛异常：返回空数组 |
| `remove` | 不抛异常：返回 `0` |
| `get_min_max_id` | 不抛异常：返回 `(-1, -1)` |
| dtype、形状、布局、字节序或整数范围不对 | `ValueError`；调用参数不对时抛出 `TypeError` |

如果搜索失败时必须抛出异常，请使用 `*_with_statistics` 变体。

## 日志

```python
pyvsag.set_logger_off()     # 关闭库的日志输出
pyvsag.set_logger_info()    # 仅输出重要信息
pyvsag.set_logger_debug()   # 输出详细调试信息
```

`pyvsag.__version__` 为包的版本号。

## 线程与进程

- 每次调用 C++ 库时都会释放 GIL，库内部的 OpenMP 线程不受 GIL 影响，线程数可通过 `OMP_NUM_THREADS` 控制。
  支持多个 Python 线程并发搜索；其他并发操作遵循 C++ 索引的规则。
- `pyvsag` 可以与 PyTorch、scikit-learn 等其他 OpenMP 使用者共存于同一进程；
  `tests/python/test_openmp_coexistence.py` 覆盖了所有导入顺序。
- 使用 `multiprocessing` 时请选择 `spawn` 启动方式，不要在 `fork` 之间共享索引。
- 对 `copy.copy(index)` 得到的浅拷贝与原对象共享同一个原生索引；最后一个引用消失时才释放原生索引。

## 与 C++ 库的关系

`pyvsag` 封装的是同一份 `vsag::Index` 实现，并与 C++ 共享底层索引二进制，因此行为和性能特征一致。
只要参数匹配，就可以在 Python 中构建索引、在 C++ 中加载（反之亦然）。大多数 C++ 参数在 Python 中以相同的
JSON 字段传递；C++ 新增的索引类型、量化方式和距离度量会随 `pyvsag` 的下一个 release 发布。

发布的 wheel 使用 `ENABLE_LIBAIO=OFF` 构建，因此不依赖 libaio。`async_io` 仍可选用，但会退化为带缓冲的
兼容后端（单次 `buffer_io` 读取，且写入后不做 fsync），并只打印一次警告；需要真正的 libaio 路径时，请在本地用
`VSAG_ENABLE_LIBAIO=ON pip install python/` 自行构建。详见[磁盘索引](../resources/disk_index.md)。

## Node.js 对应接口

Node.js 的 `calcDistancesById(query, ids)` 和旧别名 `calDistanceById(query, ids)` 返回一维
`Float32Array`。query 必须为 `Float32Array`，ID 必须为 `BigInt64Array`，其他数组类型抛出 `TypeError`；
实际 query 长度通过 C++ Dataset API 校验，底层操作失败时抛出包含 C++ 信息的 `Error`。缺失 ID 仍返回
`-1`，不属于操作失败。
