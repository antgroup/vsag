# 二进制向量与 Hamming 距离

VSAG 支持对按位压缩（bit-packed）的二进制向量建立索引，例如二进制描述符、二进制 embedding，或浮点 embedding 的符号位，并使用 Hamming 距离进行比较：

```text
hamming(a, b) = popcount(a XOR b)
```

距离等于两个向量对应位置上取值不同的 bit 数量。[`brute_force`](../indexes/brute_force.md)（精确）和 [`hgraph`](../indexes/hgraph.md)（图索引）支持二进制向量，C++ 与 `pyvsag` 均可使用。

## 数据表示

- `dim` 表示向量的 **bit 数**，而不是字节数，不要求是 8 的整数倍。
- 每个向量占用 `ceil(dim / 8)` 个连续字节，数据集中的向量首尾相接：`n` 个向量的数据集是一个 `n * ceil(dim / 8)` 字节的缓冲区。
- bit 按 **MSB-first（高位在前）** 打包：逻辑上的第 `i` 个 bit 为 `(bytes[i / 8] >> (7 - i % 8)) & 1`。第 0 个 bit 对应第 0 个字节的掩码 `0x80`，第 7 个 bit 对应 `0x01`。这与 `numpy.packbits(bits, axis=-1)` 默认的 `bitorder="big"` 布局一致，也与从左到右读取 `"10110..."` 这类 bit 串的顺序一致。
- 当 `dim % 8 != 0` 时，每个向量最后一个字节的低 `8 - dim % 8` 个 bit 是填充位。填充位的取值不会影响任何距离：计算 kernel 会屏蔽它们，存储的编码中它们也会被清零。

例如 `dim = 13` 时每个向量占 2 个字节：第 0～7 个 bit 位于第 0 个字节（第 0 个 bit 为 `0x80`，第 7 个 bit 为 `0x01`），第 8～12 个 bit 对应第 1 个字节的 `0x80`、`0x40`、`0x20`、`0x10`、`0x08`，第 1 个字节的 `0x04`、`0x02`、`0x01` 是填充位。

## 配置

通过 `"dtype": "binary"` 选择二进制向量，它必须与 `"metric_type": "hamming"` 搭配使用：

```json
{
    "dtype": "binary",
    "metric_type": "hamming",
    "dim": 256,
    "index_param": {
        "max_degree": 32,
        "ef_construction": 200
    }
}
```

- `index_param` 中不需要指定量化类型：二进制索引总是使用 `binary` 量化器原样保存打包后的 bit。也可以显式写出 `"base_quantization_type": "binary"`（HGraph 还可以写 `"precise_quantization_type": "binary"`）；其它量化类型会被拒绝，因为它们读取的是数值元素。
- `dim` 的取值范围为 1～16,777,216（2^24）bit，保证每个距离在 `float` 结果类型中都是精确整数。

以下组合会返回 `INVALID_ARGUMENT` 错误：

| 配置 | 错误信息 |
| --- | --- |
| `"metric_type": "hamming"` 搭配 `float32`、`float16`、`bfloat16`、`int8` 或 `sparse` | `parameters[metric_type]=hamming requires parameters[dtype]=binary` |
| `"dtype": "binary"` 搭配 `l2`、`ip` 或 `cosine` | `parameters[dtype]=binary requires parameters[metric_type]=hamming` |
| `"dtype": "binary"` 用于 `brute_force` 和 `hgraph` 以外的索引 | `dtype binary is supported only by brute_force and hgraph` |
| 使用 `binary` 以外的量化类型 | `parameters[...] must be binary when dtype is binary` |
| `"dtype": "binary"` 搭配非 `dense` 的 `"repr"`，或 `dim` 超过 2^24 | 指明对应字段的参数错误 |

## 支持的索引与操作

| 操作 | `brute_force` | `hgraph` |
| --- | --- | --- |
| Build、Add | 支持 | 支持 |
| k-NN 搜索 | 精确 | 近似（图搜索），返回的距离精确 |
| 范围搜索（`distance <= radius`） | 精确 | 近似候选，返回的距离精确 |
| 删除（`MARK_REMOVE`、`FORCE_REMOVE`） | 支持 | 支持（`FORCE_REMOVE` 需要 `"support_force_remove": true` 与 `"use_reverse_edges": true`） |
| 以 Dataset 作为查询的 `CalcDistanceById` / `CalcDistancesById` | 支持 | 支持 |
| `GetRawVectorByIds` | 返回打包后的 bit | 返回打包后的 bit |
| `UpdateVector` | 支持 | 支持（不带 `force_update` 时，若新向量离某个邻居比离原位置更近则拒绝更新） |
| `GetStats` / `AnalyzeIndexBySearch` | BruteForce 不提供 | 支持（分析用的查询使用二进制向量） |
| 序列化 / 反序列化 | 支持 | 支持 |
| 带 ID `Filter` 的搜索 | 支持 | 支持 |

HGraph 中依赖浮点输入的功能不支持二进制向量，会在创建索引或调用时给出明确错误：变换量化（`tq_chain`、`mrle_dim`）、RaBitQ 编码与 `rabitq_fused_datacell`、PiPNN 构图、MCI clique 以及图增强。由于二进制编码无法换成其它量化器，`Tune` 会返回 `false`。接收 `const float*` 查询的重载（`CalcDistanceById(const float*, ...)`）只用于 float32，二进制查询请放在 `Dataset` 中传入。

## 距离语义

- 距离是前 `dim` 个 bit 中取值不同的 bit 数。返回类型为 `float`，其值总是 `0` 到 `dim` 之间的精确整数。
- k-NN 搜索按距离从小到大返回结果。多个向量可能距离相同，相同距离之间的先后顺序不做保证。
- 范围搜索返回距离小于或等于半径的所有向量，因此 `radius = 0` 会返回与查询完全相同的向量。

## 输入校验

C++ 数据集通过 `Dataset::BinaryVectors(const uint8_t*)` 与 `Dataset::GetBinaryVectors()` 携带二进制向量。`Dim()` 为 bit 数，缓冲区必须包含 `NumElements() * ceil(Dim() / 8)` 个字节。二进制索引会拒绝不含二进制向量的数据集，也会拒绝 `Dim()` 与索引维度不一致的查询。该缓冲区遵循其它向量字段的所有权规则：设置 `Owner(true)` 后，由数据集的 allocator 释放，没有 allocator 时使用 `delete[]` 释放；`DeepCopy` / `Append` 按每行 `ceil(dim / 8)` 字节复制。

`pyvsag` 接受 C 连续（C-contiguous）的 `numpy.uint8` 数组作为二进制向量：

| 输入 | 接受的形状 | 抛出 `ValueError` 的情况 |
| --- | --- | --- |
| `build` / `add` 的向量 | `(num_elements, ceil(dim / 8))` 或 `(num_elements * ceil(dim / 8),)` | dtype 不符、形状或长度不符、数组不连续 |
| `knn_search` / `range_search` / `calc_distances_by_id` 的查询 | `(ceil(dim / 8),)` 或 `(1, ceil(dim / 8))` | dtype 不符或字节数不符 |

数组不会被隐式转换。不连续的视图（例如 `packed[:, ::2]`）会被拒绝，可以用 `numpy.ascontiguousarray(packed)` 先复制成连续数组。

## 示例（C++）

```cpp
const int64_t dim = 100;                   // bit 数
const int64_t bytes = (dim + 7) / 8;       // 每个向量 13 字节
std::vector<uint8_t> vectors(num_vectors * bytes);  // MSB-first 打包
std::vector<int64_t> ids(num_vectors);

auto base = vsag::Dataset::Make();
base->NumElements(num_vectors)->Dim(dim)->Ids(ids.data())->BinaryVectors(vectors.data())->Owner(false);

auto index = vsag::Factory::CreateIndex(
    "hgraph", R"({"dtype": "binary", "metric_type": "hamming", "dim": 100,
                 "index_param": {"max_degree": 32, "ef_construction": 200}})").value();
index->Build(base);

auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(dim)->BinaryVectors(query_bytes)->Owner(false);
auto result = index->KnnSearch(query, 10, R"({"hgraph": {"ef_search": 100}})").value();
// result->GetDistances()[i] 是 result->GetIds()[i] 与查询之间不同 bit 的数量
```

完整程序见 [`examples/cpp/115_index_binary_hamming.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/115_index_binary_hamming.cpp)，它会用标量 popcount 逐一校验返回的距离。

## 示例（Python）

```python
import json
import numpy as np
import pyvsag

dim = 100
bits = np.random.default_rng(0).integers(0, 2, size=(5000, dim)).astype(bool)
base = np.packbits(bits, axis=-1)            # 形状 (5000, 13)，dtype uint8
ids = np.arange(len(base), dtype=np.int64)

index = pyvsag.Index("hgraph", json.dumps(
    {"dtype": "binary", "metric_type": "hamming", "dim": dim,
     "index_param": {"max_degree": 32, "ef_construction": 200}}))
index.build(base, ids, len(base), dim)

query = np.packbits(bits[0])                 # 13 字节
result_ids, distances = index.knn_search(query, 10, json.dumps({"hgraph": {"ef_search": 100}}))
```

`"10110011..."` 这样的 bit 串可以用 `np.packbits(np.array([c == "1" for c in s]))` 转换成查询。完整示例见 [`examples/python/115_index_binary_hamming.py`](https://github.com/antgroup/vsag/blob/main/examples/python/115_index_binary_hamming.py)。

## 距离计算 kernel

Hamming kernel 位于 `src/simd/hamming_simd.{h,cpp}`。通用标量实现（64 位字加 `__builtin_popcountll`）是正确性基准和回退实现，优化实现根据 CPU 支持的指令在运行时选择：

| 实现 | 指令 | 选择条件 |
| --- | --- | --- |
| `avx512vpopcntdq` | 512 位块上的 `_mm512_popcnt_epi64`，尾部使用掩码加载 | AVX-512 VPOPCNTDQ |
| `avx512` | 512 位块上的 4 bit 查表（`vpshufb`）与 `vpsadbw` | AVX-512 F/BW/DQ/VL |
| `avx2` | 256 位块上的 4 bit 查表（`vpshufb`）与 `vpsadbw` | AVX2 |
| `sse` | 64 位字上的标量 `POPCNT` | 以 SSE4.2 编译且 CPU 报告支持 POPCNT |
| `sve` | `svcnt_u64` 与 `svwhilelt_b8` 谓词加载 | SVE |
| `neon` | `vcntq_u8` 与成对加宽累加 | NEON |
| `generic` | 可移植的标量实现 | 总是可用 |

所有实现都与通用实现返回完全相同的结果。除了每次计算一个编码，每种实现还提供两种批量形式：`HammingDistanceBatch4` 一次比较一个查询与四个任意地址上的编码，供图搜索使用；`HammingDistanceScan` 计算查询到一段连续存放的编码的距离，供 `brute_force` 扫描使用。在 AVX-512 VPOPCNTDQ 上，扫描每步处理八个编码且只做一次水平归约；当编码为 8、16 或 32 字节时，一个寄存器同时装入二到八个编码。[Hamming kernel 基准测试](../resources/hamming_benchmark.md)给出了各实现的性能，以及二进制索引的召回率与吞吐量。
