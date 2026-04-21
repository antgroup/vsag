# 创建索引

VSAG 中所有检索能力都围绕 `Index` 接口展开。要使用某种索引，首先需要通过工厂方法 `vsag::Factory::CreateIndex(name, parameters)` 创建实例，其中：

- `name` 是索引类型名称，对应 `include/vsag/constants.h` 中定义的常量；
- `parameters` 是一段 JSON 字符串，声明数据类型、距离度量、维度等构建参数。

## 当前支持的索引类型

| 名称           | `name` 字符串   | 适用场景                                            |
| -------------- | --------------- | --------------------------------------------------- |
| HNSW           | `hnsw`          | 纯内存图索引，兼容 HNSWLIB 的序列化格式（参见：`include/vsag/constants.h` 的 `INDEX_HNSW`） |
| HGraph         | `hgraph`        | VSAG 自研图索引，支持多级量化和调优（详见 `examples/cpp/103_index_hgraph.cpp`） |
| DiskANN        | `diskann`       | 内存-磁盘混合索引，适合超出内存规模的数据集         |
| IVF            | `ivf`           | 倒排索引，适合大 `k` 和批量查询                     |
| Pyramid        | `pyramid`       | 多层级索引结构                                      |
| GNO-IMI        | `gno_imi`       | 基于 GNO-IMI 的倒排索引变体                         |
| SINDI          | `sindi`         | 稀疏向量上的倒排索引                                |
| BruteForce     | `brute_force`   | 暴力搜索，用作基准或小数据集                        |

> 完整示例可在 [`examples/cpp/`](https://github.com/antgroup/vsag/tree/main/examples/cpp) 目录中按照前缀编号依次查看（`101_` ~ `109_` 为索引类型，`2xx_` 为自定义资源，`3xx_` 为功能特性）。

## 通用的构建参数

所有索引在创建时都需要声明以下字段：

- `dtype`：向量数据类型，当前常用为 `"float32"`；部分索引也支持 `"fp16"`、`"bf16"`、`"int8"`；
- `metric_type`：距离度量方式，支持 `"l2"`、`"ip"`、`"cosine"`；
- `dim`：向量维度，必须与后续写入的数据一致。

索引特有参数以嵌套对象形式提供，例如 HNSW 的 `hnsw`、HGraph 的 `index_param`。

## 示例：创建 HNSW 索引

```cpp
#include <vsag/vsag.h>

auto hnsw_build_parameters = R"(
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "hnsw": {
        "max_degree": 16,
        "ef_construction": 100
    }
}
)";
auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_parameters).value();
```

## 示例：创建 HGraph 索引

```cpp
auto hgraph_build_parameters = R"(
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "max_degree": 32,
        "ef_construction": 100,
        "base_quantization_type": "sq8"
    }
}
)";
auto index = vsag::Factory::CreateIndex("hgraph", hgraph_build_parameters).value();
```

不同索引支持的参数及其语义，请参考 [索引参数](../resources/index_parameters.md)。
