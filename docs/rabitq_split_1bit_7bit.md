# RabitQ Split 1bit + 7bit 使用说明

本文说明 HGraph 中新增的 RabitQ v4 存储和搜索路径。该路径把 RabitQ base code 拆成
one-bit 主记录和剩余补充 bit 记录，用 one-bit 记录参与图搜索，再在需要时合并完整 RabitQ
code 计算最终距离。

## 适用范围

- 索引类型：`hgraph`
- base quantization：`rabitq`
- base code 存储：`base_codes_type = "rabitq_split"`
- RabitQ 版本：`rabitq_version = "split_1bit_7bit"`
- query bits：必须是 `rabitq_bits_per_dim_query = 32`
- base bits：支持 `rabitq_bits_per_dim_base` 在 `[1, 8]`，其中 `1` 表示 `1+0bit`，推荐 `8`（即 `1+7bit`）

默认路径不变。如果不显式设置 `rabitq_version = "split_1bit_7bit"` 和
`base_codes_type = "rabitq_split"`，HGraph 仍使用已有 RabitQ/flatten datacell 路径。

## 构建参数

推荐参数如下：

```json
{
  "dtype": "float32",
  "metric_type": "l2",
  "dim": 960,
  "index_param": {
    "base_quantization_type": "rabitq",
    "base_codes_type": "rabitq_split",
    "rabitq_version": "split_1bit_7bit",
    "rabitq_bits_per_dim_query": 32,
    "rabitq_bits_per_dim_base": 8,
    "rabitq_error_rate": 1.9,
    "max_degree": 64,
    "ef_construction": 400,
    "build_thread_count": 32
  }
}
```

新增参数说明：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `base_codes_type` | string | `"flatten"` | 设为 `"rabitq_split"` 时使用 split datacell 存储 RabitQ code。 |
| `rabitq_version` | string | `"standard"` | 设为 `"split_1bit_7bit"` 时启用 v4 RabitQ split 布局。 |
| `rabitq_error_rate` | float | `1.9` | one-bit lower bound 的误差倍率，使用官方 RabitQ 风格误差项。 |

参数约束：

- `rabitq_version = "split_1bit_7bit"` 只支持 `rabitq_bits_per_dim_query = 32`。
- `rabitq_version = "split_1bit_7bit"` 支持 `rabitq_bits_per_dim_base` 在 `[1, 8]`；`1` 是合法的 `1+0bit` 配置，`8` 是推荐的 `1+7bit` 配置。
- `rabitq_error_rate` 必须是有限正数。
- `rabitq_error_rate` 会进入索引参数兼容性检查；修改该值后应重建索引。

## 搜索参数

启用 one-bit 图搜索：

```json
{
  "hgraph": {
    "ef_search": 200,
    "parallelism": 4,
    "rabitq_one_bit_search": true
  }
}
```

搜索行为：

- `rabitq_one_bit_search = false`：split datacell 会合并 one-bit 和 supplement code，按完整
  RabitQ code 计算距离。
- `rabitq_one_bit_search = true`：图搜索阶段优先使用 one-bit 记录和 lower bound 元数据，
  支持通过 `parallelism` 设置单查询内并行搜索线程数。
- 如果没有开启 `use_reorder`，搜索结束后会用 base split datacell 合并完整 code，对候选集做最终
  全码重算。
- 如果开启 `use_reorder = true`，默认最终重排使用 precise codes；设置
  `reorder_source = "base"` 时，最终重排使用 base split full code，不再依赖 high precise
  reorder codes。

## C++ 调用示例

下面示例展示创建、构建、搜索、保存和加载。实际数据准备方式与普通 HGraph 相同。

```cpp
#include <vsag/vsag.h>

#include <fstream>
#include <iostream>

int main() {
    const int64_t dim = 960;
    const int64_t count = 10000;

    // ids 和 vectors 由调用方准备。
    int64_t* ids = nullptr;
    float* vectors = nullptr;

    const std::string index_params = R"({
      "dtype": "float32",
      "metric_type": "l2",
      "dim": 960,
      "index_param": {
        "base_quantization_type": "rabitq",
        "base_codes_type": "rabitq_split",
        "rabitq_version": "split_1bit_7bit",
        "rabitq_bits_per_dim_query": 32,
        "rabitq_bits_per_dim_base": 8,
        "rabitq_error_rate": 1.9,
        "max_degree": 64,
        "ef_construction": 400
      }
    })";

    auto index = vsag::Factory::CreateIndex("hgraph", index_params).value();

    auto base = vsag::Dataset::Make();
    base->NumElements(count)->Dim(dim)->Ids(ids)->Float32Vectors(vectors);
    auto build_result = index->Build(base);
    if (not build_result.has_value()) {
        std::cerr << build_result.error().message << std::endl;
        return 1;
    }

    const std::string search_params = R"({
      "hgraph": {
        "ef_search": 200,
        "rabitq_one_bit_search": true
      }
    })";

    float* query_vector = nullptr;
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector);
    auto result = index->KnnSearch(query, 10, search_params).value();

    std::ofstream out("/tmp/hgraph-rabitq-split.index", std::ios::binary);
    auto serialize_result = index->Serialize(out);
    out.close();
    if (not serialize_result.has_value()) {
        std::cerr << serialize_result.error().message << std::endl;
        return 1;
    }

    auto loaded = vsag::Factory::CreateIndex("hgraph", index_params).value();
    std::ifstream in("/tmp/hgraph-rabitq-split.index", std::ios::binary);
    auto deserialize_result = loaded->Deserialize(in);
    in.close();
    if (not deserialize_result.has_value()) {
        std::cerr << deserialize_result.error().message << std::endl;
        return 1;
    }

    auto loaded_result = loaded->KnnSearch(query, 10, search_params).value();
    (void)result;
    (void)loaded_result;
    return 0;
}
```

## 如何存储索引

推荐使用 VSAG 的索引级序列化接口，不需要业务侧手动保存 one-bit 文件或 supplement 文件。

文件流方式：

```cpp
std::ofstream out("/path/to/index.bin", std::ios::binary);
auto ret = index->Serialize(out);
out.close();
```

KV/BinarySet 方式：

```cpp
auto binary_set = index->Serialize().value();
for (const auto& key : binary_set.GetKeys()) {
    auto binary = binary_set.Get(key);
    // 将 key、binary.data、binary.size 写入外部 KV 或对象存储。
}
```

对于 v4 split RabitQ，序列化会包含：

1. HGraph 基础信息和索引参数。
2. label table。
3. base split datacell。
4. bottom graph。
5. 可选 precise reorder codes。
6. route graph。
7. 可选 extra info、attribute index、raw vector。

其中 base split datacell 内部会按顺序保存：

1. datacell 基础信息。
2. `one_bit_io`：one-bit plane、norm、可选 MRQ/raw norm、one-bit error、low-bound error。
3. `supplement_io`：剩余 bit planes 和恢复完整 RabitQ code 所需的元数据。
4. RabitQ quantizer 模型参数，如 centroid、随机正交变换、可选 PCA。

## 如何加载索引

加载时先用同一份构建参数创建空索引，再调用 `Deserialize`：

```cpp
auto loaded = vsag::Factory::CreateIndex("hgraph", index_params).value();
std::ifstream in("/path/to/index.bin", std::ios::binary);
auto ret = loaded->Deserialize(in);
in.close();
```

使用 BinarySet 加载：

```cpp
vsag::BinarySet binary_set;
// 从外部 KV 或对象存储读回所有 key/value，并写入 binary_set。
auto loaded = vsag::Factory::CreateIndex("hgraph", index_params).value();
auto ret = loaded->Deserialize(binary_set);
```

加载要求：

- 创建空索引时必须使用与构建时兼容的参数。
- v4 split 索引必须保留 `base_codes_type = "rabitq_split"`。
- v4 split 索引必须保留 `rabitq_version = "split_1bit_7bit"`。
- `dim`、`metric_type`、`rabitq_bits_per_dim_query`、`rabitq_bits_per_dim_base`、
  `rabitq_error_rate` 等参数需要匹配。
- 旧版本 VSAG 不识别 `rabitq_split` datacell，不能加载 v4 split 索引。
- 标准 RabitQ 索引仍按 `rabitq_version = "standard"` 和 `base_codes_type = "flatten"`
  加载，默认行为不变。

## 常见配置组合

只使用 v4 split base code，不使用 precise reorder：

```json
{
  "index_param": {
    "base_quantization_type": "rabitq",
    "base_codes_type": "rabitq_split",
    "rabitq_version": "split_1bit_7bit",
    "rabitq_bits_per_dim_query": 32,
    "rabitq_bits_per_dim_base": 8,
    "use_reorder": false
  }
}
```

图搜索使用 one-bit，最终候选用 base split full code 重算。

使用 v4 split graph search，同时保留 precise reorder：

```json
{
  "index_param": {
    "base_quantization_type": "rabitq",
    "base_codes_type": "rabitq_split",
    "rabitq_version": "split_1bit_7bit",
    "rabitq_bits_per_dim_query": 32,
    "rabitq_bits_per_dim_base": 8,
    "use_reorder": true,
    "precise_quantization_type": "fp32"
  }
}
```

图搜索使用 base split one-bit，最终候选使用 precise codes 重排。

使用 v4 split graph search，并用 base split full code 做最终重排：

```json
{
  "index_param": {
    "base_quantization_type": "rabitq",
    "base_codes_type": "rabitq_split",
    "rabitq_version": "split_1bit_7bit",
    "rabitq_bits_per_dim_query": 32,
    "rabitq_bits_per_dim_base": 8,
    "use_reorder": true,
    "reorder_source": "base"
  }
}
```

图搜索使用 base split one-bit，最终候选使用 base split full code 重排。

## 注意事项

- 不要只设置 `rabitq_bits_per_dim_base = 8`。v4 split 路径必须显式设置
  `rabitq_version = "split_1bit_7bit"` 和 `base_codes_type = "rabitq_split"`。
- `rabitq_one_bit_search` 是搜索参数，不会改变已有索引的存储格式。
- `rabitq_error_rate` 会影响构建时写入的 lower-bound error 元数据，修改后需要重建索引。
- 如果需要跨进程或跨机器持久化，优先使用 `Serialize(std::ostream&)` 或 `Serialize()` 返回的
  `BinarySet`。业务侧不需要理解内部 two-IO 布局。