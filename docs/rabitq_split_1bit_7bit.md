# RabitQ Split x+y 使用说明

本文说明 HGraph 的 RabitQ split 存储和搜索路径。文件名保留了早期
`1+7bit` 设计名称，但当前外部配置已经泛化为 `x+y`：

- `rabitq_bits_per_dim_base = x`：图遍历 / lower-bound 过滤使用的 filter bits。
- `rabitq_bits_per_dim_precise = y`：重排 / full-distance 阶段额外读取的 supplement bits。
- `x + y <= 8`：最终重排使用总共 `x+y` bits 的 RabitQ 距离。

当 `base_quantization_type` 和 `precise_quantization_type` 都为 `rabitq`，并且配置了
`rabitq_bits_per_dim_precise` 时，HGraph 会自动选择 split datacell。用户侧不需要再显式设置
内部 split 版本或 datacell 类型。

## 适用范围

- 索引类型：`hgraph`
- base quantization：`rabitq`
- precise quantization：`rabitq`
- query bits：`rabitq_bits_per_dim_query = 32`
- filter bits：`rabitq_bits_per_dim_base`，范围 `[1, 8]`
- supplement bits：`rabitq_bits_per_dim_precise`，范围 `[1, 8]`
- 总 bits：`rabitq_bits_per_dim_base + rabitq_bits_per_dim_precise <= 8`

如果没有配置 `rabitq_bits_per_dim_precise`，HGraph 仍走已有 standard RabitQ 路径。

## 构建参数

`3+5` 推荐参数如下：

```json
{
  "dtype": "float32",
  "metric_type": "l2",
  "dim": 960,
  "index_param": {
    "base_quantization_type": "rabitq",
    "precise_quantization_type": "rabitq",
    "use_reorder": true,
    "rabitq_bits_per_dim_query": 32,
    "rabitq_bits_per_dim_base": 3,
    "rabitq_bits_per_dim_precise": 5,
    "rabitq_error_rate": 1.9,
    "max_degree": 64,
    "ef_construction": 400,
    "build_thread_count": 32
  }
}
```

参数说明：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `base_quantization_type` | string | - | 设为 `"rabitq"`。 |
| `precise_quantization_type` | string | - | split 模式下同样设为 `"rabitq"`。 |
| `rabitq_bits_per_dim_base` | int | `1` | split 模式下表示 `x`，即 filter 阶段使用的 bit 数。 |
| `rabitq_bits_per_dim_precise` | int | 未设置 | split 模式下表示 `y`，即 reorder 阶段额外读取的 supplement bit 数。 |
| `rabitq_error_rate` | float | `1.9` | lower-bound 误差倍率，必须为有限正数。 |
| `use_reorder` | bool | `false` | 建议开启；split 模式会用 `x+y` bits 的 base RabitQ 距离做最终重排。 |

参数约束：

- `rabitq_bits_per_dim_query` 必须是 `32`。
- `x = rabitq_bits_per_dim_base` 必须在 `[1, 8]`。
- `y = rabitq_bits_per_dim_precise` 必须在 `[1, 8]`。
- `x + y <= 8`。
- `rabitq_error_rate` 会进入索引参数兼容性检查；修改该值后应重建索引。

## 搜索参数

启用 RabitQ filter 图搜索：

```json
{
  "hgraph": {
    "ef_search": 200,
    "parallelism": 4,
    "rabitq_one_bit_search": true
  }
}
```

`rabitq_one_bit_search` 是已有搜索参数名。对 `x+y` split 索引来说，它触发的是 filter-code
lower-bound 图搜索；当 `x > 1` 时 filter 路径使用 x-bit 估计，而不是只使用 1 bit。

搜索行为：

- `rabitq_one_bit_search = false`：split datacell 会合并 filter 和 supplement code，按完整
  `x+y` RabitQ code 计算距离。
- `rabitq_one_bit_search = true`：图搜索阶段优先使用 x-bit filter 记录和 lower-bound 元数据，
  支持通过 `parallelism` 设置单查询内并行搜索线程数。
- 如果开启 `use_reorder = true`，最终重排使用 base split full code，即已有 filter bits 加上
  候选才需要读取的 y-bit supplement。

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
        "precise_quantization_type": "rabitq",
        "use_reorder": true,
        "rabitq_bits_per_dim_query": 32,
        "rabitq_bits_per_dim_base": 3,
        "rabitq_bits_per_dim_precise": 5,
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

推荐使用 VSAG 的索引级序列化接口，不需要业务侧手动保存 filter 文件或 supplement 文件。

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

对于 split RabitQ，序列化会包含：

1. HGraph 基础信息和索引参数。
2. label table。
3. base split datacell。
4. bottom graph。
5. 可选 precise reorder codes。
6. route graph。
7. 可选 extra info、attribute index、raw vector。

其中 base split datacell 内部会按顺序保存：

1. datacell 基础信息。
2. filter IO：x-bit planes、norm、可选 MRQ/raw norm、filter error、low-bound error。
3. supplement IO：y-bit planes 和恢复完整 RabitQ code 所需的元数据。
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
- `dim`、`metric_type`、`rabitq_bits_per_dim_query`、`rabitq_bits_per_dim_base`、
  `rabitq_bits_per_dim_precise`、`rabitq_error_rate` 等参数需要匹配。
- 旧版本 VSAG 不识别 split datacell，不能加载 split 索引。
- standard RabitQ 索引不配置 `rabitq_bits_per_dim_precise`，默认行为不变。

## 常见配置组合

standard RabitQ base code，使用 fp32 precise reorder：

```json
{
  "index_param": {
    "base_quantization_type": "rabitq",
    "rabitq_bits_per_dim_query": 32,
    "rabitq_bits_per_dim_base": 1,
    "use_reorder": true,
    "precise_quantization_type": "fp32"
  }
}
```

`3+5` split graph search，并用 base split full code 做最终重排：

```json
{
  "index_param": {
    "base_quantization_type": "rabitq",
    "precise_quantization_type": "rabitq",
    "use_reorder": true,
    "rabitq_bits_per_dim_query": 32,
    "rabitq_bits_per_dim_base": 3,
    "rabitq_bits_per_dim_precise": 5
  }
}
```

图搜索使用 x-bit filter，最终候选使用 `x+y` bits base split full code 重排。

## 注意事项

- 不要同时配置 `rabitq_bits_per_dim_base = 8` 和 `rabitq_bits_per_dim_precise > 0`；
  split 模式要求 `x + y <= 8`。
- `rabitq_one_bit_search` 是搜索参数，不会改变已有索引的存储格式。
- `rabitq_error_rate` 会影响构建时写入的 lower-bound error 元数据，修改后需要重建索引。
- 如果需要跨进程或跨机器持久化，优先使用 `Serialize(std::ostream&)` 或 `Serialize()` 返回的
  `BinarySet`。业务侧不需要理解内部 two-IO 布局。
