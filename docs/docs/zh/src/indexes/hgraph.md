# HGraph

HGraph 是 VSAG 的旗舰 **图索引**。它构建的是多层近邻图，
提供了丰富的量化方案、统一的构建参数 schema（`index_param`），并原生支持精排（reorder）、
增量更新、删除、以及基于 ELP 的运行时自动调优。

对于大多数稠密向量场景（文本 / 图像 / 多模态 embedding，维度 64–4096，规模从数千到数亿），
HGraph 都是推荐的默认索引。

- 源码：`src/algorithm/hgraph.{h,cpp}`
- 示例：[`examples/cpp/103_index_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/103_index_hgraph.cpp)

## 工作原理

1. **构图。** 向量被组织成层级近邻图：上层作为导航入口，底层连接每个数据点到在
   `max_degree` 预算内的最近邻。构图算法可以是 NSW 风格插入（`graph_type: "nsw"`，默认）
   或 ODescent（`graph_type: "odescent"`）。
2. **量化。** 底层存储使用可配置的量化器进行压缩（`base_quantization_type` —
   `fp32`、`fp16`、`bf16`、`sq8`、`sq4`、`sq8_uniform`、`sq4_uniform`、`pq`、`pqfs`、`rabitq`、`saq`、`tq`）。
   可选地再保留一份高精度副本（`use_reorder: true` 搭配 `precise_quantization_type`），
   用于对粗排结果进行重打分。
3. **搜索。** 自顶向下在图上做贪心 beam search，扩展候选集到 `ef_search` 个节点；如启用精排，
   最终结果会在高精度表示上重新打分。

## 快速开始

```cpp
#include <vsag/vsag.h>

std::string params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 32,
        "ef_construction": 400
    }
})";
auto index = vsag::Factory::CreateIndex("hgraph", params).value();

// 构建索引。
auto base = vsag::Dataset::Make();
base->NumElements(n)->Dim(128)->Ids(ids)->Float32Vectors(data)->Owner(false);
index->Build(base);

// 执行检索。
auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(128)->Float32Vectors(q)->Owner(false);
auto result = index->KnnSearch(
    query, /*topk=*/10, R"({"hgraph": {"ef_search": 100}})").value();
```

## 构建参数

构建参数放在 `index_param` 下。下表列出最常用的配置项；完整列表请见
[索引参数](../resources/index_parameters.md)。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_quantization_type` | string | —（必填） | `fp32`、`fp16`、`bf16`、`sq8`、`sq4`、`sq8_uniform`、`sq4_uniform`、`pq`、`pqfs`、`rabitq`、`saq`、`tq` —— 各量化器细节见[量化章节](../quantization/README.md) |
| `max_degree` | int | `64` | 图节点最大出度 |
| `ef_construction` | int | `400` | 构建阶段的候选集大小（越大召回越高，构建越慢） |
| `graph_type` | string | `"nsw"` | 构图算法：`nsw` 或 `odescent` |
| `use_reverse_edges` | bool | `false` | 跟踪入边，实现 O(1) 反向邻居查找；边存储约翻倍，且 `graph_storage_type: "compressed"` 不支持 |
| `label_remap_type` | string | `"pg"` | label 到内部 ID 的 map 实现：`"pg"` 或 `"robin"`；恢复或组合兼容索引时应保持一致 |
| `use_reorder` | bool | `false` | 是否额外保留一份高精度副本用于精排 |
| `reorder_source` | string | `"precise"` | 从 `"precise"` 编码或直接从 `"base"` 编码重排；RaBitQ x+y split（包括 `tq_chain: "mrle, rabitq"`）会自动设置为 `"base"` |
| `precise_quantization_type` | string | `"fp32"` | 精排使用的量化类型（仅在 `use_reorder: true` 时生效） |
| `base_pq_dim` | int | `1` | PQ 子空间数（`pq` / `pqfs` 时必填） |
| `mrle_dim` | int | `0` | `tq_chain` 中 MRLE 的输出维度，范围 `[0, dim]`；`0` 表示输入维度 |
| `fast_encode_rabitq` | bool | `true` | 使用多 bit RaBitQ 快速编码；设为 `false` 使用原有精确编码器 |
| `fast_encode_rabitq_rounds` | int | `6` | RaBitQ 快速编码的坐标微调轮数，范围 `[1, 32]` |
| `saq_avg_bits` | float | `4.0` | SAQ 每维平均记录位预算，范围 `[1, 8]` |
| `saq_segment_count` | int | `0` | SAQ 分段数；`0` 表示动态选择边界 |
| `saq_adjustment_rounds` | int | `6` | SAQ 码字调整轮数，范围 `[0, 32]` |
| `saq_use_pca` | bool | `true` | 使用全维 PCA 旋转对维度排序 |
| `saq_random_rotation` | bool | `true` | 在每个 SAQ 分段内使用独立正交旋转 |
| `build_thread_count` | int | `100` | 构建阶段并发线程数 |
| `support_duplicate` | bool | `false` | 是否在插入时做重复 ID 检测 |
| `deduplicate_storage` | bool | `false` | 让重复向量共享存储；需同时设置 `support_duplicate: true` |
| `duplicate_distance_threshold` | float | `0.0` | 重复判定距离阈值。大于 `0` 时按最近候选的距离判重；等于 `0` 时退化为当前编码 `memcmp` 判重 |
| `support_remove` | bool | `false` | 是否启用 mark-remove 恢复路径所需的图删除追踪元数据 |
| `support_force_remove` | bool | `false` | 是否启用 `RemoveMode::FORCE_REMOVE` 及其额外同步 |
| `store_raw_vector` | bool | `false` | 除量化副本外再保留原始向量（`cosine` 场景有用） |
| `use_elp_optimizer` | bool | `false` | 构建完成后自动调优检索参数 |
| `base_io_type` / `precise_io_type` | string | `"block_memory_io"` | 存储后端（`memory_io`、`block_memory_io`、`buffer_io`、`async_io`、`uring_io`、`mmap_io`） |
| `base_file_path` / `precise_file_path` | string | — | 磁盘后端时的文件路径（使用 `mmap_io` / `async_io` / `uring_io` / `buffer_io` 时必填） |
| `base_direct_read` / `precise_direct_read` | bool | `false` | 使用 `uring_io` 时，以 direct IO 打开对应文件而非经过页缓存 |
| `hgraph_init_capacity` | int | `100` | 初始容量提示（不会限制最终规模） |
| `persist_source_id` | bool | `false` | 序列化时保留 Source ID 元数据，使恢复后的索引仍可导出可复用的构建缓存 |
| `use_conjugate_graph` | bool | `false` | 启用 `Feedback`/`Pretrain` 图增强；详见[图索引增强](../advanced/enhance_graph.md) |
| `resize_increase_count_bit` | int | `10` | 扩容批次 slot 数的 `log2`，取值范围为 `1` 到 `31`。`1` 表示每次按 2 个 slot 对齐，`10` 表示按 1024 个 slot 对齐。较小取值减少预分配，但可能增加重分配次数。 |

`use_reverse_edges` 面向需要快速检查入邻居、图分析或图维护算法的负载。维护反向邻接表会让边
存储约翻倍，因此默认关闭。

`label_remap_type` 只改变内部 label map，不改变用户 ID。默认值为 `"pg"`；
`"robin"` 选择另一种 robin-map 实现，建议针对实际 ID 分布实测后再调整。

### 向量存储去重

同时设置 `support_duplicate: true` 和 `deduplicate_storage: true` 后，重复向量会共享
同一个物理编码槽位，但仍保留各自的标签。该选项目前仅支持使用 `graph_type: "nsw"` 的
稠密向量 HGraph 索引；`graph_type: "odescent"` 不支持。

启用存储去重后，暂不支持以下操作和配置：

- 强制删除（`support_force_remove: true`）；
- 调用 `ImportCache()` 后基于缓存加速构建；
- `Merge`；
- v0.14 旧版序列化格式。

`UpdateVector` 仅支持尚未与其他重复组成员共享向量存储的 ID。

当前序列化格式和 streaming serialization 均受支持。

## 支持的输入数据类型

顶层构建配置中的 `dtype` 字段决定 `Dataset` 如何解释原始向量字节。HGraph 支持四种输入类型，
`dtype` 值、对应的 `Dataset` setter 与演示示例见下表。

| `dtype`     | 元素类型 | `Dataset` setter         | 示例                                                                                                  |
|-------------|----------|--------------------------|-------------------------------------------------------------------------------------------------------|
| `float32`   | `float`  | `Float32Vectors`         | [`103_index_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/103_index_hgraph.cpp) |
| `int8`      | `int8_t` | `Int8Vectors`            | [`316_index_int8_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/316_index_int8_hgraph.cpp) |
| `float16`   | `uint16_t`（按 IEEE 754 binary16 位模式打包） | `Float16Vectors` | [`321_index_fp16_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/321_index_fp16_hgraph.cpp) |
| `bfloat16`  | `uint16_t`（按 Brain Float 位模式打包） | `Float16Vectors`（与 FP16 共用） | 在 `321_index_fp16_hgraph.cpp` 基础上按下文调整                                                       |

`dim` 始终表示逻辑维度（元素数量），与字节长度无关，因此四种数据类型下 `dim` 取值相同。

### `int8` 输入

量化好的 `int8` 向量直接通过 `Int8Vectors` 传入：

```cpp
std::vector<int8_t> data(num_vectors * dim);  // 填入 int8 元素
auto base = vsag::Dataset::Make();
base->NumElements(num_vectors)->Dim(dim)->Ids(ids)
    ->Int8Vectors(data.data())->Owner(false);
```

对应构建配置（注意 `dtype: "int8"`）：

```json
{
    "dtype": "int8",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "pq",
        "max_degree": 26,
        "ef_construction": 100,
        "alpha": 1.2
    }
}
```

查询时同样使用 `Int8Vectors` 和相同的 `dtype`。可运行示例：
[`316_index_int8_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/316_index_int8_hgraph.cpp)。

### `float16` / `bfloat16` 输入

FP16 与 BF16 都通过 `Float16Vectors` 传入，参数类型为 `const uint16_t*`，指向各元素的 16 位
存储。从 `float` 到 16 位格式的转换由调用方负责。VSAG 源码树内提供了便捷辅助函数
（`vsag::generic::FloatToFP16` 位于
[`src/simd/fp16_simd.h`](https://github.com/antgroup/vsag/blob/main/src/simd/fp16_simd.h)，
`vsag::generic::FloatToBF16` 位于
[`src/simd/bf16_simd.h`](https://github.com/antgroup/vsag/blob/main/src/simd/bf16_simd.h)），
但它们是**内部头文件**，并未通过 `include/vsag/` 对外安装。链接已安装版 VSAG 库的应用需要自行
完成转换（例如复制这段小工具函数、使用 `_cvtss_sh` / F16C 内置指令，或调用任意 FP16 库）。下面
的示例代码为了简洁直接使用了源码树内的辅助函数：

```cpp
// 下面的 fp16/bf16 辅助函数位于 src/simd/，并未随 VSAG 一并安装。
// 链接已安装版 VSAG 时，请替换为自行实现的 float -> uint16_t 转换。
#include "simd/fp16_simd.h"  // FloatToFP16（BF16 场景改为 simd/bf16_simd.h / FloatToBF16）

std::vector<uint16_t> data(num_vectors * dim);
for (size_t i = 0; i < data.size(); ++i) {
    data[i] = vsag::generic::FloatToFP16(some_float_source());
}
auto base = vsag::Dataset::Make();
base->NumElements(num_vectors)->Dim(dim)->Ids(ids)
    ->Float16Vectors(data.data())->Owner(false);
```

构建配置：

```json
{
    "dtype": "float16",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "pq",
        "max_degree": 26,
        "ef_construction": 100,
        "alpha": 1.2
    }
}
```

切换到 BF16 时，将 `dtype` 改为 `"bfloat16"`、把 `FloatToFP16` 替换为 `FloatToBF16` 即可；
`Float16Vectors` setter 与构建/检索流程不变。可运行 FP16 示例：
[`321_index_fp16_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/321_index_fp16_hgraph.cpp)。

> **注意。** `321_index_fp16_hgraph.cpp` 文件头注释提到 `BFloat16Vectors()`，但该 setter 并不
> 存在 —— FP16 与 BF16 都通过 `Float16Vectors` 传入。无论 `dtype` 是 `"float16"` 还是
> `"bfloat16"`，都使用同一个 setter。

### 输入类型选择建议

- 对精度要求最高、且内存预算充裕时，选 `float32`（默认）。
- 想把输入存储减半，选 `float16` / `bfloat16`。FP16 指数范围更小，BF16 尾数位更少但指数范围
  与 FP32 一致，对 embedding 类向量通常更友好。
- 数据本身已是整数量化结果（来自上游量化器或 int8 输出的模型）时，选 `int8`。此时通常仍配合
  `pq` / `sq8` 之类的索引内量化器使用。

`dtype` 仅约束**输入**表示；索引内的实际存储仍由 `base_quantization_type`（以及
`use_reorder: true` 下的 `precise_quantization_type`）决定，因此
`dtype: "float16"` + `base_quantization_type: "sq8"` 这样的组合是允许的。

## 检索参数

检索参数放在 `hgraph` 子对象下：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ef_search` | int64 | —（必填） | 正数搜索前沿大小；接受到 `INT64_MAX`，不存在与 `topk` 相关的上限。值越大，召回、延迟和前沿内存通常都越高。 |
| `hops_limit` | int | 不限 | beam search 在返回当前前沿前允许的最大跳数。 |
| `skip_ratio` | float | `0.2` | 过滤场景下的性能调优参数。控制跳过无效点的比例，取值范围 `[0.0, 1.0]`。`skip_ratio=0.2` 表示跳过 20% 的无效点，只检查 80% 的无效点。值越大性能越好但召回率可能越低。仅在带 filter 的搜索中生效。详见下文[过滤跳过策略](#过滤跳过策略skip_ratio-与-skip_strategy)。 |
| `skip_strategy` | string | `"deterministic_accumulative"` | 过滤跳过的策略。可选值：`"random"`（随机跳过）或 `"deterministic_accumulative"`（确定性累积跳过）。详见下文[过滤跳过策略](#过滤跳过策略skip_ratio-与-skip_strategy)。 |
| `brute_force_threshold` | float | `0.0` | 选择率感知的暴搜回退开关。当取值 `> 0` 且当前 filter 的 `ValidRatio()` 小于等于 `brute_force_threshold` 时，搜索会**完全跳过图遍历**，直接在通过过滤的 id 上用最佳精度的 flatten 编码做一次暴力扫描（细节见下一节）。取值范围 `[0.0, 1.0]`；默认 `0.0` 表示关闭，保持原有行为。 |
| `rabitq_one_bit_search` | bool | `false` | 启用 RaBitQ filter/lower-bound 路径；对 x+y split 索引会使用全部 x 个 filter bits，详见 [RaBitQ x+y Split](../quantization/rabitq_split.md)。 |
| `rabitq_error_rate` | float | 索引默认值 | 本次搜索使用的正数 lower-bound 误差倍率；调整它不需要重建 split 索引。 |

```cpp
auto result = index->KnnSearch(
    query, topk, R"({"hgraph": {"ef_search": 200}})").value();
```

### 高选择性过滤下的暴搜回退（`brute_force_threshold`）

图搜索在大多数候选都能通过过滤时是最优策略——图遍历能很快进入查询邻域。但是当
过滤越来越严格（只有极少数向量能通过）时，beam 需要扩展非常多的节点才能凑够
`ef_search` 个**通过过滤的**候选，此时召回率会下降，延迟反而上升。在某个临界点，
对通过过滤的 id 做一次完整暴扫既更快又精确。

`brute_force_threshold` 允许 HGraph 在每次查询时自动按 filter 选择率做这个切换：

```cpp
// 当 filter 仅保留 ≤ 1% 的 id 时，自动改走暴力扫描。
auto params = R"({"hgraph": {"ef_search": 200, "brute_force_threshold": 0.01}})";
auto result = index->KnnSearch(query, topk, params, my_filter).value();
```

工作原理（实现位于 `src/algorithm/hgraph/hgraph_search.cpp`）：

- 暴搜回退仅在**同时**满足以下条件时触发：
    - `brute_force_threshold > 0.0`，**并且**
    - 提供了 filter，**并且**
    - `filter->ValidRatio() <= brute_force_threshold`。
- `Filter::ValidRatio()` 的准确性会直接影响是否切换 —— 这是用户提供的提示值。
  详见 [带过滤的搜索](../advanced/filtered_search.md) 中关于该方法的约定。
- 暴搜会遍历所有通过过滤的内部 id，并按 64 一批用当前最精确的 flatten 存储
  计算距离（顺序：若启用了 `store_raw_vector` 则用原始向量；否则若
  `use_reorder=true` 则用精排副本；否则用基础量化编码）。
- 由于暴搜在有精排副本时本身就用了精确编码，**走暴搜分支的查询不会再做精排**。
- 该机制对 `KnnSearch`（非迭代器重载，也即 `SearchWithRequest` 与标准的
  `KnnSearch(query, k, params, filter)` 走的入口）和 `RangeSearch` 生效；对
  迭代器风格的 `KnnSearch(..., IteratorContext*&, ...)` **不生效**，因为一次
  扫描无法分页跨越多次迭代调用。

取值建议：

- 不带过滤或过滤通过率较高的场景，保持默认 `0.0`。
- 高选择性过滤（如 `ValidRatio` ≤ 0.05）下，`0.01–0.05` 是合理起点。再往上调
  实际上等于「只要带 filter 就走暴搜」。
- 暴搜的代价大致是 `O(N × dim)`，`N` 是索引内总向量数（与选择率无关，因为
  每个 id 都要走一次 `CheckValid`）。当原本需要把 `ef_search` 调到很大才能
  维持召回时，暴搜带来的收益最明显。

可运行示例：
[`322_feature_hgraph_brute_force_threshold.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/322_feature_hgraph_brute_force_threshold.cpp)。

### 过滤跳过策略（skip_ratio 与 skip_strategy）

当搜索带有 filter 时，HGraph 在图遍历过程中需要频繁调用 Filter::CheckValid() 来验证每个候选点是否有效。这个检查可能很耗时（特别是复杂过滤逻辑）。skip_ratio 和 skip_strategy 提供了一种概率性优化：通过跳过部分 filter 检查来加速搜索，但可能降低召回率。

#### 工作原理

这是一个概率性优化策略：我们事先不知道哪些点是有效的，因此按概率决定是否访问每个点。

- skip_ratio（默认 0.2）：控制跳过 filter 检查的激进程度。skip_ratio=0.2 表示跳过 20% 的无效点，只检查 80% 的无效点。值越大，跳过的越多，速度越快，但召回率可能越低。
- skip_strategy（默认 "deterministic_accumulative"）：决定如何分配跳过：
  - "random"：随机跳过。每个点被访问的概率为 `visit_ratio = valid_ratio + (1 - valid_ratio) * (1 - skip_ratio)`，大约跳过 `skip_ratio` 比例的无效点。
  - "deterministic_accumulative"：确定性累积跳过。按固定间隔做出访问决策，使长期访问比例趋近于目标 `visit_ratio`，相比 random 策略方差更小。

具体公式：
- 设 valid_ratio 为 filter 的全局有效率（来自 Filter::ValidRatio()）
- 每个点被访问的概率 = valid_ratio + (1 - valid_ratio) * (1 - skip_ratio)
- 如果 Filter::ValidRatio() 估计准确，期望跳过约 skip_ratio 比例的无效候选检查

#### 使用示例

```cpp
// 保守设置：跳过 10% 的无效候选检查，适合召回率要求高、延迟不那么关键的场景
auto params = R"({"hgraph": {"ef_search": 200, "skip_ratio": 0.1}})";
auto result = index->KnnSearch(query, topk, params, my_filter).value();

// 使用随机策略
auto params = R"({"hgraph": {"ef_search": 200, "skip_ratio": 0.2, "skip_strategy": "random"}})";
auto result = index->KnnSearch(query, topk, params, my_filter).value();

// 激进跳过：跳过 50% 的无效候选检查，以更低延迟为目标
auto params = R"({"hgraph": {"ef_search": 200, "skip_ratio": 0.5}})";
auto result = index->KnnSearch(query, topk, params, my_filter).value();
```

#### 取值建议

- 默认 0.2：适合大多数场景，在性能和召回率之间取得平衡。
- 0.1 或更低：保守设置，适合对召回率要求高、延迟不那么关键、可接受召回率下降的场景（如实时推荐系统）。
- 0.5 或更高：激进跳过，适合对延迟敏感、可接受召回率下降的场景。
- 0.0：不跳过任何点，等同于关闭此优化（所有点都会被检查）。

注意事项：
- 仅在带 filter 的搜索中生效。无 filter 时这些参数会被忽略。
- 如果 Filter::ValidRatio() 估计准确，性能优化效果更好。
- 与 brute_force_threshold 可同时使用：当 filter 非常严格（ValidRatio 很小）时，brute_force_threshold 会触发暴搜回退；否则使用图遍历 + skip 优化。

## 何时选择 HGraph

- 维度大约在 64–4096 的稠密 float 向量。
- 对延迟敏感且要求高召回的场景。
- 需要增量插入（可选通过 `support_force_remove` 打开物理删除）的混合负载。
- 内存受限环境，可用 `sq8` / `sq4_uniform` / `pq` 压缩，再配合 `use_reorder` 弥补召回。

对于 Source ID 稳定的每日构建或快照构建，参见
[HGraph 构建缓存](../advanced/build_cache.md)。其中说明了 `Dataset::SourceID`、
`ExportCache`/`ImportCache`、`persist_source_id` 与缓存命中率诊断。

如果你的业务偏向粗粒度分桶（每次查询只扫部分桶）或严重受 SSD I/O 制约，建议先对比
[IVF](ivf.md) 再决定是否选择 HGraph。

## 相关文档

- [创建索引](../guide/create_index.md)
- [图索引增强](../advanced/enhance_graph.md)
- [优化器](../advanced/optimizer.md)
- [序列化格式](../advanced/serialization.md)
