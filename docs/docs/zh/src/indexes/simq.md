# SIMQ

SIMQ（Single Index for Multi-vector Query）是 VSAG 面向 **多向量（multi-vector）**
检索的索引——适用于每篇文档由一组 token 级向量（而非单个 embedding）表示的
数据场景。这种模式常见于 ColBERT 等 late-interaction 模型，其中文档由每个 token
对应一个向量表示，相关性通过 **MaxSim**（各查询 token 的最大相似度之和）计算。


- 源码：`src/algorithm/simq/`

## 工作原理

1. **图驱动 token 分区。** token 编码存入临时池，抽样真实 token 建立初始 HGraph；其余 token 通过图搜索分配到分区，过大分区动态切分。这不是迭代 k-means。
2. **代表图粗排。** 每个分区选择真实成员 token 作为代表，用量化 HGraph 为各查询 token 路由，展开分区文档列表并聚合粗排分数。
3. **MaxSim 精排。** 读取前 `rerank_k` 个候选文档的存储编码并解码，在存储表示上计算 MaxSim。使用有损量化时，这不等于对原始 FP32 输入的精确打分。

每篇文档在一次查询中只进入粗排候选集一次，即使累计粗排分数为零，或不同查询 token 的分数贡献相互抵消。

两阶段检索在候选召回、量化误差、内存和延迟之间提供权衡。

## 快速开始

```cpp
#include <vsag/vsag.h>

std::string build_params = R"({
    "dtype": "float32",
    "metric_type": "ip",
    "dim": 256,
    "index_param": {
        "base_io_type": "async_io",
        "base_file_path": "/path/to/simq_base_codes.bin",
        "init_cluster_ratio": 0.1,
        "max_cluster_size": 160,
        "split_start_idx": 80,
        "random_seed": 42,
        "coarse_k": 50,
        "rerank_k": 1000
    }
})";
auto index = vsag::Factory::CreateIndex("simq", build_params).value();

// 使用 MultiVector 构建数据集。
// 每篇文档包含可变数量的 token 向量，每个向量维度为 dim。
std::vector<vsag::MultiVector> base_mvs(num_docs);
std::vector<int64_t> ids(num_docs);
for (int64_t i = 0; i < num_docs; ++i) {
    base_mvs[i].len_ = doc_token_counts[i];             // 文档 i 的 token 数量
    base_mvs[i].vectors_ = doc_token_vectors[i];        // 扁平数组：len_ * dim 个 float
    ids[i] = i;
}
auto base = vsag::Dataset::Make();
base->NumElements(num_docs)
    ->Dim(dim)
    ->Ids(ids.data())
    ->MultiVectors(base_mvs.data())
    ->MultiVectorDim(dim)
    ->Owner(false);
index->Build(base);

// 使用多向量查询进行检索。
vsag::MultiVector query_mv;
query_mv.len_ = query_token_count;
query_mv.vectors_ = query_token_vectors;
auto query = vsag::Dataset::Make();
query->NumElements(1)
    ->Dim(dim)
    ->MultiVectors(&query_mv)
    ->MultiVectorDim(dim)
    ->Owner(false);

std::string search_params = R"({
    "simq": {
        "coarse_k": 600,
        "rerank_k": 5000
    }
})";
auto result = index->KnnSearch(query, /*topk=*/100, search_params).value();

// 读取结果。
const int64_t* result_ids = result->GetIds();
const float* result_dists = result->GetDistances();
int64_t result_count = result->GetDim();
for (int64_t i = 0; i < result_count; ++i) {
    int64_t id = result_ids[i];
    float dist = result_dists[i];
}
```

## 构建参数

SIMQ 专属构建参数放在 `index_param` 下；通用字段保留在顶层。
无论内部采用哪种存储量化，输入仍为 FP32 `MultiVector` 数据。

### 顶层通用字段

| 参数 | 类型 | 默认值 | 约束与作用 |
|------|------|--------|------------|
| `dim` | int | 必填 | 正整数；所有文档和查询 token 的维度。 |
| `dtype` | string | 必填 | 必须为 `"float32"`；描述输入标量类型，而非存储量化器。 |
| `metric_type` | string | 必填 | 必须为 `"ip"`。 |
| `repr` | string | `"dense"`（通用解析器默认值） | 可选通用布局字段；可显式用 `"multi_vector"` 描述数据。SIMQ 的 Build/Add/检索接口仍要求 `MultiVectors`，不能传入单个稠密向量。 |
| `extra_info_size` | int | `0` | 共享解析器接受整数，无非负检查；正值会初始化继承的 extra-info 存储，但 SIMQ Build/Add 与检索不填充或返回它。本工作流请保持 `0`。 |
| `use_old_serial_format` | bool | `false` | 通用兼容标志；SIMQ 不消费此字段来选择序列化格式。 |

### `index_param` 下的字段

下表列出全部 15 个 SIMQ 公开构建字段；`index_param` 下的其他键会被拒绝。

| 参数 | 类型 | 默认值 | 约束与作用 |
|------|------|--------|------------|
| `base_io_type` | string | `"async_io"` | 文档 token 编码后端：`async_io`、`uring_io`、`buffer_io`、`mmap_io`、`memory_io`、`block_memory_io` 或 `reader_io`；注意下方 IO 限制。 |
| `base_file_path` | string | `"./default_file_path"` | 磁盘存储（`async_io`、`uring_io`、`buffer_io`、`mmap_io`）需提供可用文件路径；默认值仅为占位符。 |
| `base_enable_read_cache` | bool | `false` | 为 `async_io`、`uring_io`、`buffer_io`、`reader_io` 启用可选 LRU 页缓存。`memory_io`、`block_memory_io`、`mmap_io` 使用 `NoCache`，忽略此开关。 |
| `base_cache_total_size` | unsigned int | `268435456`（256 MiB） | 缓存字节预算，必须为非负整数；容量按 128 KiB 页向下取整，小于一页（含 `0`）时即使开启也不缓存。作用于文档编码 IO 实例，不是代表图缓存。 |
| `quantization_type` | string | `"fp32"` | 工厂支持的文档/图存储类型：`fp32`、`fp16`、`bf16`、`sq8_uniform`、`int8`；注意下方 `int8` 限制，不改变 FP32 输入要求。 |
| `build_thread_count` | unsigned int | `1` | 共享解析器接受非负整数，SIMQ 无额外正数检查；建议使用 `>= 1`。控制构建图、代表图与聚类工作，不是每次查询的并行度开关；聚类至少使用一个 worker。 |
| `init_cluster_ratio` | float | `0.2` | 范围 `(0, 1]`；初始种子 token 的采样比例。较小值通常产生更少、更大的分区。 |
| `max_cluster_size` | int | `64` | 必须 `> 1`；触发分区切分的 token 数阈值，不是文档数上限。 |
| `split_start_idx` | int | `32` | 范围 `(1, max_cluster_size)`；溢出分区按距离排序后，新分区的起始位置。 |
| `random_seed` | int | `42` | 初始 token 打乱及文档模型有界训练采样的随机种子；SIMQ 无额外范围检查。 |
| `coarse_k` | int | `8` | 必须 `> 0`；默认每个查询 token 搜索的最近代表簇数，检索时不超过可用簇数。 |
| `rerank_k` | int | `100` | 必须 `> 0`；默认进入 MaxSim 精排的候选文档数上限，检索时不超过文档总数。 |
| `split_delay_seconds` | float | `0.0` | 必须 `>= 0`；增量 `Add` 首次溢出后，等待该时长才允许切分；`0` 表示立即切分。待切分项在 `Add` 中检查，不存在后台定时器，也不推迟初次 Build 聚类切分。 |
| `support_duplicate` | bool | `false` | 相同代表向量可共享代表 HGraph 节点；不对文档 ID 去重，也不在构建内部 HGraph 中抑制重复 token。 |
| `deduplicate_storage` | bool | `false` | 要求 `support_duplicate: true`；共享代表 HGraph 中的重复向量编码槽位，不是文档 token 存储去重。 |

**IO 限制。** 原生 `uring_io` 需要 liburing；缺少时使用 buffered 后端回退。
工厂识别 `reader_io`，但它是只读的外部 Reader 后端，需要绑定非空 Reader；
仅在 JSON 中选择此类型不提供可写的 Build/Add 存储，不能把所有可识别的 IO 类型当作可互换的构建后端。

**量化限制。** 工厂支持 `int8`，但当前编码器直接复制输入原始字节，并不执行数值上的
FP32 到 int8 转换。因此不能将其宣传为 FP32 token 值的常规标量压缩。
FP32 工作流优先选择 `fp32`、`fp16`、`bf16`、`sq8_uniform`，并以实际数据验证存储精度。

> **聚类参数的选择。** `init_cluster_ratio` 与 `max_cluster_size` 共同控制
> 簇的数量与大小。较小的 `init_cluster_ratio` 搭配较大的 `max_cluster_size`
> 会产生更少的簇，粗排更快但召回降低。建议以 `init_cluster_ratio = 0.1`–`0.2`、
> `max_cluster_size = 2 × split_start_idx` 为起点，再通过检索参数调优。

## 检索参数

`simq` 子对象中的以下字段同时作用于 `KnnSearch` 与 `RangeSearch`：

| 参数 | 类型 | 默认值 | 约束与作用 |
|------|------|--------|------------|
| `coarse_k` | int | `-1` → 构建值 | 正数覆盖构建值；省略或 `<= 0` 时回退到构建值。每个查询 token 探测更多簇会扩大候选范围，通常也增加延迟。 |
| `rerank_k` | int | `-1` → 构建值 | 正数覆盖构建值；省略或 `<= 0` 时回退到构建值。更多候选会增加 MaxSim IO/计算，与请求的 `topk`/范围输出数独立。 |
| `timeout_ms` | int | 省略（无截止时间） | 毫秒计的软耗时预算。字段存在即启用计时；解析器无正数检查，因此 `<= 0` 表示预算立即耗尽，不是“关闭超时”。可能返回部分或空结果。 |

计时在粗排前开始，但不会中断代表图搜索或正在进行的 IO/MaxSim 工作。
粗排后、精排工作单元之间（KNN 每批最多 32 篇文档，范围搜索每次一篇）以及返回前检查预算。
粗排阶段超时会清空粗排候选。这不是严格的墙钟延迟上界；请检查返回统计中的 `is_timeout` 标志。

共享解析器还接受 `simq` 内的以下字段，但 SIMQ 的两条检索路径均不消费它们：

| 参数 | 类型 | 解析默认值 | SIMQ 作用 |
|------|------|------------|-----------|
| `parallelism` | int | `1` | 忽略；共享解析器会把非正值归一为 `1`。 |
| `factor` | float | `0.0` | 忽略；SIMQ 无范围检查。请改用 `rerank_k` 调节候选数。 |
| `enable_reorder` | bool | `true` | 忽略，包括 `false`；SIMQ 始终对保留候选执行 MaxSim 精排。 |

| 根级参数 | 类型 | 默认值 | 约束与作用 |
|----------|------|--------|------------|
| `threshold` | float | 省略 | 仅 KNN：最终距离的有限、包含边界的上限；详见下文。 |

**仅 KNN** 支持可选的根级 `threshold`（float，默认省略），必须是有限 JSON 数字。
它是**最终返回距离**的包含边界的上限，不是粗排截断阈值或相似度下限；仅有限且
`<= threshold` 的距离通过。允许负阈值，返回数量可能少于 `topk`。
SIMQ 距离是对每个查询 token 在存储文档 token 上求最小 `1 - inner_product` 后求和，
并非单向量 IP 距离。参见 [KNN 阈值过滤](../guide/knn_search.md)。
`RangeSearch` 使用 `radius` 参数，不消费根级 `threshold`；`limited_size` 限制范围输出数量，
不控制由 `rerank_k` 限定的候选池。

```cpp
auto result = index->KnnSearch(
    query, topk,
    R"({"threshold": 4.0, "simq": {"coarse_k": 600, "rerank_k": 5000, "timeout_ms": 100}})").value();
```

## 搜索统计

`KnnSearch` 与 `RangeSearch` 会在返回的 dataset 上附带 JSON 统计对象。可以读取完整对象，
也可以按 key 获取单项值：

```cpp
std::cout << result->GetStatistics() << '\n';
auto values = result->GetStatistics(
    {"simq_coarse_candidate_count", "simq_rerank_candidate_count"});
```

| Key | 含义 |
|-----|------|
| `dist_cmp` | 精排阶段执行的精确 MaxSim 距离计算次数 |
| `simq_coarse_dist_cmp` | 代表图搜索执行的距离计算次数 |
| `simq_coarse_probe_count` | 所有 query token 探测的代表簇数量 |
| `simq_coarse_candidate_count` | 粗排产生且尚未应用 `rerank_k` 的唯一文档候选数 |
| `simq_rerank_candidate_count` | 应用 `rerank_k` 后保留的候选数 |
| `simq_filtered_candidate_count` | 被调用方 filter 排除的精排候选数 |
| `simq_result_count` | 精排并应用 range/top-k 限制后最终返回的结果数 |
| `simq_limited_size_applied` | `RangeSearch` 是否因 `limited_size` 截断结果 |

这些计数器可以在调节召回和延迟时，区分候选池过窄、过滤、精排和输出上限各自的影响。

## 何时选择 SIMQ

- **Late-interaction 检索**：使用 ColBERT 等模型，每篇文档是一组 token 向量，
  相关性通过 MaxSim 计算。
- **多向量粒度匹配**：单 embedding 丢失过多信息，需要 token 级细粒度匹配。
- **大规模多向量语料**：暴力 MaxSim 检索过慢，需要粗排 + 精排的两阶段管线
  来平衡召回与延迟。

SIMQ 仅接受 `float32` 多向量数据，仅支持内积相似度。
**不支持** 单稠密向量或稀疏向量（请使用 HGraph 或 SINDI）。

## 实践建议

- **调整 `coarse_k` 与 `rerank_k`。** 增大 `coarse_k` 扩大簇级候选范围；
  增大 `rerank_k` 让更多文档进入基于存储表示的 MaxSim 打分。实践中 `rerank_k` 对召回的影响
  更大，但每个额外候选都需要访问存储和完整的 MaxSim 计算，延迟也会增加。
- **IO 类型选择。** 语料规模超出内存时使用 `async_io`；多向量数据可以放入
  内存时，使用 `memory_io` 或 `block_memory_io` 获得最低的精排延迟。
- **簇大小配置。** 将 `max_cluster_size` 设为 `split_start_idx` 的约两倍。切分位置
  决定了簇溢出时 token 向量的分配方式，居中设置可使两半保持平衡。

## MultiVector 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `len_` | `uint32_t` | 当前文档或查询包含的 token 向量数量 |
| `vectors_` | `float*` | `len_ * dim` 个 float 的连续数组 |

## 相关文档

- [创建索引](../guide/create_index.md)
- [索引参数](../resources/index_parameters.md)
- [k-近邻搜索](../guide/knn_search.md)

## 量化构建与增量导入

即使 `quantization_type` 为 `fp16`、`bf16` 或 `sq8_uniform`，Dataset 输入仍然必须是 FP32 `MultiVector`。SIMQ 在分区构建前编码 token，通过量化器原生 code-to-code 距离维护分区；构建图与代表图均使用量化 base 存储，不配置 FP32 精排副本。图插入/查询接口和均值累加仍使用有界 FP32 临时解码，不是完全免解码构建。

构建不再创建全量 FP32 token 打平副本，而使用临时编码池、token 地址元数据及每个最多 4096 个 token 的训练/暂存缓冲。编码池在 Build 返回前与文档编码存储重复存在；用户持有的 FP32 输入及 O(tokens) 元数据仍占内存。Build 将最多 4096 个 token 的确定性样本传给文档量化器的 Train 方法（数据更多时有放回抽样）；是否学习数据相关状态取决于量化器。后续 Add 复用文档模型，不重新训练。两个 HGraph 各自拥有量化模型，不混用不同模型的编码字节。

可以使用 `Build(首批)` 后分批 `Add(后续批次)`，首批应覆盖后续数据分布。量化和增量构建不保证与全量 FP32 构建产生相同分区或召回，请使用真实负载评估。SIMQ 输入仍仅支持 FP32，metric 仍仅支持内积；输入类型与内部存储类型是不同概念。

### 序列化兼容性

新版本可加载历史无版本标记的 SIMQ 索引，构造时维度与文档量化格式必须一致。历史代表图保持 FP32，保留加载前搜索行为；加载时以有界向量缓冲重建文档模型下的代表编码，并通过 O(tokens) 遍历更新距离。后续增量分区轨迹不保证与旧实现一致。重新保存会记录实际代表图量化格式和新元数据。新写入采用 SIMQ 格式版本 1，不支持的版本及维度/量化不匹配会被拒绝；不保证旧程序读取新文件。
