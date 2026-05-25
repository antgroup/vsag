# SINDI 稀疏索引研究报告

生成日期：2026-05-19

研究范围：
VSAG 当前仓库中的 SINDI 文档、C++ 实现、测试、示例，
以及论文页面 `arXiv:2509.08395` v3 的公开摘要。
本文面向后续研发、调参、文档校对和用户支持。

## 摘要

SINDI 是 VSAG 的稀疏向量 MIPS 索引。
它适合 BM25、SPLADE、BGE-M3 sparse embedding、
uniCOIL 等 term/value 形式的检索向量。

SINDI 不把 sparse vector 转成 dense embedding。
它直接围绕非零 term 构建窗口化倒排表，
查询时扫描相关 posting list，累积分数，
并用候选堆维护 top-k 或 range 结果。

论文将 SINDI 展开为：
Sparse Inverted Non-redundant Distance Index。
公开摘要强调三类优化：

- 高效 inner product 计算。
- 顺序访问倒排表，减少原始稀疏向量随机访问。
- 保留高权重非零项的 vector pruning。

论文摘要称，在 MsMarco 数据集上，
当 Recall@50 超过 99% 时，
SINDI 的单线程 QPS 相比 SEISMIC 和 PyANNs
提升 4.2x 到 26.4x，并已集成到 VSAG。

VSAG 当前实现主要入口：

- `src/algorithm/sindi/`
- `src/datacell/sparse_term_datacell.*`
- `src/quantization/sparse_quantization/sparse_term_computer.h`

当前实现支持构建、增量 Add、KNN、Range Search、
ID filter、序列化、按 ID 计算距离、raw sparse vector 获取、
GetStats、AnalyzeIndexBySearch、内存估算，
以及可选 quantization、rerank 和 term ID remap。

## 背景与问题定义

稀疏向量检索常见于文本检索和 RAG 多路召回。
每个文档或查询表示为若干非零项：

```text
SparseVector = [(term_id_0, value_0), ...]
score(query, doc) = inner_product(query, doc)
distance = 1 - inner_product
```

稀疏 MIPS 的主要工程难点：

- 维度空间可能很大。
- 每条向量通常只有少量非零项。
- 原始稀疏向量压缩格式容易导致随机访问。
- 随机访问会削弱缓存局部性。
- 倒排索引减少无关文档扫描。
- posting list 扫描和 candidate 管理又带来新开销。
- 线上 RAG 通常要求高召回、低延迟和可增量写入。

SINDI 的目标是为 sparse vector 提供比 brute-force 更高的吞吐。
它通过剪枝、量化和可选 rerank，
在准确率、延迟和内存之间做可调平衡。

## 命名口径

仓库中存在三种历史口径：

- `docs/sindi.md`：Sparse Inverted Index。
- `docs/docs/en/src/indexes/sindi.md`：
  Sparse INverted Dense Index。
- 论文与 `research_papers.md`：
  Sparse Inverted Non-redundant Distance Index。

建议后续以论文题名为准。
也就是 Sparse Inverted Non-redundant Distance Index。
官网索引页中的 Dense 很可能是旧文档或笔误。

## 数据模型

SINDI 只接受：

- `dtype: "sparse"`
- `metric_type: "ip"`

两个参数最容易混淆：

- `dim`：单条 sparse vector 的最大非零项数。
  它不是词表大小。
- `term_id_limit`：term id 上限。
  它应大于数据集最大 term id。
  当前解析限制是 `(0, 10_000_000]`。
  默认值是 `1_000_000`。

例如 `{0: 0.1, 2: 0.5, 177: 0.8}`：

- `dim` 是 3。
- `term_id_limit` 至少应为 178。

## 索引结构

SINDI 将数据按固定数量文档切成多个 window。
每个 window 内维护一个 `SparseTermDataCell`。

`window_size` 控制每个 window 的文档数。
当前合法范围是 `[10_000, 60_000]`。
默认值是 `50_000`。

单个 window 内的数据结构：

```text
SINDI
  window_term_list_[window_id] -> SparseTermDataCell
    term_ids_[term_id]   -> uint16 doc offsets
    term_datas_[term_id] -> float bytes or uint8 values
    term_sizes_[term_id] -> posting length
```

因为每个 window 最多 60,000 条文档，
window 内部 doc offset 可以用 `uint16_t` 存储。
posting list 只保存局部 doc id。
最终结果通过 window 起始偏移和 label table
还原为用户外部 ID。

## 构建流程

构建入口是 `SINDI::Build`。
它实际复用 `SINDI::Add`。

主要步骤：

1. 校验 dataset 非空。
2. 读取 sparse vectors、ids 和 extra info。
3. 如启用 `use_quantization` 且索引为空，
   基于首批 Add 数据统计 `min_val/max_val/diff`。
4. 根据新增数据量扩展 window。
5. 每个 window 创建一个 `SparseTermDataCell`。
6. 对每条 sparse vector 检查重复 ID 和空向量。
7. 失败向量的外部 ID 会进入 failed IDs。
8. 如启用 `remap_term_ids`，先映射到 compact id。
9. 调用 `SparseTermDataCell::InsertVector` 写倒排表。
10. 更新 label table、extra info 和元素计数。
11. 如启用 `use_reorder`，同步写内部 `SparseIndex`。

`SparseTermDataCell::InsertVector` 会先检查最大 term id。
如果超过 `term_id_limit`，插入会失败。
随后它按 value 降序排序 sparse vector。

若 `doc_prune_ratio > 0`，实现不是简单按数量截断。
它会保留高权重 prefix，直到累计权重达到目标质量比例。
目标比例是 `1 - doc_prune_ratio`。

## 查询流程

SINDI 当前要求每次搜索只有一个 query：

```text
query->GetNumElements() == 1
```

KNN search 的核心路径：

1. 解析 `SINDISearchParameter`。
2. 参数包括 `n_candidate`、`query_prune_ratio`、
   `term_prune_ratio` 和 `use_term_lists_heap_insert`。
3. 如果有外部 ID filter，封装成 inner id filter。
4. 通过 filter 中的有效 ID 估计要访问的 window 范围。
5. 如启用 `remap_term_ids`，查询只保留已见过的 term。
6. 如果查询 term 全部未命中，直接返回空结果。
7. `SparseTermComputer` 按 value 降序排序 query。
8. query value 会乘以 -1。
9. heap 中的 distance 因此表示为 `-inner_product`。
10. 对每个 window 调用 `SparseTermDataCell::Query`。
11. Query 扫描对应 posting list 并累加到 `dists` 数组。
12. `dists` 数组长度是 `window_size`。
13. 根据 `use_term_lists_heap_insert` 选择插堆路径。
14. 可从 term list 触达的 doc 插堆。
15. 也可扫描整个 `dists` 数组插堆。
16. 插入候选后，将对应 `dists[id]` 清零。
17. 如果启用 `use_reorder`，候选交给内部 `SparseIndex`。
18. 内部 `SparseIndex` 用原始精度重新算距离。
19. 如果不启用 rerank，返回 `1 - inner_product`。

Range Search 复用同一个 `search_impl`。
区别是使用 `radius` 阈值和可选 `limited_size`。

## 剪枝、量化和重排

### 文档剪枝

`doc_prune_ratio` 是构建期参数。
实现先按 term value 从大到小排序。
然后保留高权重 prefix。
保留直到累计权重达到 `1 - doc_prune_ratio`。

该策略适合 BM25、SPLADE、BGE-M3 这类
非负权重稀疏向量。
剪枝越强，posting list 越短。
内存和延迟通常降低，但 recall 风险升高。

### 查询剪枝

`query_prune_ratio` 是搜索期参数。
它由 `SparseTermComputer` 控制。
查询项按 value 降序排序。
搜索只扫描保留比例内的最高权重 term。
非空 query 至少保留一个 term。

### Term List 剪枝

`term_prune_ratio` 是搜索期参数。
它对每个 query term 只扫描 posting list 前缀。

需要注意：
posting list 插入顺序来自文档写入顺序。
它不是显式按 value 排序。
所以该参数更像扫描比例，
不能简单理解为保留最高权重 posting。

### 量化

`use_quantization` 将 posting value 线性量化成 `uint8_t`。
量化比例来自首批 Add 数据的 min/max。

这会把每个 posting value 从 4 字节 float
降到 1 字节 uint8。
但后续 Add 若出现超出首批范围的值，
会被 clamp 到量化范围。

因此大规模增量写入时，
首批样本代表性会影响量化误差。

### 高精度重排

`use_reorder` 会额外维护一个内部 `SparseIndex`。
候选生成后，它用原始精度重新计算距离。

这个选项能提升剪枝和量化后的准确率。
代价是内存大致翻倍，
构建与 Add 也要写两份结构。

当前实现已把 rerank 存储抽象为后端：

- `rerank_type: "fp32"` 是默认后端，行为等同历史 `SparseIndex`。
- `rerank_type: "dmq"` 是初始压缩后端，保留精确 term id，
  用每条向量的 lower/step 元数据和 bit-packed code 压缩 value。
- `dmq_bits` 支持 `1`、`2`、`4`、`8`，默认 `8`。
- `rerank_type: "dmq"` 要求 `use_reorder: true`。

这个后端优先替换 SINDI 内存热点中的 fp32 rerank 存储。
它还不是 DMQ-main 的完整 IVF/估计器实现，
因此需要通过 Wholenet 等数据集实测 recall、RSS、QPS 和 latency。

## Term ID Remap

`remap_term_ids` 用于处理 term id 极稀疏的场景。
例如存在大 gap、hash-based token、外部词表 ID 很大。

构建时，`TermIdMapper` 会为首次出现的原始 term id
分配连续 compact id。
查询时，未在构建阶段见过的 term 不参与搜索。

序列化保存 compact-to-original 映射。
反序列化时会重建 forward map。

该功能能减少大量空 posting list 造成的结构开销。
启用后，`term_id_limit` 表达 compact id 容量上限，
不再表达原始 term id 最大值。

## 参数速查

构建参数：

- `dtype`
  - 默认：`sparse`
  - 约束：必须为 `sparse`
  - 说明：数据类型
- `metric_type`
  - 默认：`ip`
  - 约束：必须为 `ip`
  - 说明：内积距离
- `dim`
  - 默认：必填
  - 约束：正整数
  - 说明：单条 sparse vector 最大非零项数
- `term_id_limit`
  - 默认：`1_000_000`
  - 约束：`(0, 10_000_000]`
  - 说明：term id 或 compact term id 容量上限
- `window_size`
  - 默认：`50_000`
  - 约束：`[10_000, 60_000]`
  - 说明：每个 window 的文档数
- `doc_prune_ratio`
  - 默认：`0.0`
  - 约束：`[0.0, 0.9]`
  - 说明：构建期文档项剪枝比例
- `use_quantization`
  - 默认：`false`
  - 约束：bool
  - 说明：posting value 线性 SQ8 量化
- `use_reorder`
  - 默认：`false`
  - 约束：bool
  - 说明：保留高精度 flat rerank 索引
- `rerank_type`
  - 默认：`fp32`
  - 约束：`fp32` 或 `dmq`
  - 说明：`use_reorder` 使用的 rerank 存储后端
- `dmq_bits`
  - 默认：`8`
  - 约束：`1`、`2`、`4` 或 `8`
  - 说明：DMQ rerank 后端的 value 量化位宽
- `remap_term_ids`
  - 默认：`false`
  - 约束：bool
  - 说明：原始 term id 映射为 compact id
- `avg_doc_term_length`
  - 默认：`100`
  - 约束：`> 0`
  - 说明：仅用于内存估算

搜索参数：

- `n_candidate`
  - 默认：`0`
  - 约束：`<= SPARSE_AMPLIFICATION_FACTOR * topk`
  - 说明：候选堆大小
- `query_prune_ratio`
  - 默认：`0.0`
  - 约束：`[0.0, 0.9]`
  - 说明：查询项剪枝比例
- `term_prune_ratio`
  - 默认：`0.0`
  - 约束：`[0.0, 0.9]`
  - 说明：posting list 扫描比例
- `use_term_lists_heap_insert`
  - 默认：`true`
  - 约束：bool
  - 说明：从 term list 触达候选插堆

注意：
官网文档描述 `n_candidate=0` 时，
会自动取 `SPARSE_AMPLIFICATION_FACTOR * topk`。
但当前代码设置 `inner_param.ef = max(n_candidate, k)`。
因此 `n_candidate=0` 实际表现为至少 top-k 候选。
这个实现/文档差异建议后续确认。

## 内存模型

`SINDI::EstimateMemory` 的估算包括：

- label table：约 `2 * sizeof(int64_t) * num_elements`。
- 未量化 posting：
  `avg_doc_term_length * num_elements * (sizeof(float) + sizeof(uint16_t))`。
- 量化 posting：
  `avg_doc_term_length * num_elements * (sizeof(uint8_t) + sizeof(uint16_t))`。
- 启用 `use_reorder` 时，核心结构估算乘以 2。
- term list 元数据：
  `sizeof(std::vector<float>) * 2 * term_id_limit`。
- 启用 `remap_term_ids` 时，额外估算约 `54 * term_id_limit`。

真实内存还受 allocator、vector capacity、window 数量、
posting 分布、序列化 buffer 和 extra info 影响。
对于 SINDI，建议构建完成后调用 `GetMemoryUsage()`。

## 功能覆盖

从当前实现和 `tests/test_sindi.cpp` 看，
SINDI 已覆盖以下能力：

- 构建和继续 Add。
- KNN search。
- Range Search。
- ID filter search。
- 搜索 allocator 参数。
- 并发 KNN search。
- Add/Search 并发场景。
- binary、file、reader set 序列化。
- duplicate ID 检测。
- 非法 sparse vector 失败 ID 返回。
- `GetRawVectorByIds`。
- `CalcDistanceById`。
- batch distance by ID。
- `GetStats`。
- `AnalyzeIndexBySearch`。
- posting 分布分析。
- doc prune recall 分析。
- quantization recall 分析。
- `remap_term_ids` 的构建、查询和序列化组合。

## 当前限制和风险点

1. 只支持 sparse vector 与 inner product。
2. 不支持 dense vector、L2 或 Cosine。
3. 搜索接口当前要求单 query。
4. 批量 query 需要上层循环。
5. `UpdateVector` 当前只检查旧向量是否为新向量子集。
6. `UpdateVector` 不实际更新 SINDI posting。
7. `n_candidate=0` 的文档行为与代码行为不一致。
8. 该差异可能影响默认召回。
9. `term_prune_ratio` 只按 posting list 前缀截断。
10. posting list 未按 value 排序。
11. 所以它更接近扫描比例。
12. `use_quantization` 的 min/max 来自首批 Add。
13. 后续增量数据分布漂移会增加量化误差。
14. `window_size` 受 `uint16_t` doc offset 约束。
15. 当前解析限制为 60,000。
16. 示例中 100000 optimal 的注释与代码限制不一致。
17. 启用 `remap_term_ids` 后，新 term 会被查询侧忽略。
18. 在线新词多的场景需要关注召回。
19. 论文摘要提到 SIMD acceleration。
20. 当前可见代码主要是顺序 posting 扫描和 prefetch hint。
21. 显式 SIMD 路径需要进一步确认。
22. SINDI 当前是 sparse leg。
23. 它尚未在索引内部融合 dense+sparse 双路打分。

## 调参建议

### 高召回基线

适合首次验证正确性和建立 recall/QPS 基线：

```json
{
  "dtype": "sparse",
  "metric_type": "ip",
  "dim": 1024,
  "index_param": {
    "term_id_limit": 1000000,
    "window_size": 50000,
    "doc_prune_ratio": 0.0,
    "use_quantization": false,
    "use_reorder": false,
    "remap_term_ids": false
  }
}
```

搜索时先使用保守参数：

```json
{
  "sindi": {
    "n_candidate": 200,
    "query_prune_ratio": 0.0,
    "term_prune_ratio": 0.0
  }
}
```

### Wholenet Sparse 10M 推荐配置

针对当前数据集：

```text
/root/data/wholenet-sparse-10m-ip.hdf5
```

推荐构建参数：

```json
{
  "dtype": "sparse",
  "dim": 301,
  "metric_type": "ip",
  "index_param": {
    "term_id_limit": 300000,
    "window_size": 50000,
    "doc_prune_ratio": 0.2,
    "use_quantization": true,
    "use_reorder": true,
    "rerank_type": "fp32",
    "dmq_bits": 8,
    "avg_doc_term_length": 50
  }
}
```

推荐搜索参数：

```json
{
  "sindi": {
    "query_prune_ratio": 0.2,
    "term_prune_ratio": 0,
    "n_candidate": 100,
    "use_term_lists_heap_insert": true
  }
}
```

建议 sweep：

- `query_prune_ratio`：优先测试 `0.1` 和 `0.2`。
- `n_candidate`：优先测试 `50` 到 `500`。
- 可选点位：`50`、`100`、`200`、`300`、`500`。
- 用户提到的 `c_candidate` 应对应 SINDI 搜索参数 `n_candidate`。
- `term_prune_ratio` 建议先保持 `0`。
- `use_term_lists_heap_insert` 建议保持 `true`。

直接运行 SINDI 的脚本入口：

```bash
bash benchs/run_sindi_wholenet_direct.sh
```

测试 DMQ rerank 后端时可切换环境变量：

```bash
RERANK_TYPE=dmq DMQ_BITS=8 bash benchs/run_sindi_wholenet_direct.sh
```

当前实测对比：

- fp32 result：`sindi_wholenet_sparse_10m_20260519_123839`。
- dmq 8-bit result：`sindi_wholenet_sparse_10m_20260520_024238`。

| Backend | Recall | QPS | Index Memory | Search Peak RSS | Rerank Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 | 0.9482 | 796.5264 | 4.87 GB | 10.29 GB | 3.29 GB |
| dmq 8-bit | 0.9425 | 603.0951 | 3.39 GB | 7.97 GB | 1.81 GB build |

DMQ 8-bit 当前能明显降低 index memory、index file 和 RSS，
其中 bit-packed term id 修正后，rerank backend 从 3.29 GB 降到 1.81 GB。
但 recall 比 fp32 rerank 低约 0.0057 绝对值，QPS 也因逐项 bit unpack 降低。
后续应优先调 `dmq_bits`、`n_candidate`、query pruning 和解码路径，
在保留内存收益的同时追回 recall 与吞吐。

脚本内部会使用仓库 Makefile 构建工具，不直接调用 cmake：

```bash
VSAG_ENABLE_TOOLS=ON make release
```

脚本不会调用 `eval_performance`。
它调用的是 `sindi_direct_benchmark`，
直接通过 VSAG SINDI API 完成 build、serialize、deserialize 和 KNN search。

默认结果目录形如：

```text
benchs/results/sindi_wholenet_sparse_10m_<timestamp>/
```

默认产物包括：

- `sindi.index`：序列化后的 SINDI 索引。
- `build_metrics.json`：build time、build peak RSS、index memory/detail 等。
- `search_metrics.json`：load time、search peak RSS、QPS、recall、latency、
  index memory/detail 等。
- `build.log` 和 `search.log`：runner 原始输出。
- `summary.md`：本次跑数的 Markdown 总结。

常用环境变量：

- `SEARCH_QUERY_COUNT`：搜索评估 query 数，默认 `10000`，设为 `0` 表示全量 query。
- `TOPK`：KNN top-k，默认 `10`。
- `OUTPUT_DIR`：指定结果目录。
- `INDEX_PATH`：指定索引保存路径。
- `BUILD_PARAMETER`：覆盖默认构建参数。
- `SEARCH_PARAMETER`：覆盖默认搜索参数。
- `RERANK_TYPE`：默认 `fp32`，可设为 `dmq`。
- `DMQ_BITS`：默认 `8`，支持 `1`、`2`、`4`、`8`。
- `SAMPLE_MS`：RSS 采样间隔，默认 `50` ms。

### 内存优先配置

适合大规模语料或内存紧张场景：

- 开启 `use_quantization`。
- 逐步提高 `doc_prune_ratio`。
- 可从 `0.1`、`0.2`、`0.4` 做 sweep。
- 若 recall 下降明显，开启 `use_reorder`。
- 如果 fp32 rerank 内存过高，测试 `rerank_type: "dmq"`。
- 优先从 `dmq_bits: 8` 开始，再按 recall 结果尝试更低 bit。
- 同时提高 `n_candidate`。
- 构建后用 `GetMemoryUsage()` 验证真实内存。
- 用 `AnalyzeIndexBySearch` 验证 recall 和剪枝影响。

### 大词表或稀疏 Term ID

如果原始 term id 范围大或 gap 多，
优先测试 `remap_term_ids: true`。

此时把 `term_id_limit` 设置为预估唯一 term 数量上限。
不要把它设成原始 term id 最大值。

### 查询延迟优先

延迟优先时可以做以下 sweep：

- 降低 `n_candidate`，观察 recall knee point。
- 提高 `query_prune_ratio`。
- 优先保留高权重 query term。
- 谨慎提高 `term_prune_ratio`。
- 必须用目标数据集验证 `term_prune_ratio`。
- 保持 `use_term_lists_heap_insert: true`。
- 只有 benchmark 证明更快时，才切到扫描 `dists`。

## 建议的后续工作

1. 统一 SINDI 全称。
2. 优先采用论文中的全称。
3. 校准 `n_candidate=0` 的实现与文档。
4. 如果希望默认 500x topk，应修改代码。
5. 如果当前行为是预期，应修正文档。
6. 修正示例中 `window_size=100000` 的过期注释。
7. 明确 `UpdateVector` 的对外语义。
8. 判断它应只做兼容性校验，还是应真正更新倒排结构。
9. 为 `term_prune_ratio` 补充实际语义文档。
10. 避免用户误以为它按 posting value 选择高权重项。
11. 对增量 Add + quantization 的分布漂移做 benchmark。
12. 或者至少补充文档警示。
13. 确认论文中的 SIMD 优化对应代码入口。
14. 如果尚未落地，应在研发计划中明确差距。

## 参考入口

- `docs/docs/en/src/indexes/sindi.md`
  - 官网英文 SINDI 用户文档源。
- `docs/sindi.md`
  - 顶层历史 SINDI 文档。
- `docs/docs/en/src/resources/research_papers.md`
  - SINDI 论文摘要和 VSAG 集成说明。
- `src/algorithm/sindi/sindi.cpp`
  - 核心构建、查询、序列化、统计实现。
- `src/algorithm/sindi/sindi_parameter.cpp`
  - 构建和搜索参数解析、校验、兼容性判断。
- `src/datacell/sparse_term_datacell.cpp`
  - window 内倒排表、剪枝、量化、posting 扫描和插堆。
- `src/quantization/sparse_quantization/sparse_term_computer.h`
  - query 排序、剪枝和累积分数计算。
- `src/algorithm/sindi/term_id_mapper.*`
  - term id remap 实现。
- `tests/test_sindi.cpp`
  - 功能测试、序列化、并发、analyze、remap 覆盖。
- `examples/cpp/109_index_sindi.cpp`
  - C++ 使用示例。
- `https://arxiv.org/abs/2509.08395`
  - SINDI ICDE 2026 论文页面。
