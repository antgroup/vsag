# SINDI 正排 Reorder / DMQ Memory

## 背景

当前分支的 SINDI 索引有两级检索路径：

1. 召回阶段：窗口化倒排 `SparseTermDataCell` 计算候选分数。
2. Reorder 阶段：当建索引参数 `use_reorder=true` 时，把候选 inner id 交给
   `SINDIRerankBackend` 用正排数据重新算距离，再输出最终 topk 或 range 结果。

核心文件：

- `src/algorithm/sindi/sindi_parameter.{h,cpp}`
- `src/algorithm/sindi/sindi.{h,cpp}`
- `src/algorithm/sindi/sindi_rerank_backend.{h,cpp}`
- `src/algorithm/sindi/sindi_sparse_dmq.{h,cpp}`
- `src/algorithm/sindi/sindi_test.cpp`

## 参数入口

`SINDIParameter` 负责解析 reorder 相关参数：

- `use_reorder`: 是否启用正排 rerank，默认走 `DEFAULT_USE_REORDER`。
- `rerank_type`: `fp32` 或 `dmq`，默认 `fp32`。
- `dmq_bits`: 当前 DMQ 只允许 8 bit；`rerank_type=dmq` 且 `dmq_bits!=8` 会抛异常。
- `rerank_type=dmq` 要求 `use_reorder=true`，否则参数校验失败。
- `remap_term_ids`: 可与 reorder 共同使用；倒排使用 compact term id，rerank 查询需要保留原始 query。

兼容性检查会比较 `use_reorder`、`rerank_type`、`dmq_bits` 等字段。已序列化索引加载时，如果这些参数与当前创建参数不一致，会被 footer 兼容性校验拦住。

## 构建与正排写入

`SINDI::Add` 同时写两份数据：

- 倒排召回数据：写入 `window_term_list_[cur_window]->InsertVector(...)`。
- 正排 rerank 数据：如果 `use_reorder_` 为 true，写入 `rerank_flat_` 或
  `dmq_rerank_backend_`。

backend 选择在 `SINDI` 构造函数完成：

- `rerank_type=fp32`: 使用 `SparseVectorDataCell` / `FlattenInterface` 保存排序后的
  fp32 正排。
- `rerank_type=dmq`: 使用 `SINDIDmqRerankBackend`，存压缩后的 direct 8-bit DMQ 正排。

DMQ 有一个批量写入优化：`Add` 中会先收集本批成功插入的 `SparseVector` 和 label，循环结束后一次调用 `dmq_rerank_backend_->Add(dmq_base)`。FP32 正排则逐条写入 `rerank_flat_`。

## 查询 Reorder 流程

`KnnSearch` / `RangeSearch` 先构造 `SparseTermComputer` 做倒排召回，再进入
`search_impl`。

`search_impl` 中的关键逻辑：

1. 遍历 window，调用 `term_list->Query(dists.data(), computer)`。
2. 将候选 inner id 写入低精度 heap。
3. 如果 `use_reorder_` 为 true：
   - FP32 调用 `sort_sparse_vector(...)` 后访问 `rerank_flat_`。
   - DMQ 调用 `dmq_rerank_backend_->PrepareQuery(...)` 生成 query context。
   - 遍历低精度 heap 中的所有候选 inner id。
   - 用 `CalDistanceByInnerId` 重新计算高精度/近似高精度距离。
   - KNN 保留 topk；RangeSearch 按 radius 和 limit 保留结果。
4. 如果 `use_reorder_` 为 false，直接返回倒排阶段的低精度结果。

距离语义是 sparse inner product 转距离：`distance = 1.0 - inner_product`。非 reorder 低精度分支最后也会用 `1 + heap.top().first` 把 `-ip` 转成 `1 - ip`。

`remap_term_ids=true` 时，倒排 query 使用 remap 后的 compact term id；但 reorder 正排存的是原始向量，所以 `search_impl` 会通过 `original_query` 把原始 query 传给 rerank。

## FP32 Rerank Backend

FP32 reorder 是精确正排：

- `Add`: 将排序后的 sparse vector 写入 `SparseVectorDataCell`。
- `PrepareQuery`: 对 query sparse vector 按 term id 排序。
- `CalDistanceByInnerId`: 调用 `SparseVectorDataCell::ComputePairVectors` 或
  `calc_distance_by_id_unsafe`。
- `Serialize` / `Deserialize`: 通过 `FlattenInterface` 的 IO 序列化正排。
- `GetSparseVectorByInnerId` / `CalcDistanceById` / `CalDistanceById`: 走
  `rerank_flat_`。

基础测试会用精确 sparse inner product 或关闭 reorder 的 SINDI 作为 baseline，验证开启 reorder 后的 SINDI 结果一致。

## DMQ Rerank Backend

`SINDIDmqRerankBackend` 是当前分支的新增重点。它不保存原始 float 正排，而保存：

- `encoded_vectors_`: 每个向量的 `term_offset`、`len`、`DirectDmqVectorFactors{mean, alpha}`。
- `id_codes_`: bit-packed term id 序列，按每个向量 term id 排序。
- `value_codes_`: direct 8-bit residual code；8 bit 情况下一个 term 一个 byte。
- `codebook_term_ids_` / `codebooks_`: 每个 term 独立的 direct DMQ codebook。
- `codebook_index_by_term_id_` 和 `codebook_index_lookup_`: term id 到 codebook index 的查找结构。

### 训练与编码

`TrainCodebooks(base, train_missing_only)`：

- 先计算每个文档的均值。
- 对每个 term 收集 `value - doc_mean` 的 residual samples。
- 每个 term 用 `BuildDirectCodebook` 构建 256 桶 codebook。
- 增量 Add 时如果已有 codebook，则只训练缺失 term 的 codebook。

`Add(base)`：

- 对每个 sparse vector 按 term id 排序。
- 计算文档 mean。
- 对每个 residual 用对应 term codebook 编码成 8-bit code。
- 计算 alpha，对 codebook decoded residual 做一次线性缩放校正。
- term id 写入 packed `id_codes_`，value code 写入 `value_codes_`。

### 查询解码

`PrepareQuery` 对 query 排序，并在最大 term id 不超过 `kMaxDmqQueryLookupValues`
时建立 dense lookup，提升按 base term 扫描时的查询值查找效率。

`CalDistanceByInnerId`：

- 如果 query 有 dense lookup，则按 base 向量 term id block 扫描。
- 否则走 sorted query ids 与 sorted base ids 的双指针匹配。
- 对命中的 term 只累加 `query_sum` 和 `query_value * codebook.value[code]`。
- 最终近似 inner product 为：

```text
ip ~= encoded.mean * query_sum + encoded.alpha * qualifier_product
distance = 1.0 - ip
```

large term id 场景会自动退回双指针路径，避免构造过大的 dense lookup。测试
`SINDI DMQ Rerank Large Term ID Fallback Test` 覆盖了该行为。

### 序列化

DMQ backend 序列化写入：

- magic: `0x53444D51`
- version: `4`
- `total_bits_`
- `id_bits_`
- `cur_element_count_`
- `total_term_count_`
- `encoded_vectors_`
- `id_codes_`
- `value_codes_`
- `codebook_term_ids_`
- `codebooks_`

反序列化会校验 magic、version、bits、id_bits，并重建 `codebook_index_by_term_id_` 与 lookup。

## 距离与原始向量接口

`SINDI::CalcDistanceById` 和 `SINDI::CalDistanceById` 在 `use_reorder_ && calculate_precise_distance`
时走 rerank backend。

注意：DMQ backend 的 `calculate_precise_distance=true` 返回的是 DMQ 近似距离，不是原始 float 的严格精确距离，因为原始 float 已经没有保存。

`GetSparseVectorByInnerId`：

- `use_reorder=true`: 直接从 rerank backend 取正排。
- FP32 backend 返回原始稀疏向量。
- DMQ backend 返回 decode 后的近似稀疏向量。
- `use_reorder=false`: 从倒排 `SparseTermDataCell` 反查；如果启用 remap，会 reverse map 回原始 term id。

## 内存估算

`SINDI::EstimateMemory` 对 reorder 分支分别估算：

- FP32: label/ids/float values 的正排开销。
- DMQ: packed ids、packed value codes、每向量 metadata、每 term codebook 与 lookup。

`GetMemoryUsageDetail` 会输出：

- 顶层 `rerank_type`
- `rerank_backend`
- `rerank_backend_detail`

DMQ detail 中包括 `backend_type=sparse_dmq_direct8`、`total_bits`、`id_bits`、`num_vectors`、`num_terms`、`num_codebooks`、`id_codes`、`value_codes`、`codebooks` 等字段。

## 已有测试覆盖

重点测试：

- `SINDI Basic Test`: `use_reorder=true` + FP32 backend，对齐精确 sparse baseline，并覆盖序列化。
- `SINDI Direct DMQ Scalar Oracle Test`: 校验 direct DMQ codebook、编码、近似 inner product 的基础数值。
- `SINDI DMQ Rerank Backend Test`: 构建 DMQ reorder，检查 memory detail、序列化、搜索、自距离和 decoded vector。
- `SINDI DMQ Rerank Direct 8-Bit Mode Test`: 校验非 8-bit 参数失败，8-bit 搜索和序列化成功。
- `SINDI DMQ Rerank Large Term ID Fallback Test`: 覆盖 query dense lookup 不适用时的双指针 fallback。
- `SINDI Remap with Reorder Test`: 覆盖 `remap_term_ids=true` 与 reorder 一起使用时，搜索结果与精确 baseline 一致。

## 风险点与 Review Checklist

- `rerank_type=dmq` 与 `use_reorder=false` 必须保持非法，否则 backend 不会创建。
- DMQ 只支持 direct 8-bit；如果未来支持 4/2/1 bit，需要补齐 packed value code 的测试和序列化兼容策略。
- DMQ `CalcDistanceById(..., calculate_precise_distance=true)` 名义上是 precise path，但实际是 DMQ 近似值；对外文档或注释需要避免误导。
- DMQ 增量 Add 对已有 term 不重训 codebook，只为新 term 训练 codebook；这会影响长生命周期索引的量化质量。
- `remap_term_ids=true` 时，倒排和 rerank 使用的 term id 空间不同；查询必须把原始 query 传入 rerank backend。
- DMQ `GetSparseVectorByInnerId` 返回 decode 后的近似值，不是原始值。
- 序列化兼容依赖 footer 参数和 DMQ backend version；改 DMQ layout 必须 bump version 并补兼容/失败测试。
- `id_bits_` 由 `term_id_limit` 或 remap 场景的 `uint32_t::max()` 推导；加载不同 `term_id_limit` 的索引会失败，这是预期行为。
