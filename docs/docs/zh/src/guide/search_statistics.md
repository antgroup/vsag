# 查询统计信息

VSAG 可以随搜索结果返回查询阶段的统计信息，用于可观测性、问题排查以及搜索参数对比。

## 读取统计信息

结果 `Dataset` 提供同一份统计数据的两种视图：

- `GetSearchMetrics()` 返回类型化快照，新的 C++ 代码建议优先使用；
- `GetStatistics()` 返回兼容已有用户的 JSON；
- `GetStatistics(keys)` 返回指定字段对应的 JSON 值，字段不存在时返回空字符串。

```cpp
vsag::SearchRequest request;
request.query_ = query;
request.topk_ = 10;
request.params_str_ = R"({})";  // 默认启用统计。

auto result = index->SearchWithRequest(request);
if (!result.has_value()) {
    // 处理 result.error()。
}

auto metrics = result.value()->GetSearchMetrics();
if (metrics.has_value()) {
    std::cout << metrics->distance_evaluations << std::endl;
    std::cout << metrics->DistanceEvaluations(
                     vsag::DistanceEvaluationPhase::APPROXIMATE)
              << std::endl;
}

// 兼容的 JSON 视图仍然可用。
std::cout << result.value()->GetStatistics() << std::endl;
```

存在类型化快照时，它是权威数据源。JSON 由快照按需生成，并缓存在结果 Dataset 中。

## 启用和关闭结果统计

默认启用统计。若要为某次 `SearchWithRequest` 关闭统计，在 `params_str_` 顶层加入框架保留键：

```cpp
request.params_str_ = R"({"want_statistics":false})";
```

`want_statistics` 设置为 `false` 时：

- `GetSearchMetrics()` 返回 `std::nullopt`；
- `GetStatistics()` 返回 `"{}"`；
- 搜索结果 ID 和距离不受影响。

关闭统计不保证降低延迟：解析并移除显式开关、规范化请求存在固定开销，对很小的搜索可能超过节省的采集成本。
条件允许时优先读取类型化数据而不是 JSON，并以实际业务负载测量。

BruteForce 在关闭时会跳过统计采集。其他索引可能仍执行内部记账，但不会在结果中暴露统计信息。

旧的 `KnnSearch` 和 `RangeSearch` overload 不提供请求级 `want_statistics` 关闭开关，因此继续保持已有 JSON
统计行为。已接入类型化统计的索引也可能在这些旧路径上附加类型化快照；需要按请求关闭统计时，应使用
`SearchWithRequest`。

## 类型化统计支持范围

类型化统计按索引逐步接入。当前支持矩阵如下：

| 索引 | `SearchWithRequest` 类型化快照 | `"want_statistics": false` 的结果行为 | 跳过统计采集 |
| --- | --- | --- | --- |
| BruteForce | 支持 | `nullopt` 和 `"{}"` | 支持 |
| 其他索引 | 暂不支持 | `nullopt` 和 `"{}"` | 不保证 |

尚未支持类型化快照的索引，在启用统计时继续暴露原有 JSON 统计。

## 基线字段

基线 schema 由 Statistics 框架拥有，索引扩展不得覆盖这些字段。

| 字段 | C++ 类型 | 含义 |
| --- | --- | --- |
| `is_timeout` | `bool` | 搜索是否超时 |
| `dist_cmp` | `uint32_t` | 历史距离比较次数 |
| `hops` | `uint32_t` | 历史图遍历跳数 |
| `io_cnt` | `uint32_t` | 搜索 IO 操作次数 |
| `io_time_ms` | `uint32_t` | 搜索 IO 耗时，单位毫秒 |
| `reorder_distance_count` | `uint32_t` | reorder 阶段计算的精确距离数 |
| `reorder_candidate_count` | `uint32_t` | reorder 阶段处理的候选数 |
| `reorder_lower_bound_probe_count` | `uint32_t` | reorder 使用的 lower-bound probe 数 |
| `rabitq_filter_count` | `uint32_t` | RaBitQ filter 距离次数 |
| `rabitq_full_count` | `uint32_t` | RaBitQ full distance 次数 |
| `rabitq_filter_fallback_full_count` | `uint32_t` | filter 回退到 full distance 的次数 |
| `rabitq_reorder_hint_full_count` | `uint32_t` | reorder hint 请求 full distance 的次数 |
| `rabitq_reorder_fallback_full_count` | `uint32_t` | reorder 回退到 full distance 的次数 |
| `query_computer_count` | `uint32_t` | 创建的 query computer 数量 |
| `parallel_search_fallback_count` | `uint32_t` | 并行搜索回退为串行搜索的次数 |
| `distance_evaluations` | `uint64_t` | 已记录的距离计算总数 |
| `distance_evaluations_by_phase` | `array<uint64_t, 3>` | routing、approximate 和 rerank 阶段计数 |
| `distance_evaluations_by_backend` | `array<uint64_t, 16>` | 按距离计算 backend 分类的计数 |
| `complete` | `bool` | 框架是否认为当前记账完整 |

统计值为零和没有统计是不同状态。已采集的空搜索有类型化快照，只是计数为零；关闭统计的搜索没有
类型化快照。

## 如何理解计数

- `APPROXIMATE` 表示主要候选搜索阶段，不表示距离精度。BruteForce 的精确 FP32 扫描也记录在此阶段；
  当前示例的 routing 和 rerank 计数为零。
- `distance_evaluations` 统计已记录的距离计算次数，不是去重候选数或返回邻居数。同一候选被再次计算可以
  再次计数。phase 和 backend 数组从不同维度划分同一组计算，不能将两个数组的计数再相加。
- `dist_cmp` 保留历史上各索引自己的比较次数口径，可能与新计数不同，不能跨索引、跨搜索路径视为新总数的别名。
- `complete` 描述诊断性距离记账，不代表召回率、搜索成功或未超时。未知 backend 或距离计数饱和会使其为
  false；true 也不是所有执行路径均已插桩的证明。请结合支持矩阵使用，并独立检查 `is_timeout`。
- collector 的 64 位距离计数在 `UINT64_MAX` 饱和而不是回绕，无法完整记录增量时标记记账不完整。
  历史 32 位字段继续使用原有更新语义，并不普遍保证饱和。JSON 使用无符号 64 位 setter 仅保存已有值，
  不会扩大采集计数的位宽，也不能恢复已溢出的信息。

可运行且包含运行时契约检查的示例：
[`330_feature_search_statistics.cpp`](../../../../../examples/cpp/330_feature_search_statistics.cpp)。

## 索引扩展字段

索引可以增加以下类型的扩展值：

- `uint32_t`
- `uint64_t`
- `bool`
- `double`
- `std::string`

扩展字段是否存在取决于索引和搜索路径。应用通过 `SearchResultMetrics::Extensions()` 返回的 const map
读取，并在读取前检查字段。基线字段名是保留名称；只有内部 collector 可以增加扩展，且会拒绝重名字段。

## 兼容性

- 已有 `GetStatistics()` 和 `GetStatistics(keys)` API 保持可用；
- 默认返回统计信息，保持已有搜索行为；
- 没有采集统计的 Dataset 继续从 `GetStatistics()` 返回 `"{}"`；
- 调用历史 `Statistics(string)` setter 会清除类型化快照，并使传入 JSON 成为权威数据；
- `GetSearchMetrics()` 是 nonvirtual bridge，不会给已有 `Dataset` vtable 增加 slot；
- 请求级控制复用已有 `params_str_` 成员，因此 `SearchRequest` 保持原有对象布局，兼容旧版二进制调用方。

## 维护契约

修改搜索实现时必须保持以下规则：

1. 类型化快照是与查询状态脱离的结果数据，查询期间的原子计数器不得逃逸到结果中；快照在搜索线程完成后生成，
   计数器用于诊断，不为仍有并发写入的场景提供跨字段事务一致性；
2. Dataset 已挂接类型化快照时，JSON 必须从类型化数据派生；
3. 索引扩展不得覆盖基线字段；
4. 空结果和提前返回路径必须遵守框架保留的 `want_statistics` 参数；
5. 关闭统计必须表示为 `std::nullopt` 和 `"{}"`，不能伪装成已采集的全零快照；
6. 新增基线字段、phase、backend 或扩展类型时，必须在同一改动中更新序列化、文档、单元测试和功能测试；
7. 为新索引增加类型化支持时，必须更新支持矩阵，并执行 `tests/test_search_statistics.cpp` 的公共行为契约。

聚焦测试命令：

```bash
./build-release/tests/unittests "[search_metrics]"
./build-release/tests/functests "[search_statistics]"
```
