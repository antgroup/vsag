# HGraph 构建缓存

HGraph 可以从一次构建中导出图邻居信息，并在下一次 `Build()` 前导入。向量通过稳定的
字符串 Source ID 匹配，因此未变化的数据可以从旧图 warm-start，新数据或已变化数据
仍走正常的 refine 流程。

该流程适用于周期性快照构建，例如每天从大部分重叠的语料重建索引。它是构建加速器，
不是索引序列化格式；可搜索索引仍应使用普通[序列化](serialization.md)接口持久化。

## 使用条件

- 索引类型必须是 HGraph。
- 流程中的每个 base Dataset 都必须设置 `Dataset::SourceID`。
- 希望跨构建匹配的逻辑记录必须使用稳定且唯一的 Source ID。
- 在调用 `Build()` 前，把缓存导入全新、为空且兼容的 HGraph。
- 两次构建的维度、度量和存储/量化参数应保持兼容。

不同快照中的 `Dataset::Ids` 数字 label 可以变化；缓存使用对应的 `SourceID`
字符串匹配。

## 首次构建并导出缓存

构建源索引时，为每个向量提供一个 Source ID：

```cpp
std::vector<std::string> source_ids = load_stable_source_ids();

auto base = vsag::Dataset::Make();
base->NumElements(count)
    ->Dim(dim)
    ->Ids(ids.data())
    ->Float32Vectors(vectors.data())
    ->SourceID(source_ids.data())
    ->Owner(false);

auto index = vsag::Factory::CreateIndex("hgraph", build_params).value();
index->Build(base).value();

std::ofstream cache_out("hgraph.cache", std::ios::binary);
index->ExportCache(cache_out).value();
```

在 `Build()` 执行期间要保持 `std::string` 数组有效。缓存包含后续构建使用的
Source ID 到邻居信息映射，本身不可直接用于搜索。

## Warm-start 下一次构建

创建空的兼容 HGraph，导入旧缓存，再构建新快照：

```cpp
auto next_index = vsag::Factory::CreateIndex("hgraph", build_params).value();

std::ifstream cache_in("hgraph.cache", std::ios::binary);
next_index->ImportCache(cache_in).value();

auto next_base = vsag::Dataset::Make();
next_base->NumElements(next_count)
    ->Dim(dim)
    ->Ids(next_ids.data())
    ->Float32Vectors(next_vectors.data())
    ->SourceID(next_source_ids.data())
    ->Owner(false);

next_index->Build(next_base).value();
```

调用 `ImportCache()` 后，`Build()` 会自动进入缓存辅助路径。两个快照中都存在的
Source ID 使用缓存邻居 warm-start；未匹配记录作为 cache miss 走正常构建 refine。
缓存辅助 `Build()` 未设置 `Dataset::SourceID` 时会返回参数错误。

## 随索引持久化 Source ID

HGraph 默认不会在序列化中写入 Source ID 元数据。如果反序列化后的索引还需要
调用 `ExportCache()`，请在 `index_param` 中设置 `persist_source_id: true`：

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 32,
        "ef_construction": 400,
        "persist_source_id": true
    }
}
```

该选项会给序列化索引增加 Source ID 元数据。如果缓存始终从首次内存构建中导出，
恢复后的索引不需要 Source ID 映射，则无需开启。

## 度量缓存复用

warm-start 构建完成后调用 `GetStats()`，检查：

| 字段 | 含义 |
| --- | --- |
| `build_cache_hit_rate` | 构建节点中成功匹配并从导入缓存 warm-start 的比例 |
| `build_cache_hit_nodes` | 成功匹配的节点数量 |
| `build_cache_missed_nodes` | 没有匹配缓存条目、按正常路径构建的节点数量 |

如果上一次构建没有使用导入缓存，统计会输出 `skipped_reason`。其他 HGraph
指标见[索引分析](../resources/analyze_index.md)。

## 限制与运维建议

- 必须先 Import 再 `Build()`；缓存辅助构建要求索引为空。
- 构建缓存依赖兼容的 HGraph 配置；参数发生不兼容变化后应重新生成。
- `deduplicate_storage: true` 不能与缓存辅助构建组合。
- 生产使用前，应和完整重建对比召回率与构建时间。
- 检查文件流是否成功打开，并处理 API 返回值。缓存文件与可搜索的序列化索引相互独立，
  应和生成它的快照一起版本化，或通过原子方式替换。

API 签名见 [Index 缓存接口](../api/index_class.md#缓存构建加速)，Source ID 字段见
[Dataset](../api/dataset.md)。

### PiPNN reservoir 缓存

HGraph 和 Pyramid 的 `graph_type: "pipnn"` 现在也支持 `ExportCache` / `ImportCache`。
调用顺序与 NSW 相同：构建时提供非空且唯一的 `Dataset::SourceID`，导出缓存，
在新的空索引中导入，再构建下一批快照。Pyramid 按层级名称及节点路径分别匹配缓存，路由层正常重建。

PiPNN 导出带版本的 reservoir 快照，包含稳定 Source ID、原始 FP32 输入的 64 位指纹、
候选 ID 和 BF16 距离键，以及最终邻接表和相应的源点距离键。输入顺序和数值标签可以变化。已删除或向量已变化的候选会被丢弃；
新增、内容变化或剩余候选不足的行重新计算叶内邻居。可复用行跳过叶内 Top-K，
缺失行通过矩形 GEMM 计算距离并更新两端候选。未受影响且仍有效的邻接表直接复用；变化行在固定 alpha 的剪枝证明成立时也可复用，其余行重新剪枝。
旧 reservoir 只在收到新候选时按需装入，当前数据重新分区。
这是近似增量构建，不保证图结构或 Recall 与冷构建相同，需要用业务数据实测 Recall 和 QPS。

维度、距离类型、最大度数、随机投影签名和全部 PiPNN 策略参数必须一致。Source ID 缺失或不唯一、
策略不兼容，以及不含 reservoir 的旧缓存，都会回退为 PiPNN 冷构建。每个图中实际可复用行不足 80% 时，该图也回退为冷构建。
`support_duplicate: true` 的重复组图也走冷构建，本版本不导出其分组 reservoir。
已有 fused 存储缓存接口限制继续适用。PiPNN 自适应剪枝在策略一致时可以复用 reservoir，
由于阈值依赖候选数量，所有自适应行都重新剪枝。指纹用于检测意外内容变化，不提供对抗哈希碰撞的保证，缓存文件应与索引文件处于相同可信边界。

快照在 `Build` 后保留于内存，额外空间与点数乘 `reservoir_size` 成正比。大型快照缓冲区使用索引传入的 allocator。
`GetMemoryUsage()` 计入保留快照的容量，`GetMemoryUsageDetail()` 通过 `pipnn_build_cache`
字段给出该分项，也包括 `ImportCache` 导入的快照。该分项统计快照对象、缓冲区容量及字符串堆容量，
与 allocator 元数据和进程 RSS 的统计口径不同。

导出的是最近一次 PiPNN 批量构建的快照，后续增删改不会重新生成 reservoir。
普通索引序列化不会保留 reservoir，即使设置 `persist_source_id: true` 也应在构建后单独导出缓存。
恢复后的索引如果没有 reservoir，只能导出旧图种子，PiPNN 会冷构建。
仅包含 NSW 信息的缓存保留现有二进制编码；旧读取器不能读取新版 PiPNN payload。
不支持的版本、非法偏移/候选以及截断文件会被拒绝，且不会覆盖此前已导入的有效缓存。

`GetStats()` 的 `build_cache_hit_nodes` 表示实际复用的行数，`build_cache_hit_rate` 为其比例。
Pyramid 的 `build_cache_hit_memberships` 按子图分别计数，同一向量在多个图中复用时，
节点总数只计一次；重新计算的图计入成员未命中。`build_cache_restored_edges` 统计直接恢复的图边，
不包含 reservoir 候选；PiPNN 图行复用单独记录为 `pipnn_build.reused_graph_rows`，其中包含
`pipnn_build.certificate_rows`。Pyramid 的这两个字段按子图计数。
临时诊断对象 `pipnn_build.phase_seconds` 记录准备、缓存匹配、分区、叶内计算及合并、
图写入和缓存生成的耗时，不包含索引训练、路由层和流 I/O。

导入时，最终邻接中的每条边必须存在于对应的候选行，且距离键一致；自环和重复的最终邻接目标也会被拒绝，原缓存不会被无效导入替换。
