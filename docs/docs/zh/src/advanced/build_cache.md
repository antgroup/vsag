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
