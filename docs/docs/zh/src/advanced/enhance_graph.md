# 图索引增强

图类索引（HNSW、HGraph）在"困难查询"（与真实近邻连通性较弱）下可能出现召回率下降。
VSAG 通过 **Conjugate Graph**（共轭图）机制对这类查询进行在线/离线修补，在几乎不增加索引体积的
情况下显著改善尾部召回。

## 启用共轭图

构建时开启：

```json
{
    "hnsw": {
        "max_degree": 32,
        "ef_construction": 400,
        "use_conjugate_graph": true
    }
}
```

搜索时通过 `SearchParam` 或直接参数控制是否启用：

```cpp
auto result = index->KnnSearch(query, k, search_param_json, /*use_conjugate_graph=*/true);
```

## 工作原理

共轭图由原图在训练数据上的"失败路径"反向构建而成，在搜索时作为补充的候选边参与贪心扩展。
它相当于对主图的一层轻量索引补丁，典型体积 < 主图 10%。

## 示例

`examples/cpp/304_feature_enhance_graph.cpp` 给出了从构建、训练到对比召回率的完整流程。

## 适用场景

- 数据分布存在稀疏簇或离群点；
- 对 P99 召回敏感的在线场景；
- 希望在不重建索引的前提下小幅提升召回。

## 注意事项

- 启用后构建时间会略有增加。
- 共轭图数据会随索引一并序列化。
- 与 `Tune` 可以叠加使用，分别作用于路由质量与运行期参数。
