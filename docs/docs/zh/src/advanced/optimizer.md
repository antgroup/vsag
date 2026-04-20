# 优化器（Tune）

对于图类索引（HNSW、HGraph），VSAG 提供 `Tune` 接口，根据给定查询集自动调整运行期参数以在**召回率**
与**延迟**之间取得更好的权衡。其底层实现即历史版本中的 ELP Optimizer。

## 基本用法

```cpp
#include <vsag/vsag.h>

auto index = vsag::Factory::CreateIndex("hgraph", build_params).value();
index->Build(base_dataset);

std::string tune_params = R"(
{
    "queries_dataset": "path/or/inline/queries",
    "target_recall": 0.95,
    "top_k": 10
}
)";
auto ret = index->Tune(tune_params);
```

`Tune` 的第二个参数 `disable_future_tuning=false` 默认允许后续多次调用继续调整；设为 `true` 会冻结参数。

## 与 ELP Optimizer 的关系

历史文献（见 [科研论文](../resources/research_papers.md)）中提到的 "ELP Optimizer" 对应实现键
`use_elp_optimizer`，现已收敛到统一的 `Tune` 接口背后，用户无需直接操作。

## 适用索引

| 索引类型 | 支持 Tune |
|---------|----------|
| hnsw | 是 |
| hgraph | 是 |
| diskann | 部分参数 |
| ivf / sindi / brute_force | 否 |

## 示例

`examples/cpp/318_feature_tune.cpp` 给出了端到端的调优流程：

1. 构造索引并 `Build`；
2. 使用一份代表性查询集调用 `Tune`；
3. 序列化调优后的索引供生产环境使用。

## 注意事项

- 调优依赖查询集的分布，建议使用真实业务分布下的样本。
- 调优后的参数会随索引元信息一起 `Serialize` / `Deserialize`，部署后仍然生效。
