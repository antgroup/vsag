# 优化器（Tune）

对于图类索引（HGraph），VSAG 提供 `Tune` 接口，根据给定查询集自动调整运行期参数以在**召回率**
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
| hgraph | 是 |
| ivf / sindi / brute_force | 否 |

## 示例

`examples/cpp/318_feature_tune.cpp` 给出了端到端的调优流程：

1. 构造索引并 `Build`；
2. 使用一份代表性查询集调用 `Tune`；
3. 序列化调优后的索引供生产环境使用。

## 注意事项

- 重建 HGraph 向量编码时，`Tune` 需要覆盖完整内部 ID 范围的原始向量或 FP32 数据源。如果失败的 `Add` 导致数据源不完整，且没有完整的备用数据源，`Tune` 会返回 `false`，保持索引不变。后续成功的 `Add` 不会填补之前未写入的 ID。去重存储要求每个 ID 都有已发布的映射；普通存储会在序列化时保留未完成写入的信息和已预留的 ID，确保恢复后的 Add 不会复用缺失向量的 ID。此行为使用现有计数元数据，不改变二进制格式；旧快照中未记录的未完成写入信息无法恢复。`Merge` 在复制任何输入之前检查所有输入及目标索引；若存在未完成的向量编码写入，则返回 `INVALID_ARGUMENT`，需先使用原始数据重建不完整索引。反序列化时，若预留 ID 数超过序列化容量或内部 ID 类型上限，则返回 `INVALID_BINARY`；旧格式元数据缺少预留数量时，仍以已写入编码数量为准。它不会恢复失败的 `Add` 中的向量；请先使用原始数据重建索引，再进行调优。
- 调优依赖查询集的分布，建议使用真实业务分布下的样本。
- 调优后的参数会随索引元信息一起 `Serialize` / `Deserialize`，部署后仍然生效。
