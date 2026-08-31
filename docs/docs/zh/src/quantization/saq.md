# SAQ（分段码字调整量化）

`saq` 是面向高维稠密向量的训练型低比特量化器。它组合了全维 PCA 旋转、按方差进行的维度
分段、动态比特分配和码字调整量化（CAQ）。实现依据论文
[SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension
Segmentation](https://arxiv.org/abs/2509.12086)。

> 实现：`src/quantization/saq_quantization/saq_quantizer.cpp`；参数：
> `saq_quantizer_parameter.cpp`。

## 处理流程

训练阶段执行以下步骤：

1. 学习全维 PCA 旋转，并按方差递减排列主成分。L2 数据使用训练均值中心化，与 SAQ 参考实现
   的预处理一致；内积和余弦不会中心化，因为平移会改变这两种度量。投影保留全部维度。
2. 以 64 维为边界候选进行分段。动态规划在记录预算内联合选择边界和每段 1–13 位的位宽。显式
   指定分段数时，边界等距划分，只优化各段比特数。
3. 默认给每段生成独立伪随机正交旋转，以均衡段内坐标。SAQ 使用固定种子 `20260825` 并结合
   分段起点派生矩阵，因此相同输入的重复训练会得到相同的持久化模型。
4. 每段先做对称均匀标量量化，再用 CAQ 坐标调整改进量化向量的方向。

检索时只投影一次查询，并为投影坐标建立逐字节查找表。标量码按位平面存储，由专用的四记录
批量查表内核扫描（可用时走 AVX512，否则回退 AVX2 或通用实现），无需为每个候选重建 FP32
向量。L2 阈值扫描按方差顺序处理各段；当已累计的非负分段距离超过当前阈值时可以提前停止。

## 参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `saq_avg_bits` | float | `4.0` | 每个输入维度的平均 payload 目标，范围 `[1, 8]`。完整记录预留与同码长 multi-bit RaBitQ 相同的 12 字节开销；SAQ 在该记录内按每段 8 字节计入元数据。 |
| `saq_segment_count` | integer | `0` | `0` 表示由动态规划选择段数和边界；正数表示请求对应数量、按 64 维对齐的分段。预算无法容纳时训练失败。 |
| `saq_adjustment_rounds` | integer | `6` | CAQ 坐标调整最大轮数，范围 `[0, 32]`；`0` 表示只使用标量量化初始化。 |
| `saq_use_pca` | bool | `true` | 学习并应用全维 PCA 旋转；建议只在消融实验中关闭。 |
| `saq_random_rotation` | bool | `true` | 对每个分段应用正交旋转以均衡坐标。 |

相同平均位数下，完整记录长度与 multi-bit RaBitQ 相同。每段元数据为 8 字节，包含合并后的
调整尺度和与度量相关的平方范数：L2 保存非对称查询到码距离所需的原始投影范数；cosine 保存
重建范数，使查询到码与码到码路径都严格采用解码向量的余弦语义。因此多分段会在固定记录预算
内减少实际 payload 位数。位平面码记录宽度在训练前已经固定，不会因训练得到的分段方案改变
内存或磁盘布局。

## HGraph 示例

```json
{
    "dim": 768,
    "dtype": "float32",
    "metric_type": "l2",
    "index_param": {
        "base_quantization_type": "saq",
        "saq_avg_bits": 4,
        "saq_segment_count": 0,
        "saq_adjustment_rounds": 6,
        "saq_use_pca": true,
        "saq_random_rotation": true,
        "max_degree": 32,
        "ef_construction": 400,
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
}
```

当 `base_quantization_type` 为 `saq` 时，IVF 和 Pyramid 也接受同样五个 `saq_*` 参数。IVF
在基础码输入上训练 SAQ（启用 `use_residual` 时输入为残差）；HGraph 和 Pyramid 通过共享的
flatten datacell 直接使用该量化器。

## 训练、持久化与限制

- SAQ 至少需要两个训练向量。可直接调用 `Build`，或在 `Add` 前显式调用 `Train`。
- 序列化内容包括 PCA 矩阵、L2 投影均值、分段方案、比特分配和各段旋转；加载时无需重新训练。
  若旋转矩阵元素数与分段尺寸不一致，或包含非有限值，加载会直接拒绝该状态。
- 全维 PCA 模型占用 O(dim²) 空间。量化器会从输入中确定性地均匀采样最多 65,536 条向量来
  训练 PCA 和分段方案；全部输入仍用于后续编码与索引构建。
- L2 消融关闭 PCA 时仍保留均值中心化，因此该对比只隔离 PCA 排序，而不会同时改变坐标原点。
- 段内旋转使用固定实现种子以保证构建可复现；训练所得矩阵会被序列化，加载时不会重新生成。
- 支持全部稠密度量（`l2`、`ip`、`cosine`）；余弦输入会在 PCA 和编码前归一化。任一操作数
  为零范数时，余弦距离定义为有限值 `1.0`，与 VSAG 的约定一致。
- 渐进阈值提前终止目前仅用于 L2，因为其分段距离可以形成安全、单调的部分下界。

## 参数选择建议

建议从 `saq_avg_bits: 4`、自动分段、6 轮调整、PCA 和随机段内旋转开始。在相同码长下与 RaBitQ
对比；业务要求高召回时配合 FP32 重排存储。`saq_segment_count` 更适合可复现的消融实验，自动
分段通常能更好地适应具体数据集的方差分布。

## 相关页面

- [量化总览](README.md)
- [RaBitQ](rabitq.md)
- [HGraph](../indexes/hgraph.md)
- [IVF](../indexes/ivf.md)
- [Pyramid](../indexes/pyramid.md)
