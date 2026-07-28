# Issue #2533：MRLE + RaBitQ Split 实施计划

## 1. 目标

在 HGraph 与 Pyramid 中支持以下组合：

```json
{
    "base_quantization_type": "tq",
    "tq_chain": "mrle, rabitq",
    "mrle_dim": 768,
    "precise_quantization_type": "rabitq",
    "use_reorder": true,
    "rabitq_bits_per_dim_base": 3,
    "rabitq_bits_per_dim_precise": 5
}
```

数据路径保持严格分层：

```text
原始向量
  -> TransformQuantizer 执行 MRLE 截断
  -> 末端 RaBitQuantizer 执行自身的 FHT/ROM 随机旋转与 RaBitQ 编码
  -> RaBitQSplitDataCell 将 RaBitQ code 拆成 x-bit filter 与 y-bit supplement
```

MRLE 与 split storage 解耦。MRLE 只改变送入末端量化器的向量；split 编码、lower
bound、full distance、scalar-code optimized build 仍全部由 `RaBitQuantizer` 实现。

## 2. 架构决策

### 2.1 不新增 bridge 或 codec

`TransformQuantizer` 已能组合任意末端量化器，因此不引入
`RaBitQSplitBridge`、`TransformFlattenDataCell` 或另一套 transform chain。

采用 `BottomQuantizerAccessor` 静态策略：

```text
RaBitQuantizer
  -> bottom quantizer/computer 是自身

TransformQuantizer<RaBitQuantizer>
  -> bottom quantizer 是 inner RaBitQuantizer
  -> bottom computer 是 inner Computer<RaBitQuantizer>
  -> optimized build 输入先经过 TransformBaseVector
```

该策略没有热路径虚调用，也不复制 RaBitQ 编码逻辑。

### 2.2 支持范围

split + TQ 首版只接受精确链：

```text
mrle, rabitq
```

不接受外部 PCA、ROM 或 FHT。RaBitQ 内部已有的 FHT/ROM 随机旋转保持不变，顺序为
MRLE 降维后再执行 RaBitQ 内部旋转。

### 2.3 code layout

`TransformQuantizer` 的 terminal quantizer code 位于 full code 起始位置，transform metadata
位于其后。MRLE 当前 metadata size 为 0，因此 split datacell 合并 full code 时将外层 buffer
清零，再让底层 RaBitQ 写入 code 起始区域即可。

## 3. 代码修改

### 3.1 BottomQuantizerAccessor

新增：

- `src/quantization/bottom_quantizer_accessor.h`

职责：

- direct quantizer 返回自身 quantizer/computer；
- `TransformQuantizer` 返回 inner quantizer/computer；
- optimized scalar build 时为 TQ 准备变换后的 base input。

### 3.2 TransformQuantizer

修改：

- `src/quantization/transform_quantization/transform_quantizer.h`
- `src/quantization/transform_quantization/transform_quantizer_parameter.cpp`

内容：

- 新增 `GetTransformedDim()` 与 `TransformBaseVector()`；
- 训练 batch 按 transformed dim 紧凑分配和寻址，修复降维后第二条向量起 stride 错误；
- 无 metadata 输出时允许 transform chain 不编码 meta；
- compatibility 同时比较 transformer 参数与末端 quantizer 完整参数。

### 3.3 RaBitQSplitDataCell

修改：

- `src/datacell/rabitq_split_datacell.h`
- `src/datacell/rabitq_split_datacell_factory*.{h,cpp}`
- `src/datacell/flatten_interface.cpp`

内容：

- datacell 增加 `QuantizerT` 模板参数，默认仍为 direct `RaBitQuantizer`；
- 用 `BottomQuantizerAccessor` 获取底层 RaBitQ 与 query computer；
- split、merge、lower bound、full distance、scalar code 均调用底层 RaBitQ；
- 普通 Train/Encode/FactoryComputer/Serialize 仍调用外层 quantizer；
- factory 静态实例化 direct 与 `TransformQuantizer<RaBitQuantizer>` 两条路径；
- factory 校验 TQ split 只能使用 `mrle,rabitq`。

现有 direct split 类型、序列化布局和行为保持不变。

### 3.4 共享参数映射

修改：

- `src/algorithm/inner_index_parameter.h/.cpp`
- `src/algorithm/hgraph/hgraph_param_mapping.cpp`
- `src/algorithm/pyramid/pyramid.cpp`
- `src/algorithm/pyramid/pyramid_zparameters.cpp`

新增共享内部函数：

- `MapRaBitQSplitParam`：校验 exact chain、x/y bits、terminal RaBitQ，并生成 split 内部参数；
- `ValidateMRLEDim`：统一校验 `mrle_dim` 范围。

HGraph 与 Pyramid 都将 split reorder source 固定为 `base`。

### 3.5 Pyramid base reorder

Pyramid 增加 `reorder_by_base_` 与两个小型 helper：

- `has_precise_codes()`：仅 precise reorder 时返回 true；
- `graph_codes()`：direct split/TQ split 返回 base，传统 reorder 返回 precise。

base reorder 不创建、不训练、不插入、不序列化 precise datacell；`FlattenReorder` 直接包装
base split datacell，从 supplement code 完成最终重排。

## 4. 测试

单元测试覆盖：

- MRLE + split factory 创建；
- 多向量降维训练的紧凑 batch 布局；
- optimized scalar-code build 的 base transform；
- split query 与 merged full-code distance 一致；
- 非 `mrle,rabitq` chain 被拒绝；
- HGraph/Pyramid 映射为 `rabitq_split` 与 `reorder_source=base`；
- TQ compatibility 比较 MRLE 与 RaBitQ 参数；
- direct split 回归。

功能测试覆盖：

- HGraph MRLE + RaBitQ split 构建与 KNN 搜索；
- Pyramid MRLE + RaBitQ split 构建与 KNN 搜索。

## 5. 验收标准

- `clang-format-15` 通过；
- `git diff --check` 通过；
- debug library build 通过；
- `[MRLE]` 单元和功能测试通过；
- direct split、optimized build、TQ compute 与既有参数映射回归通过；
- 英文和中文量化文档同步说明 exact chain 与 RaBitQ 内部旋转顺序。
