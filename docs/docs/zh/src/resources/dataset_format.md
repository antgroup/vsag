# HDF5 数据集格式

VSAG 的评测与基准工具（尤其是 [`eval_performance`](eval.md)）使用与
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks) 一致的 HDF5
数据集格式。本页说明 VSAG 期望的具体布局，便于你准备自定义数据集或排查评测
失败的问题。

下文描述的是 **dense（稠密）** 布局（对应全局属性 `type="dense"`，或省略该属性）。
对于 **sparse（稀疏）** 数据集（`type="sparse"`），`/train` 与 `/test` 是 1 维
`INT8` 字节流，由 VSAG 的稀疏向量序列化接口生成（由
`tools/eval/eval_dataset.cpp` 中的 `parse_sparse_vectors` 解析）；其余数据集与
全局属性的约束仍然适用。

## 必选数据集

### `/train`（底库向量）

- **类型**：`INT8` 或 `FLOAT32`
- **形状**：`(N, D)`
    - `N` —— 底库向量数量（`number_of_base`）
    - `D` —— 向量维度（`dim`）
- **说明**：元素类型由 HDF5 自动推断：
    - `H5T_INTEGER`（1 字节）→ `INT8`
    - `H5T_FLOAT`（4 字节）→ `FLOAT32`

### `/test`（查询向量）

- **类型**：必须与 `/train` 一致
- **形状**：`(Q, D)`
    - `Q` —— 查询向量数量（`number_of_query`）
    - `D` —— 必须等于 `/train` 的 `D`

### `/neighbors`（真实近邻索引）

- **类型**：`INT64`
- **形状**：`(Q, K)`
    - `K` —— 每个查询的真实近邻个数
- **内容**：预先计算好的 Top-`K` 索引，指向 `/train` 中的向量。

### `/distances`（真实近邻距离）

- **类型**：`FLOAT32`
- **形状**：`(Q, K)`，与 `/neighbors` 相同
- **要求**：与 `/neighbors` 中对应位置的近邻一一对齐。

## 全局属性

### `type`（向量类型）

- **类型**：ASCII 字符串
- **必填**：否（缺失时默认为 `"dense"`）
- **可选值**：
    - `"dense"` —— 稠密向量，按标准矩阵布局存放在 `/train` 与 `/test`
    - `"sparse"` —— 稀疏向量，使用 VSAG 稀疏向量辅助接口的序列化格式

### `distance`（距离度量）

评测工具会将 `distance` 视为**距离**（数值越小越好），并与 `/distances` 中的真值进行
对比。请按下方公式准备真值距离。

- **类型**：ASCII 字符串
- **必填**：是
- **稠密向量可选值**：
    - `"euclidean"` —— L2 距离，以 `sqrt(L2Sqr)` 计算
    - `"ip"` —— 内积距离（`1 - 内积`），自动识别数据类型
    - `"angular"` —— 余弦距离（`1 - 余弦相似度`）
- **稀疏向量可选值**：
    - `"ip"` —— 稀疏内积距离（`1 - 稀疏内积`），稀疏向量暂不支持其他度量

## 可选数据集

### `/train_labels` 与 `/test_labels`

- **类型**：`INT64`
- **形状**：
    - `/train_labels`：`(N,)`
    - `/test_labels`：`(Q,)`
- **要求**：若使用标签，两个数据集必须同时存在。

### `/valid_ratios`

- **类型**：`FLOAT32`
- **形状**：`(L,)`
- **用途**：保存每个类别的验证比例。评测工具会以**原始 label 值**作为下标
  （`valid_ratio_[label]`，见 `tools/eval/eval_dataset.h:71`），因此 label 必须为
  非负整数，且 `L` 必须严格大于最大 label 值（通常为 `L > max(label)`，下标范围
  `0..L-1`）。数据集作者需自行保证该数组足够大，能覆盖 `/train_labels` 与
  `/test_labels` 中出现的所有 label。

## 结构性要求

1. **维度一致性**
    - `train_shape[1] == test_shape[1]`（`D` 相同）
    - `neighbors.shape == distances.shape`
2. **类型映射**

   | HDF5 规格              | 内部类型      | 大小    | 出现于                                              |
   |------------------------|---------------|---------|-----------------------------------------------------|
   | `H5T_INTEGER`（size=1）| `INT8`        | 1 字节  | `/train`、`/test`                                   |
   | `H5T_FLOAT`（size=4）  | `FLOAT32`     | 4 字节  | `/train`、`/test`、`/distances`、`/valid_ratios`    |
   | `H5T_INTEGER`（size=8）| `INT64`       | 8 字节  | `/neighbors`、`/train_labels`、`/test_labels`       |

3. **内存布局**
    - 所有矩阵按行优先（row-major）存储。
    - 向量元素连续存放：
        - `/train` 总大小 = `N × D × element_size`（每元素 1 或 4 字节）。

## 参考

- 与该格式兼容的公开基准数据集可在
  [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
  获取（如 `sift-128-euclidean.hdf5`、`gist-960-euclidean.hdf5`）。
- 关于该格式如何被消费，参见 [性能评估工具](eval.md)。
