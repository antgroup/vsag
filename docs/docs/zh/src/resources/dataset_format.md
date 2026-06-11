# HDF5 数据集格式

VSAG 的评测与基准工具（尤其是 [`eval_performance`](eval.md)）使用与
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks) 一致的 HDF5
数据集格式。本页说明 VSAG 期望的具体布局，便于你准备自定义数据集或排查评测
失败的问题。

下文描述的是 **dense（稠密）** 布局（对应全局属性 `type="dense"`，或省略该属性）。
对于 **sparse（稀疏）** 数据集（`type="sparse"`），`/train` 与 `/test` 是形状为
`(X,)` 的扁平 `INT8` 字节流，由 VSAG 的稀疏向量序列化接口生成（由
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
- **多向量可选值**：
    - 与稠密向量相同（`"euclidean"`、`"ip"`、`"angular"`）；多向量使用与稠密
      向量相同的逐子向量距离函数

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

## 多向量数据集

当 `type="multi_vector"` 时，文件采用平坦展开布局：将每个文档的子向量拼接为一个
二维矩阵，并辅以 `vector_counts` 数组记录每个文档包含多少子向量。

### 额外全局属性

| 属性               | 类型    | 必填 | 说明                                  |
|--------------------|---------|------|---------------------------------------|
| `multi_vector_dim` | `INT64` | 是   | 子向量维度（每个子向量的 float 个数） |

### 额外数据集

| 数据集                 | 形状                    | 类型      | 说明                         |
|------------------------|-------------------------|-----------|------------------------------|
| `/train_multi_vectors` | `(sum_counts_train, D)` | `FLOAT32` | 所有训练子向量，按行平坦拼接 |
| `/test_multi_vectors`  | `(sum_counts_test, D)`  | `FLOAT32` | 所有查询子向量，按行平坦拼接 |
| `/train_vector_counts` | `(N,)`                  | `UINT32`  | 每个训练文档的子向量数       |
| `/test_vector_counts`  | `(Q,)`                  | `UINT32`  | 每个查询文档的子向量数       |

> `D` 等于 `multi_vector_dim`。`sum_counts_train` 是 `/train_vector_counts` 所有值
> 之和，`sum_counts_test` 是 `/test_vector_counts` 所有值之和。

当 `type="multi_vector"` 时，标准的 `/train` 和 `/test` 数据集**不是必需的**，
文档数量（`N`、`Q`）分别从 `/train_vector_counts` 和 `/test_vector_counts` 推导。
其余数据集（`/neighbors`、`/distances`、可选标签）仍然是必填的。

评测工具会从平坦数组和 counts 重建每个文档的 `vsag::MultiVector`，然后将完整数组
传递给 `vsag::Dataset::MultiVectors()`、`VectorCounts()` 和 `MultiVectorDim()`。

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

## Sparse 布局

当全局属性 `type` 取值为 `"sparse"` 时，`/train` 与 `/test` **不再**遵循 `(N, D)` 的
稠密矩阵布局，而是以扁平的 `INT8`（`H5T_INTEGER`，size 1）数据集存储原始字节流。
通过 h5py 调用 `f["/train"].shape` 得到的形状是 `(X,)`，其中 `X` 为字节流总长度；
此处的 `int8` 仅是传输形式，**字节本身并不是 int8 向量元素**。

### `/train`、`/test`（稀疏字节流）

- **HDF5 类型**：`H5T_INTEGER`，size 1（`INT8`）
- **HDF5 形状**：`(X,)`，其中 `X` 为字节流总长度（所有向量记录大小之和）
- **字节序**：小端（little-endian）
- **内容**：按向量顺序首尾相接的记录序列，每条记录包含以下字段，紧密拼接，
  无填充、无分隔符：

  | 字段        | 类型        | 大小            | 说明                              |
  |-------------|-------------|-----------------|-----------------------------------|
  | `len`       | `uint32`    | 4 字节          | 该向量的非零项个数                |
  | `ids[len]`  | `uint32[]`  | `4 * len` 字节  | 非零项的特征下标（column ids）    |
  | `vals[len]` | `float32[]` | `4 * len` 字节  | 与 `ids` 对应的取值               |

  允许 `len == 0` 的记录，此时仅占 4 字节的长度字段。

- **键的顺序**：eval 工具在读取时会对每条向量按 `ids` 升序排序（`vals` 同步重排）。
  写入侧可以输出无序键，但读取侧不应假设无序。

### `/train_offsets`、`/test_offsets`（随机访问索引，可选）

这两个数据集分别记录 `/train`、`/test` 字节流中每条记录的起始字节偏移，
使得按下标取第 i 条稀疏向量可以做到 **O(1)**，无需顺序扫描整条字节流。

- **HDF5 类型**：`H5T_INTEGER`，size 8（`UINT64`）
- **HDF5 形状**：`/train_offsets` 为 `(N + 1,)`，`/test_offsets` 为 `(Q + 1,)`
- **内容**：`offsets[i]` 是第 i 条记录在对应字节流中的起始字节偏移，
  `offsets[N]` 是哨兵，等于字节流总长度，因此第 i 条记录的长度等于
  `offsets[i + 1] - offsets[i]`，整个数组非递减。

两个数据集都是**可选**的。VSAG 写入端在产出稀疏文件时默认写入它们；
但只包含 `/train`、`/test` 而不带 offsets 的旧版稀疏 HDF5 文件依然可以
正常加载——读取端会顺序扫描一遍字节流并自动重建索引。文件中如果已有
on-disk offsets，会与重建结果做交叉校验，不一致时直接报错以防止误用
损坏的索引。

### `/train_token_sequences`、`/test_token_sequences`（可选）

这两个数据集保存生成每条稀疏向量的**原始分词序列**。它们是完全可选的：
不包含这两个数据集的稀疏 HDF5 文件依然可以正常加载。当存在时，它们必须
与 `/train`、`/test` 一一对应：`/train_token_sequences` 中的第 i 条记录对应
`/train` 中的第 i 条稀疏向量（`/test` 同理）。

- **HDF5 类型**：`H5T_INTEGER`，size 1（`INT8`）
- **HDF5 形状**：`(X,)`，其中 `X` 为字节流总长度（所有记录大小之和）
- **字节序**：小端（little-endian）
- **内容**：按与 `/train` / `/test` 相同的向量顺序依次首尾相接的记录序列，
  每条记录字段如下，紧密拼接，无填充、无分隔符：

  | 字段                | 类型        | 大小                  | 说明                                         |
  |---------------------|-------------|-----------------------|----------------------------------------------|
  | `seq_len`           | `uint32`    | 4 字节                | 原始文档的 token 个数                         |
  | `term_ids[seq_len]` | `uint32[]`  | `4 * seq_len` 字节    | 按原始分词顺序的 term id（保留重复与顺序） |

  允许 `seq_len == 0` 的记录，此时仅占 4 字节的长度字段,读取端应当将其
  视为"该向量没有原始分词信息"。

- **记录数量**：必须与对应分组（train/test）中稀疏向量的数量一致，
  否则读取端会报错。
- **与 `ids` 的顺序差异**：`term_ids` 按原始分词顺序存储（可能包含重复），
  这与 `ids`（被读取端排序去重）**不同**。

### `/train_token_sequences_offsets`、`/test_token_sequences_offsets`（在分词序列存在时必填）

当 `/train_token_sequences`（或 `/test_token_sequences`）存在时，对应的
`UINT64` 偏移索引**必须**也存在。

- **HDF5 类型**：`H5T_INTEGER`，size 8（`UINT64`）
- **HDF5 形状**：`(N + 1,)`（或 `(Q + 1,)`）
- **内容**：与 `/train_offsets` 完全相同的契约，使得按下标访问第 i 条分词
  记录可以 O(1) 完成。

契约：分词字节流与其偏移数据集**必须同时出现，缺一即视为文件损坏**——
即只要 `*_token_sequences` 与 `*_token_sequences_offsets` 中只出现了一个，
读取端就会拒绝该文件。两者同时存在时，读取端会将磁盘上的 offsets 与从
字节流重建的 offsets 做交叉校验，不一致即抛异常拒绝加载。

### 真值与距离

`/neighbors` 与 `/distances` 的形状与类型规则与上文稠密布局相同。`distance` 属性
对稀疏向量仅支持 `"ip"`（稀疏内积距离，`1 - 稀疏内积`）。

### Python 辅助函数

Python 包 `pyvsag` 在 [`pyvsag.sparse`](https://github.com/antgroup/vsag/blob/main/python/pyvsag/sparse.py)
中提供解码工具：

```python
from pyvsag.sparse import load_sparse_hdf5

data = load_sparse_hdf5("sparse.hdf5")
# data["type"]      -> "sparse"
# data["distance"]  -> "ip"
# data["train"]     -> list[dict[int, float]]   每条稀疏向量一个字典，键升序
# data["test"]      -> list[dict[int, float]]
# data["neighbors"] -> numpy.ndarray  shape (Q, K) int64
# data["distances"] -> numpy.ndarray  shape (Q, K) float32
```

如果已经拿到原始字节流，可以直接调用 `pyvsag.sparse.decode_sparse_bytes(buffer)`。

### 参考实现

字节流的编码/解码逻辑位于
[`tools/eval/eval_dataset.cpp`](https://github.com/antgroup/vsag/blob/main/tools/eval/eval_dataset.cpp)
（参见 `parse_sparse_vectors` 与 `serialize_sparse_vectors`）。

## 参考

- 与该格式兼容的公开基准数据集可在
  [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
  获取（如 `sift-128-euclidean.hdf5`、`gist-960-euclidean.hdf5`）。
- 关于该格式如何被消费，参见 [性能评估工具](eval.md)。
