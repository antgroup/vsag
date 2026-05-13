# HGraph Build Cache 文件格式说明

本文说明 HGraph `ExportBuildCache` 导出的二进制缓存文件格式，并结合实际样本文件做定点验证。

分析样本文件：

- `/home/gubaoyuan.gby/vector-data/indexes/whole-30m-768-euclidean-day1/hgraph_fp32_d64_ef400_with_featureids.build_cache`

对应写出逻辑位于：

- `src/algorithm/hgraph.cpp`

适用范围：

- HGraph Build Cache
- 当前格式版本 `version = 1`
- 当前 FeatureID 编码模式 `feature_id_mode = 1`

## 1. 总体布局

整个文件按小端序写出，布局固定为四段：

1. `BuildCacheHeader`，固定 128 字节
2. `FeatureIdIndexEntry[node_count]`，每项 16 字节
3. `feature_blob`，变长字节区，保存所有 FeatureID 的原始文本拼接结果
4. `neighbor_records[node_count]`，每个节点 1 条邻接记录

可表示为：

```text
| 128B Header |
| node_count * 16B FeatureIdIndexEntry |
| feature_id_bytes B FeatureID Blob |
| node_count * (2 + max_degree * 4) B Neighbor Records |
```

文件总大小公式为：

```text
total_size = 128
           + node_count * 16
           + feature_id_bytes
           + node_count * (2 + max_degree * 4)
```

其中：

- `2` 来自每条邻接记录开头的 `uint16 degree`
- `max_degree * 4` 来自固定槽位数的 `uint32 neighbor_id[max_degree]`

## 2. 头部结构

源码中的头部结构定义如下：

```cpp
struct BuildCacheHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t feature_id_mode;
    uint64_t node_count;
    uint32_t max_degree;
    uint32_t reserved0;
    uint64_t feature_id_count;
    uint64_t feature_id_bytes;
    uint64_t build_param_hash;
    uint64_t create_time;
    uint8_t reserved[64];
};
```

字段说明如下：

| 偏移 | 大小 | 类型 | 字段 | 含义 |
| --- | --- | --- | --- | --- |
| 0x00 | 8 | `uint64_t` | `magic` | 文件魔数，当前常量为 `0x5653414743484500` |
| 0x08 | 4 | `uint32_t` | `version` | 格式版本，当前为 `1` |
| 0x0C | 4 | `uint32_t` | `feature_id_mode` | FeatureID 编码模式，当前为 `1`，表示变长文本 |
| 0x10 | 8 | `uint64_t` | `node_count` | 节点总数 |
| 0x18 | 4 | `uint32_t` | `max_degree` | 每个节点导出的固定邻居槽位数 |
| 0x1C | 4 | `uint32_t` | `reserved0` | 预留字段，当前写出为 `0` |
| 0x20 | 8 | `uint64_t` | `feature_id_count` | FeatureID 条数，当前要求等于 `node_count` |
| 0x28 | 8 | `uint64_t` | `feature_id_bytes` | FeatureID 字节区总大小 |
| 0x30 | 8 | `uint64_t` | `build_param_hash` | 构建参数摘要，用于导入时校验参数一致性 |
| 0x38 | 8 | `uint64_t` | `create_time` | 导出时间，Unix 秒级时间戳 |
| 0x40 | 64 | `uint8_t[64]` | `reserved` | 预留空间 |

说明：

- 文件使用小端序，所以 `magic = 0x5653414743484500` 在文件里的首 8 字节表现为 `00 45 48 43 47 41 53 56`
- 当前实现没有额外校验和，也没有压缩头部

## 3. FeatureID 索引表

头部之后紧跟 `node_count` 条 `FeatureIdIndexEntry`：

```cpp
struct FeatureIdIndexEntry {
    uint64_t offset;
    uint32_t length;
    uint32_t reserved;
};
```

字段说明：

- `offset`：该节点 FeatureID 在 `feature_blob` 中的起始偏移
- `length`：该节点 FeatureID 的字节长度
- `reserved`：预留字段，当前写出为 `0`

这一段的用途是把“节点顺序”映射到“FeatureID 文本切片”。

第 `i` 个节点的 FeatureID 读取方式为：

```text
feature_id_i = feature_blob[offset : offset + length]
```

注意：

- FeatureID 不带终止符 `\0`
- FeatureID 之间也没有分隔符
- 能否正确切分，完全依赖索引表中的 `offset + length`

## 4. FeatureID 字节区

`feature_blob` 是一个连续字节区，内容就是所有 FeatureID 的原始文本直接拼接。

例如索引表前 3 项若为：

1. `(offset=0, length=37)`
2. `(offset=37, length=37)`
3. `(offset=74, length=37)`

则表示：

- 第 0 个节点的 FeatureID 占前 37 字节
- 第 1 个节点的 FeatureID 紧接着占后 37 字节
- 第 2 个节点继续从第 74 字节开始

这也说明当前格式支持变长文本 FeatureID，不要求固定 32 字节或固定 64 字节。

## 5. 邻接记录区

FeatureID 字节区之后，是按节点顺序写出的邻接记录区。每个节点一条记录，结构如下：

```cpp
struct NeighborRecordHeader {
    uint16_t degree;
};

uint32_t neighbors[max_degree];
```

单条记录总长度固定为：

```text
neighbor_record_bytes = 2 + max_degree * 4
```

字段含义：

- `degree`：该节点实际邻居数
- `neighbors`：固定长度槽位数组，长度总是 `max_degree`

写出规则：

1. 先写 `degree`
2. 再循环写 `max_degree` 个 `uint32`
3. 若真实邻居数不足 `max_degree`，剩余槽位写 `UINT32_MAX`

因此：

- 每条邻接记录长度固定，便于随机定位
- 实际有效邻居数量只由 `degree` 决定
- 读取时只消费前 `degree` 个槽位作为有效邻居，其余槽位只是填充

第 `i` 条邻接记录的偏移为：

```text
neighbor_offset + i * (2 + max_degree * 4)
```

## 6. 样本文件实测结果

针对样本文件：

- `/home/gubaoyuan.gby/vector-data/indexes/whole-30m-768-euclidean-day1/hgraph_fp32_d64_ef400_with_featureids.build_cache`

通过二进制抽样和大小校验，得到以下实测值：

| 字段 | 实测值 |
| --- | --- |
| 文件大小 | `9,850,887,626` bytes |
| `version` | `1` |
| `feature_id_mode` | `1` |
| `node_count` | `31,669,206` |
| `max_degree` | `64` |
| `feature_id_count` | `31,669,206` |
| `feature_id_bytes` | `1,173,525,054` |
| `build_param_hash` | `0xcba8b4bef3cb5ade` |
| `create_time` | `1778294121` |

由此可推导：

```text
index_table_offset = 128
index_table_bytes  = 31,669,206 * 16 = 506,707,296
feature_blob_offset = 128 + 506,707,296 = 506,707,424
neighbor_offset = 506,707,424 + 1,173,525,054 = 1,680,232,478
neighbor_record_bytes = 2 + 64 * 4 = 258
expected_total_size = 128
                    + 506,707,296
                    + 1,173,525,054
                    + 31,669,206 * 258
                    = 9,850,887,626
```

计算值与实际文件大小完全一致，说明布局与源码定义严格匹配。

### 6.1 头部抽样

文件前 64 字节的关键部分可概括为：

```text
00 45 48 43 47 41 53 56   magic (little-endian u64)
01 00 00 00               version = 1
01 00 00 00               feature_id_mode = 1
d6 3b e3 01 00 00 00 00   node_count = 31669206
40 00 00 00               max_degree = 64
00 00 00 00               reserved0 = 0
d6 3b e3 01 00 00 00 00   feature_id_count = 31669206
3e 92 f2 45 00 00 00 00   feature_id_bytes = 1173525054
de 5a cb f3 be b4 a8 cb   build_param_hash
69 9d fe 69 00 00 00 00   create_time = 1778294121
```

### 6.2 FeatureID 索引表抽样

样本文件前 3 条索引项可解读为：

1. `(offset=0, length=37)`
2. `(offset=37, length=37)`
3. `(offset=74, length=37)`

这说明至少在该样本中，前几条 FeatureID 的文本长度都是 37 字节。

### 6.3 FeatureID 字节区抽样

从 `feature_blob_offset = 506,707,424` 处抽样，可见连续可打印文本，例如：

```text
438b28dd3e02e8325f752dcf61e1d2d6_NEWS...
```

这证明：

- FeatureID 不是二进制定长字段
- 也不是单独一条条带分隔符的字符串记录
- 而是原始文本直接拼接，再通过索引表切片恢复

### 6.4 邻接区抽样

从 `neighbor_offset = 1,680,232,478` 处抽样：

- 前 2 字节为 `33 00`，解读为 `degree = 51`
- 后续为连续的 `uint32` 邻居 ID 槽位

这与源码中的导出逻辑一致，即每条邻接记录为：

```text
uint16 degree + uint32 neighbors[64]
```

## 7. 读取建议

如果需要自己编写解析器，建议按以下顺序处理：

1. 读取 128 字节头部
2. 校验 `magic/version/feature_id_mode`
3. 计算并读取 `node_count` 条 FeatureID 索引项
4. 读取 `feature_id_bytes` 大小的字节区
5. 通过 `(offset, length)` 恢复每个旧节点的 FeatureID
6. 按固定记录长度遍历邻接区
7. 每条记录只取前 `degree` 个邻居作为有效值

## 8. 设计特点与限制

这个格式的优点：

- 结构简单，顺序写出，导出实现直接
- FeatureID 支持变长文本
- 邻接记录定长，方便随机定位
- 可通过 `build_param_hash` 快速拒绝参数不兼容的缓存文件

当前限制：

- 只支持当前定义的 `feature_id_mode = 1`
- 没有额外校验和，损坏检测主要依赖边界和字段一致性检查
- 邻接区按 `max_degree` 固定槽位写出，空间占用较大
- 该格式保存的是“旧图节点顺序 + FeatureID + 邻接信息”，不直接保存向量数据本体

## 9. 与 BuildWithCache 的对应关系

`BuildWithCache` 导入时，主要依赖以下映射关系：

1. 从缓存文件读出 `old inner_id -> FeatureID`
2. 从新数据集构建 `FeatureID -> new inner_id`
3. 把旧邻接表中的 `old neighbor_id` 映射成新的 `new neighbor_id`
4. 命中节点直接复用旧邻接关系，未命中节点走补建与 refine 流程

因此，这个文件格式的核心价值不是存储完整索引，而是提供 warm start 所需的“稳定键 + 邻接骨架”。