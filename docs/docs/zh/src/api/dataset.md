# Dataset

`vsag::Dataset`（声明于 `vsag/dataset.h`）是 VSAG 用于**输入**（要构建/添加的基础向量、要搜索的查询
向量）与**输出**（搜索结果、取回的向量）的通用容器。你始终通过 `DatasetPtr` 持有它：

```cpp
using DatasetPtr = std::shared_ptr<Dataset>;
```

## Builder 模式

`Dataset` 采用流式 builder：`Make()` 创建实例，每个 setter 都返回同一个 `DatasetPtr`，因此调用可以链式
书写。setter 只存储指针/值 —— 它们**不会**拷贝你的缓冲区。

```cpp
auto base = vsag::Dataset::Make()
                ->Dim(128)
                ->NumElements(10000)
                ->Ids(ids)                 // const int64_t*
                ->Float32Vectors(vectors)  // const float*
                ->Owner(false);            // 由调用方保留 ids/vectors 的所有权
```

### 所有权

所有权决定由谁释放底层缓冲区：

| 调用 | 含义 |
|------|------|
| `Owner(true)` | dataset 拥有其缓冲区，并在析构时释放（使用默认 allocator）。 |
| `Owner(true, allocator)` | dataset 拥有其缓冲区，并通过所提供的 [`Allocator`](resource.md#allocator) 释放。 |
| `Owner(false)` | 由调用方保留所有权；dataset 只借用这些指针。它们必须比 dataset 活得更久。 |

对于你已经持有的构建/查询输入，使用 `Owner(false)`。索引返回的搜索结果使用 `Owner(true)`，因此你读取
之后可以让 `DatasetPtr` 释放全部内容。

```cpp
DatasetPtr Make();               // 静态工厂

DatasetPtr Owner(bool is_owner, Allocator* allocator);
DatasetPtr Owner(bool is_owner);              // 使用默认 allocator
DatasetPtr Append(const DatasetPtr& other);   // 拼接另一个 dataset
DatasetPtr DeepCopy(Allocator* allocator = nullptr) const;  // 独立副本
```

## 元信息

| Setter | Getter | 类型 | 含义 |
|--------|--------|------|------|
| `NumElements(int64_t)` | `GetNumElements()` | `int64_t` | 元素（向量/行）数量。 |
| `Dim(int64_t)` | `GetDim()` | `int64_t` | 稠密向量维度。 |
| `Ids(const int64_t*)` | `GetIds()` | `const int64_t*` | 逐元素 id（长度为 `NumElements`）。 |
| `Distances(const float*)` | `GetDistances()` | `const float*` | 距离（搜索输出；长度取决于 `k`/命中数）。 |

## 向量负载

一个 dataset 只携带一种向量表示，需与索引的 `dtype` 匹配：

| Setter | Getter | 元素类型 | 配合使用 |
|--------|--------|----------|----------|
| `Float32Vectors(const float*)` | `GetFloat32Vectors()` | `float` | `dtype: float32` |
| `Float16Vectors(const uint16_t*)` | `GetFloat16Vectors()` | `uint16_t` | `dtype: float16` **及** `bfloat16`（原始 16 位负载） |
| `Int8Vectors(const int8_t*)` | `GetInt8Vectors()` | `int8_t` | `dtype: int8` |
| `SparseVectors(const SparseVector*)` | `GetSparseVectors()` | [`SparseVector`](#sparsevector) | `dtype: sparse`（SINDI） |

稠密向量按行主序排列：元素 `i` 的维度 `j` 位于 `vectors[i * dim + j]`。

### 多向量负载

用于每篇文档包含多个稠密子向量的场景：

| Setter | Getter | 类型 | 含义 |
|--------|--------|------|------|
| `MultiVectors(const MultiVector*)` | `GetMultiVectors()` | [`MultiVector`](#multivector) | 每篇文档一个条目。 |
| `MultiVectorDim(int64_t)` | `GetMultiVectorDim()` | `int64_t` | 每个子向量的 float 数（独立于 `Dim`）。 |
| `VectorCounts(const uint32_t*)` | `GetVectorCounts()` | `const uint32_t*` | 每篇文档的子向量数量。 |

## 元数据负载

| Setter | Getter | 类型 | 含义 |
|--------|--------|------|------|
| `AttributeSets(const AttributeSet*)` | `GetAttributeSets()` | [`AttributeSet`](types.md#attributeset) | 用于混合搜索的逐元素属性。 |
| `ExtraInfos(const char*)` | `GetExtraInfos()` | `const char*` | 打包的 extra-info 数据块。 |
| `ExtraInfoSize(int64_t)` | `GetExtraInfoSize()` | `int64_t` | 每个 extra-info 数据块的字节数。 |
| `Paths(const std::string*)` | `GetPaths()` | `const std::string*` | 层级路径（Pyramid）。默认层级。 |
| `Paths(const std::string& hierarchy, const std::string*)` | `GetPaths(const std::string& hierarchy)` | `const std::string*` | 命名层级的路径。 |
| `SourceID(const std::string*)` | `GetSourceID()` | `const std::string*` | 可选的来源标识。 |

见 [属性过滤（混合搜索）](../advanced/attribute_filter.md) 与
[Extra Info（附加信息）](../advanced/extra_info.md)。

## 诊断负载

| Setter | Getter | 类型 | 含义 |
|--------|--------|------|------|
| `Statistics(const std::string&)` | `GetStatistics()` / `GetStatistics(keys)` | `std::string` / `std::vector<std::string>` | 序列化的统计信息；带键的 getter 返回所请求键的值。 |
| `Reasoning(const std::string&)` | `GetReasoning()` | `std::string` | 解释 `expected_labels_` 召回情况的推理报告（JSON）。 |

## 读取搜索结果

搜索方法返回一个 `DatasetPtr`，你用 getter 读回：

```cpp
auto result = index->KnnSearch(query, 10, search_params);
if (result.has_value()) {
    auto r = result.value();
    for (int64_t i = 0; i < r->GetDim(); ++i) {
        int64_t id = r->GetIds()[i];
        float dist = r->GetDistances()[i];
    }
}
```

对 KNN，`GetNumElements()` 为 `1`，ids/distances 数组长度为 `k`。对范围搜索，命中数通过结果的维度报告。
见 [k-近邻搜索](../guide/knn_search.md)。

## `SparseVector`

```cpp
struct SparseVector {
    uint32_t len_ = 0;         // 非零项的数量
    uint32_t* ids_ = nullptr;  // term id，长度 len_（索引内部按升序排列）
    float* vals_ = nullptr;    // term 权重，长度 len_

    // 可选的原始分词（保留顺序/重复，与 ids_ 不同）
    uint32_t token_seq_len_ = 0;
    uint32_t* token_sequence_ = nullptr;
};
```

建议在插入前把 `ids_` 按升序排序。`token_sequence_` 是可选的，仅被消费原始 token 顺序的索引使用。

## `MultiVector`

```cpp
struct MultiVector {
    uint32_t len_ = 0;          // 本文档中的子向量数量
    float* vectors_ = nullptr;  // len_ * MultiVectorDim 个 float 的扁平数组
};
```

当设置了 `Owner(true)` 时，每个元素的 `vectors_` 必须各自独立分配，因为析构函数会分别释放每个
`vectors_`。

## 参见

- [Index](index_class.md) —— 消费并返回 dataset 的方法。
- [搜索请求与过滤器](search.md) —— 把查询 dataset 包进 `SearchRequest`。
- [辅助类型](types.md) —— `AttributeSet` 与属性值类型。
