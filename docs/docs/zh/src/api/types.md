# 辅助类型

本页汇总其余的公有类型：用于混合（属性过滤）搜索的[属性](#attributes)系统、
[`IndexFeature`](#indexfeature) 能力标志、[索引细节信息](#索引细节信息)自省类型、`utils.h` 中的
[工具函数](#工具函数)，以及 `constants.h` 中的字符串[常量](#常量)。

## Attributes

声明于 `vsag/attribute.h`。属性是附加到每个向量上的、带类型且具名的元数据，可在搜索时进行 SQL 风格的
过滤（见 [属性过滤（混合搜索）](../advanced/attribute_filter.md)）。

### `AttrValueType`

```cpp
enum AttrValueType {
    INT32 = 1, UINT32 = 2, INT64 = 3, UINT64 = 4,
    INT8 = 5, UINT8 = 6, INT16 = 7, UINT16 = 8,
    STRING = 9,
};
```

属性所携带的元素类型。

### `Attribute`

```cpp
class Attribute {
public:
    std::string name_{};

    virtual AttrValueType GetValueType() const = 0;
    virtual uint64_t GetValueCount() const = 0;
    virtual Attribute* DeepCopy() const = 0;
    virtual bool Equal(const Attribute* other) const = 0;
};
using AttributePtr = std::shared_ptr<Attribute>;
```

一个抽象的、具名的属性。每个属性可持有多个值（`GetValueCount()`），因此单个字段可以表示一个多值的标签
集合。

| 成员 | 说明 |
|------|------|
| `name_` | 属性（字段）名。 |
| `GetValueType()` | 所存值的 [`AttrValueType`](#attrvaluetype)。 |
| `GetValueCount()` | 所持有的值的数量。 |
| `DeepCopy()` | 分配一个独立副本。 |
| `Equal(other)` | 与另一个属性做值相等比较。 |

### `AttributeValue<T>`

```cpp
template <class T>
class AttributeValue : public Attribute {
public:
    AttrValueType GetValueType() const override;
    uint64_t GetValueCount() const override;
    std::vector<T>& GetValue();
    const std::vector<T>& GetValue() const;
    Attribute* DeepCopy() const override;
    bool Equal(const Attribute* other) const override;
};
```

`Attribute` 的具体、带类型实现。用与目标 [`AttrValueType`](#attrvaluetype) 匹配的 C++ 类型实例化它
（如 `AttributeValue<int32_t>`、`AttributeValue<std::string>`），设置 `name_`，并把值 push 进
`GetValue()`。

```cpp
auto tag = std::make_shared<vsag::AttributeValue<int32_t>>();
tag->name_ = "category";
tag->GetValue().push_back(7);
```

### `AttributeSet`

```cpp
struct AttributeSet {
    std::vector<Attribute*> attrs_;
};
```

描述单个元素的一组属性。可通过 `AttributeSets(...)` 把逐元素的 `AttributeSet` 数组附加到
[`Dataset`](dataset.md#元数据负载)，或把一个传给
[`Index::UpdateAttribute`](index_class.md#更新与删除)。

## `IndexFeature`

声明于 `vsag/index_features.h`。一个可选能力的枚举，你可以在调用某个可选方法前用
[`Index::CheckFeature`](index_class.md#checkfeature) 探测它。

```cpp
enum IndexFeature {
    NEED_TRAIN = 1,
    SUPPORT_BUILD,
    SUPPORT_ADD_AFTER_BUILD,
    SUPPORT_KNN_SEARCH,
    SUPPORT_RANGE_SEARCH,
    SUPPORT_DELETE_BY_ID,
    SUPPORT_SERIALIZE_BINARY_SET,
    SUPPORT_CAL_DISTANCE_BY_ID,
    SUPPORT_MERGE_INDEX,
    SUPPORT_CLONE,
    /* ... 还有很多 ... */
    INDEX_FEATURE_COUNT   // 哨兵；始终是最后一个值
};
```

该枚举把能力分为几个族：

| 族 | 示例 |
|----|------|
| 生命周期 | `NEED_TRAIN`、`SUPPORT_BUILD`、`SUPPORT_ADD_AFTER_BUILD`、`SUPPORT_ADD_FROM_EMPTY`、`SUPPORT_RESET` |
| 搜索 | `SUPPORT_KNN_SEARCH`、`SUPPORT_RANGE_SEARCH`、`SUPPORT_*_WITH_ID_FILTER`、`SUPPORT_KNN_ITERATOR_FILTER_SEARCH`、`SUPPORT_BATCH_SEARCH` |
| 度量 | `SUPPORT_METRIC_TYPE_L2`、`SUPPORT_METRIC_TYPE_INNER_PRODUCT`、`SUPPORT_METRIC_TYPE_COSINE` |
| 序列化 | `SUPPORT_SERIALIZE_FILE` / `_BINARY_SET` / `_WRITE_FUNC`、`SUPPORT_DESERIALIZE_FILE` / `_BINARY_SET` / `_READER_SET` |
| 并发 | `SUPPORT_ADD_CONCURRENT`、`SUPPORT_SEARCH_CONCURRENT`、`SUPPORT_ADD_SEARCH_DELETE_CONCURRENT`，以及 `SUPPORT_*_WITH_MULTI_THREAD` 的构建/训练变体 |
| 自省与运维 | `SUPPORT_ESTIMATE_MEMORY`、`SUPPORT_GET_MEMORY_USAGE`、`SUPPORT_CHECK_ID_EXIST`、`SUPPORT_MERGE_INDEX`、`SUPPORT_CLONE`、`SUPPORT_EXPORT_MODEL`、`SUPPORT_EXPORT_IDS`、`SUPPORT_TUNE`、`SUPPORT_CAL_DISTANCE_BY_ID`、`SUPPORT_GET_*_BY_ID(S)` |

`INDEX_FEATURE_COUNT` 标记枚举末尾，并非真正的能力。见
`examples/cpp/307_feature_check_features.cpp`。

## 索引细节信息

声明于 `vsag/index_detail_info.h`。这些类型描述并承载
[`Index::GetIndexDetailInfos`](index_class.md#数据获取) 与
[`Index::GetDetailDataByName`](index_class.md#数据获取) 返回的结构化数据。见
[索引自省](../advanced/introspection.md) 与 `examples/cpp/317_feature_get_detail_data.cpp`。

### `IndexDetailDataType`

```cpp
enum class IndexDetailDataType {
    TYPE_2DArray_INT64,
    TYPE_1DArray_INT64,
    TYPE_SCALAR_INT64,
    TYPE_SCALAR_DOUBLE,
    TYPE_SCALAR_STRING,
    TYPE_SCALAR_BOOL,
};
```

告诉你对某个字段哪个 `DetailData` getter 是有效的。

### `IndexDetailInfo`

```cpp
class IndexDetailInfo {
public:
    std::string name;
    std::string description;
    IndexDetailDataType type;
};
```

单个可自省字段的描述符：其 `name`、人类可读的 `description`，以及负载 `type`。

### `DetailData`

```cpp
class DetailData {
public:
    virtual std::vector<int64_t> GetData1DArrayInt64();
    virtual std::vector<std::vector<int64_t>> GetData2DArrayInt64();
    virtual std::string GetDataScalarString();
    virtual bool GetDataScalarBool();
    virtual int64_t GetDataScalarInt64();
    virtual double GetDataScalarDouble();
    // ... const 重载 ...
};
using DetailDataPtr = std::shared_ptr<DetailData>;
```

负载本身。用与描述符 [`IndexDetailDataType`](#indexdetaildatatype) 匹配的 getter 读取它；调用不匹配的
getter 没有意义。

## 工具函数

声明于 `vsag/utils.h`。用于聚类与召回评估的自由辅助函数。

### `kmeans_clustering`

```cpp
float kmeans_clustering(uint64_t d, uint64_t n, uint64_t k, const float* x,
                        float* centroids, const std::string& dis_type);
```

对 `n` 个维度为 `d` 的点运行 k-means，将 `k` 个聚类中心写入预分配的 `centroids`（大小 `k * d`）。
`dis_type` 是 `"l2"`、`"cosine"`、`"ip"` 之一。返回最终的量化误差。

### `l2_and_filtering`

```cpp
BitsetPtr l2_and_filtering(int64_t dim, int64_t nb, const float* base,
                           const float* query, float threshold);
```

返回一个 [`Bitset`](search.md#bitset)：当基础向量 `i` 落在与 `query` 的 L2 距离 `threshold` *之内*时，
将第 `i` 位置为 `true`。这是 `range_search_recall` 使用的 ground truth。注意其置位极性与搜索预过滤
**相反**：预过滤中置位表示该 id 被*排除*（见 [`Bitset`](search.md#bitset)）；若要作为 `invalid` /
`bitset_filter_` 掩码复用，需先取反。

### `knn_search_recall` / `range_search_recall`

```cpp
float knn_search_recall(const float* base, const int64_t* id_map, int64_t base_num,
                        const float* query, int64_t data_dim,
                        const int64_t* result_ids, int64_t result_size);

float range_search_recall(const float* base, const int64_t* base_ids, int64_t num_base,
                          const float* query, int64_t dim,
                          const int64_t* result_ids, int64_t result_size, float threshold);
```

针对由基础向量导出的 ground truth，计算 KNN 或范围搜索结果的召回率。便于测试与基准评测；见
[标准环境性能参考](../resources/performance.md)。

## 常量

声明于 `vsag/constants.h`。一大批 `extern const char* const` 字符串常量，对应贯穿于基于 JSON 的配置中
所用的键与枚举字符串值。用这些常量而非裸字符串字面量可避免拼写错误。它们分为若干组：

| 组 | 示例 |
|----|------|
| 索引类型名 | `INDEX_HGRAPH`、`INDEX_IVF`、`INDEX_DISKANN`、`INDEX_BRUTE_FORCE`、`INDEX_SINDI`、`INDEX_PYRAMID` |
| Dataset 字段名 | `DIM`、`NUM_ELEMENTS`、`IDS`、`DISTS`、`FLOAT32_VECTORS`、`SPARSE_VECTORS` |
| 度量名 | `METRIC_L2`、`METRIC_COSINE`、`METRIC_IP` |
| 数据类型名 | `DATATYPE_FLOAT32`、`DATATYPE_FLOAT16`、`DATATYPE_BFLOAT16`、`DATATYPE_INT8`、`DATATYPE_SPARSE` |
| 顶层参数 | `PARAMETER_DTYPE`、`PARAMETER_DIM`、`PARAMETER_METRIC_TYPE`、`INDEX_PARAM` |
| 各索引参数 | `HGRAPH_*`、`IVF_*`、`DISKANN_PARAMETER_*`、`PYRAMID_*`、`BRUTE_FORCE_*` |
| 统计键 | `STATSTIC_MEMORY`、`STATSTIC_KNN_TIME`、`STATSTIC_RANGE_TIME` |

各参数键的含义见 [索引参数](../resources/index_parameters.md) 与各 [索引页面](../indexes/README.md)。

## 参见

- [Index](index_class.md) —— `CheckFeature`、`GetIndexDetailInfos`、`UpdateAttribute`。
- [Dataset](dataset.md) —— 把 `AttributeSet` 数据附加到元素上。
- [搜索请求与过滤器](search.md) —— 属性过滤表达式与 `Bitset`。
