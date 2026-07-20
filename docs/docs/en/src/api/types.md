# Auxiliary Types

This page gathers the remaining public types: the [attribute](#attributes) system used for hybrid
(attribute-filtered) search, the [`IndexFeature`](#indexfeature) capability flags, the
[index detail info](#index-detail-info) introspection types, the [utility functions](#utility-functions)
in `utils.h`, and the string [constants](#constants) in `constants.h`.

## Attributes

Declared in `vsag/attribute.h`. Attributes are typed, named metadata attached to each vector, enabling
SQL-style filtering during search (see
[Attribute Filter (Hybrid Search)](../advanced/attribute_filter.md)).

### `AttrValueType`

```cpp
enum AttrValueType {
    INT32 = 1, UINT32 = 2, INT64 = 3, UINT64 = 4,
    INT8 = 5, UINT8 = 6, INT16 = 7, UINT16 = 8,
    STRING = 9,
};
```

The element type carried by an attribute.

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

An abstract, named attribute. Each attribute may hold multiple values (`GetValueCount()`), so a single
field can represent a multi-valued tag set.

| Member | Description |
|--------|-------------|
| `name_` | The attribute (field) name. |
| `GetValueType()` | The [`AttrValueType`](#attrvaluetype) of the stored values. |
| `GetValueCount()` | Number of values held. |
| `DeepCopy()` | Allocate an independent copy. |
| `Equal(other)` | Value equality against another attribute. |

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

The concrete, typed implementation of `Attribute`. Instantiate it with the C++ type matching the
desired [`AttrValueType`](#attrvaluetype) (e.g. `AttributeValue<int32_t>`, `AttributeValue<std::string>`),
set `name_`, and push values into `GetValue()`.

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

A bag of attributes describing one element. Attach a per-element array of `AttributeSet` to a
[`Dataset`](dataset.md#metadata-payloads) via `AttributeSets(...)`, or pass one to
[`Index::UpdateAttribute`](index_class.md#update--remove).

## `IndexFeature`

Declared in `vsag/index_features.h`. An enum of optional capabilities you can probe with
[`Index::CheckFeature`](index_class.md#checkfeature) before calling an optional method.

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
    /* ... many more ... */
    INDEX_FEATURE_COUNT   // sentinel; always the last value
};
```

The enum groups capabilities into families:

| Family | Examples |
|--------|----------|
| Lifecycle | `NEED_TRAIN`, `SUPPORT_BUILD`, `SUPPORT_ADD_AFTER_BUILD`, `SUPPORT_ADD_FROM_EMPTY`, `SUPPORT_RESET` |
| Search | `SUPPORT_KNN_SEARCH`, `SUPPORT_RANGE_SEARCH`, `SUPPORT_*_WITH_ID_FILTER`, `SUPPORT_KNN_ITERATOR_FILTER_SEARCH`, `SUPPORT_BATCH_SEARCH` |
| Metric | `SUPPORT_METRIC_TYPE_L2`, `SUPPORT_METRIC_TYPE_INNER_PRODUCT`, `SUPPORT_METRIC_TYPE_COSINE` |
| Serialization | `SUPPORT_SERIALIZE_FILE` / `_BINARY_SET` / `_WRITE_FUNC`, `SUPPORT_DESERIALIZE_FILE` / `_BINARY_SET` / `_READER_SET` |
| Concurrency | `SUPPORT_ADD_CONCURRENT`, `SUPPORT_SEARCH_CONCURRENT`, `SUPPORT_ADD_SEARCH_DELETE_CONCURRENT`, and the `SUPPORT_*_WITH_MULTI_THREAD` build/train variants |
| Introspection & ops | `SUPPORT_ESTIMATE_MEMORY`, `SUPPORT_GET_MEMORY_USAGE`, `SUPPORT_CHECK_ID_EXIST`, `SUPPORT_MERGE_INDEX`, `SUPPORT_CLONE`, `SUPPORT_EXPORT_MODEL`, `SUPPORT_EXPORT_IDS`, `SUPPORT_TUNE`, `SUPPORT_CAL_DISTANCE_BY_ID`, `SUPPORT_GET_*_BY_ID(S)` |

`INDEX_FEATURE_COUNT` marks the end of the enum and is not a real feature. See
`examples/cpp/307_feature_check_features.cpp`.

## Index detail info

Declared in `vsag/index_detail_info.h`. These types describe and carry the structured data returned by
[`Index::GetIndexDetailInfos`](index_class.md#data-retrieval) and
[`Index::GetDetailDataByName`](index_class.md#data-retrieval). See
[Index Introspection](../advanced/introspection.md) and
`examples/cpp/317_feature_get_detail_data.cpp`.

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

Tells you which `DetailData` getter is valid for a given field.

### `IndexDetailInfo`

```cpp
class IndexDetailInfo {
public:
    std::string name;
    std::string description;
    IndexDetailDataType type;
};
```

A descriptor for one introspectable field: its `name`, a human-readable `description`, and the payload
`type`.

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
    // ... const overloads ...
};
using DetailDataPtr = std::shared_ptr<DetailData>;
```

The payload itself. Read it through the getter matching the descriptor's
[`IndexDetailDataType`](#indexdetaildatatype); calling a mismatched getter is not meaningful.

## Utility functions

Declared in `vsag/utils.h`. Free helper functions for clustering and recall evaluation.

### `kmeans_clustering`

```cpp
float kmeans_clustering(uint64_t d, uint64_t n, uint64_t k, const float* x,
                        float* centroids, const std::string& dis_type);
```

Runs k-means over `n` points of dimension `d`, writing `k` centroids into the pre-allocated
`centroids` (size `k * d`). `dis_type` is one of `"l2"`, `"cosine"`, `"ip"`. Returns the final
quantization error.

### `l2_and_filtering`

```cpp
BitsetPtr l2_and_filtering(int64_t dim, int64_t nb, const float* base,
                           const float* query, float threshold);
```

Returns a [`Bitset`](search.md#bitset) in which bit `i` is set (`true`) when base vector `i` falls
*within* `threshold` L2 distance of `query` — the ground truth consumed by `range_search_recall`.
Note the polarity is the **opposite** of a search pre-filter, where a set bit *excludes* an id (see
[`Bitset`](search.md#bitset)); invert it before reusing it as an `invalid` / `bitset_filter_` mask.

### `knn_search_recall` / `range_search_recall`

```cpp
float knn_search_recall(const float* base, const int64_t* id_map, int64_t base_num,
                        const float* query, int64_t data_dim,
                        const int64_t* result_ids, int64_t result_size);

float range_search_recall(const float* base, const int64_t* base_ids, int64_t num_base,
                          const float* query, int64_t dim,
                          const int64_t* result_ids, int64_t result_size, float threshold);
```

Compute the recall of a KNN or range search result against the ground truth derived from the base
vectors. Handy for tests and benchmarks; see [Benchmarks](../resources/performance.md).

## Constants

Declared in `vsag/constants.h`. A large set of `extern const char* const` string constants for the
keys and enumerated string values used throughout the JSON-based configuration. Using the constants
instead of raw string literals avoids typos. They fall into several groups:

| Group | Examples |
|-------|----------|
| Index type names | `INDEX_HGRAPH`, `INDEX_IVF`, `INDEX_DISKANN`, `INDEX_BRUTE_FORCE`, `INDEX_SINDI`, `INDEX_PYRAMID` |
| Dataset field names | `DIM`, `NUM_ELEMENTS`, `IDS`, `DISTS`, `FLOAT32_VECTORS`, `SPARSE_VECTORS` |
| Metric names | `METRIC_L2`, `METRIC_COSINE`, `METRIC_IP` |
| Data type names | `DATATYPE_FLOAT32`, `DATATYPE_FLOAT16`, `DATATYPE_BFLOAT16`, `DATATYPE_INT8`, `DATATYPE_SPARSE` |
| Top-level params | `PARAMETER_DTYPE`, `PARAMETER_DIM`, `PARAMETER_METRIC_TYPE`, `INDEX_PARAM` |
| Per-index params | `HGRAPH_*`, `IVF_*`, `DISKANN_PARAMETER_*`, `PYRAMID_*`, `BRUTE_FORCE_*` |
| Statistics keys | `STATSTIC_MEMORY`, `STATSTIC_KNN_TIME`, `STATSTIC_RANGE_TIME` |

For the meaning of each parameter key, see [Index Parameters](../resources/index_parameters.md) and
the individual [index pages](../indexes/README.md).

## See also

- [Index](index_class.md) — `CheckFeature`, `GetIndexDetailInfos`, `UpdateAttribute`.
- [Dataset](dataset.md) — attaching `AttributeSet` data to elements.
- [Search Request & Filters](search.md) — attribute-filter expressions and `Bitset`.
