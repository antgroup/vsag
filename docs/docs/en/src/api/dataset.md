# Dataset

`vsag::Dataset` (declared in `vsag/dataset.h`) is the universal container VSAG uses for **inputs**
(base vectors to build/add, query vectors to search) and **outputs** (search results, retrieved
vectors). You always hold it through `DatasetPtr`:

```cpp
using DatasetPtr = std::shared_ptr<Dataset>;
```

## Builder pattern

`Dataset` uses a fluent builder: `Make()` creates an instance, and every setter returns the same
`DatasetPtr` so calls chain. Setters only store pointers/values — they do **not** copy your buffers.

```cpp
auto base = vsag::Dataset::Make()
                ->Dim(128)
                ->NumElements(10000)
                ->Ids(ids)                 // const int64_t*
                ->Float32Vectors(vectors)  // const float*
                ->Owner(false);            // caller keeps ownership of ids/vectors
```

### Ownership

Ownership controls who frees the underlying buffers:

| Call | Meaning |
|------|---------|
| `Owner(true)` | The dataset owns its buffers and frees them on destruction (using the default allocator). |
| `Owner(true, allocator)` | The dataset owns its buffers and frees them via the supplied [`Allocator`](resource.md#allocator). |
| `Owner(false)` | The caller keeps ownership; the dataset only borrows the pointers. They must outlive the dataset. |

Use `Owner(false)` for build/query inputs you already hold. Search results returned by the index use
`Owner(true)`, so you can read them and let the `DatasetPtr` free everything.

```cpp
DatasetPtr Make();               // static factory

DatasetPtr Owner(bool is_owner, Allocator* allocator);
DatasetPtr Owner(bool is_owner);              // uses the default allocator
DatasetPtr Append(const DatasetPtr& other);   // concatenate another dataset
DatasetPtr DeepCopy(Allocator* allocator = nullptr) const;  // independent copy
```

## Metadata

| Setter | Getter | Type | Meaning |
|--------|--------|------|---------|
| `NumElements(int64_t)` | `GetNumElements()` | `int64_t` | Number of elements (vectors/rows). |
| `Dim(int64_t)` | `GetDim()` | `int64_t` | Dense vector dimensionality. |
| `Ids(const int64_t*)` | `GetIds()` | `const int64_t*` | Per-element ids (length `NumElements`). |
| `Distances(const float*)` | `GetDistances()` | `const float*` | Distances (search output; length depends on `k`/matches). |

## Vector payloads

A dataset carries exactly one vector representation, chosen to match the index's `dtype`:

| Setter | Getter | Element type | Use with |
|--------|--------|--------------|----------|
| `Float32Vectors(const float*)` | `GetFloat32Vectors()` | `float` | `dtype: float32` |
| `Float16Vectors(const uint16_t*)` | `GetFloat16Vectors()` | `uint16_t` | `dtype: float16` **and** `bfloat16` (raw 16-bit payload) |
| `Int8Vectors(const int8_t*)` | `GetInt8Vectors()` | `int8_t` | `dtype: int8` |
| `SparseVectors(const SparseVector*)` | `GetSparseVectors()` | [`SparseVector`](#sparsevector) | `dtype: sparse` (SINDI) |

Dense vectors are laid out row-major: element `i`, dimension `j` lives at `vectors[i * dim + j]`.

### Multi-vector payloads

For documents that hold several dense sub-vectors each:

| Setter | Getter | Type | Meaning |
|--------|--------|------|---------|
| `MultiVectors(const MultiVector*)` | `GetMultiVectors()` | [`MultiVector`](#multivector) | One entry per document. |
| `MultiVectorDim(int64_t)` | `GetMultiVectorDim()` | `int64_t` | Floats per sub-vector (independent of `Dim`). |
| `VectorCounts(const uint32_t*)` | `GetVectorCounts()` | `const uint32_t*` | Sub-vector count per document. |

## Metadata payloads

| Setter | Getter | Type | Meaning |
|--------|--------|------|---------|
| `AttributeSets(const AttributeSet*)` | `GetAttributeSets()` | [`AttributeSet`](types.md#attributeset) | Per-element attributes for hybrid search. |
| `ExtraInfos(const char*)` | `GetExtraInfos()` | `const char*` | Packed extra-info blobs. |
| `ExtraInfoSize(int64_t)` | `GetExtraInfoSize()` | `int64_t` | Bytes per extra-info blob. |
| `Paths(const std::string*)` | `GetPaths()` | `const std::string*` | Hierarchy paths (Pyramid). Default hierarchy. |
| `Paths(const std::string& hierarchy, const std::string*)` | `GetPaths(const std::string& hierarchy)` | `const std::string*` | Paths for a named hierarchy. |
| `SourceID(const std::string*)` | `GetSourceID()` | `const std::string*` | Optional source identifier. |

See [Attribute Filter (Hybrid Search)](../advanced/attribute_filter.md) and
[Extra Info](../advanced/extra_info.md).

## Diagnostics payloads

| Setter | Getter | Type | Meaning |
|--------|--------|------|---------|
| `Statistics(const std::string&)` | `GetStatistics()` / `GetStatistics(keys)` | `std::string` / `std::vector<std::string>` | Serialized statistics; the keyed getter returns values for the requested keys. |
| `Reasoning(const std::string&)` | `GetReasoning()` | `std::string` | Reasoning report (JSON) explaining recall of `expected_labels_`. |

## Reading search results

Search methods return a `DatasetPtr` you read back with the getters:

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

For KNN, `GetNumElements()` is `1` and the ids/distances arrays have length `k`. For range search,
the number of matches is reported through the result's dimension. See
[k-Nearest Neighbor Search](../guide/knn_search.md).

## `SparseVector`

```cpp
struct SparseVector {
    uint32_t len_ = 0;         // number of non-zero entries
    uint32_t* ids_ = nullptr;  // term ids, length len_ (sorted ascending inside the index)
    float* vals_ = nullptr;    // term weights, length len_

    // optional original tokenization (order/duplicates preserved, unlike ids_)
    uint32_t token_seq_len_ = 0;
    uint32_t* token_sequence_ = nullptr;
};
```

Sorting `ids_` ascending before insertion is recommended. `token_sequence_` is optional and only
used by indexes that consume raw token order.

## `MultiVector`

```cpp
struct MultiVector {
    uint32_t len_ = 0;          // number of sub-vectors in this document
    float* vectors_ = nullptr;  // flat array of len_ * MultiVectorDim floats
};
```

When `Owner(true)` is set, each element's `vectors_` must be independently allocated, because the
destructor frees each `vectors_` separately.

## See also

- [Index](index_class.md) — the methods that consume and return datasets.
- [Search Request & Filters](search.md) — wrapping a query dataset in a `SearchRequest`.
- [Auxiliary Types](types.md) — `AttributeSet` and attribute value types.
