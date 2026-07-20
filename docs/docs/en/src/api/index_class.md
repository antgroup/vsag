# Index

`vsag::Index` (declared in `vsag/index.h`) is the central abstraction of the library. Every concrete
index type — HGraph, IVF, DiskANN, BruteForce, SINDI, Pyramid, and so on — implements this
interface. You never instantiate `Index` directly; obtain one from
[`Factory::CreateIndex`](factory_engine.md#createindex) or [`Engine::CreateIndex`](factory_engine.md#createindex-1)
and hold it through `IndexPtr` (`std::shared_ptr<Index>`).

```cpp
using IndexPtr = std::shared_ptr<Index>;
```

## How to read this reference

`Index` exposes many optional capabilities. The base class provides a **default implementation** for
almost every method:

- Most methods return `tl::unexpected(Error(ErrorType::UNSUPPORTED_INDEX_OPERATION, ...))` when the
  concrete index does not implement them.
- A handful of statistics accessors instead **throw `std::runtime_error`** (called out explicitly
  below). Wrap those in `try/catch` if you call them on an index that may not support them.

Because "unsupported" is a normal, expected outcome, probe capabilities up front with
[`CheckFeature`](#checkfeature) rather than assuming a method works. Methods marked *(pure virtual)*
must be implemented by every index and are always safe to call.

Pointer/handle types used throughout this page: `DatasetPtr` ([Dataset](dataset.md)), `FilterPtr`
([Filter](search.md#filter)), `BitsetPtr` ([Bitset](search.md#bitset)), `BinarySet` /
`ReaderSet` ([Serialization Types](serialization.md)).

## Enumerations and helper types

### `IndexType`

```cpp
enum class IndexType {
    HNSW, DISKANN, HGRAPH, IVF, PYRAMID, BRUTEFORCE, SPARSE, SINDI, WARP, LAZY_HGRAPH, SIMQ
};
```

Returned by [`GetIndexType`](#getindextype).

### `RemoveMode`

```cpp
enum class RemoveMode {
    MARK_REMOVE = 0,   // mark as deleted; no shrink/repair — fast
    FORCE_REMOVE = 1,  // physically remove and repair the graph — heavy
};
```

Passed to [`Remove`](#remove).

### `MergeUnit` and `IdMapFunction`

```cpp
using IdMapFunction = std::function<std::tuple<bool, int64_t>(int64_t)>;

struct MergeUnit {
    IndexPtr index = nullptr;         // source sub-index to merge from
    IdMapFunction id_map_func = nullptr;  // per-id filter + remap
};
```

For each source id, `id_map_func` returns `{keep, new_id}`: `keep == true` includes the vector under
target id `new_id`. Used by [`Merge`](#merge).

### `Checkpoint`

```cpp
struct Index::Checkpoint {
    BinarySet data;       // intermediate state
    bool finish = false;  // true once the build is complete
};
```

Returned by [`ContinueBuild`](#continuebuild) to drive incremental builds.

### Data-selection flags

Bit flags for [`GetDataByIdsWithFlag`](#getdatabyidswithflag), combined with bitwise OR:

| Macro | Value | Selects |
|-------|-------|---------|
| `DATA_FLAG_FLOAT32_VECTOR` | `0x01` | float32 vectors |
| `DATA_FLAG_INT8_VECTOR` | `0x02` | int8 vectors |
| `DATA_FLAG_SPARSE_VECTOR` | `0x04` | sparse vectors |
| `DATA_FLAG_EXTRA_INFO` | `0x10` | extra info blobs |
| `DATA_FLAG_ATTRIBUTE` | `0x20` | attributes |
| `DATA_FLAG_ID` | `0x40` | ids |

### `WriteFuncType`

```cpp
using OffsetType = uint64_t;
using SizeType = uint64_t;
using WriteFuncType = std::function<void(OffsetType, SizeType, const void*)>;
```

A sink callback for streaming [`Serialize`](#serialize). Each call asks you to persist `SizeType`
bytes (at the given source pointer) at logical `OffsetType` in the output.

## Build & train

| Method | Signature | Notes |
|--------|-----------|-------|
| `Build` | `tl::expected<std::vector<int64_t>, Error> Build(const DatasetPtr& base)` | *(pure virtual)* Builds the index from all vectors. Returns the ids that failed to insert. |
| `Train` | `tl::expected<void, Error> Train(const DatasetPtr& data)` | Trains an index (e.g. IVF centroids, quantizer) without inserting. |
| `Tune` | `tl::expected<bool, Error> Tune(const std::string& parameters, bool disable_future_tuning = false)` | Applies runtime tuning. See [Optimizer (Tune)](../advanced/optimizer.md). |
| `ContinueBuild` | `tl::expected<Checkpoint, Error> ContinueBuild(const DatasetPtr& base, const BinarySet& binary_set)` | Adds dynamism to indexes that cannot insert incrementally; drive it with the returned [`Checkpoint`](#checkpoint). |
| `Add` | `tl::expected<std::vector<int64_t>, Error> Add(const DatasetPtr& base)` | Inserts new vectors into an already-built index. Returns ids that failed to insert. |

See [Build and Train](../advanced/build_and_train.md) and `examples/cpp/311_feature_train.cpp`.

## Update & remove

| Method | Signature | Notes |
|--------|-----------|-------|
| `Remove` | `tl::expected<uint32_t, Error> Remove(const std::vector<int64_t>& ids, RemoveMode mode = RemoveMode::MARK_REMOVE)` | Removes many ids; returns the count removed. |
| `Remove` | `tl::expected<uint32_t, Error> Remove(int64_t id, RemoveMode mode = RemoveMode::MARK_REMOVE)` | Single-id convenience overload. |
| `UpdateId` | `tl::expected<bool, Error> UpdateId(int64_t old_id, int64_t new_id)` | Relabels a base point. |
| `UpdateVector` | `tl::expected<bool, Error> UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update = false)` | Replaces the vector for `id`. `force_update = false` performs a connectivity check. |
| `UpdateExtraInfo` | `tl::expected<bool, Error> UpdateExtraInfo(const DatasetPtr& new_base)` | Updates stored extra-info blobs. |
| `UpdateAttribute` | `tl::expected<void, Error> UpdateAttribute(int64_t id, const AttributeSet& new_attrs)` | Replaces attributes of `id`. |
| `UpdateAttribute` | `tl::expected<void, Error> UpdateAttribute(int64_t id, const AttributeSet& new_attrs, const AttributeSet& origin_attrs)` | Same, but supplies the previous attributes for a faster in-place update. |

See `examples/cpp/303_feature_remove.cpp`.

## Search

The recommended entry point is [`SearchWithRequest`](#searchwithrequest), which takes a single
[`SearchRequest`](search.md#searchrequest) carrying the query, mode, top-k / radius, and any filters.
The older per-argument `KnnSearch` / `RangeSearch` overloads remain for compatibility.

Every search returns a `DatasetPtr`: for KNN, `num_elements == 1` and `ids` / `distances` have
length `k`; for range search, the result length is the number of matches. See [Dataset](dataset.md)
for how to read results.

### `SearchWithRequest`

```cpp
[[nodiscard]] tl::expected<DatasetPtr, Error>
SearchWithRequest(const SearchRequest& request) const;
```

Unified KNN or range search driven by [`SearchRequest`](search.md#searchrequest). This is the
preferred API for new code; it supports attribute filters, callback filters, bitset filters, a
per-search allocator, and iterator search through one struct.

### `KnnSearch` overloads

```cpp
// (1) bitset pre-filter — pure virtual
tl::expected<DatasetPtr, Error>
KnnSearch(const DatasetPtr& query, int64_t k, const std::string& parameters,
          BitsetPtr invalid = nullptr) const;

// (2) callback pre-filter — pure virtual
tl::expected<DatasetPtr, Error>
KnnSearch(const DatasetPtr& query, int64_t k, const std::string& parameters,
          const std::function<bool(int64_t)>& filter) const;

// (3) Filter object
tl::expected<DatasetPtr, Error>
KnnSearch(const DatasetPtr& query, int64_t k, const std::string& parameters,
          const FilterPtr& filter) const;

// (4) Filter + iterator context
tl::expected<DatasetPtr, Error>
KnnSearch(const DatasetPtr& query, int64_t k, const std::string& parameters,
          const FilterPtr& filter, IteratorContext*& iter_ctx, bool is_last_search) const;

// (5) SearchParam — [[deprecated]], use SearchWithRequest
tl::expected<DatasetPtr, Error>
KnnSearch(const DatasetPtr& query, int64_t k, SearchParam& search_param) const;
```

Notes on the filter argument:

- In overloads (1)/(2) the predicate/bitset marks vectors **filtered out**. For a `bitset`,
  `Test(id) == true` means id is excluded. For the `std::function` predicate, returning `true` means
  the id is excluded.
- Overload (3)/(4) take a [`Filter`](search.md#filter) object, whose `CheckValid(id)` uses the
  opposite convention (`true` means *keep*). See [Filtered Search](../advanced/filtered_search.md)
  for the full semantics, and `examples/cpp/301_feature_filter.cpp`.
- Overload (4) powers [Iterator Search](../advanced/iterator_search.md); pass the same `iter_ctx`
  across calls and set `is_last_search` on the final call.

### `RangeSearch` overloads

```cpp
// (1) plain — pure virtual
tl::expected<DatasetPtr, Error>
RangeSearch(const DatasetPtr& query, float radius, const std::string& parameters,
            int64_t limited_size = -1) const;

// (2) bitset pre-filter — pure virtual
tl::expected<DatasetPtr, Error>
RangeSearch(const DatasetPtr& query, float radius, const std::string& parameters,
            BitsetPtr invalid, int64_t limited_size = -1) const;

// (3) callback pre-filter — pure virtual
tl::expected<DatasetPtr, Error>
RangeSearch(const DatasetPtr& query, float radius, const std::string& parameters,
            const std::function<bool(int64_t)>& filter, int64_t limited_size = -1) const;

// (4) Filter object
tl::expected<DatasetPtr, Error>
RangeSearch(const DatasetPtr& query, float radius, const std::string& parameters,
            const FilterPtr& filter, int64_t limited_size = -1) const;
```

`radius` bounds the distance; `limited_size` caps the result count (`<= 0` means no limit, `0` is an
error). See [Range Search](../advanced/range_search.md) and `examples/cpp/302_feature_range_search.cpp`.

## Distance by id

| Method | Signature | Notes |
|--------|-----------|-------|
| `CalcDistanceById` | `tl::expected<float, Error> CalcDistanceById(const float* vector, int64_t id, bool calculate_precise_distance = true) const` | Distance from a dense query to the stored vector `id`. |
| `CalcDistanceById` | `tl::expected<float, Error> CalcDistanceById(const DatasetPtr& vector, int64_t id, bool calculate_precise_distance = true) const` | Same, accepting a `DatasetPtr` (works for sparse indexes such as SINDI). |
| `CalDistanceById` | `tl::expected<DatasetPtr, Error> CalDistanceById(const float* query, const int64_t* ids, int64_t count, bool calculate_precise_distance = true) const` | Batch variant; `-1` in the result marks an invalid distance. |
| `CalDistanceById` | `tl::expected<DatasetPtr, Error> CalDistanceById(const DatasetPtr& query, const int64_t* ids, int64_t count, bool calculate_precise_distance = true) const` | Batch variant accepting a `DatasetPtr` query. |

`calculate_precise_distance = true` may load full-precision vectors (possibly from disk) instead of
quantized codes. See [Calculate Distance by ID](../advanced/calc_distance_by_id.md) and
`examples/cpp/306_feature_calculate_distance_by_id.cpp`.

## Conjugate-graph enhancement

| Method | Signature | Notes |
|--------|-----------|-------|
| `Pretrain` | `tl::expected<uint32_t, Error> Pretrain(const std::vector<int64_t>& base_tag_ids, uint32_t k, const std::string& parameters)` | Enhances chosen base vectors by searching generated queries. Returns successful insertions. |
| `Feedback` | `tl::expected<uint32_t, Error> Feedback(const DatasetPtr& query, int64_t k, const std::string& parameters, int64_t global_optimum_tag_id = INT64_MAX)` | Feeds a known optimum back into the conjugate graph. |

See [Graph Enhancement](../advanced/enhance_graph.md).

## Data retrieval

| Method | Signature | Notes |
|--------|-----------|-------|
| `GetMinAndMaxId` | `tl::expected<std::pair<int64_t, int64_t>, Error> GetMinAndMaxId() const` | Smallest and largest ids in the index. |
| `GetExtraInfoByIds` | `tl::expected<void, Error> GetExtraInfoByIds(const int64_t* ids, int64_t count, char* extra_infos) const` | Copies extra-info blobs for `ids` into a caller-provided buffer. |
| `GetRawVectorByIds` | `tl::expected<DatasetPtr, Error> GetRawVectorByIds(const int64_t* ids, int64_t count, Allocator* specified_allocator = nullptr) const` | Returns stored vectors. Values are *close to* the originals but not guaranteed bit-identical (quantization/precision). |
| `GetDataByIds` | `tl::expected<DatasetPtr, Error> GetDataByIds(const int64_t* ids, int64_t count) const` | Returns all stored data (vectors, attributes, extra info) for `ids`. |
| `GetDataByIdsWithFlag` | `tl::expected<DatasetPtr, Error> GetDataByIdsWithFlag(const int64_t* ids, int64_t count, uint64_t selected_data_flag) const` | Like `GetDataByIds` but selects fields via [`DATA_FLAG_*`](#data-selection-flags). |
| `GetIndexDetailInfos` | `tl::expected<std::vector<IndexDetailInfo>, Error> GetIndexDetailInfos() const` | Lists the introspectable detail fields. See [`IndexDetailInfo`](types.md#index-detail-info). |
| `GetDetailDataByName` | `tl::expected<DetailDataPtr, Error> GetDetailDataByName(const std::string& name, IndexDetailInfo& info) const` | Fetches one detail-data payload by name. |

See [Index Introspection](../advanced/introspection.md) and
`examples/cpp/317_feature_get_detail_data.cpp`.

## Capabilities, merge, clone, export

| Method | Signature | Notes |
|--------|-----------|-------|
| `CheckFeature` | `bool CheckFeature(IndexFeature feature) const` | Probes whether an optional capability is supported. See [`IndexFeature`](types.md#indexfeature). |
| `Merge` | `tl::expected<void, Error> Merge(const std::vector<MergeUnit>& merge_units)` | Merges same-type sub-indexes with id remapping. See [`MergeUnit`](#mergeunit-and-idmapfunction). |
| `Clone` | `tl::expected<IndexPtr, Error> Clone(const std::shared_ptr<Allocator>& allocator = nullptr) const` | Deep-copies the index. |
| `ExportModel` | `tl::expected<IndexPtr, Error> ExportModel() const` | Returns an empty index carrying only the trained model. |
| `ExportIDs` | `tl::expected<DatasetPtr, Error> ExportIDs() const` | Returns all ids as a dataset. |
| `SetImmutable` | `tl::expected<void, Error> SetImmutable()` | Freezes the index; further add/delete is rejected. |

See `examples/cpp/309_feature_clone.cpp`, `310_feature_export_model.cpp`, and
`315_feature_hgraph_merge.cpp`, plus [Index Lifecycle Management](../advanced/index_lifecycle.md).

## Serialization

| Method | Signature | Notes |
|--------|-----------|-------|
| `Serialize` | `tl::expected<BinarySet, Error> Serialize() const` | *(pure virtual)* Serializes to an in-memory [`BinarySet`](serialization.md#binaryset). |
| `Serialize` | `tl::expected<void, Error> Serialize(WriteFuncType write_func) const` | Streams the serialized index through a [`WriteFuncType`](#writefunctype) sink. |
| `Serialize` | `tl::expected<void, Error> Serialize(std::ostream& out_stream)` | Serializes to an open output stream. |
| `Deserialize` | `tl::expected<void, Error> Deserialize(const BinarySet& binary_set)` | *(pure virtual)* Restores from a `BinarySet`. Fails if the index is not empty. |
| `Deserialize` | `tl::expected<void, Error> Deserialize(const ReaderSet& reader_set)` | *(pure virtual)* Restores from a [`ReaderSet`](serialization.md#readerset) (e.g. on-disk readers). |
| `Deserialize` | `tl::expected<void, Error> Deserialize(std::istream& in_stream)` | Restores from an open input stream. |

Deserializing onto a non-empty index yields `INDEX_NOT_EMPTY`. See [Serialization](../advanced/serialization.md)
and `examples/cpp/401_persistent_kv.cpp` / `402_persistent_streaming.cpp`.

## Cache (build acceleration)

| Method | Signature | Notes |
|--------|-----------|-------|
| `ExportCache` | `tl::expected<void, Error> ExportCache(std::ostream& out_stream) const` | Writes a build-time cache (e.g. graph neighbors) that can accelerate a later `Build`. |
| `ImportCache` | `tl::expected<void, Error> ImportCache(std::istream& in_stream)` | Loads a previously exported cache; the next `Build` reuses it. |

## Statistics & introspection

Unless noted, these return values directly. **The methods marked "throws" raise
`std::runtime_error`** (not `tl::expected`) when the index does not support them.

| Method | Signature | Notes |
|--------|-----------|-------|
| `GetIndexType` | `IndexType GetIndexType() const` | **Throws** if unsupported. |
| `GetNumElements` | `int64_t GetNumElements() const` | *(pure virtual)* Live element count. |
| `GetNumberRemoved` | `int64_t GetNumberRemoved() const` | **Throws** if unsupported. Count of removed elements. |
| `GetMemoryUsage` | `int64_t GetMemoryUsage() const` | *(pure virtual)* Bytes occupied by the index. |
| `GetMemoryUsageDetail` | `std::string GetMemoryUsageDetail() const` | **Throws** if unsupported. Per-component memory as JSON. |
| `EstimateMemory` | `uint64_t EstimateMemory(uint64_t num_elements) const` | **Throws** if unsupported. Estimated bytes for `num_elements`. |
| `GetEstimateBuildMemory` | `int64_t GetEstimateBuildMemory(int64_t num_elements) const` | **Throws** if unsupported. Estimated peak build memory. |
| `GetStats` | `std::string GetStats() const` | **Throws** if unsupported. Runtime statistics as JSON. |
| `AnalyzeIndexBySearch` | `std::string AnalyzeIndexBySearch(const SearchRequest& request)` | **Throws** if unsupported. Analysis JSON for a probe search. |
| `CheckIdExist` | `bool CheckIdExist(int64_t id) const` | **Throws** if unsupported. Whether `id` is present. |

See `examples/cpp/308_feature_estimate_memory.cpp`, `319_feature_get_memory_usage.cpp`, and the
[Index Analysis Tool](../resources/analyze_index.md).

## See also

- [Dataset](dataset.md) — construct query/base inputs and read search results.
- [Search Request & Filters](search.md) — the `SearchRequest` fields and filter types.
- [Serialization Types](serialization.md) — `BinarySet`, `Binary`, `Reader`, `ReaderSet`.
- [Auxiliary Types](types.md) — `IndexFeature`, `IndexDetailInfo`, `AttributeSet`.
