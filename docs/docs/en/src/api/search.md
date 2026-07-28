# Search Request & Filters

This page covers the types that describe *how* to search: the unified
[`SearchRequest`](#searchrequest), the filtering primitives [`Filter`](#filter) and
[`Bitset`](#bitset), and the [`IteratorContext`](#iteratorcontext) used for incremental search. The
deprecated [`SearchParam`](#searchparam-deprecated) is documented at the end for migration.

## `SearchRequest`

Declared in `vsag/search_request.h`. `SearchRequest` is a plain struct that bundles every option for
[`Index::SearchWithRequest`](index_class.md#searchwithrequest). Fill in the fields you need and leave
the rest at their defaults.

```cpp
vsag::SearchRequest request;
request.query_ = query;      // DatasetPtr with one query vector
request.mode_ = vsag::SearchMode::KNN_SEARCH;
request.topk_ = 10;
request.params_str_ = R"({"hgraph": {"ef_search": 100}})";

auto result = index->SearchWithRequest(request);
```

### `SearchMode`

```cpp
enum class SearchMode {
    KNN_SEARCH = 1,    // return the top-k nearest vectors
    RANGE_SEARCH = 2,  // return all vectors within radius_
};
```

### Basic fields

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `query_` | `DatasetPtr` | `nullptr` | The query. Exactly one query vector is allowed. |
| `mode_` | `SearchMode` | `KNN_SEARCH` | KNN vs. range search. |
| `topk_` | `int64_t` | `10` | Neighbors to return (KNN mode). Must be positive. |
| `radius_` | `float` | `0.5` | Distance threshold (range mode). Non-negative. |
| `limited_size_` | `int64_t` | `-1` | Cap on range results; `-1` means no limit. |
| `params_str_` | `std::string` | `""` | Algorithm-specific search params as JSON (e.g. `ef_search`). |

### IVF bucket routing

IVF accepts `{"ivf":{"scan_buckets_count":N,"disable_bucket_scan":true}}` through
`params_str_`. This routing-only mode returns the `N` selected bucket IDs per query in the
result `Dataset` instead of vector labels. `NumElements()` equals the number of queries,
`Dim()` equals `scan_buckets_count`, `GetIds()` contains bucket IDs (with `-1` for empty
slots), and `GetDistances()` has distances to bucket centroids. No vector scan is performed,
so filters, `topk`, range limits, reordering, and reasoning options are ignored.

### Filtering fields

Three filtering mechanisms are available and are combined with logical **AND** when more than one is
enabled.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `enable_attribute_filter_` | `bool` | `false` | Enable SQL-style attribute filtering. |
| `attribute_filter_str_` | `std::string` | `""` | The filter expression (see below). Requires `enable_attribute_filter_`. |
| `enable_filter_` | `bool` | `false` | Enable a custom [`Filter`](#filter) callback. |
| `filter_` | `FilterPtr` | `nullptr` | The filter object. Requires `enable_filter_`. |
| `enable_bitset_filter_` | `bool` | `false` | Enable a [`Bitset`](#bitset) filter. |
| `bitset_filter_` | `BitsetPtr` | `nullptr` | The bitset. `Test(id) == true` **excludes** id. Requires `enable_bitset_filter_`. |

The `attribute_filter_str_` grammar is SQL-like. Examples:

```text
category = 'electronics' AND price != 1000
multi_in(category, ['electronics', 'clothing']) AND multi_notin(color, ['red', 'blue'])
```

See [Attribute Filter (Hybrid Search)](../advanced/attribute_filter.md) and
[Filtered Search](../advanced/filtered_search.md).

### Resource & iterator fields

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `search_allocator_` | `Allocator*` | `nullptr` | Per-search allocator; falls back to the index allocator when null. |
| `enable_iterator_search_` | `bool` | `false` | Enable incremental (iterator) search. |
| `p_iter_ctx_` | `IteratorContext**` | `nullptr` | Handle to the iterator state, reused across calls. |
| `is_last_search_` | `bool` | `false` | Marks the final call of an iterator sequence. |
| `expected_labels_` | `std::vector<int64_t>` | `{}` | Ids expected in the result; enables reasoning analysis of missed recalls. |

See [Per-Search Allocator](../advanced/search_allocator.md) and
[Iterator Search](../advanced/iterator_search.md), plus `examples/cpp/313`/`314` for the allocator.

## `Filter`

Declared in `vsag/filter.h`. Implement this abstract class to express arbitrary "keep this id?"
logic. Hold it through `FilterPtr` (`std::shared_ptr<Filter>`).

```cpp
class Filter {
public:
    enum class Distribution { NONE = 0, RELATED_TO_VECTOR };

    virtual bool CheckValid(int64_t id) const = 0;          // true  => KEEP the id
    virtual bool CheckValid(const char* data) const;         // extra-info variant (default true)
    virtual float ValidRatio() const;                        // fraction kept (default 1.0)
    virtual Distribution FilterDistribution() const;         // hint (default NONE)
    virtual void GetValidIds(const int64_t** valid_ids, int64_t& count) const;
};
```

> **Convention:** `Filter::CheckValid(id)` returns `true` to **keep** a vector. This is the opposite
> of the `bitset` / `std::function<bool(int64_t)>` pre-filter overloads on
> [`Index`](index_class.md#knnsearch-overloads), where `true` means *filtered out*. Keep this
> distinction in mind when choosing an overload.

| Member | Purpose |
|--------|---------|
| `CheckValid(int64_t id)` | Core predicate. `true` keeps the id in results. |
| `CheckValid(const char* data)` | Predicate over an element's extra-info bytes. Defaults to `true`. |
| `ValidRatio()` | Estimated fraction of vectors that pass; lets the engine pick a strategy. |
| `FilterDistribution()` | `RELATED_TO_VECTOR` hints validity correlates with vector position. |
| `GetValidIds(...)` | Optionally expose the explicit valid-id set. |

See `examples/cpp/301_feature_filter.cpp`.

## `Bitset`

Declared in `vsag/bitset.h`. A compact set of bit flags keyed by position, held through `BitsetPtr`.
It is used both as a filtering input and as a utility (e.g. the result of
[`l2_and_filtering`](types.md#utility-functions)).

```cpp
static BitsetPtr Random(int64_t length);  // random bitset of the given length
static BitsetPtr Make();                  // empty bitset

void Set(int64_t pos, bool value);
void Set(int64_t pos);       // = Set(pos, true)
bool Test(int64_t pos) const;
uint64_t Count();            // number of set bits
std::string Dump();          // debug dump
```

> When a `Bitset` is used as a search pre-filter (`bitset_filter_`, or the `invalid` argument of
> `KnnSearch` / `RangeSearch`), `Test(id) == true` means the id is **filtered out**.

## `IteratorContext`

Declared in `vsag/iterator_context.h`. An opaque handle that stores the position of an in-progress
iterator search so that subsequent calls resume where the previous one stopped.

```cpp
class IteratorContext {
public:
    virtual ~IteratorContext() = default;
};
```

You do not construct or inspect it directly. VSAG allocates it on the first iterator search; pass the
same handle back (via `SearchRequest::p_iter_ctx_`, or the `KnnSearch` iterator overload) on each
subsequent call, and set the last-search flag on the final call so the engine can release it. See
[Iterator Search](../advanced/iterator_search.md).

## `SearchParam` (deprecated)

Declared in `vsag/search_param.h`. `SearchParam` predates `SearchRequest` and is retained only for
the deprecated `KnnSearch(query, k, SearchParam&)` overload.

```cpp
struct SearchParam {  // [[deprecated]] use SearchRequest
    bool is_iter_filter{false};
    bool is_last_search{false};
    const std::string& parameters;
    FilterPtr filter{nullptr};
    Allocator* allocator{nullptr};
    IteratorContext* iter_ctx{nullptr};
};
```

**Prefer [`SearchRequest`](#searchrequest) + [`SearchWithRequest`](index_class.md#searchwithrequest)
for all new code.** `SearchParam` holds `parameters` by reference, so the referenced string must
outlive the call.

## See also

- [Index](index_class.md) — the search methods that consume these types.
- [Dataset](dataset.md) — building the `query_` and reading results.
- [Auxiliary Types](types.md) — attribute value types used by attribute filters.
