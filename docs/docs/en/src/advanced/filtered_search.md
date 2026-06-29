# Filtered Search

Filtered search restricts the result set of a `KnnSearch` or `RangeSearch` to vectors that
satisfy an application-defined predicate. VSAG applies the predicate **during** index
traversal whenever the underlying algorithm supports it, so you avoid the recall loss and
extra latency of post-filtering top-`k` results.

This page covers the three id-based filter APIs:

- **Bitset filter** — a compact bit array indexed by vector id.
- **Function-callback filter** — a `std::function<bool(int64_t)>`.
- **`Filter` object** — a `vsag::Filter` subclass that can also expose hints (valid ratio,
  distribution) to the search algorithm.

For attribute / "hybrid" search where the predicate is an SQL-like expression over typed
fields, see [Attribute Filter (Hybrid Search)](attribute_filter.md). For filtering against an
opaque per-vector byte payload during graph traversal, see [Extra Info](extra_info.md).

## Truth-value Conventions

The three APIs disagree on how to spell "exclude this id". Read this table carefully before
mixing them.

| API                   | Method                | Returning `true` means … |
|-----------------------|-----------------------|--------------------------|
| `Bitset`              | `Test(id)`            | id is **filtered out**   |
| `std::function`       | `f(id)`               | id is **filtered out**   |
| `Filter::CheckValid`  | `CheckValid(id)`      | id is **kept**           |

The bitset and `std::function` overloads are wrapped internally as a `BlackListFilter`
(`src/impl/filter/black_list_filter.cpp`): the bit being set, or the callback returning
`true`, marks the id as excluded. The `Filter::CheckValid` API inverts that polarity — `true`
keeps the id. If you maintain your own deletion bitmap, the bitset/function APIs are a
natural fit. If you want predicate logic with hints, the `Filter` form is clearer.

## Bitset Filter

`vsag::Bitset` (`include/vsag/bitset.h`) is a growable, ordinal-indexed bit array.

```cpp
auto invalid = vsag::Bitset::Make();
for (int64_t i = 0; i < num_vectors; ++i) {
    if (ids[i] % 2 == 0) {
        invalid->Set(ids[i]);    // even ids are excluded
    }
}

auto search_params = R"({ "hgraph": { "ef_search": 100 } })";
auto result = index->KnnSearch(query, /*topk=*/10, search_params, invalid).value();
```

The bitset is indexed by vector id, but ids are masked to their low 32 bits before lookup
(`bit_index = id & ROW_ID_MASK` in `src/impl/filter/black_list_filter.cpp`, where
`ROW_ID_MASK = 0xFFFFFFFFLL`). Two ids that share the same low 32 bits will collide in the
bitset, so keep ids within `[0, 2^32)` if you rely on this filter; otherwise switch to the
`Filter` form. The bitset is indexed by id, not by insertion order, so reused/recycled ids
must be handled by your application.

## Function-callback Filter

A plain lambda or `std::function<bool(int64_t)>` works directly. The callback must return
`true` for ids that should be **excluded** (it is wrapped as a `BlackListFilter`):

```cpp
// Drop even ids: return true to exclude.
std::function<bool(int64_t)> drop_even = [](int64_t id) { return id % 2 == 0; };
auto result = index->KnnSearch(query, 10, search_params, drop_even).value();
```

This is the easiest way to drop in a small amount of custom logic without subclassing. If
you prefer the "return true to keep" polarity, use the `Filter` object instead.

## `Filter` Object

The richest API is `vsag::Filter` (`include/vsag/filter.h`). Subclass it when the search
algorithm can benefit from hints about the predicate:

```cpp
class MyFilter : public vsag::Filter {
public:
    bool CheckValid(int64_t id) const override {
        return id % 2 == 1;
    }

    // Approximate fraction of ids that pass the predicate. The search uses this to
    // size internal candidate buffers; an accurate estimate improves latency and recall.
    float ValidRatio() const override { return 0.5F; }

    // Hint whether passing ids cluster spatially. NONE means "no correlation"; use
    // RELATED_TO_VECTOR if the predicate correlates with vector position (e.g. region tags).
    Distribution FilterDistribution() const override { return Distribution::NONE; }
};

auto filter = std::make_shared<MyFilter>();
auto result = index->KnnSearch(query, 10, search_params, filter).value();
```

Important methods:

| Method | Default | Purpose |
|---|---|---|
| `CheckValid(int64_t id)` | pure virtual | Required. `true` keeps the id. |
| `CheckValid(const char* data)` | returns `true` | Used for in-graph filtering against the per-vector byte payload; see [Extra Info](extra_info.md). |
| `ValidRatio()` | `1.0F` | Hint, in `[0, 1]`, of the fraction of ids that pass. |
| `FilterDistribution()` | `NONE` | `NONE` or `RELATED_TO_VECTOR`. |
| `GetValidIds(...)` | empty | Optional whitelist for very selective filters. |

Passing the wrong `ValidRatio` is not a correctness bug, but a poor estimate may either
inflate latency (overestimate) or hurt recall (underestimate).

## Available Overloads

`KnnSearch` and `RangeSearch` both expose four filter shapes (`include/vsag/index.h`):

```cpp
// KnnSearch
index->KnnSearch(query, topk, params);                                    // no filter
index->KnnSearch(query, topk, params, BitsetPtr invalid);
index->KnnSearch(query, topk, params, std::function<bool(int64_t)> f);
index->KnnSearch(query, topk, params, FilterPtr filter);

// RangeSearch
index->RangeSearch(query, radius, params, limited_size);                  // no filter
index->RangeSearch(query, radius, params, BitsetPtr invalid, limited_size);
index->RangeSearch(query, radius, params, std::function<bool(int64_t)> f, limited_size);
index->RangeSearch(query, radius, params, FilterPtr filter, limited_size);
```

`limited_size` is the maximum number of results returned by `RangeSearch`:

- `limited_size < 0`: no limit (the default `-1`).
- `limited_size == 0`: rejected explicitly by the API
  (`CHECK_ARGUMENT(limited_size != 0, ...)`); pass `-1` for "no limit".
- `limited_size > 0`: cap the result list at this many entries.

A filtered iterator-style search is also exposed:

```cpp
vsag::IteratorContext* ctx = nullptr;
index->KnnSearch(query, topk, params, filter, ctx, /*is_last_search=*/false);
// repeat with the same ctx; pass true on the final call to release resources
```

## Index Support Matrix

All index types accept the bitset, function, and `FilterPtr` overloads — the inner
implementation wraps bitsets and lambdas into a `FilterPtr` automatically. The columns below
reflect the **capability flags** each index registers (see `include/vsag/index_features.h`),
which is what runtime feature checks return.

| Index        | `_KNN_SEARCH_WITH_ID_FILTER` | `_RANGE_SEARCH_WITH_ID_FILTER` | `_KNN_ITERATOR_FILTER_SEARCH` |
|--------------|:----------------------------:|:------------------------------:|:-----------------------------:|
| HGraph       |              Yes             |               Yes              |              Yes              |
| IVF          |              Yes             |               Yes              |               —               |
| BruteForce   |              Yes             |               Yes              |               —               |
| Pyramid      |              Yes             |               Yes              |               —               |
| SINDI / WARP |              Yes             |               Yes              |               —               |

For id-based filtering, query support at runtime via
`index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_ID_FILTER)`,
`SUPPORT_RANGE_SEARCH_WITH_ID_FILTER`, and `SUPPORT_KNN_ITERATOR_FILTER_SEARCH`. The flag
`SUPPORT_KNN_SEARCH_WITH_EX_FILTER` is unrelated — it covers extra-info (byte-payload)
filtering, see [Extra Info](extra_info.md).

## Performance Notes

- The more selective the filter (smaller `ValidRatio`), the more candidates the search has
  to expand. For graph indexes, increase `ef_search` proportionally when the filter is very
  selective; otherwise recall will drop sharply below ~1% selectivity.
- **HGraph also offers a selectivity-aware brute-force fallback**: set
  `brute_force_threshold` (e.g. `0.01–0.05`) in the search params so that, when
  `Filter::ValidRatio()` is small enough, HGraph automatically skips graph
  traversal and runs an exact scan over the surviving ids. This is often a
  better choice than chasing recall by raising `ef_search` to very large values.
  See the [HGraph index page](../indexes/hgraph.md#brute-force-fallback-under-highly-selective-filters-brute_force_threshold)
  and example [`322_feature_hgraph_brute_force_threshold.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/322_feature_hgraph_brute_force_threshold.cpp).
- Bitset filters are fastest because `Test()` is a single bit lookup. A `Filter` object that
  performs heavy work in `CheckValid` will be called many times per query.
- For `RangeSearch`, set a finite `limited_size` when filters can let through millions of
  ids — otherwise the result set may grow unbounded.
- Filters compose cheaply with [Attribute Filter](attribute_filter.md) when using
  `SearchRequest`: all enabled filters are combined with logical AND.

## Combining Filters via `SearchRequest`

`SearchRequest` (`include/vsag/search_request.h`) is the unified entry point used by
`SearchWithRequest`. It can carry a bitset filter, a `Filter` object, and an attribute
expression simultaneously; all are ANDed together.

```cpp
vsag::SearchRequest req;
req.query_                = query;
req.mode_                 = vsag::SearchMode::KNN_SEARCH;
req.topk_                 = 10;
req.params_str_           = R"({ "hgraph": { "ef_search": 200 } })";
req.enable_filter_        = true;
req.filter_               = std::make_shared<MyFilter>();
req.enable_bitset_filter_ = true;
req.bitset_filter_        = invalid;
auto result = index->SearchWithRequest(req).value();
```

See [Attribute Filter](attribute_filter.md) for the `attribute_filter_str_` field.

## Examples

- C++: [`examples/cpp/301_feature_filter.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/301_feature_filter.cpp)
  — bitset, function, and `Filter`-object styles.
- C++: [`examples/cpp/320_feature_extra_info.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/320_feature_extra_info.cpp)
  — in-graph filtering using the `CheckValid(const char*)` byte-buffer overload.

## Python Status

Python bindings for the filter APIs are not yet exposed; the placeholder at
`examples/python/todo_examples/301_feature_filter.py` is intentionally empty. Use the C++
API for filtered search today.
