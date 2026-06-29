# Iterator Search

VSAG supports **iterator-based search** (also called *iterative search*): instead of asking for
the top-`k` results in one shot, the caller can request results in successive chunks while VSAG
preserves the internal search state between calls. Each subsequent call resumes from where the
previous one left off and returns *new*, non-overlapping results.

This is useful when:

- The application implements an external re-ranker or post-filter and wants to keep pulling more
  candidates until enough survivors are collected.
- Result consumption is lazy / streaming (e.g. UI pagination, server-side cursor).
- The eventual `k` is unknown up front and may grow on demand.

## How It Works

Iterator search relies on a long-lived `IteratorContext` object that holds:

- the current candidate heap / visited bitmap, and
- the cursor into the underlying graph or inverted lists.

The first call creates the context (when the pointer is `nullptr`); follow-up calls reuse it so
the search continues instead of restarting. When the caller is done, the `IteratorContext` object
itself must be deleted by the caller — that is what releases the iterator's internal state.

The `is_last_search` flag is *optional*: when set to `true`, the index drains the candidates that
are still buffered inside the context (the "discard heap") and returns them as the result of that
call. This is useful when the caller wants the long tail of explored-but-not-yet-emitted
candidates; if you don't need them, you can simply skip the final call and `delete` the context
directly. Note that the returned set is still capped to `k`, so if you want all tail candidates,
pass a sufficiently large `k` on the finalize call.

## Basic Usage (`SearchParam` API)

```cpp
#include <vsag/vsag.h>

// 1. Build an index (HGraph in this example)
auto index = vsag::Factory::CreateIndex("hgraph", hgraph_build_params).value();
index->Build(dataset);

// 2. Prepare query
auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(dim)->Float32Vectors(query_vec)->Owner(false);

// 3. Configure SearchParam in iterator mode
nlohmann::json search_parameters = {
    {"hgraph", {{"ef_search", 100}}},
};
std::string param_str = search_parameters.dump();

vsag::SearchParam search_param(
    /*iter_filter_flag=*/true,   // enable iterator mode
    param_str,
    /*filter=*/nullptr,
    /*allocator=*/&allocator,
    /*iter_ctx=*/nullptr,        // first call: context is created internally
    /*last_search_flag=*/false);

// 4. First page
auto page1 = index->KnnSearch(query, /*k=*/10, search_param).value();

// 5. Next page — context carries over, results do not overlap with page1
auto page2 = index->KnnSearch(query, /*k=*/10, search_param).value();

// 6. (Optional) drain the candidates still buffered in the context.
//    Skip this call if you don't need the tail candidates; cleanup
//    happens through `delete` below either way.
search_param.is_last_search = true;
auto page3 = index->KnnSearch(query, /*k=*/10, search_param).value();

// 7. The caller owns the context object — this is what releases resources.
delete search_param.iter_ctx;
```

> Reference: `examples/cpp/313_feature_search_allocator.cpp` and
> `examples/cpp/314_feature_hgraph_search_allocator.cpp`.

## Alternative: Explicit `IteratorContext` Argument

The lower-level `KnnSearch` overload accepts the context pointer directly. This is the form used
by VSAG's own tests (`tests/test_index/test_index_search.cpp`) when calling `KnnSearch` several
times in a row:

```cpp
vsag::IteratorContext* iter_ctx = nullptr;

auto r1 = index->KnnSearch(query, k1, param_str, filter, iter_ctx, /*is_last_search=*/false);
auto r2 = index->KnnSearch(query, k2, param_str, filter, iter_ctx, /*is_last_search=*/false);
auto r3 = index->KnnSearch(query, k3, param_str, filter, iter_ctx, /*is_last_search=*/false);

delete iter_ctx;
```

Each call advances `iter_ctx`; the union of the returned ids is a non-overlapping continuation of
the search ordered by distance. Pass `is_last_search=true` on a trailing call instead if you want
the index to also emit the candidates still buffered in the context.

> **`SearchRequest` API.** `SearchRequest` declares `enable_iterator_search_`, `p_iter_ctx_`, and
> `is_last_search_` fields, but no in-tree `SearchWithRequest` implementation currently consults
> them. Until that wiring lands, use one of the two `KnnSearch` forms above to drive iterator
> search.

## Combining With Filters

Iterator search composes with regular filters (label filter, attribute filter, bitset filter).
A common use case is "keep iterating until enough results pass my external check":

```cpp
size_t needed = 50;
std::vector<int64_t> kept;
vsag::IteratorContext* ctx = nullptr;

while (kept.size() < needed) {
    auto page = index->KnnSearch(query, 32, param_str, filter, ctx, /*is_last_search=*/false);
    if (!page.has_value() || page.value()->GetDim() == 0) break;

    for (int64_t i = 0; i < page.value()->GetDim(); ++i) {
        if (external_check(page.value()->GetIds()[i])) {
            kept.push_back(page.value()->GetIds()[i]);
        }
    }
}

// Release the iterator state. No `is_last_search=true` call is required —
// add one only if you also want the candidates still buffered in `ctx`.
delete ctx;
```

The HGraph index supports an additional runtime parameter — `skip_ratio` — that controls how
aggressively the iterator skips already-explored regions during continuation. See
`examples/cpp/314_feature_hgraph_search_allocator.cpp`.

## Support Matrix

Indexes that advertise the `SUPPORT_KNN_ITERATOR_FILTER_SEARCH` feature (queryable via
`Index::CheckFeature`):

| Index type | Supports iterator search |
|------------|--------------------------|
| hgraph     | yes |
| ivf        | no  |
| brute_force| no  |
| sindi      | no  |

Always check `index->CheckFeature(vsag::SUPPORT_KNN_ITERATOR_FILTER_SEARCH)` at runtime before
relying on this capability — coverage may expand in future releases.

## Notes and Pitfalls

- **Ownership.** The `IteratorContext` is owned by the caller. Forgetting to `delete` it leaks
  the internal search state (heap, visited bitmap, allocator scratch). Resource release is driven
  entirely by `delete`, not by `is_last_search`.
- **Optional last call.** `is_last_search = true` is *not* required for cleanup. Its only effect
  is to make the index drain the candidates that are still buffered in the context and return
  them as that call's result, still capped to `k`. Use it only when you want those tail
  candidates, and pick a `k` large enough not to truncate them.
- **Parameter stability.** Do not change the query vector, distance metric, or filter between
  calls that share a context — results are only meaningful when the search state is reused for
  the same logical query.
- **`k` per call.** The `k` argument applies to each call individually; the returned chunks are
  disjoint, so the cumulative result size grows by `k` (or less if the index is exhausted) each
  iteration.
- **Thread safety.** A single `IteratorContext` must not be used concurrently from multiple
  threads. Different queries should each have their own context.
