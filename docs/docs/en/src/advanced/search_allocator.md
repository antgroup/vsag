# Per-Search Allocator

VSAG exposes a **per-call** `Allocator` hook that is separate from the index's own allocator,
intended for use cases such as:

- isolating per-query memory from the index's long-lived heap;
- backing high-concurrency online traffic with a thread-local arena that has no atomic
  contention with neighbours;
- accounting or capping each query's footprint independently of the index.

The hook is exposed through two surfaces — `SearchRequest::search_allocator_` (recommended) and
the legacy `SearchParam::allocator` — but **how much of a search actually consumes that
allocator depends on the index and the entry point**. HGraph, IVF, Pyramid, and SINDI use a
non-null `SearchRequest::search_allocator_` for non-empty result buffers as well as search
scratch state. BruteForce (including WARP mode) uses it for selected temporary work, while its
result `Dataset` still uses the index allocator. See
[Relationship to the Index's Allocator](#relationship-to-the-indexs-allocator) below for the
per-surface breakdown.

> **Scope.** The allocator hook is exposed through `KnnSearch` (`SearchParam` overload) and
> `SearchWithRequest`. The dedicated `RangeSearch` overloads do not accept an allocator, but a
> range-mode `SearchRequest` does, and supporting indexes consult `search_allocator_` on that
> path too.

## Recommended API — `SearchRequest::search_allocator_`

```cpp
#include "vsag/search_request.h"

vsag::SearchRequest req;
req.query_ = query;
req.mode_ = vsag::SearchMode::KNN_SEARCH;
req.topk_ = 10;
req.params_str_ = R"({"hgraph":{"ef_search":100}})";
req.search_allocator_ = thread_local_allocator.get();  // optional, may stay nullptr

auto result = index->SearchWithRequest(req).value();
```

`SearchRequest` (`include/vsag/search_request.h`) is the recommended, non-deprecated way to drive
a single search call. The `search_allocator_` field is optional — when left at `nullptr`, the
index falls back to the allocator that was attached to its owning `Resource`.

> **Availability.** `Index::SearchWithRequest` has a default implementation that returns an
> *unsupported* error. HGraph, IVF, BruteForce, WARP, SINDI and Pyramid implement it today.
> Pyramid supports KNN request searches, including `expected_labels_` reasoning reports; range
> requests with expected labels are not supported. For indexes that do not yet override
> `SearchWithRequest` (SINDI_V2), use the legacy `SearchParam`
> path described below.

## Legacy API — `SearchParam::allocator` *(deprecated)*

```cpp
#include "vsag/search_param.h"

nlohmann::json search_params = {{"hgraph", {{"ef_search", 100}}}};
std::string param_str = search_params.dump();

vsag::SearchParam search_param(/*iter_filter=*/false,
                               param_str,
                               /*filter=*/nullptr,
                               /*allocator=*/thread_local_allocator.get());
auto result = index->KnnSearch(query, /*k=*/10, search_param).value();
```

`SearchParam` is documented as deprecated in `include/vsag/search_param.h` ("Use SearchRequest
instead") and remains only for source compatibility. The wording is currently a doc comment —
the struct itself does not carry the C++ `[[deprecated]]` attribute, so the compiler will not
emit deprecation warnings, but new code should still target `SearchRequest` /
`SearchWithRequest` on indexes that support it. The example
`examples/cpp/314_feature_hgraph_search_allocator.cpp` (HGraph) demonstrates the legacy form.

## Result Ownership

The result-`Dataset` ownership contract depends on which index implements `SearchWithRequest`:

- **HGraph / IVF / Pyramid / SINDI** pass the selected query allocator into their non-empty
  result builders. When `request.search_allocator_` is non-null, the returned `Dataset` owns its
  `ids` / `distances` buffers through that allocator and releases them from its destructor.
- **BruteForce / WARP** currently build the result through
  `pack_knn_result_with_extra_info(heap)` without an allocator argument, so the result uses the
  index allocator. `request.search_allocator_` is still used for selected temporary work.
- **Empty results** use the shared empty-dataset path and do not allocate result buffers through
  the per-search allocator.

What this means in practice:

- **Do not manually `Deallocate` the result buffers.** Letting the `Dataset` go out of scope is
  enough; double-freeing through both manual `Deallocate(...)` and the destructor is undefined
  behaviour.
- **Whichever allocator owns the result must outlive that result `Dataset`.** For HGraph, IVF,
  Pyramid, and SINDI this can be the per-search allocator; for BruteForce / WARP it is the index
  allocator.

The simplest safe pattern is "one allocator per thread, reset between batches":

```cpp
ArenaAllocator arena;       // thread-local, big enough for one batch

for (const auto& q : batch) {
    vsag::SearchRequest req;
    req.query_ = q;
    req.topk_ = topk;
    req.params_str_ = params;
    req.search_allocator_ = &arena;
    auto result = index->SearchWithRequest(req).value();
    consume(result);
    // result Dataset destroyed here; arena frees ids/distances via its Deallocate.
}
arena.reset();              // drops every per-query buffer at once
```

## Relationship to the Index's Allocator

| Surface | Allocator used |
|---|---|
| Index build, insert, persistent state | `Resource`'s allocator (or the default if none was passed). |
| HGraph / IVF / Pyramid / SINDI `SearchWithRequest` non-empty result | `search_allocator_` if set, otherwise the index allocator. |
| BruteForce / WARP `SearchWithRequest` result | The index allocator; `search_allocator_` is used only for selected temporary work. |
| `SearchWithRequest` scratch state | Index-specific; the implementations above consult `search_allocator_` for at least part of the query path. |
| `KnnSearch(query, k, SearchParam)` (legacy) | `SearchParam::allocator` on indexes whose legacy overload honors it; otherwise the index allocator. |
| `KnnSearch(query, k, parameters_str)` | No per-search allocator hook; uses the index allocator. |
| Dedicated `RangeSearch(...)` overloads | No allocator parameter; use the index allocator. |
| Range-mode `SearchWithRequest` | Follows the same index-specific rules as KNN-mode `SearchWithRequest`. |

Setting a per-search allocator never affects the index's permanent data structures. It only
narrows the lifetime of memory touched by one specific search call, and only to the extent that
the index/entry point actually consumes it (see the per-row notes above).

## Requirements

- The allocator must be thread-safe **only if** it is shared across threads. A thread-local
  arena does not need internal synchronization.
- The allocator's lifetime must outlive every result `Dataset` it produced.
- `Reallocate(nullptr, size)` must behave like `Allocate(size)`. VSAG relies on this contract for
  its internal containers.

## Runnable Examples

- `examples/cpp/314_feature_hgraph_search_allocator.cpp` — HGraph (`sq8`) + custom allocator.

See also [Memory Management](memory.md) for the index-level `Allocator` / `Resource` setup, and
[Filtered Search](filtered_search.md) for combining a per-search allocator with custom filtering
in a `SearchRequest`.
