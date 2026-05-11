# Per-Search Allocator

VSAG exposes a **per-call** `Allocator` hook that is separate from the index's own allocator,
intended for use cases such as:

- isolating per-query memory from the index's long-lived heap;
- backing high-concurrency online traffic with a thread-local arena that has no atomic
  contention with neighbours;
- accounting or capping each query's footprint independently of the index.

The hook is exposed through two surfaces — `SearchRequest::search_allocator_` (recommended) and
the legacy `SearchParam::allocator` — but **how much of a search actually consumes that
allocator depends on the index and the entry point**. As of today, only `HGraph::SearchWithRequest`
plumbs `search_allocator_` end-to-end (scratch buffers **and** the result `Dataset`); the other
`SearchWithRequest` implementations (IVF / BruteForce / WARP) use it for some scratch
state but still allocate the result `Dataset` from the index's own allocator. See
[Relationship to the Index's Allocator](#relationship-to-the-indexs-allocator) below for the
per-surface breakdown.

> **Scope.** The allocator hook is currently exposed through `KnnSearch` (`SearchParam` overload)
> and `SearchWithRequest`. `RangeSearch` does not have an allocator-bearing overload at this
> time, and `SearchRequest::search_allocator_` is not consulted by the range-search path.

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
> *unsupported* error. Only HGraph, IVF, BruteForce and WARP implement it today
> (`src/algorithm/{hgraph,ivf,brute_force,warp}.cpp`). For indexes that do not yet override
> `SearchWithRequest` (HNSW, DiskANN, SINDI, Pyramid, SparseIndex), use the legacy `SearchParam`
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
`examples/cpp/313_feature_search_allocator.cpp` (HNSW) and
`examples/cpp/314_feature_hgraph_search_allocator.cpp` (HGraph) demonstrate the legacy form.

## Result Ownership

The result-`Dataset` ownership contract depends on which index implements `SearchWithRequest`:

- **HGraph** is the only index that currently plumbs `request.search_allocator_` into
  `create_fast_dataset` (see `src/algorithm/hgraph.cpp` — `ctx.alloc = request.search_allocator_`).
  The resulting `Dataset` is marked `Owner(true, allocator)` and its destructor will call
  `allocator->Deallocate(...)` on `ids` / `distances` automatically.
- **IVF / BruteForce / WARP** currently construct the result `Dataset` via
  `create_fast_dataset(..., allocator_)` — i.e. the index's own allocator
  (`src/algorithm/ivf.cpp`, `src/algorithm/brute_force.cpp`, `src/algorithm/warp.cpp`).
  `request.search_allocator_` is only consulted for scratch state on those paths today; the
  result buffers are owned by the index's allocator. Treat the result `Dataset`'s lifetime as
  tied to the index's allocator on these indexes.

What this means in practice:

- **Do not manually `Deallocate` the result buffers.** Letting the `Dataset` go out of scope is
  enough; double-freeing through both manual `Deallocate(...)` and the destructor is undefined
  behaviour.
- **Whichever allocator owns the result must outlive that result `Dataset`.** For HGraph that is
  the per-search allocator; for IVF / BruteForce / WARP that is the index allocator (always
  alive while the index is alive).
- **`examples/cpp/314_feature_hgraph_search_allocator.cpp` currently makes the deallocation
  explicit.** That pattern is left over from earlier API iterations; new code that targets the
  current owner-tracking behaviour should rely on the `Dataset` destructor instead.

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

| Surface                                                                            | Allocator used                                                                                                                                |
|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Index build, insert, persistent state                                              | `Resource`'s allocator (or default if none was passed).                                                                                       |
| `HGraph::SearchWithRequest` scratch + result `Dataset`                             | `search_allocator_` if set, otherwise the `Resource`'s allocator. HGraph is the only index that plumbs `search_allocator_` into the result.   |
| `IVF` / `BruteForce` / `WARP` `SearchWithRequest` result `Dataset`                 | Always the index's own allocator (`allocator_`). `search_allocator_` is *not* consulted for result buffers today.                             |
| `IVF` / `BruteForce` / `WARP` `SearchWithRequest` scratch state                    | Uses `search_allocator_` for some intermediate buffers when set; otherwise the index's allocator.                                             |
| `KnnSearch(query, k, SearchParam)` (legacy)                                        | Uses `SearchParam::allocator` if set, on indexes whose `KnnSearch` honors it (e.g. HNSW, HGraph examples). Otherwise the `Resource` allocator. |
| `KnnSearch(query, k, parameters_str)`                                              | No per-search allocator hook — uses the `Resource` allocator.                                                                                 |
| `RangeSearch(...)` (all forms)                                                     | Uses the `Resource` allocator; no per-search allocator hook.                                                                                  |

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

- `examples/cpp/313_feature_search_allocator.cpp` — HNSW + custom allocator (legacy
  `SearchParam`).
- `examples/cpp/314_feature_hgraph_search_allocator.cpp` — HGraph (`sq8`) + custom allocator.

See also [Memory Management](memory.md) for the index-level `Allocator` / `Resource` setup, and
[Filtered Search](filtered_search.md) for combining a per-search allocator with custom filtering
in a `SearchRequest`.
