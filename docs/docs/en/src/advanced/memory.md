# Memory Management

VSAG uses custom `Allocator` and `Resource` objects on its hot paths, allowing users to:

- plug in existing in-house memory pools;
- measure and cap index memory usage;
- route allocations precisely in multi-process or NUMA environments.

## Custom Allocator

```cpp
class MyAllocator : public vsag::Allocator {
public:
    std::string Name() override { return "my_allocator"; }
    void* Allocate(size_t size) override;
    void Deallocate(void* p) override;
    void* Reallocate(void* p, size_t size) override;
    // ...
};

auto allocator = std::make_shared<MyAllocator>();
auto resource = std::make_shared<vsag::Resource>(allocator, /*thread_pool=*/nullptr);
auto engine = vsag::Engine(resource);

auto index = engine.CreateIndex("hgraph", build_params).value();
```

See `examples/cpp/201_custom_allocator.cpp` for a full example.

## Per-Search Temporary Allocator

`KnnSearch` / `RangeSearch` can take a per-call `Allocator` that lives in a thread-local arena,
avoiding contention with the global heap:

```cpp
vsag::SearchParam search_param;
search_param.allocator = thread_local_allocator.get();
auto result = index->KnnSearch(query, k, search_param);
```

See `examples/cpp/313_feature_search_allocator.cpp` and
`examples/cpp/314_feature_hgraph_search_allocator.cpp`.

## Estimating and Querying Memory

### `EstimateMemory(data_num)`

`Index::EstimateMemory(data_num)` returns a byte-level estimate of the memory the index will
occupy once `data_num` vectors have been inserted. It is computed from the build parameters
(dimension, quantization, `max_degree`, etc.) without allocating any vector storage, so it is
safe to call on an empty index and is the recommended way to size a node before ingest:

```cpp
if (index->CheckFeature(vsag::SUPPORT_ESTIMATE_MEMORY)) {
    uint64_t estimated = index->EstimateMemory(1'000'000);  // bytes
}
```

For HGraph, a count of zero returns 0, while a rounded element count or byte total that is
unrepresentable in `uint64_t` throws an exception derived from `std::exception` instead of
silently wrapping.

**Random-level failure handling.** Before reserving IDs or publishing labels, the NSW,
batch-graph and imported-cache build paths validate the input-sized random-level prefix
using a copied generator. This adds O(N) time and constant storage, without consuming the
live RNG. NSW holds the Add preparation lock through validation and level assignment;
row insertion still runs after that lock is released. An invalid level therefore leaves
IDs, labels and RNG state available for retry. This is not general rollback: NSW training
may already have occurred, and other failures can still leave partial state. Checking all
input rows is conservative when duplicate labels would consume fewer draws.

See `examples/cpp/308_feature_estimate_memory.cpp` for a full run.

### `EstimateBuildMemory(num_elements)`

`Index::EstimateBuildMemory(num_elements)` returns the estimated peak memory (in bytes)
attributable to VSAG during a **successful `Build`** of `num_elements` vectors, starting
from the index's current state. Unlike `EstimateMemory`, which estimates the steady-state
size of the completed index, this estimate covers the retained constructor memory plus the
storage `Build` creates: sampling and training buffers, prepared rows, construction
scratch, and scheduler objects. `EstimateMemory` is used as a floor, so the result is never
lower than the estimated final footprint.

The estimate accounts for two allocation routes: requests made through the index's
configured `Allocator`, and standard-library allocations (shared-state control blocks,
`std::function` targets, container internals) made through the global `operator new`. Each
allocation family is charged once, including old-plus-new overlap while vectors and tables
grow. Excluded by contract: caller-owned input vectors, pre-existing `Engine`/`Resource`/
pool ownership, unrelated work on a shared pool, allocator internals, thread stacks and
runtime state, and arbitrary failure-path exception payloads. The result is a documented
model estimate for the supported configurations below, not a universal process-heap upper
bound:

```cpp
if (index->CheckFeature(vsag::SUPPORT_ESTIMATE_BUILD_MEMORY)) {
    uint64_t peak = index->EstimateBuildMemory(1000000);  // bytes
}
```

**Preconditions.** The documented guarantee assumes the estimated `Build` runs alone on the
index — no other operation, including concurrent const queries that can lease visited lists
or allocate index-owned scratch, overlaps it. Separately, the caller must not invoke
`EstimateBuildMemory` or its `CheckFeature` guard concurrently with same-index mutation or
associated resource resizing. The result describes the current state; it does not certify a
later `Build` after state or resource changes.

**Supported configurations.** HGraph advertises `SUPPORT_ESTIMATE_BUILD_MEMORY` only when
its active configuration and state have the documented memory model: a float32 scalar NSW
graph with flat in-memory block code and graph stores; `fp32`, `sq8` or `sq8_uniform` base
quantization; and ordinary numeric labels under the default `pg` label remap
(`label_remap_type: "robin"` is unsupported). The public default `Engine`, ordinary internal
pools, and the known built-in supplied-pool arrangement — including the configuration used
by `examples/cpp/308_feature_estimate_memory.cpp` — are covered. Eligibility is also
compile-time: the standard-library allocation model is compiled only for libc++ 15.0.7 or
21.1.6 (`_LIBCPP_VERSION` 15007 or 210106, ABI v1, without the optimized `std::function`
ABI) or for libstdc++, so other C++ standard libraries report the flag as false regardless
of configuration. The measured-peak comparison has completed on the native Apple Clang 21
/ SDK 26.5 / libc++ 210106 build and on the aarch64 Linux build under GCC 11.4 headers /
libstdc++; eligibility on another toolchain means the model is available, not that its
runtime is validated — the LLVM 15 clang-tidy parse used for lint is not runtime
validation. Unsupported configurations
and states report the flag as false and throw a typed unsupported-operation error (an
exception derived from `std::exception`) instead of returning a number. This includes
advanced transforms, RaBitQ, MCI, removal or reverse edges, reorder/raw storage, duplicate
or deduplicated storage, ELP, and attribute or extra-info paths, as well as any index that
is nonempty, immutable, holds an imported or retained build cache, or has entered `Merge`,
`Tune`, `Deserialize` or `ParallelDeserialize` — including failed calls, and including a
count-zero `Tune`, which can replace quantizers and storage even on an empty index.

**Sizing and errors.** Internally, the estimator rounds the element count up to the
configured capacity step, derives per-component envelopes, and replays the route-level
generator in O(N) time with fixed storage — no allocation proportional to `num_elements`.
A `num_elements` whose rounded capacity exceeds the internal ID range, or any intermediate
result unrepresentable in `uint64_t`, produces a typed invalid-argument error; unsupported
configurations and states produce the unsupported-operation error even for `num_elements`
0. On a supported empty index, `EstimateBuildMemory(0)` returns a positive
retained-baseline estimate — the constructed index still owns memory — while
`EstimateMemory(0)` returns 0.

**Model limitations.** The reverse-label map uses the nominal prime-table growth rule,
justified for at most 8193 distinct labels and nonnegative dense IDs `0..N-1`; other label
patterns can trigger extra displacement growth the model does not cover, and the
large-count route-map key-range ceiling is conservative and can be loose. The modeled input
is vectors with numeric labels; additional source-ID or metadata payloads that `Build`
copies into index-owned tables are outside the guarantee — they remain VSAG-owned and are
not part of the caller-owned exclusion. For opaque custom
pools or additional wrapper nesting, `Build`-created queuing and wrapper costs beyond the
documented arrangement are unmodeled — not guaranteed, and not relabeled as pre-existing
ownership. The estimate's comparison target is the measured allocator phase peak plus a
derived standard-allocation envelope, over a documented 24-cell matrix
(`fp32`/`sq8`/`sq8_uniform` × dimensions 16/128 × `max_degree` 8/32 × serial and
width-8-pool/four-requested-worker modes; N=1537, `ef_construction` 64, fixed 256 KiB
blocks).

**Parallel insertion note.** During parallel `Build`/`Add`, row insertion is dispatched to
at most `min(max(build_thread_count, 1), rows)` workers, so a supplied pool wider than the
effective worker count adds no row concurrency, and per-`Build` scheduler objects scale
with that bound. With a supplied pool, a nonempty operation uses the parallel path even
when `build_thread_count` is 0 or 1 — a single worker runs, with parallel-path error
semantics (the exception from the lowest failing row is rethrown). This applies to `Add`
and every other parallel insertion caller, not only `Build`; if worker submission fails
after some workers have launched, those workers may drain all remaining rows, so mixed
submission/row-failure ordering can differ from earlier versions.

See `examples/cpp/308_feature_estimate_memory.cpp` for a run that checks both flags.

### `GetMemoryUsage()`

`Index::GetMemoryUsage()` returns the **current** memory footprint of an index in bytes:

```cpp
uint64_t bytes = index->GetMemoryUsage();
```

Properties:

- Implemented by every index type, but only indexes that advertise
  `vsag::SUPPORT_GET_MEMORY_USAGE` via `CheckFeature` are formally guaranteed to return a
  meaningful value. HGraph, IVF, BruteForce, Pyramid and WARP set the flag
  (see `src/algorithm/{hgraph,ivf,brute_force,pyramid,warp}.cpp`); SINDI implements the call
  (since the method is pure-virtual on `Index`) but does not currently set the feature flag, so
  treat its value as informational only.
- Thread-safe; can be polled concurrently with searches.
- Latency is on the order of microseconds — suitable for production-grade real-time
  monitoring loops.
- Reports memory attributable to the index itself (vectors, graph, quantizer state). The number
  is typically smaller than the resident set size observed at the OS level, which also includes
  allocator overhead, scratch buffers, and any data held outside the index (e.g. user-owned input
  vectors). For SINDI in particular, call `GetMemoryUsage()` **after** the build completes to get
  a representative value.

See `examples/cpp/319_feature_get_memory_usage.cpp` for a runnable example, including a helper
that compares the interface value with the process resident size.

### `GetMemoryUsageDetail()`

`Index::GetMemoryUsageDetail()` returns a breakdown of the **current** memory usage by
component:

```cpp
std::unordered_map<std::string, uint64_t> detail = index->GetMemoryUsageDetail();
for (const auto& [component, bytes] : detail) {
    std::cout << component << ": " << bytes << " bytes\n";
}
```

The returned map keys are component names and values are memory in bytes. This is useful for
understanding *where* the memory is going inside an index.

Currently only HGraph provides a meaningful implementation, returning components such as
`basic_flatten_codes`, `bottom_graph`, `route_graph`, `neighbors_mutex`, `pool`,
`label_table`, `high_precise_codes`, `extra_infos`, and `raw_vector`. SINDI returns an empty
map. Other index types throw an exception by default.

### Capability Flags

| Flag                                  | Meaning                                                          |
|---------------------------------------|------------------------------------------------------------------|
| `vsag::SUPPORT_ESTIMATE_MEMORY`       | `EstimateMemory(data_num)` is available.                         |
| `vsag::SUPPORT_ESTIMATE_BUILD_MEMORY` | `EstimateBuildMemory(num_elements)` is available for the index's active configuration and state. |
| `vsag::SUPPORT_GET_MEMORY_USAGE`      | `GetMemoryUsage()` is available.                                 |

These flags can be checked via `index->CheckFeature(...)` — see
[Index Introspection](introspection.md).

## Thread Pool

`Resource` also accepts a user-supplied `ThreadPool`, which combined with a custom allocator gives
full control over parallelism and resource ownership. See
`examples/cpp/203_custom_thread_pool.cpp`.

## Notes

- A custom allocator must be thread-safe.
- The allocator's lifetime must outlive any index and result object referencing it.
- If nothing is configured, VSAG falls back to a default `malloc`-based allocator.
