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

See `examples/cpp/308_feature_estimate_memory.cpp` for a full run.

### `EstimateBuildMemory(num_elements)`

`Index::EstimateBuildMemory(num_elements)` returns the estimated memory (in bytes) required
**during the build process** for `num_elements` vectors. Unlike `EstimateMemory`, which
estimates the steady-state size of the final index, this accounts for temporary buffers and
intermediate data structures that exist only while `Build` is running. The peak memory during
build is typically higher than the post-build footprint:

```cpp
uint64_t peak = index->EstimateBuildMemory(1000000);  // bytes
```

Currently only DiskANN provides a non-trivial implementation; other index types throw
an exception by default.

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

| Flag                          | Meaning                                          |
|-------------------------------|--------------------------------------------------|
| `vsag::SUPPORT_ESTIMATE_MEMORY` | `EstimateMemory(data_num)` is available.       |
| `vsag::SUPPORT_GET_MEMORY_USAGE` | `GetMemoryUsage()` is available.              |

Both flags can be checked via `index->CheckFeature(...)` — see
[Index Introspection](introspection.md).

## Thread Pool

`Resource` also accepts a user-supplied `ThreadPool`, which combined with a custom allocator gives
full control over parallelism and resource ownership. See
`examples/cpp/203_custom_thread_pool.cpp`.

## Notes

- A custom allocator must be thread-safe.
- The allocator's lifetime must outlive any index and result object referencing it.
- If nothing is configured, VSAG falls back to a default `malloc`-based allocator.
