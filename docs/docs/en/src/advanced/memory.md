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

- `Index::EstimateMemory(data_num)` — estimate memory usage before building
  (`examples/cpp/308_feature_estimate_memory.cpp`).
- `Index::GetMemoryUsage()` — query the current memory footprint
  (`examples/cpp/319_feature_get_memory_usage.cpp`).

## Thread Pool

`Resource` also accepts a user-supplied `ThreadPool`, which combined with a custom allocator gives
full control over parallelism and resource ownership. See
`examples/cpp/203_custom_thread_pool.cpp`.

## Notes

- A custom allocator must be thread-safe.
- The allocator's lifetime must outlive any index and result object referencing it.
- If nothing is configured, VSAG falls back to a default `malloc`-based allocator.
