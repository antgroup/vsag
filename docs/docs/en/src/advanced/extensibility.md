# Extensibility

VSAG exposes a small set of stable C++ extension points so applications can plug
in their own infrastructure without forking the library. This page summarizes
**what is extensible** and **what is not**, and links to runnable examples.

## Public extension points

| Extension point | Header | Purpose |
|---|---|---|
| `vsag::Allocator` | `vsag/allocator.h` | Custom memory allocation strategy. |
| `vsag::Logger` | `vsag/logger.h` | Redirect VSAG logs to your logging stack. |
| `vsag::ThreadPool` | `vsag/thread_pool.h` | Reuse an external worker pool for builds and IO. |
| `vsag::Filter` | `vsag/filter.h` | Custom pre-filter for `KnnSearch` / `RangeSearch`. |
| `vsag::Reader` (+ `ReaderSet`) | `vsag/readerset.h` | Custom IO backend for deserialization. |

All five are abstract base classes. Each declares at least one pure-virtual method that you
must implement; some also declare non-pure-virtual methods with sensible defaults (for example,
`Filter::CheckValid(const char*)`, `Filter::ValidRatio()`, `Filter::FilterDistribution()`,
`Filter::GetValidIds()`, and `Reader::MultiRead()`) that you can override only when you need
custom behaviour. Implement the required methods, wrap your instance in a `std::shared_ptr`
(or pass a raw pointer where the API requires it), and hand it to VSAG.

## Wiring extensions into an index

There are two main entry points.

### 1. Per-index resources via `Engine`

`vsag::Engine` (`vsag/engine.h`) is the recommended way to bind a custom
`Allocator` and `ThreadPool` to every index it creates:

```cpp
auto allocator   = std::make_shared<MyAllocator>();
auto thread_pool = std::make_shared<MyThreadPool>();
vsag::Resource resource(allocator, thread_pool);
vsag::Engine engine(&resource);

auto index = engine.CreateIndex("hgraph", parameters).value();
// ... use index ...
engine.Shutdown();
```

`Engine(Resource*)` takes a non-owning pointer — the caller is responsible for
keeping the `Resource` alive for at least as long as the engine and every index
it produced (until `Shutdown()` returns / those indexes are destroyed). The
`Resource` itself owns the `Allocator` / `ThreadPool` shared pointers. See
[Memory Management](memory.md) for the full ownership model, and
[Per-Search Allocator](search_allocator.md) for scoping an allocator to a
single search call.

For quick prototypes, `Engine::CreateDefaultAllocator()` and
`Engine::CreateThreadPool(num_threads)` return ready-to-use implementations.

### 2. `Factory::CreateIndex` with a raw allocator

`vsag::Factory::CreateIndex(name, params, allocator)`
(`vsag/factory.h`) accepts an optional `Allocator*`. This path does not take a
thread pool; new code should prefer `Engine`.

## Filter

Implement `vsag::Filter` and pass a `FilterPtr` through `SearchRequest::filter_`
**and** set `SearchRequest::enable_filter_ = true` (the filter is ignored when
the flag is off). The legacy `SearchParam::filter` path remains supported.
Only `CheckValid(int64_t id)` is required; the other hooks are optional
optimizations:

- `CheckValid(const char* data)` — filter on per-vector extra info.
- `ValidRatio()` — hint the planner about selectivity.
- `FilterDistribution()` — hint about the spatial distribution of the valid
  ids: `NONE` (default) means no hint, `RELATED_TO_VECTOR` means the valid ids
  are correlated with vector position. See `vsag/filter.h`.
- `GetValidIds(...)` — expose a precomputed valid-id list for very selective
  filters.

Runnable example: `examples/cpp/301_feature_filter.cpp`. The
[Filtered Search](filtered_search.md) page describes filter integration in
detail.

## Reader / ReaderSet

`Index::Deserialize(const ReaderSet&)` lets you stream an index from any storage
backend (local file, object storage, remote FS, …) by providing a `Reader` per
named binary stream. Implement `Read`, `AsyncRead`, and `Size` at minimum;
`MultiRead` is optional and improves throughput when the backend supports
batched IO. `vsag::Factory::CreateLocalFileReader` is a reference
implementation for local files.

Runnable example: `examples/cpp/102_index_diskann.cpp` (DiskANN deserialization
uses `ReaderSet`). See [Serialization](serialization.md) for the full
serialize / deserialize matrix.

## Logger

VSAG uses a single global logger configured through the `Options` singleton:

```cpp
class MyLogger : public vsag::Logger { /* implement Trace/Debug/Info/... */ };
static MyLogger my_logger;
vsag::Options::Instance().set_logger(&my_logger);
```

The logger pointer is **not** owned by VSAG — keep it alive for the duration of
any VSAG call. Pass `nullptr` to fall back to the built-in logger.

Runnable example: `examples/cpp/202_custom_logger.cpp`.

## Global tuning via `Options`

`vsag::Options::Instance()` (`vsag/options.h`) is a process-wide singleton for
settings that do not belong to a specific index:

| Setter | Default | Notes |
|---|---|---|
| `set_num_threads_io(n)` | `8` | Threads used for disk-index IO during search. Must be in `[1, 200]`. |
| `set_num_threads_building(n)` | `4` | Threads used while building disk indexes. |
| `set_block_size_limit(bytes)` | `128 MiB` | Maximum size of a single allocation block. Must be `≥ 256 KiB` (`src/options.cpp:53-57`). |
| `set_direct_IO_object_align_bit(bits)` | `9` | Direct-IO alignment, in bits. Must be `≤ 21` (alignment size up to 2 MiB; `src/options.cpp:40-46`). |
| `set_logger(Logger*)` | built-in | See [Logger](#logger). |

These options affect every index in the process; set them once at startup. They
do **not** override per-index parameters such as HGraph's `build_thread_count`.

## What is *not* publicly extensible

VSAG does not currently provide stable public interfaces for the following:

- **Quantizers.** Concrete quantizer types (SQ8, PQ, RaBitQ, …) are selected
  via index parameters; subclassing them from user code is not supported.
- **Distance computers / metric types.** Distance metrics are fixed to
  `l2`, `ip`, and `cosine` per index.
- **DataCell / IO / storage backends inside an index.** These are
  implementation details. Use the `Reader` interface for custom IO at the
  deserialization boundary.

If you need one of these, please open an issue describing the use case.

## A note on `vsag::ext`

The `vsag/vsag_ext.h` header defines a thin handle-based API (`IndexHandler`,
`DatasetHandler`, `BitsetHandler`, …) intended for **language bindings and FFI**. It is not a
user-facing extension surface; prefer the standard `vsag::Index` API for C++
applications.

## Related examples

- `examples/cpp/201_custom_allocator.cpp`
- `examples/cpp/202_custom_logger.cpp`
- `examples/cpp/203_custom_thread_pool.cpp`
- `examples/cpp/301_feature_filter.cpp`
- `examples/cpp/102_index_diskann.cpp`
