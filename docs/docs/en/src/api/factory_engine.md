# Factory & Engine

Every VSAG workflow begins by obtaining an `Index`. There are two entry points:

- [`Factory`](#factory) — the simplest way to create an index or a file reader. It uses a default
  (or caller-supplied) allocator and manages resources internally.
- [`Engine`](#engine) — an explicit owner of shared resources (allocator + thread pool). Use it
  when you want several indexes to share one memory allocator / thread pool, or when you need
  deterministic control over resource lifetime.

This page also documents the process-level [initialization helpers](#library-initialization) and the
[top-level helper functions](#top-level-helper-functions) for parameter generation and validation.

## Library initialization

```cpp
#include <vsag/vsag.h>

vsag::init();                       // call once, before any other API
std::string ver = vsag::version();  // e.g. the git-derived build string
```

| Function | Signature | Notes |
|----------|-----------|-------|
| `vsag::init` | `bool init()` | One-time process initialization. Returns `true`. |
| `vsag::version` | `std::string version()` | Build version derived from the git revision. |

## Factory

Declared in `vsag/factory.h`. `Factory` is a stateless utility class with only static methods; it
cannot be instantiated.

### `CreateIndex`

```cpp
static tl::expected<std::shared_ptr<Index>, Error>
CreateIndex(const std::string& name,
            const std::string& parameters,
            Allocator* allocator = nullptr);
```

Creates an index of the given type.

| Parameter | Description |
|-----------|-------------|
| `name` | Index type name, e.g. `"hgraph"`, `"ivf"`, `"brute_force"`, `"sindi"`, `"pyramid"`. Removed names such as `"hnsw"`, `"fresh_hnsw"`, and `"diskann"` return `UNSUPPORTED_INDEX`. |
| `parameters` | A JSON string describing the index configuration (dtype, dim, metric, index-specific keys). See [Index Parameters](../resources/index_parameters.md). |
| `allocator` | Optional custom [`Allocator`](resource.md#allocator). When `nullptr`, VSAG uses a built-in default allocator. The caller must keep the allocator alive for the whole lifetime of the returned index. |

Returns a `std::shared_ptr<Index>` on success, or an `Error` (typically `UNSUPPORTED_INDEX` for an
unknown `name`, or `INVALID_ARGUMENT` for malformed `parameters`).

```cpp
auto index = vsag::Factory::CreateIndex("hgraph", R"(
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": { "base_quantization_type": "sq8" }
})");
if (not index.has_value()) {
    std::cerr << index.error().message << std::endl;
    return;
}
std::shared_ptr<vsag::Index> hgraph = index.value();
```

### `CreateLocalFileReader`

```cpp
static std::shared_ptr<Reader>
CreateLocalFileReader(const std::string& filename, int64_t base_offset, int64_t size);
```

Creates a [`Reader`](serialization.md#reader) that reads a window of a local file starting at
`base_offset` for `size` bytes. This is most often used to build a [`ReaderSet`](serialization.md#readerset)
for streaming deserialization of on-disk indexes. Unlike the methods above, it returns a plain
`std::shared_ptr` (there is no fallible `Error` channel).

## Engine

Declared in `vsag/engine.h`. An `Engine` binds a [`Resource`](resource.md#resource) (allocator +
thread pool) and lets you create indexes that share it. The engine never takes ownership of a
`Resource*` passed to it; you control its lifetime.

```cpp
vsag::Resource resource(vsag::Engine::CreateDefaultAllocator().get(), nullptr);
vsag::Engine engine(&resource);

auto index = engine.CreateIndex("hgraph", params);
// ... use index ...

engine.Shutdown();   // release engine-held state; warns on dangling references
```

### Constructor & lifecycle

| Member | Signature | Description |
|--------|-----------|-------------|
| Constructor | `explicit Engine(Resource* resource)` | Binds an externally-owned `Resource`. The `Resource` is **not** managed by the engine. |
| `Shutdown` | `void Shutdown()` | Gracefully tears down engine-held state. Warns if external references to engine resources still exist, guarding against dangling references. |

### `CreateIndex`

```cpp
[[nodiscard]] tl::expected<std::shared_ptr<Index>, Error>
CreateIndex(const std::string& name, const std::string& parameters);
```

Same semantics as [`Factory::CreateIndex`](#createindex), except the index is created against the
engine's shared `Resource` (allocator and thread pool) instead of a per-call allocator.

### Static resource helpers

| Member | Signature | Description |
|--------|-----------|-------------|
| `CreateDefaultAllocator` | `static std::shared_ptr<Allocator> CreateDefaultAllocator()` | Creates VSAG's built-in allocator. Returns an empty pointer on failure — check for null. |
| `CreateThreadPool` | `static tl::expected<std::shared_ptr<ThreadPool>, Error> CreateThreadPool(uint32_t num_threads)` | Creates a thread pool with `num_threads` workers. Returns an `Error` for an invalid count. |

See [Resource Management](resource.md) for how `Resource`, `Allocator`, and `ThreadPool` fit
together, and `examples/cpp/201_custom_allocator.cpp` / `203_custom_thread_pool.cpp` for runnable
samples.

## See also

- [Index](index_class.md) — what you can do with the index once it is created.
- [Resource Management](resource.md) — allocator, thread pool, and `Resource` details.
- [Index Parameters](../resources/index_parameters.md) — the JSON schema accepted by `CreateIndex`.
