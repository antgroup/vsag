# Resource Management

VSAG lets you take control of the memory and threads it uses. This page covers
[`Allocator`](#allocator) (custom memory management), [`ThreadPool`](#threadpool) (custom
concurrency), [`Resource`](#resource) (a bundle of the two shared by an [`Engine`](factory_engine.md#engine)),
the process-wide [`Options`](#options) singleton, and the pluggable [`Logger`](#logger).

Runnable samples: `examples/cpp/201_custom_allocator.cpp`, `202_custom_logger.cpp`, and
`203_custom_thread_pool.cpp`. See also [Memory Management](../advanced/memory.md) and
[Extensibility](../advanced/extensibility.md).

## `Allocator`

Declared in `vsag/allocator.h`. An abstract interface for custom memory management. Implement it to
route all of an index's allocations through your own pool, arena, or accounting layer, then pass it to
[`Factory::CreateIndex`](factory_engine.md#createindex) or a [`Resource`](#resource).

```cpp
class Allocator {
public:
    virtual std::string Name() = 0;
    virtual void* Allocate(uint64_t size) = 0;
    virtual void Deallocate(void* p) = 0;
    virtual void* Reallocate(void* p, uint64_t size) = 0;

    template <typename T, typename... Args> T* New(Args&&... args);  // Allocate + construct
    template <typename T> void Delete(T* p);                          // destruct + Deallocate
};
```

| Member | Description |
|--------|-------------|
| `Name()` | Identifier for the allocator implementation (used in diagnostics). |
| `Allocate(size)` | Return a block of at least `size` bytes. |
| `Deallocate(p)` | Free a block previously returned by this allocator. |
| `Reallocate(p, size)` | Resize a block, preserving contents. |
| `New<T>(args...)` | Helper: allocate and construct a `T`; frees and rethrows if the constructor throws. |
| `Delete<T>(p)` | Helper: destruct `*p` and free its storage (null-safe). |

An allocator passed to an index must outlive that index. VSAG's built-in allocator is available via
[`Engine::CreateDefaultAllocator`](factory_engine.md#static-resource-helpers).

## `ThreadPool`

Declared in `vsag/thread_pool.h`. An abstract task executor. Supply your own to make VSAG share your
application's threads instead of spawning its own.

```cpp
class ThreadPool {
public:
    virtual void WaitUntilEmpty() = 0;
    virtual void SetQueueSizeLimit(std::uint64_t limit) = 0;
    virtual void SetPoolSize(std::uint64_t limit) = 0;
    virtual std::future<void> Enqueue(std::function<void(void)> task) = 0;
};
```

| Member | Description |
|--------|-------------|
| `WaitUntilEmpty()` | Block until all enqueued tasks finish. |
| `SetQueueSizeLimit(limit)` | Cap the pending-task queue; behavior past the cap is implementation-defined. |
| `SetPoolSize(limit)` | Cap the number of worker threads. |
| `Enqueue(task)` | Submit a task; returns a `std::future<void>` for its completion. |

A ready-made pool can be created with
[`Engine::CreateThreadPool`](factory_engine.md#static-resource-helpers).

## `Resource`

Declared in `vsag/resource.h`. A `Resource` bundles an [`Allocator`](#allocator) and a
[`ThreadPool`](#threadpool) so that an [`Engine`](factory_engine.md#engine) — and every index it
creates — can share them.

```cpp
class Resource {
public:
    explicit Resource(Allocator* allocator, ThreadPool* thread_pool);
    explicit Resource(const std::shared_ptr<Allocator>& allocator,
                      const std::shared_ptr<ThreadPool>& thread_pool);
    explicit Resource();  // default allocator, no thread pool

    std::shared_ptr<Allocator> GetAllocator() const;
    std::shared_ptr<ThreadPool> GetThreadPool() const;
};
```

| Constructor / method | Description |
|----------------------|-------------|
| `Resource(Allocator*, ThreadPool*)` | Use raw pointers you own. A null allocator means "create and own a default"; a null thread pool means "no pool". |
| `Resource(shared_ptr, shared_ptr)` | Same, with shared ownership. |
| `Resource()` | Default allocator, no thread pool. |
| `GetAllocator()` | The resource's allocator (a default one if none was supplied). |
| `GetThreadPool()` | The resource's thread pool, or null if none was supplied. |

```cpp
auto alloc = vsag::Engine::CreateDefaultAllocator();
auto pool = vsag::Engine::CreateThreadPool(4).value();
vsag::Resource resource(alloc, pool);
vsag::Engine engine(&resource);
auto index = engine.CreateIndex("hgraph", params);
```

## `Options`

Declared in `vsag/options.h`. A process-wide singleton for global configuration, accessed via
`Options::Instance()`. Thread-safe. `Option` is a type alias for `Options`.

```cpp
vsag::Options::Instance().set_num_threads_building(8);
vsag::Options::Instance().set_logger(&my_logger);
```

| Setting | Accessors | Default | Meaning |
|---------|-----------|---------|---------|
| IO threads | `num_threads_io()` / `set_num_threads_io(n)` | `8` | Threads for disk-index IO during search (1–200). |
| Build threads | `num_threads_building()` / `set_num_threads_building(n)` | `4` | Threads for constructing an index. |
| Block size limit | `block_size_limit()` / `set_block_size_limit(bytes)` | `128 MB` | Max bytes per allocation block (must be > 2 MB). |
| Direct-IO align | `direct_IO_object_align_bit()` / `set_direct_IO_object_align_bit(bits)` | `9` | Direct-IO object alignment, in bits (< 21). |
| Logger | `logger()` / `set_logger(Logger*)` | `nullptr` | Active [`Logger`](#logger); returns `true` on set. |

## `Logger`

Declared in `vsag/logger.h`. An abstract logging sink. Implement it and register it via
`Options::set_logger` to route VSAG's log output through your application's logging system.

```cpp
class Logger {
public:
    enum Level : int {
        kTRACE = 0, kDEBUG = 1, kINFO = 2, kWARN = 3, kERR = 4, kCRITICAL = 5, kOFF = 6, kN_LEVELS
    };

    virtual void SetLevel(Level log_level) = 0;
    virtual void Trace(const std::string& msg) = 0;
    virtual void Debug(const std::string& msg) = 0;
    virtual void Info(const std::string& msg) = 0;
    virtual void Warn(const std::string& msg) = 0;
    virtual void Error(const std::string& msg) = 0;
    virtual void Critical(const std::string& msg) = 0;
};
```

| Member | Description |
|--------|-------------|
| `SetLevel(level)` | Only messages at or above `level` are emitted. `kOFF` disables logging. |
| `Trace` / `Debug` / `Info` / `Warn` / `Error` / `Critical` | Emit a message at the corresponding severity. |

See `examples/cpp/202_custom_logger.cpp`.

## See also

- [Factory & Engine](factory_engine.md) — how `Engine` consumes a `Resource`.
- [Index](index_class.md) — per-call allocators on retrieval and search methods.
- [Search Request & Filters](search.md#resource--iterator-fields) — the per-search allocator field.
