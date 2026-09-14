# User-Defined IO

User-defined IO (`user_defined_io`) lets callers inject their own read, write, and resize
callbacks so that VSAG builds an index directly into external storage without an intermediate
serialization step. This is useful when the precise codes must live in a managed store
(e.g. a distributed filesystem, an object store, or a custom block device) that VSAG does not
natively support.

## Supported configuration

- **Index**: HGraph with `use_reorder: true`.
- **Base codes**: `sq8` in `block_memory_io`.
- **Precise codes**: `fp32` in `user_defined_io`.
- **Graph**: `block_memory_io`.

Other graph / base / bucket storage combinations are not supported in this version.

## Callback contract

Three synchronous callbacks must be provided:

| Callback | Signature | Responsibilities |
|----------|-----------|------------------|
| Read | `void(uint64_t offset, uint64_t len, void* dest)` | Copy `len` bytes from `offset` into `dest`. Throw on out-of-bounds. |
| Write | `void(uint64_t offset, uint64_t len, const void* src)` | Consume `len` bytes from `src` at `offset`. Grow the backing store if `offset + len` exceeds the current size. |
| Resize | `void(uint64_t new_size)` | Truncate or extend the backing store to `new_size` while preserving the retained prefix. |

Constraints:

- All three callbacks are serialized by a mutex and must not reenter the same adapter.
- Writes must be immediately readable by a subsequent Read call.
- The `initial_size` argument to `Factory::CreateUserDefinedIO` must match the existing
  backing store (use `0` for a fresh store).
- Throw on failure; partial writes are not rolled back.
- Each independent `user_defined_io` instance needs independent backing bytes; registering
  the same store under two names does not provide isolation.

## Example

```cpp
auto store = std::make_shared<ByteStore>();

auto read_func = [store](uint64_t offset, uint64_t len, void* dest) {
    std::lock_guard lock(store->mutex);
    if (offset > store->bytes.size() || len > store->bytes.size() - offset)
        throw std::runtime_error("read out of bounds");
    if (len > 0)
        std::memcpy(dest, store->bytes.data() + offset, len);
    ++store->reads;
};

auto write_func = [store](uint64_t offset, uint64_t len, const void* src) {
    std::lock_guard lock(store->mutex);
    if (offset > UINT64_MAX - len)
        throw std::runtime_error("write overflow");
    if (offset + len > store->bytes.size())
        store->bytes.resize(offset + len);
    if (len > 0)
        std::memcpy(store->bytes.data() + offset, src, len);
    ++store->writes;
};

auto resize_func = [store](uint64_t size) {
    std::lock_guard lock(store->mutex);
    store->bytes.resize(size);
    ++store->resizes;
};

auto storage = vsag::Factory::CreateUserDefinedIO(read_func, write_func, resize_func, 0);

vsag::UserDefinedIOSet storages;
storages.Set("hgraph_precise", storage.reader, storage.writer);

auto index = engine.CreateIndex("hgraph", R"({
    "dtype": "float32", "metric_type": "l2", "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 26, "ef_construction": 100,
        "use_reorder": true,
        "precise_quantization_type": "fp32",
        "precise_io_type": "user_defined_io",
        "precise_user_defined_io": "hgraph_precise"
    }
})", storages).value();

index->Build(base);
auto result = index->KnnSearch(query, 10, R"({"hgraph": {"ef_search": 50}})");
```

A runnable version is at `examples/cpp/200_user_defined_io.cpp`.

## Index parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `precise_io_type` | `"user_defined_io"` | Required |
| `precise_user_defined_io` | **string** | Must match a name in the `UserDefinedIOSet` passed to `CreateIndex`. Non-empty, unique. |
| `use_reorder` | `true` | Required |
| `precise_quantization_type` | `"fp32"` | Required |
| `base_quantization_type` | `"sq8"` | Required |
| `graph_io_type` | `"block_memory_io"` | Implicit default |
| `base_io_type` | `"block_memory_io"` | Implicit default |

## Limitations

- Graph, base, and bucket storage must remain in memory.
- No asynchronous writes, crash durability, transactions, or validated writable-storage
  restore.
- Realtime construction performance over a network is not measured; ODescent may issue
  many small reads.
- Callback backing residency is not counted in the backend memory estimate.

## See also

- [Extensibility](extensibility.md) — other VSAG extension points.
- [Example `200_user_defined_io`](https://github.com/antgroup/vsag/blob/main/examples/cpp/200_user_defined_io.cpp)
  (runnable).
- [`include/vsag/readerset.h`](https://github.com/antgroup/vsag/blob/main/include/vsag/readerset.h) —
  the `UserDefinedIOSet` API.
- [Disk-Based Index Best Practices](../resources/disk_index.md) — using `user_defined_io`
  as a disk-tiering backend.