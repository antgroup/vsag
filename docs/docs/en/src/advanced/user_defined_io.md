# User-Defined IO

User-defined IO (`user_defined_io`) lets callers inject their own read, write, and resize callbacks so that VSAG builds an index directly into external storage without an intermediate serialization step. This is useful when the precise codes must live in a managed store (e.g. a distributed filesystem, an object store, or a custom block device) that VSAG does not natively support.

## Supported configuration

- **Index**: HGraph with `use_reorder: true`.
- **Base codes**: `sq8` in `block_memory_io`.
- **Precise codes**: `fp32` in `user_defined_io`.
- **Graph**: `block_memory_io`.

Other graph / base / bucket storage combinations are not supported in this version.

## What `precise_user_defined_io` means

`precise_io_type` selects the IO backend for **HGraph precise codes**. When set to `"user_defined_io"`, `precise_user_defined_io` is the **registered name** used to look up a Reader/Writer pair. It is not a file path, a connection address, or another IO type, and it does not create the backing store. The name is resolved in the `UserDefinedIOSet` supplied to this `CreateIndex` call; there is no process-wide registry.

Bind the pair in this order:

1. Provide a Reader and Writer accessing the same backing bytes. You can wrap synchronous
   callbacks with `Factory::CreateUserDefinedIO(read_func, write_func, resize_func, initial_size)`, or implement the Reader/Writer interfaces yourself. 2. Register the pair with `storages.Set("hgraph_precise", storage.reader, storage.writer)`. The application chooses this non-empty name; it must be unique **within the same set**. A name does not imply a separate on-disk file. 3. Set both `"precise_io_type": "user_defined_io"` and `"precise_user_defined_io": "hgraph_precise"` inside HGraph's `index_param`, then call `engine.CreateIndex("hgraph", parameters, storages)` (or the corresponding Factory overload). VSAG looks up the pair by name for construction writes and subsequent reads of precise codes.

The two names must **match exactly**. Registering `"hgraph_precise"` but specifying `"other_store"` causes index creation to fail with “user defined IO not found”. When `user_defined_io` is selected, a missing, non-string, or empty name is rejected; omitting the pair set also fails. Conversely, if `precise_io_type` does **not** select `user_defined_io`, pre-populating `precise_user_defined_io` does not activate custom storage: the chosen IO type is used instead. Registration and parameter selection are both necessary.

This setting affects only the `fp32` **precise codes**. The `sq8` base codes and graph remain in `block_memory_io`, and `use_reorder: true` is required. It cannot redirect the whole index or base/graph/bucket storage to custom IO. The constructed index retains shared ownership of its bound pair; changing or destroying the set later does not rebind that index to another name.

## Callback contract

Three synchronous callbacks must be provided:

| Callback | Signature | Responsibilities |
|----------|-----------|------------------|
| Read | `void(uint64_t offset, uint64_t len, void* dest)` | Copy `len` bytes from `offset` into `dest`. Throw on out-of-bounds. |
| Write | `void(uint64_t offset, uint64_t len, const void* src)` | Consume `len` bytes from `src` at `offset`. Grow the backing store if `offset + len` exceeds the current size. |
| Resize | `void(uint64_t new_size)` | Truncate or extend the backing store to `new_size` while preserving the retained prefix. |

Constraints:

- All three callbacks are serialized by a mutex and must not reenter the same adapter.
- Serialize writes and resize/shrink operations at the index/application level. Per-callback
  locking does not make a size-check followed by resize atomic. Custom Reader/Writer pairs must share backing bytes and provide compatible synchronization. Do not mutate the backing store outside the pair while the index is using it.
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
// "hgraph_precise" is a registry name, not a path; it must match the JSON value below.
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
| `precise_io_type` | `"user_defined_io"` | Select this IO type to enable custom precise-code storage; the name below has no effect otherwise. |
| `precise_user_defined_io` | **non-empty string** | Required when that IO type is selected; must exactly match a `UserDefinedIOSet::Set` name supplied to this `CreateIndex` call. Names cannot be duplicated within one set. |
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
