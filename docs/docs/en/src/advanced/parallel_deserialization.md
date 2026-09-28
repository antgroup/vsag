# Parallel Deserialization

Loading a large index is often I/O bound: the sequential `Deserialize` path reads and decompresses
the file with a single thread, so multi-core machines and high-bandwidth storage stay
underutilized. `ParallelDeserialize` restores an index from a
[chunked file](chunked_serialization.md) with one task per frame, driven by a caller-supplied
positioned reader and the thread pool bound to the index.

The parallel entry points are currently implemented for HGraph; other index types return an
unsupported-operation error.

## DeserializeReader

The caller supplies the data source. VSAG never opens the file itself, which keeps object storage,
distributed file systems, and local files behind one interface:

```cpp
class MyReader : public vsag::DeserializeReader {
public:
    uint64_t Size() const override { /* total size of the index data */ }

    void Read(uint64_t offset, uint64_t len, void* dest) override {
        /* positioned read; must throw on short reads and out-of-range offsets */
    }

    void ReadDecompressed(uint64_t offset, uint64_t compressed_size,
                          const std::function<void(std::istream&)>& consume) override {
        /* decompress the frame at [offset, offset + compressed_size) into a
           sequential stream and hand it to consume; required only for files
           written through a compressing writer */
    }
};
```

The interface has two levels. A plain source implements `Size` and `Read` only and can load
uncompressed chunked files; a source that additionally implements `ReadDecompressed` can load
compressed ones.

`Read` and `ReadDecompressed` are invoked concurrently from worker threads on non-overlapping
ranges, so an implementation must be **thread-safe for positioned reads**. A handle whose reads
share a cursor, or serialize on an internal lock, silently degrades the load to a single I/O
stream even when the pool is large; prefer a per-call positioned read (`pread`-style) or a pool of
handles. Each decompressed stream is consumed by exactly one thread.

## Loading an Index

```cpp
MyReader reader;                   // wraps a file / object-storage handle
auto index = vsag::Factory::CreateIndex("hgraph", build_params).value();

index->ParallelDeserialize(reader).value();        // internal default pool
```

To run the load on your own executor, bind the pool to the `Resource` the index is created from:

```cpp
auto allocator = vsag::Engine::CreateDefaultAllocator();
auto pool = vsag::Engine::CreateThreadPool(16).value();

vsag::Resource resource(allocator, pool);
vsag::Engine engine(&resource);
auto index = engine.CreateIndex("hgraph", build_params).value();

index->ParallelDeserialize(reader).value();        // runs on the bound pool
```

Binding a pool lets the caller reuse an existing executor and bound the concurrency of the whole
process. The pool must outlive the call.

## Load Phases

1. **Prepare (main thread).** Parse the footer and, for every chunked component, read its head and
   pre-allocate the io extent.
2. **Fill (thread pool).** One task per frame: decompress the frame when compressed and write it
   into the pre-allocated extent at its recorded offset. Tasks touch disjoint byte ranges, so no
   locking is required.
3. **Finalize (main thread).** Read the tails and run the same post-load setup as the sequential
   path.

Pre-allocating in phase 1 is what makes phase 2 lock-free: the extent never grows while workers
are writing. Before any task is dispatched, the recorded layout is validated against the body
extent: component names must be unique, every frame must lie inside the body without overflowing,
no two frames may overlap, and an uncompressed codec must record a physical size equal to the
logical size. Per-frame byte counts are validated against the layout and each whole component must
consume exactly its recorded logical size, so a corrupted or tampered frame fails with an error
instead of corrupting memory.

## Loading Files Without a Layout

An index written before the chunked form existed carries no `chunked_layout` in its footer, but it
can still be loaded concurrently as long as its body is uncompressed — including the all-in-one
output of `Serialize(ostream)`. `ParallelDeserialize` detects the missing layout and
switches to a probe path:

1. The main thread walks the body in serialization order. For every component that exposes the
   parallel hooks it reserves the io extent, records the extent's file offset, skips over the io
   bytes, and reads the tail; components without the hooks are deserialized in place.
2. The recorded extents are then filled by the pool, split into tasks of `DEFAULT_SERIALIZE_CHUNK_SIZE`
   bytes each.

The body must be uncompressed for this to work: without a recorded layout there is no way to locate
the individual frames of a compressed body, so only files whose body bytes are stored verbatim can
be probed.

Because the head and tail of every component are still read sequentially by the main thread, the
probe path scales less than the layout path, but it needs no re-serialization of existing files.

Peak memory is decided by the io type chosen for each component, not by whether a layout is
recorded: a `SkipDeserialize` io such as `reader_io` only records the extent and seeks past the
bytes on both paths, a file-backed io writes into its backing file, and only an in-memory io holds
the component on the heap. What the recorded layout buys is concurrency and support for compressed
frames.

## Concurrency and IO Notes

- **Allocator.** Workers may allocate through the index allocator; a custom allocator must be
  thread-safe.
- **Memory.** Frames are streamed, not accumulated: a task holds a bounded read window and a codec
  context rather than a whole chunk, so peak memory scales with the thread count instead of
  `chunk_size`.
- **Whole components** are handled by a single task each. For compressed files they are read
  forward-only, so a component's deserialization must not seek backwards inside its frame.
- **File-backed io.** In-memory, mmap, buffered, and asynchronous io types fill their large
  components concurrently. `reader_io` is excluded — its write path is a no-op — and its
  components fall back to whole frames loaded by one task each.
- **Thread count.** The benefit comes mainly from I/O concurrency. Because worker threads spend
  most of their time blocked in reads, sizing the pool after the storage bandwidth and the number
  of independent frames is usually more effective than matching the CPU core count.

## Performance

1M x 1024 vectors, `sq8` base with `fp32` reorder, 32 MiB frames, zstd, 4.6 GiB index file, loaded
from page cache on a 48-core machine. Speedup over the sequential `Deserialize`, whose wall-clock
baseline is given in the second column:

| io type | sequential | 4 threads | 8 threads | 16 threads |
| --- | --- | --- | --- | --- |
| block_memory_io | 12.53 s | 1.70x | 2.98x | 4.64x |
| memory_io | 13.19 s | 1.83x | 3.25x | 5.19x |
| mmap_io | 14.51 s | 1.75x | 3.05x | 3.50x |
| buffer_io | 11.83 s | 1.67x | 2.74x | 2.68x |

Recall is unaffected: every configuration above returns results identical to the in-memory index,
query by query.

The probe path gives a smaller but still useful speedup on files that predate the chunked form. The
same index written by `Serialize(ostream)` (5.0 GiB, uncompressed, block_memory_io) loads as
follows, with the single-argument `Deserialize(istream)` as the baseline:

| mode | wall time | speedup |
| --- | --- | --- |
| `Deserialize(istream)` | 7.42 s | 1.00x |
| `ParallelDeserialize`, 4 threads | 3.79 s | 1.96x |
| `ParallelDeserialize`, 8 threads | 3.06 s | 2.43x |
| `ParallelDeserialize`, 16 threads | 2.77 s | 2.67x |

Note that a single-threaded parallel load is *slower* than the sequential path. Each frame builds
its own codec session and issues its own positioned read, and with one worker that overhead has
nothing to overlap with. The crossover is around four threads.

## See Also

- [Chunked Serialization](chunked_serialization.md) — the file format this path consumes.
- [Serialization](serialization.md) — the sequential interfaces and the footer-based format.
