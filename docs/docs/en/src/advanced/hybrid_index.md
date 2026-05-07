# Memory + Disk Hybrid Index (DiskANN)

For billion-scale vector datasets, fitting the full graph index in memory is expensive and
wasteful. VSAG's `diskann` index splits storage:

- **Compressed vectors (PQ)** are kept in memory for fast pruning.
- **Full-precision vectors** and the **graph structure** live on disk and are fetched
  asynchronously along the search path.

This lets a single machine serve billion-scale nearest-neighbor queries under a limited memory
budget.

## Building DiskANN

```cpp
std::string build_params = R"(
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "diskann": {
        "max_degree": 32,
        "ef_construction": 400,
        "pq_sample_rate": 0.1,
        "pq_dims": 32,
        "use_async_io": true
    }
}
)";
auto index = vsag::Factory::CreateIndex("diskann", build_params).value();
index->Build(dataset);
```

Complete example: `examples/cpp/102_index_diskann.cpp`.

## Asynchronous IO (libaio)

On Linux, set `use_async_io` in the build parameters to dispatch concurrent reads through libaio.
This requires compiling with `VSAG_ENABLE_LIBAIO=ON` (see [Building](../development/building.md)).

## File Layout

`diskann` produces two file kinds on disk:

- `*.index` — the graph structure.
- `*.data` — the full-precision vectors.

Both files must be reachable at deserialization time.

## Notes

- Prefer NVMe SSDs; on HDDs query latency degrades dramatically.
- The compression ratio and accuracy of the in-memory PQ depend on `pq_dims`; setting it too low
  hurts recall.
- Warm up the index files on cold start (read a few MB at random) to populate the page cache.
- DiskANN does not currently support online insert/delete; rebuild the index when updates are
  needed.
