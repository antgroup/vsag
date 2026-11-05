# HGraph Build Cache

HGraph can export graph-neighbor information from one build and import it before a later
`Build()`. Vectors are matched by stable string source IDs, so unchanged data can warm-start from
the previous graph while new or changed data is refined normally.

This workflow is intended for recurring snapshot builds, such as rebuilding an index every day
from a mostly overlapping corpus. It is a build accelerator, not an index serialization format:
use the normal [Serialization](serialization.md) APIs to persist a searchable index.

## Requirements

- The index type must be HGraph.
- Every base dataset used in the workflow must set `Dataset::SourceID`.
- Source IDs must be stable and unique for the logical records you want to match across builds.
- Import the cache into a fresh, empty, compatible HGraph before calling `Build()`.
- Keep dimensions, metrics, and storage/quantization parameters compatible between builds.

The numeric label in `Dataset::Ids` may change between snapshots. Cache matching uses the
corresponding `SourceID` string instead.

## First build and cache export

Provide one source ID per vector when building the source index:

```cpp
std::vector<std::string> source_ids = load_stable_source_ids();

auto base = vsag::Dataset::Make();
base->NumElements(count)
    ->Dim(dim)
    ->Ids(ids.data())
    ->Float32Vectors(vectors.data())
    ->SourceID(source_ids.data())
    ->Owner(false);

auto index = vsag::Factory::CreateIndex("hgraph", build_params).value();
index->Build(base).value();

std::ofstream cache_out("hgraph.cache", std::ios::binary);
index->ExportCache(cache_out).value();
```

Keep the `std::string` array alive for the duration of `Build()`. The cache contains source-ID to
neighbor information used by a later build; it is not directly searchable.

## Warm-starting a later build

Create an empty compatible HGraph, import the previous cache, and then build with the new snapshot:

```cpp
auto next_index = vsag::Factory::CreateIndex("hgraph", build_params).value();

std::ifstream cache_in("hgraph.cache", std::ios::binary);
next_index->ImportCache(cache_in).value();

auto next_base = vsag::Dataset::Make();
next_base->NumElements(next_count)
    ->Dim(dim)
    ->Ids(next_ids.data())
    ->Float32Vectors(next_vectors.data())
    ->SourceID(next_source_ids.data())
    ->Owner(false);

next_index->Build(next_base).value();
```

`Build()` automatically takes the cache-assisted path after `ImportCache()`. Source IDs found in
both snapshots are warm-started from cached neighbors; unmatched records are treated as cache
misses and refined by the normal build path. Calling cache-assisted `Build()` without
`Dataset::SourceID` returns an invalid-argument error.

## Persisting source IDs with the index

Source-ID metadata is not included in HGraph serialization by default. Set
`persist_source_id: true` in `index_param` when a deserialized index must later call
`ExportCache()`:

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 32,
        "ef_construction": 400,
        "persist_source_id": true
    }
}
```

This option adds source-ID metadata to the serialized index. It is unnecessary if cache export
always happens from the original in-memory build and the restored index never needs the source
mapping.

## Measuring cache reuse

After the warm-started build, call `GetStats()` and inspect:

| Field | Meaning |
| --- | --- |
| `build_cache_hit_rate` | Fraction of nodes warm-started from the imported cache |
| `build_cache_hit_nodes` | Matched node count |
| `build_cache_missed_nodes` | Nodes built without a matching cache entry |

When no imported cache participated in the last build, the statistics include a
`skipped_reason` instead. See [Index Analysis](../resources/analyze_index.md) for the rest of the
HGraph statistics.

## Limitations and operational guidance

- Import before `Build()`; cache-assisted build requires an empty index.
- A build cache is tied to compatible HGraph configuration and should be regenerated after
  incompatible parameter changes.
- `deduplicate_storage: true` cannot be combined with cache-assisted build.
- Validate recall and build time against a full rebuild before adopting the cache in production.
- Check stream-open and API return values. A cache file is separate from the searchable serialized
  index and should be versioned or replaced atomically with the snapshot that produced it.

API signatures are listed under [Index Cache](../api/index_class.md#cache-build-acceleration), and
the source-ID dataset field is documented on the [Dataset](../api/dataset.md) page.
