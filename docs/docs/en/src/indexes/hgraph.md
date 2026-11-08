# HGraph

HGraph is VSAG's flagship **graph-based** index. It builds a hierarchical proximity graph
and offers a rich set of quantization options, a unified
build-parameter schema (`index_param`), and first-class support for reordering,
incremental updates, deletion, and ELP-based runtime tuning.

For most dense-vector workloads (text / image / multimodal embeddings, 64–4096 dims,
from a few thousand up to hundreds of millions of points), HGraph is the recommended
default.

- Source: `src/algorithm/hgraph.{h,cpp}`
- Example: [`examples/cpp/103_index_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/103_index_hgraph.cpp)

## How it works

1. **Graph construction.** Vectors are organised in a layered proximity graph; upper
   layers act as navigation aids, the bottom layer connects every data point to its
   nearest neighbours within a `max_degree` budget. The construction algorithm can be
   NSW-style insertion (`graph_type: "nsw"`, the default), ODescent
   (`graph_type: "odescent"`), or PiPNN (`graph_type: "pipnn"`).
2. **Quantization.** The base storage is compressed with a configurable quantizer
   (`base_quantization_type` — `fp32`, `fp16`, `bf16`, `sq8`, `sq8_local`, `sq4_local`, `sq4`, `sq8_uniform`, `sq4_uniform`,
   `pq`, `pqfs`, `rabitq`, `saq`, `tq`). Optionally, a second high-precision copy is kept
   (`use_reorder: true` with `precise_quantization_type`) and used to re-rank the
   candidates returned by the coarse search.
3. **Search.** Greedy beam search traverses the graph top-down, expanding the current
   frontier up to `ef_search` candidates. When reordering is enabled, the final list is
   re-scored against the precise representation.

## Quick start

```cpp
#include <vsag/vsag.h>

std::string params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 32,
        "ef_construction": 400
    }
})";
auto index = vsag::Factory::CreateIndex("hgraph", params).value();

// Build.
auto base = vsag::Dataset::Make();
base->NumElements(n)->Dim(128)->Ids(ids)->Float32Vectors(data)->Owner(false);
index->Build(base);

// Search.
auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(128)->Float32Vectors(q)->Owner(false);
auto result = index->KnnSearch(
    query, /*topk=*/10, R"({"hgraph": {"ef_search": 100}})").value();
```

## Disk read-cache example

[`406_feature_read_cache.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/406_feature_read_cache.cpp)
uses 1,000 64-dimensional vectors, in-memory SQ8 base codes, and disk-backed FP32
precise codes (`async_io`) for query reranking. It sets `build_by_base: true` so
graph construction reads the in-memory base codes, avoiding repeated disk-cache
refills during this small usage example. This changes the graph construction
quantization; it is not a recall benchmark or an example of precise-code graph
construction.

Read caching remains enabled with a 131,072-byte budget (one 128-KiB page).
The 256,000-byte FP32 payload spans two pages. The example serializes to a
`BinarySet`, restores a new index, and checks the restored count, top-10 result
shape, ID range, and finite distances. Run it with a writable file path in a
fresh working directory. Writes and durability behavior are unchanged.
Focused AsyncIO tests cover one-page eviction/refill, write invalidation, and
invalid reads; the HGraph read-cache functional test retains precise-code graph
construction coverage.

## Build parameters

Build-time parameters live under `index_param`. The table below is the complete list of public
build keys accepted by HGraph; [Index Parameters](../resources/index_parameters.md) additionally
records cross-index defaults and the top-level common fields.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_quantization_type` | string | `"fp32"` | `fp32`, `fp16`, `bf16`, `sq8`, `sq8_local`, `sq4_local`, `sq4`, `sq8_uniform`, `sq4_uniform`, `pq`, `pqfs`, `rabitq`, `saq`, `tq` — see the [Quantization chapter](../quantization/) for per-quantizer details |
| `max_degree` | int | `64` | Maximum out-degree per graph node |
| `ef_construction` | int | `400` | Candidate list size during build (higher = better recall, slower build) |
| `alpha` | float | `1.0` | Final robust-pruning factor; PiPNN requires a finite value at least `1.0`. |
| `graph_type` | string | `"nsw"` | Graph algorithm: `nsw`, `odescent`, or `pipnn` |
| `pipnn_max_leaf_size` | int | `1024` | Maximum partition leaf size. Larger leaves increase candidate coverage and leaf distance work. |
| `pipnn_min_leaf_size` | int | `64` | Target used when merging undersized leaves. |
| `pipnn_leader_sample_rate` | float | `0.005` | Fraction sampled as partition leaders; each partition uses at least `2` and at most `1000` leaders, bounded by its point count. |
| `pipnn_fanout` | int[] | `[10, 2]` | Number of nearest leader partitions joined at each listed partition level; deeper levels use `1`. |
| `pipnn_leaf_neighbor_count` | int | `5` | Nearest candidates contributed per point in each leaf. This is the primary PiPNN quality/build-work knob; leaves smaller than `pipnn_min_leaf_size` use at least `4`. |
| `pipnn_hash_plane_count` | int | `12` | Direction-hash bits, in `[1, 15]`; `max_degree <= 2^pipnn_hash_plane_count`. |
| `pipnn_reservoir_size` | int | `64` | Candidate slots retained per point before final pruning; effective capacity is at least `max_degree`. |
| `pipnn_adaptive_pruning` | bool | `false` | Experimental per-point alpha adjustment in PiPNN final pruning; L2 only. |
| `pipnn_adaptive_pruning_adjust_step` | float | `0.06` | Alpha adjustment step; finite, non-negative, `alpha - 2*step > 0`, and `alpha + 3*step` finite when enabled. |
| `use_reverse_edges` | bool | `false` | Track incoming neighbors for O(1) reverse-edge lookup. Roughly doubles edge storage and is unsupported with `graph_storage_type: "compressed"`. |
| `label_remap_type` | string | `"pg"` | Label-to-inner-ID map implementation: `"pg"` or `"robin"`. Keep the same value when restoring or combining compatible indexes. |
| `use_reorder` | bool | `false` | Keep a high-precision copy and re-rank after the coarse search |
| `reorder_source` | string | `"precise"` | Reorder from `"precise"` codes or directly from `"base"` codes. RaBitQ x+y split, including `tq_chain: "mrle, rabitq"`, sets `"base"` automatically. |
| `precise_quantization_type` | string | `"fp32"` | Quantizer used for reordering (takes effect only with `use_reorder: true`) |
| `base_pq_dim` | int | `1` | Number of PQ subspaces. When using `pq` / `pqfs`, set this explicitly instead of relying on the default. Must not exceed `dim` in a way the quantizer rejects. |
| `mrle_dim` | int | `0` | Output dimension for an MRLE transform in `tq_chain`; allowed range `[0, dim]`, where `0` means the input dimension. |
| `fast_encode_rabitq` | bool | `true` | Use the fast multi-bit RaBitQ encoder; set to `false` for the previous exact encoder. |
| `fast_encode_rabitq_rounds` | int | `6` | Fast RaBitQ coordinate-refinement rounds, in `[1, 32]`. |
| `saq_avg_bits` | float | `4.0` | SAQ average record bits per dimension, in `[1, 8]`. |
| `saq_segment_count` | int | `0` | SAQ segment count; `0` selects boundaries dynamically. |
| `saq_random_rotation` | bool | `true` | Apply an independent orthogonal rotation inside each SAQ segment. |
| `rabitq_fused_datacell` | bool | `false` | Fuse the bottom HGraph node and RaBitQ split codes into one in-memory record. Requires L2/IP, flat in-memory graph storage, RaBitQ x+y split codes with x in `[1, 4]`, and the other constraints described in [RaBitQ x+y split](../quantization/rabitq_split.md). |
| `train_sample_count` | int | `65536` | Maximum number of vectors sampled for quantizer training; must be at least `512` when set explicitly. |
| `build_thread_count` | int | `100` | Threads used to parallelise build |
| `support_duplicate` | bool | `false` | Group duplicate vectors while retaining their individual labels |
| `deduplicate_storage` | bool | `false` | Share vector storage between duplicates; requires `support_duplicate: true` |
| `duplicate_distance_threshold` | float | `0.0` | Duplicate-detection distance threshold. When greater than `0`, deduplicate by the nearest candidate distance; when `0`, compare codes with the nearest candidate first, then check the remaining returned build candidates until a match is found |
| `support_remove` | bool | `false` | Enable graph delete-tracking metadata used by mark-remove recovery paths |
| `support_force_remove` | bool | `false` | Enable `RemoveMode::FORCE_REMOVE` and its extra synchronization on the built index |
| `store_raw_vector` | bool | `false` | Keep the raw vector in addition to the quantized copy (useful for `cosine`) |
| `use_elp_optimizer` | bool | `false` | Auto-tune search parameters after build |
| `base_io_type` / `precise_io_type` | string | `"block_memory_io"` | Storage backend (`memory_io`, `block_memory_io`, `buffer_io`, `async_io`, `uring_io`, `mmap_io`) |
| `base_file_path` / `precise_file_path` | string | `"./default_file_path"` | File path used when the corresponding `*_io_type` is disk-backed (`buffer_io`, `async_io`, `uring_io`, `mmap_io`). The parser never requires it, but two indexes sharing the default path will collide, so give disk-backed indexes distinct paths. |
| `base_direct_read` / `precise_direct_read` | bool | `false` | With `uring_io`, open the corresponding file using direct IO instead of the page cache. |
| `hgraph_init_capacity` | int | `100` | Initial capacity hint (doesn't cap the final size) |
| `persist_source_id` | bool | `false` | Persist source-ID metadata during serialization so a restored index can later export a reusable build cache. |
| `use_conjugate_graph` | bool | `false` | Enable `Feedback`/`Pretrain` graph enhancement; see [Graph Index Enhancement](../advanced/enhance_graph.md). |
| `resize_increase_count_bit` | int | `10` | `log2` of the slot-growth batch. Valid range is `1` to `31`; `1` grows in 2-slot batches and `10` in 1,024-slot batches. Smaller values reduce preallocation but can increase reallocations. |
| `graph_storage_type` | string | `"flat"` | Graph storage layout: `flat` or `compressed`. `compressed` is unsupported together with `use_reverse_edges`. |
| `graph_iter_turn` | int | `30` | ODescent only: graph iteration rounds. |
| `neighbor_sample_rate` | float | `0.2` | ODescent only: neighbour-sampling rate used while refining the graph. |
| `min_in_degree` | int | `1` | ODescent only: minimum in-degree enforced while building. |
| `build_block_size` | int | `10000` | ODescent only: number of points processed per build block. |
| `build_by_base` | bool | `false` | Build the graph from base codes instead of precise codes when precise reorder storage exists. Pair with `use_reorder: true` and `reorder_source: "base"` to trade graph quality for memory. |
| `ignore_reorder` | bool | `false` | Keep the precise codes in storage but never read them at query time. The effective reorder flag is `use_reorder && !ignore_reorder`, and `use_reorder`/`ignore_reorder` must agree at compatibility-check time. |
| `use_attribute_filter` | bool | `false` | Build the attribute inverted index so `extra_info` predicates can be pushed down into the search. Configure posting-list layout through the index defaults; `attr_params` is **not** a public key. |
| `graph_io_type` | string | `"block_memory_io"` | Storage backend for the graph structure (`memory_io`, `block_memory_io`, `buffer_io`, `async_io`, `uring_io`, `mmap_io`). |
| `graph_file_path` | string | `"./default_file_path"` | File path used when `graph_io_type` is disk-backed. |
| `raw_vector_io_type` | string | `"block_memory_io"` | Storage backend for the retained raw vectors (`store_raw_vector: true`). |
| `raw_vector_file_path` | string | `"./default_file_path"` | File path used when `raw_vector_io_type` is disk-backed. |
| `base_supplement_io_type` | string | unset | IO backend for the RaBitQ split supplement codes. Only meaningful with split base codes: the split data cell accepts the same backend for both halves (`block_memory_io`, `memory_io`, `buffer_io`, `async_io`, `mmap_io`; `uring_io` is never supported) or the hybrid base `block_memory_io` + supplement `async_io` (`buffer_io` when libaio is unavailable). `rabitq_fused_datacell` additionally restricts both halves to `block_memory_io` / `memory_io`. |
| `base_supplement_file_path` | string | derived | File path for the split supplement backend; defaults to the base path with a `_supplement` suffix. |
| `precise_user_defined_io` | string | unset | Registered user-defined IO class name. Required only when `precise_io_type: "user_defined_io"`, which also demands `use_reorder: true`, precise `fp32`, base `sq8` codes and `block_memory_io` base plus graph storage. |
| `base_enable_read_cache` | bool | `false` | Enable the block read cache for base codes. Only meaningful with a block-based disk backend. |
| `precise_enable_read_cache` | bool | `false` | Enable the block read cache for precise codes. |
| `graph_enable_read_cache` | bool | `false` | Enable the block read cache for graph data. |
| `raw_vector_enable_read_cache` | bool | `false` | Enable the block read cache for retained raw vectors. |
| `base_cache_total_size` | uint64 | `268435456` | Read-cache budget in bytes for base codes (256 MiB). |
| `precise_cache_total_size` | uint64 | `268435456` | Read-cache budget in bytes for precise codes. |
| `graph_cache_total_size` | uint64 | `268435456` | Read-cache budget in bytes for graph data. |
| `raw_vector_cache_total_size` | uint64 | `268435456` | Read-cache budget in bytes for retained raw vectors. |
| `tq_chain` | string | unset | Transform chain, for example `"mrle, rabitq"`. Only valid when `base_quantization_type: "tq"`; setting it with any other base quantizer is rejected. |
| `rabitq_pca_dim` | int | `0` | RaBitQ PCA output dimension; `0` disables PCA. Use `dim`-compatible values only. |
| `rabitq_bits_per_dim_query` | int | `32` | RaBitQ query bits per dimension. Must be `4` or `32`, and split storage requires `32`. |
| `rabitq_bits_per_dim_base` | int | `1` | RaBitQ base bits per dimension, in `[1, 8]`. With split storage this is the `x` of `x + y`. |
| `rabitq_bits_per_dim_precise` | int | unset | RaBitQ precise bits per dimension, the `y` of a split. Requires `x + y <= 8` plus reorder with precise RaBitQ codes. |
| `rabitq_error_rate` | float | `1.9` | Finite positive lower-bound error multiplier baked into the built RaBitQ codes. Search-level `rabitq_error_rate` overrides it per query. |
| `rabitq_use_fht` | bool | `false` | Use the fast Hadamard transform inside RaBitQ. |
| `sq4_uniform_trunc_rate` | float | `0.05` | Truncation rate for the `sq4_uniform` base quantizer; must be finite and in `[0, 0.5]`. |
| `remove_flag_bit` | uint32 | `8` | Bit position of the delete-tracking flag in the graph record. Consistent with `support_remove` metadata; changing it on an existing index breaks compatibility. |
| `use_mci` | bool | `false` | Enable Multi-Cluster Indexing. Setting any `mci_*` build key also turns MCI on. |
| `mci_mcs` | int | `200` | MCI target cluster size; must be positive. |
| `mci_clique_max` | int | `50` | MCI clique size floor used by the full-clique phase — a minimum, not a cap on clique size. |
| `mci_alpha` | float | `1.2` | MCI pruning alpha; at least `1.0`. |
| `mci_knng_source` | string | `"hgraph"` | k-NNG source graph for MCI: `"hgraph"` or `"odescent"`. |
| `mci_incremental_join_ratio_threshold` | float | `0.6` | Join-ratio threshold for the incremental MCI phase, in `[0, 1]`. |
| `mci_incremental_degree_min` | int | `50` | Minimum degree in the incremental MCI phase; must be positive. |
| `mci_incremental_degree_n_divisor` | int | `10000` | Divisor applied to point count when sizing incremental degrees; must be positive. |
| `mci_incremental_degree_mcs_divisor` | int | `2` | Divisor applied to MCI cluster size when sizing incremental degrees; must be positive. |
| `mci_incremental_clique_max` | int | `50` | Clique ceiling in the incremental MCI phase; integer at least `2`. |
| `mci_delete_clique_size_threshold` | int | `30` | Clique size below which MCI delete bookkeeping triggers; must be positive. |
| `mci_delete_node_mct_threshold` | int | `3` | Node MCT threshold for MCI delete bookkeeping; must be positive. |
| `mci_clique_activation_order` | bool | `false` | Visit MCI cliques in activation order instead of index order during build. |

Keys that look related but are **not** public HGraph build parameters: `attr_params`
(an internal object built from index defaults), `fill_rejected` /
`adaptive_pruning_fill_rejected` (accepted by the mapper but rejected downstream by the
adaptive-pruning parser), `mci` (rejected marker), `mci_incremental_added_mct` (removed old
MCI key), `mci_knng_path` (internal only), `precise_enable_prefetch_hint`, `rabitq_version`
and `rabitq_bits_per_dim_filter` (derived by the split helpers). Passing any of them raises
`invalid config param` or a dedicated argument error.

`sq8_local` and `sq4_local` support float32/float16/bfloat16 input with L2/IP/cosine base and precise stores (including RaBitQ + local SQ4/SQ8 reranking). See [layout, numeric limits and Tune compatibility](../quantization/sq.md#local-sq4--sq8-float32-l2-ip-and-cosine).

`use_reverse_edges` is intended for workloads that need fast incoming-neighbor inspection, graph
analysis, or future graph-maintenance algorithms. It is disabled by default because maintaining
the reverse adjacency approximately doubles edge storage.

`label_remap_type` changes the internal label map, not user-visible IDs. `"pg"` is the default;
`"robin"` selects the alternate robin-map implementation. Benchmark the target ID distribution
before changing it.

### PiPNN build boundary

Set `graph_type: "pipnn"` to use [PiPNN](https://arxiv.org/abs/2602.21247) for the initial,
full `Build`. The PiPNN builder accepts dense `dtype: "float32"`, `"float16"`, or `"bfloat16"`
input with `metric_type` set to `"l2"`, `"ip"`, or `"cosine"`; sparse and `int8` inputs are not
supported. For FP16/BF16 input, it borrows the full input rows in their original half format
rather than materializing a full FP32 input copy. Bounded, local partition/leaf operands are
converted to FP32 for BLAS matrix operations, with FP32 accumulation; this is not a native
half-precision matrix backend. Final vector-storage quantization is configured independently
of the input dtype, subject to the index and quantizer support restrictions.
Quantizer training is separate from PiPNN graph construction: training may still materialize the full selected training sample in an FP32 buffer (potentially the full dataset when all rows are selected). Borrowing half input therefore does not guarantee that the complete build avoids a full-sample FP32 buffer. Validation of these half-input paths is in progress; no performance or non-regression claim is made.
It builds the bottom graph from the original build vectors, then reuses
HGraph's route layers, storage, search, filtering, reordering, incremental `Add`, removal, and
serialization paths. The persistent base storage may use a supported quantizer such as `sq8`,
including RaBitQ with SQ8 reorder. Cache-assisted build is not supported with PiPNN.
The `pipnn_*` parameters in the table above tune this builder;
`ef_construction` does not. The existing `alpha` parameter controls PiPNN's final robust pruning.

`build_thread_count` parallelizes vector preparation, partitioning, candidate generation, and
final pruning. Keep `OPENBLAS_NUM_THREADS=1` when benchmarking so BLAS threads do not obscure
builder scaling. The reproducible `tools/eval/pipnn_parallel.yaml` workload compares NSW,
ODescent, and PiPNN at a matched Recall@10 target on SIFT1M and query-validates every graph.


### Experimental PiPNN adaptive alpha

Set `pipnn_adaptive_pruning: true` under `index_param` to apply [PR #2933](https://github.com/antgroup/vsag/pull/2933)'s per-node count-based schedule to the final PiPNN reservoir pruning. This opt-in experiment supports L2 and affects the initial batch `Build` only. It leaves partitioning, leaf neighbors, relative hashes, reservoir contents, incremental `Add`, and ODescent route-layer construction unchanged.

For target degree K, scan the existing ordered reservoir at the configured `alpha`, yielding accepted A and scanned rejects B. Unscanned tail candidates do not count toward B. With a nonzero step, if `0 < |A| < K`, keep A and retry B with alpha increased by 1, 2, or 3 steps for `K/|A| <= 1.5`, `(1.5, 3]`, or `> 3`. If `|A| = K`, clear A and rescan the complete original reservoir, including the tail, with alpha decreased by 2, 1, or 0 steps for `|B|/K < 2.5`, `[2.5, 5)`, or `>= 5`. If the tightened pass is short, keep its accepted neighbors and retry its new rejects at the original alpha. Rejected neighbors are never appended unconditionally.

The strict rejection rule remains `alpha * d(selected, candidate) < d(source, candidate)`. PiPNN retains its BF16 source-distance cache, FP32 pairwise squared-L2 calculations, and `(distance, source-ID gap, ID)` order. Unlike the NSW selector's disabled fast path, PiPNN already prunes short reservoirs, so zero adjustment step reproduces its original single-pass adjacency. Greedy selection and the degree cap mean recall need not improve monotonically with alpha.

The policy and step are serialized. Loading an enabled index requires matching policy, base alpha, and step. Missing fields mean disabled. The disabled path preserves legacy behavior. This is experimental: compare build cost and query throughput at matched recall.

### Deduplicating vector storage

Set both `support_duplicate: true` and `deduplicate_storage: true` to let duplicate
vectors share one physical code slot while retaining their individual labels. This option
supports dense-vector HGraph indexes using `graph_type: "nsw"` or `"pipnn"`;
`graph_type: "odescent"` is not supported.

For PiPNN, initial `Build` groups byte-identical FP32 input rows and encodes each
representative once. All labels, attributes, and extra information remain independent.
This also works with adaptive alpha. `duplicate_distance_threshold` does not affect
PiPNN batch grouping; later incremental `Add` retains its existing threshold/code-based
duplicate detection. Quantizer training still uses the original training sample.

The following operations and configurations are not supported while storage deduplication
is enabled:

- force removal (`support_force_remove: true`);
- cache-assisted build after `ImportCache()`;
- `Merge`;
- legacy v0.14 serialization.

`UpdateVector` is supported only for IDs whose vector storage is not shared with another
duplicate-group member.

Current serialization and streaming serialization are supported.

## Supported input data types

The `dtype` field in the top-level build config selects how `Dataset` interprets the raw vector
bytes. HGraph supports four input types; the `dtype` value, the corresponding `Dataset` setter,
and the example demonstrating each combination are summarised below.

| `dtype`     | Element type | `Dataset` setter         | Example                                                                                                |
|-------------|--------------|--------------------------|--------------------------------------------------------------------------------------------------------|
| `float32`   | `float`      | `Float32Vectors`         | [`103_index_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/103_index_hgraph.cpp) |
| `int8`      | `int8_t`     | `Int8Vectors`            | [`316_index_int8_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/316_index_int8_hgraph.cpp) |
| `float16`   | `uint16_t` (IEEE 754 binary16, bit-pattern packed) | `Float16Vectors` | [`321_index_fp16_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/321_index_fp16_hgraph.cpp) |
| `bfloat16`  | `uint16_t` (Brain Float, bit-pattern packed) | `Float16Vectors` (shared with FP16) | adapt `321_index_fp16_hgraph.cpp` per the notes below                                                  |

The `dim` value is the logical vector dimensionality (number of elements), not the byte length, so
the same `dim` is reused across all four data types.

### `int8` input

Quantized `int8` vectors are passed directly via `Int8Vectors`:

```cpp
std::vector<int8_t> data(num_vectors * dim);  // populate with int8 elements
auto base = vsag::Dataset::Make();
base->NumElements(num_vectors)->Dim(dim)->Ids(ids)
    ->Int8Vectors(data.data())->Owner(false);
```

Build config (note `dtype: "int8"`):

```json
{
    "dtype": "int8",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "pq",
        "max_degree": 26,
        "ef_construction": 100,
        "alpha": 1.2
    }
}
```

Queries use the same `Int8Vectors` setter and the same `dtype`. A runnable example is
[`316_index_int8_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/316_index_int8_hgraph.cpp).

### `float16` / `bfloat16` input

FP16 and BF16 vectors are both passed through `Float16Vectors`, which takes a `const uint16_t*`
that points at the 16-bit storage of each element. Conversion from `float` is up to the caller;
inside the VSAG source tree there are convenience helpers (`vsag::generic::FloatToFP16` in
[`src/simd/fp16_simd.h`](https://github.com/antgroup/vsag/blob/main/src/simd/fp16_simd.h)
and `vsag::generic::FloatToBF16` in
[`src/simd/bf16_simd.h`](https://github.com/antgroup/vsag/blob/main/src/simd/bf16_simd.h)),
but these are **internal headers** that are not installed under `include/vsag/`. Application code
linking against an installed VSAG library should provide its own conversion (for example, copy
the small helper, use `_cvtss_sh` / F16C intrinsics, or any FP16 library of choice). The snippet
below uses the in-tree helper for brevity:

```cpp
// The fp16/bf16 helpers below live in src/simd/ and are not part of the public
// installed headers. Replace with your own float -> uint16_t conversion when
// linking against an installed VSAG.
#include "simd/fp16_simd.h"  // FloatToFP16 (for BF16, use simd/bf16_simd.h / FloatToBF16)

std::vector<uint16_t> data(num_vectors * dim);
for (size_t i = 0; i < data.size(); ++i) {
    data[i] = vsag::generic::FloatToFP16(some_float_source());
}
auto base = vsag::Dataset::Make();
base->NumElements(num_vectors)->Dim(dim)->Ids(ids)
    ->Float16Vectors(data.data())->Owner(false);
```

Build config:

```json
{
    "dtype": "float16",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "pq",
        "max_degree": 26,
        "ef_construction": 100,
        "alpha": 1.2
    }
}
```

To switch the example to BF16, change `dtype` to `"bfloat16"` and replace `FloatToFP16` with
`FloatToBF16`; the `Float16Vectors` setter and the rest of the build/search flow stay the same.
A runnable FP16 example is
[`321_index_fp16_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/321_index_fp16_hgraph.cpp).

> **Note.** The header comment at the top of `321_index_fp16_hgraph.cpp` currently mentions a
> `BFloat16Vectors()` setter, but no such setter exists — `Float16Vectors` is the single entry
> point for both FP16 and BF16. Use it for both `dtype: "float16"` and `dtype: "bfloat16"`.

FP16/BF16 input supports the same originally legal FP32 storage-quantizer configurations, not a separate limited quantizer list: `fp32`, `fp16`, `bf16`, ordinary `sq4`/`sq8`, `sq4_uniform`/`sq8_uniform`, `sq4_local`/`sq8_local`, `pq`, `pqfs`, `rabitq`, and eligible `tq` chains. This applies to base storage and, when enabled, precise storage; it does not make every base/precise combination legal. Existing metric, dimension, training, graph, and IO restrictions still apply. Local SQ4/SQ8 remain prohibited inside TQ chains. Split RaBitQ supports 3+5 bits, including MRLE; a split TQ chain is restricted to `"mrle, rabitq"`. HGraph fused RaBitQ remains L2/IP-only, requires flat in-memory graph storage and its existing IO constraints, and does not support TQ. Half input does not relax any of these restrictions.

### Choosing an input type

- Pick `float32` when accuracy matters most and memory budget allows; this is the default.
- Pick `float16` / `bfloat16` to halve the input storage. FP16 has a smaller exponent range; BF16
  has fewer mantissa bits but the same exponent range as FP32, which is often preferable for
  embedding-style vectors.
- Pick `int8` when your data is already integer-quantised (e.g. produced by an upstream quantiser
  or by a model with int8 outputs). With `int8` input you typically still combine a coarse
  quantizer such as `pq` / `sq8` for the in-index storage.

The chosen `dtype` only constrains the **input** representation. The on-disk / in-memory storage is
still controlled by `base_quantization_type` (and optionally `precise_quantization_type` when
`use_reorder: true`), so e.g. `dtype: "float16"` + `base_quantization_type: "sq8"` is valid.

## Experimental adaptive pruning

See [algorithm details, examples and related work](adaptive_pruning.md).

Parameters are flat fields under `index_param` and `adaptive_pruning` is a boolean; the former
experimental nested object and the removed `fill_rejected` option are rejected. Indexes created
with that experimental schema must be rebuilt.

`index_param.adaptive_pruning` enables a multi-pass L2 neighbor selector for NSW bottom-layer
**new-node forward edges**. It is disabled by default. Ordinary `Build`, parallel insertion,
`Add`, and the NSW optimized build path (including RaBitQ split/fused) use this selector.
Set `adaptive_pruning_apply_to_reverse: true` to also prune full bottom-layer reverse neighbor lists adaptively.
A reverse list with free slots still appends the new edge directly. When full, the existing
neighbor is the selection center and its old neighbors plus the new node form the candidates;
the same alpha/step policy applies under the existing neighbor lock. Upper layers and
update/refinement operations retain their fixed-alpha policy. Enabled ODescent, imported-cache
builds, inner product/cosine, and `adaptive_pruning_apply_to_upper: true` remain unsupported.

The following fields belong inside `index_param`. This is a GIST reference experiment,
not a dimension-dependent default or a claim of improved performance:

```json
{
    "alpha": 1.06,
    "graph_type": "nsw",
    "adaptive_pruning": true,
    "adaptive_pruning_adjust_step": 0.06,
    "adaptive_pruning_apply_to_reverse": false,
    "adaptive_pruning_apply_to_upper": false
}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `adaptive_pruning` | `false` | Enable the adaptive selector |
| `adaptive_pruning_adjust_step` | `0.06` | Alpha adjustment step, delta |
| `adaptive_pruning_apply_to_reverse` | `false` | Also use adaptive pruning for full bottom-layer reverse lists; requires `adaptive_pruning: true` |
| `adaptive_pruning_apply_to_upper` | `false` | Reserved; `true` is unsupported when enabled |

The existing `alpha` is the baseline. Enabled mode requires finite alpha and delta,
nonnegative delta, positive `alpha - 2*delta`, and finite `alpha + 3*delta`.
Distances must be finite nonnegative L2 values from the existing build distance provider;
this setting does not change the build quantizer.

Candidates are sorted by `(distance, internal ID)`, deduplicated, and stripped of self edges.
A candidate c is rejected if any selected s satisfies `alpha * d(s,c) < d(center,c)`.
The first pass stops at the graph's actual maximum degree K, retaining the unscanned tail.

- If accepted A is below K, keep A and rescan rejects with `alpha + k*delta`.
  Choose k=3 for `K/|A| > 3`, k=2 for `1.5 < K/|A| <= 3`, otherwise k=1.
  Remaining rejected candidates are not appended.
- If A reaches K, clear it and rescan the full sorted candidate list with `alpha - k*delta`.
  Choose k=0 for `|B|/K >= 5`, k=1 for `2.5 <= |B|/K < 5`, otherwise k=2,
  where B contains first-pass rejects. If still below K, rescan the new rejects at baseline
  alpha. This branch never fills unconditionally.
- With zero delta, use one baseline pass without filling. This differs from disabled
  mode, whose original shortcut for fewer than K candidates is preserved.

The selector holds only per-call scratch data and exposes optional per-call statistics for
tests and profiling; it does not collect global counters or change graph locking.
Serialized indexes retain the configuration. Reload using the same enabled state and, when
enabled, the same alpha, delta and scope settings; mismatches fail compatibility checks.
Indexes written before this option existed default to disabled mode.

## Search parameters

Search-time parameters live under the `hgraph` sub-object:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ef_search` | int64 | — (required) | Positive search-frontier size. Any value up to `INT64_MAX` is accepted; there is no `topk`-relative upper bound. Larger values increase recall, latency, and frontier memory. Omit it and parsing fails; the `30` in the parameter struct is not a usable query default. |
| `hops_limit` | int | unlimited | Cap on the number of hops the beam search performs before returning the current frontier. A value that is **not greater** than `ef_search` is not applied: the parser logs a warning and restores the unlimited default, so the hop cap only takes effect above `ef_search`. |
| `skip_ratio` | float | `0.2` | Performance tuning parameter for filtered search. Controls the ratio of invalid points to skip, in range `[0.0, 1.0]`. `skip_ratio=0.2` means skip 20% of invalid points and only check 80%. Higher values improve performance but may reduce recall. Only applies to searches with filters. See [Filter Skip Strategy](#filter-skip-strategy-skip_ratio-and-skip_strategy) below. |
| `skip_strategy` | string | `"deterministic_accumulative"` | Strategy for filter skipping. Options: `"random"` (random skipping) or `"deterministic_accumulative"` (deterministic cumulative skipping). See [Filter Skip Strategy](#filter-skip-strategy-skip_ratio-and-skip_strategy) below. |
| `brute_force_threshold` | float | `0.0` | Selectivity-aware brute-force fallback. When `> 0` and the supplied filter's `ValidRatio()` is `≤ brute_force_threshold`, the search **bypasses the graph traversal entirely** and runs an exact scan over the valid ids using the best available flatten codes (see the section below). Must lie in `[0.0, 1.0]`; the default `0.0` disables the feature and preserves legacy behavior. |
| `rabitq_one_bit_search` | bool | `false` | Enables the RaBitQ filter/lower-bound path. On an x+y split index it uses all x filter bits; see [RaBitQ x+y Split](../quantization/rabitq_split.md). |
| `rabitq_error_rate` | float | index default | Positive lower-bound error multiplier for this search. It can be tuned without rebuilding the split index. |
| `enable_reorder` | bool | `true` | Whether to re-rank coarse candidates against the precise codes. Set to `false` to skip reordering and return the approximate results directly. Use `use_reorder: true` in `index_param` to store the precise codes needed for reordering. |
| `factor` | float | `0.0` | Reorder-candidate multiplier. The candidate pool starts at `max(ef_search, topk)`; when `> 1.0` it is capped at `min(max(ef_search, topk), floor(topk * factor))`, so it can end up **below** `ef_search`. The default `0.0` keeps the `ef_search` pool. A conjugate graph raises the final pool to at least its look-at-`k` floor. Ignored when reorder is not available. |
| `parallelism` | int | `1` | Number of parallel workers for a single HGraph search. Values `<= 0` are reset to `1` during parsing; parallel execution requires a configured executor, otherwise the search falls back to sequential execution. |
| `timeout_ms` | int | unlimited | Per-query wall-clock timeout in milliseconds. The timer starts after routing; once exceeded, the search returns the current frontier. Best-effort only — it does not preempt a distance-computation batch. |
| `use_extra_info_filter` | bool | `false` | Create the query filter from extra-info rather than plain label membership. Requires matching extra-info columns at build time. |
| `use_conjugate_graph_search` | bool | `true` | Search enhancement pass. When `use_conjugate_graph: true` was set at build time, the search expands results through the conjugate graph for potentially higher recall; set to `false` to skip it. Only takes effect when a conjugate graph was built. |
| `use_mci` | bool | `true` | Use the built MCI routing phase during traversal. Requires an index built with MCI enabled; otherwise the flag is harmless. |
| `use_hybrid_traversal` | bool | `false` | Blend MCI routing with graph traversal instead of choosing one path. |
| `search_no_lock` | bool | `false` | Skip the search-side lock. Requires `parallelism` to be `1`; the caller must guarantee no concurrent modification or destruction of the index while the query runs. |
| `mci_packed_filter` | bool | `true` | Evaluate MCI filters against packed candidate blocks. |
| `mci_batch_distance` | bool | `true` | Batch MCI distance computations across a candidate batch. |
| `mci_activation_cap` | uint32 | `0` | Maximum number of MCI cliques activated per query; `0` means unlimited. |
| `mci_hops_limit` | uint32 | `4294967295` | Hop cap applied to the MCI routing phase specifically. |
| `mci_expansion_idle_window` | uint32 | `0` | Stop MCI expansion after this many consecutive candidates without improvement; `0` disables early stop. The parser limits the value to the `uint32` range even though the field is `uint64`. |
| `hybrid_vob` | float | `0.0` | Virtual out-degree budget for hybrid traversal; finite and non-negative. |
| `hybrid_filter_cost_ratio` | float | `1.0` | Relative cost weight of filter evaluation against graph expansion during hybrid traversal; finite and non-negative. |
| `mci_seed_ratio` | float | `0.1` | Multiplier on `sqrt(N)` that sets the base MCI seed budget; finite and non-negative. Setting it together with `mci_seed_coverage` to `0` disables seeding. |
| `mci_seed_coverage` | float | `1.0` | Fraction of the valid set the seed budget is raised to cover. The raised target is used only when it fits `mci_seed_max_count` and does not exceed the live count; otherwise the coverage term is dropped entirely rather than truncated. Finite and non-negative. |
| `mci_seed_max_count` | int64 | `32768` | Cap for the `mci_seed_coverage` term above; `0` means unlimited. Replacement for the removed `mci_seed_count`, which is rejected outright. |
| `hgraph_valid_ratio_threshold` | float | `0.05` | Filter selectivity below which MCI routing may be skipped, in `[0, 1]`. |

The root-level `threshold` key (sibling of `hgraph`, not a member of it) is also honoured: a
finite numeric value discards KNN candidates whose distance exceeds it. It is read by
`ParseSearchThreshold` before the `hgraph` object is parsed.

```cpp
auto result = index->KnnSearch(
    query, topk, R"({"hgraph": {"ef_search": 200}})").value();
```

### Brute-force fallback under highly selective filters (`brute_force_threshold`)

Graph traversal is the right strategy when most candidates pass the filter — the
graph quickly reaches the neighborhood of the query. As filter selectivity
increases (only a tiny fraction of vectors survive), the beam has to expand far
more nodes just to fill `ef_search` with **valid** candidates, and recall drops.
At some point an exhaustive scan over the surviving ids is both faster *and*
exact.

`brute_force_threshold` lets HGraph make that switch automatically on a
per-query basis:

```cpp
// When the active filter keeps ≤ 1% of ids, run an exact scan instead.
auto params = R"({"hgraph": {"ef_search": 200, "brute_force_threshold": 0.01}})";
auto result = index->KnnSearch(query, topk, params, my_filter).value();
```

How it works (`src/algorithm/hgraph/hgraph_search.cpp`):

- The fallback only fires when **all** of the following hold:
    - `brute_force_threshold > 0.0`, **and**
    - a filter is supplied, **and**
    - `filter->ValidRatio() <= brute_force_threshold`.
- The accuracy of `Filter::ValidRatio()` matters — it is the user-supplied hint
  the dispatcher checks against the threshold. See
  [Filtered Search](../advanced/filtered_search.md) for the API contract.
- The scan iterates every valid inner id and computes distances in batches of
  64 using the most precise flatten storage available (raw vectors if
  `store_raw_vector` was set, otherwise the high-precision reorder codes when
  `use_reorder=true`, otherwise the base quantized codes).
- Because the scan already uses precise codes when present, the post-search
  reorder pass is **skipped** for queries that took the brute-force branch.
- Applies to `KnnSearch` (the non-iterator overload, which is what
  `SearchWithRequest` and the standard `KnnSearch(query, k, params, filter)`
  call) and to `RangeSearch`. It does **not** apply to the iterator-style
  `KnnSearch(..., IteratorContext*&, ...)`, because a single sweep cannot be
  paged across multiple iterator calls.

Picking a value:

- Leave at `0.0` (default) for unfiltered or weakly filtered workloads.
- For highly selective filters, `0.01–0.05` is a reasonable starting point.
  Setting it higher than that effectively turns the index into a brute-force
  scanner whenever a filter is present.
- The cost of the brute-force scan is roughly `O(N × dim)` where `N` is the
  total number of indexed vectors (regardless of selectivity, because every id
  is visited to check `CheckValid`). The benefit grows when graph search would
  otherwise need a much larger `ef_search` to recover recall.

See
[`322_feature_hgraph_brute_force_threshold.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/322_feature_hgraph_brute_force_threshold.cpp)
for a runnable brute-force fallback example.

### Filter Skip Strategy (skip_ratio and skip_strategy)

When searching with a filter, HGraph needs to frequently call Filter::CheckValid() during graph traversal to verify whether each candidate point is valid. This check can be expensive (especially for complex filter logic). skip_ratio and skip_strategy provide a probabilistic optimization: they skip some filter checks to speed up the search, but may reduce recall.

#### How It Works

This is a probabilistic optimization strategy: we don't know in advance which points are valid, so we decide probabilistically whether to visit each point.

- skip_ratio (default 0.2): Controls the aggressiveness of skipping filter checks. skip_ratio=0.2 means skip 20% of candidate checks and only check 80%. Higher values skip more, making search faster but potentially reducing recall.
- skip_strategy (default "deterministic_accumulative"): Determines how skipping is distributed:
  - "random": Random skipping. Each point is visited independently with probability `visit_ratio = valid_ratio + (1 - valid_ratio) * (1 - skip_ratio)`, so roughly a `1 - skip_ratio` fraction of invalid points are skipped.
  - "deterministic_accumulative": Deterministic cumulative skipping. Emits visit decisions at fixed intervals so that the long-run visit ratio matches the target `visit_ratio`, with lower variance than the random strategy.

The specific formula:
- Let valid_ratio be the filter's global validity rate (from Filter::ValidRatio())
- Probability of visiting each point = valid_ratio + (1 - valid_ratio) * (1 - skip_ratio)
- In expectation, this targets skipping about skip_ratio of invalid candidate checks when Filter::ValidRatio() is accurate

#### Usage Examples

```cpp
// Conservative setting: skip 10% of invalid candidate checks, suitable for high-recall
// scenarios where latency is less critical
auto params = R"({"hgraph": {"ef_search": 200, "skip_ratio": 0.1}})";
auto result = index->KnnSearch(query, topk, params, my_filter).value();

// Use random strategy
auto params = R"({"hgraph": {"ef_search": 200, "skip_ratio": 0.2, "skip_strategy": "random"}})";
auto result = index->KnnSearch(query, topk, params, my_filter).value();

// Aggressive skipping: skip 50% of invalid candidate checks for lower latency
auto params = R"({"hgraph": {"ef_search": 200, "skip_ratio": 0.5}})";
auto result = index->KnnSearch(query, topk, params, my_filter).value();
```

#### Choosing Values

- Default 0.2: Suitable for most scenarios, balancing performance and recall.
- 0.1 or lower: Conservative setting, suitable for scenarios with high recall requirements where latency is less critical.
- 0.5 or higher: Aggressive skipping, suitable for latency-sensitive scenarios where recall degradation is acceptable (e.g., real-time recommendation systems).
- 0.0: Don't skip any points, equivalent to disabling this optimization (all points will be checked).

Important notes:
- Only applies to searches with filters. These parameters are ignored when no filter is present.
- Performance optimization works better when Filter::ValidRatio() is accurately estimated.
- Can be used together with brute_force_threshold: when the filter is very strict (ValidRatio is very small), brute_force_threshold will trigger brute-force fallback; otherwise, graph traversal + skip optimization is used.

## When to use HGraph

- Dense float vectors with dimensions roughly between 64 and 4096.
- Latency-sensitive queries where high recall matters.
- Mixed workloads with incremental insertion (optionally force removal via `support_force_remove`).
- Memory-constrained deployments that benefit from `sq8` / `sq4_uniform` / `pq` — often
  in combination with `use_reorder` to recover recall.

For repeated daily or snapshot builds with stable source identifiers, see
[HGraph Build Cache](../advanced/build_cache.md). It documents `Dataset::SourceID`,
`ExportCache`/`ImportCache`, `persist_source_id`, and cache hit-rate diagnostics.

If your workload is partition-heavy (coarse-grained buckets scanned per query) or
strongly I/O-bound on a SSD, compare against [IVF](ivf.md) before committing to HGraph.

## FGIM merge (opt-in)

Set `merge_strategy: "fgim"` in the destination HGraph parameters to use FGIM
through the existing `Index::Merge(vector<MergeUnit>)` API. The default remains
`"odescent"`. `fgim_refinement_iterations` is a non-negative integer (default 4);
zero disables refinement. This is an iteration budget, not a guarantee of
convergence. Setting this option alone keeps the default `odescent`
strategy; VSAG reports a warning when the option is supplied for another strategy.
The persistent k-NNG degree bound is the destination
`max_degree`; CrossQuery uses `ceil(k / (number_of_sources - 1))` for both top-k
and search width.

The non-empty destination, if present, is the first source. At least two non-empty
source graphs are required. Sources must have successful builds, homogeneous
float32/L2/FP32 storage, and no concurrent mutation. This first implementation
rejects deletion, duplicate tracking/storage, conjugate graphs, MCI, attributes,
extra information and persisted source IDs. Set `support_duplicate: false` and
`deduplicate_storage: false` explicitly.

Existing destination labels are retained. Each MergeUnit callback maps external
labels once; a null callback means identity mapping. Filtering through a false
callback result and collisions in the resulting labels are rejected before
destination data is changed. Source indexes are not modified.

FGIM builds an initial k-NNG, refines it, selects bottom neighbors, adds reverse
connections and rebuilds upper layers using existing HGraph insertion primitives.
It does not concatenate old upper layers. Performance and recall depend on the
data and degree/iteration settings; this option does not guarantee a speedup or
full graph connectivity. Allocation or storage failures during publication do
not provide transactional rollback.

Refinement and reconstruction each use an additional FP32 vector buffer of
`N * dimension * sizeof(float)` bytes. These buffers are released between stages;
peak memory also includes the source indexes and intermediate neighbor lists.
The destination graph must be writable; read-only `reader_io` graph storage is
rejected before destination data is changed. Queries on the destination normally
wait for its structural lock during Merge. Between the lock scopes around capacity
growth, readers can still access the unchanged, previously published index.

## See also

- [Creating an Index](../guide/create_index.md)
- [Graph Enhancement](../advanced/enhance_graph.md)
- [Optimizer (Tune)](../advanced/optimizer.md)
- [Serialization](../advanced/serialization.md)
