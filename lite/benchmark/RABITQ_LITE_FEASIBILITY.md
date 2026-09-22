# Lite RaBitQ 3+5 feasibility boundary

This document records the source-based boundary for a possible standalone Lite
RaBitQ backend. It is an experiment design, not a public API commitment. The
Full HGraph reference measurements are recorded in `README.md` and
`FINAL_REPORT.md`.

## Source evidence

The proposed direction follows the repository's existing implementation rather
than defining a new quantizer:

- `docs/docs/en/src/quantization/rabitq_split.md` defines x+y split storage:
  x filter bits drive traversal, y supplement bits are read for reorder, and
  the final distance uses x+y bits.
- `src/algorithm/hgraph/hgraph_parameter_test.cpp` verifies that external
  `rabitq_bits_per_dim_base=3` and `rabitq_bits_per_dim_precise=5` map to an
  eight-bit RaBitQ quantizer with `rabitq_bits_per_dim_filter=3`.
- `src/quantization/rabitq_quantization/rabitq_quantizer.cpp` trains a centroid
  and random transform, encodes normalized vectors, lays out split bit planes
  and metadata, and combines filter and supplement contributions.
- `src/simd/generic.cpp` and `src/simd/rabitq_simd.h` provide the scalar
  split-code inner product, supplement inner product, and scalar-to-plane pack
  operations together with dispatched SIMD variants.
- `src/datacell/rabitq_split_datacell.h` owns two independent fixed layouts and
  performs filter-only traversal followed by supplement-backed full distance.
- `src/impl/transform/fht_kac_rotate_transformer.cpp` applies four rounds of
  persisted random sign flips and FHT/Kac transforms. Its model is not a dense
  orthogonal matrix.
- `include/vsag/lite/index.h`, `src/lite/backend.h`, and
  `src/lite/graph_backend.cpp` show that Lite currently exposes FP32/FP16
  storage and a single-stage graph distance. Add, Update, Remove, filtered
  Search, and versioned snapshots all operate through this smaller contract.

## Exact 128-dimensional L2 layout

For the measured x=3, y=5 configuration, Full maps the quantizer to total
bits B=8 and filter bits x=3. With dimension d=128, one bit plane is
`ceil(d/8)=16` bytes.

The ordinary split layout derived from `RefreshSplitLayout` is:

| Record | Planes | L2 metadata | Bytes/vector |
| --- | ---: | --- | ---: |
| Filter | 3 x 16 | norm, filter-code norm, lower-bound error, filter error | 64 |
| Supplement | 5 x 16 | full-code norm, vector norm, full error, lower-bound error, filter error | 100 |
| Combined | 8 x 16 | metadata above | 164 |

The combined code payload is 68.0% smaller than a 512-byte FP32 vector and
35.9% smaller than a 256-byte FP16 vector before IDs, graph links, container
capacity, and model state. The model also needs a 128-float centroid and four
128-bit FHT sign masks. These arithmetic sizes explain why the candidate is
plausible, but they are not a prediction of complete Lite snapshot or RSS.

## Dependency decision

### Behavior and small kernels to reuse

- Preserve the official x+y bit ordering, centering, norm/error metadata, and
  lower-bound equations.
- Reuse or minimally extract the repository's generic plane pack and split
  inner-product kernels; add ISA translation units only after the scalar path
  is correct and measured.
- Preserve the official FHT transform shape and persist its sign masks so a
  loaded index reproduces the build-time model.
- Keep the existing Lite error contract, physical remove/fill-hole behavior,
  external-ID filter semantics, and little-endian bounded snapshot parsing.

### Full components that must stay outside Lite

- `Quantizer`/`Computer` CRTP infrastructure, `Allocator`, Full `Stream`, JSON
  parameter factories, DataCell/Layout/IO classes, HGraph/Pyramid, thread
  pools, PCA/MRQ, fused residual clusters, disk IO, and Full serialization.
- The dense random orthogonal matrix path. The measured configuration uses
  FHT; importing the alternative ROM/BLAS dependency would defeat the
  standalone boundary.
- Full raw-vector or FP32 reorder storage. The reference one-bit mode showed
  that retaining FP32 reorder data misses the Lite storage objective.

The entire `RaBitQuantizer` or `RaBitQSplitDataCell` cannot be linked into the
standalone Lite target: their direct contract brings the Full allocator,
transform, parameter, stream, IO, and datacell closure. Only the required
math and byte layout should cross the boundary, with repository provenance
kept in comments and tests.

## Proposed minimal change

The next implementation should remain opt-in under `lite/benchmark` and should
not change `VectorStorage`, `Index`, or snapshot versions.

1. Add a standalone L2-only 3+5 codec probe with a fixed internal model
   representation: dimension, centroid, four FHT sign-mask rounds, and the
   official filter/supplement records.
2. Train once from a batch, encode the batch, and support query transform,
   filter estimate/lower bound, and full 8-bit reorder distance. The probe must
   use a reproducible model input or persist the generated masks; it must not
   claim byte identity with a separately trained Full model.
3. Differentially test plane packing and scalar distances against direct
   decoded-code formulas, including non-multiple-of-eight dimensions, constant
   vectors, zero norm, non-finite input rejection, truncated model/code data,
   and round-trip persistence.
4. Run the existing prepared SIFT 10k/100k inputs in fresh processes and report
   encoded bytes, model bytes, Recall@10, build/search latency, and RSS. Compare
   with the current Lite FP32/FP16 evidence and with the Full reference, while
   keeping algorithm-level comparisons separate.

Only after that probe meets an agreed quality and latency budget should a
feature design extend the graph backend. That later design must add two-stage
traversal/reorder rather than substituting an eight-bit distance everywhere;
otherwise it would not implement the official split search pipeline.

## Excluded from the next implementation

- No change to PR #2904, the public Lite API, `VectorStorage`, or v1/v2/v3
  snapshots.
- No PCA, MRQ, cosine/IP, disk supplement IO, mmap, fused datacell, concurrent
  search, or Full HGraph linkage.
- No SIMD-first implementation and no performance claim from a scalar probe.
- No reuse of the old dirty graph exploration workspace.

## Validation and promotion gates

The experiment must pass clang-format-15, clang-tidy-15, its focused codec
tests, the existing Release Lite CTest suite, malformed-input tests, and
fresh-process round trips. Formal measurements require seven alternating-order
runs at both prepared scales with exact commit and binary hashes and empty
stderr.

Promotion to a public backend remains blocked until the maintainer/mentor
accepts a Recall@10 floor and latency budget, the Lite-specific snapshot and
steady-RSS results materially improve on FP16, and CRUD behavior with a fixed
trained model is specified and tested. The current Full reference supports
continuing the 3+5 experiment; it does not by itself satisfy those gates.

## Prototype gate status at `a99b0e6`

The standalone probe now covers deterministic batch training, FHT masks, fast 8-bit CAQ encoding, 3+5 plane packing, the official L2 lower-bound calculation, candidate pruning, and supplement-only reranking. Its self-test checks deterministic model/code generation, scalar-versus-split inner-product parity, and filtered-versus-full-code Top-10 equality. It also rejects non-finite vectors and malformed, duplicate, or out-of-range SIFT ground truth.

Seven fresh SIFT-100k processes all produced 0.985 Recall@10 and exact Top-10 agreement between filtered search and an exhaustive full-code scan. The filter read the supplement for a mean 0.2273% of records. These results pass the standalone functional and quality gate for this fixed dataset and seed. The probe now also round-trips its FHT model, split payloads, and metadata through an independent bounded little-endian format, with truncation, magic, metadata, size, and trailing-byte rejection. It does not yet pass the public-backend promotion gate: in-memory records still use per-record allocations, and graph integration, CRUD model lifecycle, SIMD filter kernels, and owned-memory measurement remain open. The next bounded implementation is contiguous in-memory storage, followed by graph traversal integration only if the mentor accepts the measured quality floor.

## Contiguous-storage gate

The in-memory probe now stores all filter planes, supplement planes, and fixed-size metadata in three contiguous owned arrays. It retains no per-record vectors after encoding, verifies fixed record strides, and preserves the `VSLRBQ01` v1 snapshot bytes across load/save. Single fresh-process SIFT-10k/100k checks retained Recall@10 of 0.994/0.985, exact filtered/full agreement, and the previous reorder ratios. The 100k peak RSS was 69,048 KiB, 12.0% below the earlier seven-run median of 78,456 KiB; this is directional rather than a stable performance result because the new value is a single run and source vectors remain resident.

The public-backend promotion gate remains open: graph traversal integration, CRUD behavior for a fixed trained model, SIMD filter kernels, and isolated owned-memory measurement are not yet complete. The next bounded implementation is an experiment-only graph traversal adapter that uses the contiguous records for filter-first exploration and supplement-backed result reorder. It must remain outside `VectorStorage` and public snapshots until its quality and latency are measured and the mentor accepts the promotion boundary.

## Filter-first graph traversal gate

The experiment now reuses the official Lite FP32 graph builder as a reference topology and exports its adjacency into contiguous CSR storage. Search traverses with the 3-bit filter distance and reads the 5-bit supplement only for the final `ef_search=128` candidate set. Snapshot-loaded encoded records produce byte-identical graph results on the same topology.

At SIFT-10k/100k, graph Recall@10 was 0.966/0.940, with mean visited counts of 829.15/1,150.83 and graph-search P50 of 281.383/406.009 us. The corresponding scalar full-scan P50 values were 1,461.610/14,266.271 us. The 100k result is close to the existing public FP32 graph Recall@10 of 0.946, so the traversal experiment passes its bounded quality gate.

This does not yet establish a standalone RaBitQ graph backend. The graph topology is built through temporary FP32 Lite backends, topology persistence is not part of `VSLRBQ01`, CRUD/model lifecycle is unspecified, and the filter kernel is scalar. The next bounded step is to persist and restore the CSR topology with strict validation, then define Add/Update/Remove behavior under one fixed trained model before considering a public `VectorStorage` value.
