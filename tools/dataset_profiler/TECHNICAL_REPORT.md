# Dataset Feature Visualization Tool — Technical Report

This report documents the design decisions behind `tools/dataset_profiler`
(`vsag-dataset-profiler`), implemented for issue #2862: metric selection, the
sampling design, privacy boundaries, the AI prompt/response schema, the
validation strategy, and known limitations.

## 1. Metric selection

The profiler computes the metric families requested in issue #2862. Every
metric carries a `definition`, a `scope` (`full`, `sample`, or `mixed`), and
the `sample_size` when applicable, so full-dataset facts and estimates are
never conflated.

| Metric | Scope | Why it matters for index choice |
| --- | --- | --- |
| `norms` | full | Norm spread drives quantization range selection (SQ/PQ/FastScan) and the value of normalization or OPQ-style rotation. |
| `dimensions` (per-coordinate mean/std/min/max/zero-ratio + top lists) | full | Shows scale mismatch between coordinates (a warning sign for Euclidean indexes) and dimensions carrying no signal (candidates for pruning). |
| `sparsity` | full | Exact-zero ratio for dense files; stored entries, non-zero counts per vector (`nnz`), and density for sparse files. Predicts SINDI-style sparse index behavior and compression headroom. |
| `invalid_values` | full | NaN/Inf counts per dataset and per affected vector; indexes and quantizers silently degrade on invalid input. |
| `duplicates` | full (exact) + sample (near) | Exact duplicates are byte-identical vectors (full coverage via hashing). Near-duplicates are a bounded indicator: sampled pairs closer than `epsilon x median(sampled duplicate-detection distance)`, using Euclidean distance for IP inputs to keep the threshold non-negative. |
| `pairwise_distances` | sample | Distribution shape (median, spread, tails) informs distance-based thresholds and graph vs. IVF trade-offs. |
| `nearest_neighbor_distances` | sample | Small NN-distance gaps relative to typical distances indicate hard recall problems for graph indexes. |
| `outliers` | full | Robust modified z-score over norms (median/MAD, IQR fallback) — norm outliers distort shared quantization ranges. |
| `anisotropy` | sample | Eigenvalue concentration of the centered sample covariance (top-1 share, top-10 share, effective rank, mean pairwise cosine). Highly anisotropic data hurts cosine/IP quality and motivates transformation or different quantization. |
| `clustering` | sample + optional | Deterministic PCA-2 projection of the sample (scatter plot) and optional seeded k-means summary when `--clusters K` is given. |

Excluded on purpose: ground-truth/kNN recall estimation (requires the VSAG
C++ library and belongs to `tools/eval`), and query-distribution analysis
(no queries exist in the dataset files).

## 2. Sampling design

Requirements: reproducible reports, no `O(N^2)` allocation over the full
dataset, and bounded memory for streaming sparse files.

* **Generator**: `numpy.random.Generator(PCG64(seed))`, seeded explicitly by
  `--seed` (default 42). No global NumPy state is used or modified.
* **Index selection**: Floyd's algorithm (`sample_indices`) draws `k` distinct
  indices from `[0, n)` with memory/time bounded by `k`, not `n` — important
  for sparse files without `*_offsets`, which are scanned streaming.
* **Pairwise bound**: a seeded permutation of the uniformly random sample
  selects `min(pairwise_sample_size, sample_size, N)` rows for the pairwise
  matrix. Sorting the selected positions afterward restores deterministic row
  order without biasing the subset toward low dataset indices. The report
  records requested vs. effective sizes and the exact method string.
* **Streaming**: dense files are read in row blocks sized from a 128 MiB
  working-byte budget (at most 65536 rows), so a wide FLOAT32 matrix cannot
  retain gigabytes per block; sparse files are decoded record-by-record from
  bounded byte chunks (default 64 MiB).
  Full-dataset statistics (norms, per-dimension aggregates, duplicates,
  invalid counts, sparsity) come from a single streaming pass. Norm state is
  preallocated once (8 bytes per vector), never accumulated as block lists and
  concatenated. Norm state and sparse nnz state (8 bytes per vector each) have
  an explicit 256 MiB state budget. Exact-duplicate detection preallocates a
  16-byte digest row per vector and uses a peak-aware 64-byte-per-vector budget
  that also covers `np.unique`'s sorted copy, sort index, unique output, and
  counts. Each row-count estimate is checked before allocation; when it exceeds
  the budget, the affected full-split norms/outliers, nnz distribution, or
  exact-duplicate fields are marked unavailable with a reason and warning,
  while streaming totals and sampled metrics still appear. Duplicate groups
  are counted with NumPy over the fixed-width digest rows rather than a Python
  `Counter`.
* **Distance computations**: Euclidean pairwise distances use the expanded
  norm/dot form when it is numerically stable and fall back to direct
  coordinate differences for pairs whose norms dwarf their distance (for
  example rows near `1e10` differing by `1`, where the expansion rounds to
  zero). The sparse Euclidean branch always subtracts coordinates directly.
  The mean pairwise cosine is derived from normalized-vector sums
  (`||sum(u)||² - sum(||u||²)`) instead of an `O(M²)` Gram matrix.
* **Per-dimension arrays**: sparse absent coordinates contribute implicit zeros
  to per-dimension statistics. Sparse accumulation stops at dimensionality
  10,000,000 (other metrics continue) to bound memory. Series longer than
  `max_dimension_points` (default 1024) are bucket-averaged for storage, with
  `source_length`, `bucket_size`, and `downsampled` flags recorded so charts
  stay honest.
* **Determinism**: two runs with the same file, configuration, sample sizes,
  and seed produce byte-identical JSON (`sort_keys`, fixed indent,
  `allow_nan=False`). The committed example reports under `examples/output/`
  are byte-reproducible. The one environment-dependent piece is SVD sign
  convention (BLAS-dependent), which only affects projection scatter
  orientation, not the scalar anisotropy statistics.
* **Sparse offsets**: results must not change between the `*_offsets`-indexed
  and scan-only encodings of the same byte stream. Offset validation reads
  bounded windows. An index larger than 64 MiB is not retained; the reader
  uses the bounded sequential decoder and cross-checks its decoded record
  count. Tests assert full-dataset metric equality between the paths and
  exercise the oversized-index fallback.
* **Invalid values**: NaN/Inf entries are excluded from numeric summaries and
  treated as zero in norms, pairwise math, and PCA input (documented per
  metric). An all-zero vector is a *valid* vector: it contributes to counts,
  norms (0), and the all-zero-vector counter, but is excluded from outlier
  detection by the robust method's behavior.

## 3. Privacy boundaries

* **Default offline**: JSON/HTML generation never opens a network connection.
* **Aggregate-only AI payload**: the request is built from the report —
  aggregate metrics, sampling config, warnings, and non-sensitive metadata
  (split name, vector type, dtype, count/dim, distance). A field allowlist
  drops per-vector entries (`clustering.projection_points`, which holds one
  PCA coordinate pair per sampled vector, and `clustering.labels`, which holds
  one cluster assignment per sampled vector). Raw vectors are not in the
  report and therefore structurally cannot be sent; tests assert that source
  rows never appear in the request body and that those keys are absent.
* **Credentials**: the API key is read from the environment (`VSAG_AI_API_KEY`,
  fallback `OPENAI_API_KEY`, or a custom variable via `VSAG_AI_KEY_ENV`),
  sent only in the `Authorization` header over https (plaintext URLs are
  rejected), and never written to reports. Redirects are rejected so the
  bearer token cannot be forwarded to another endpoint. Error messages are
  redacted (`sk-…` patterns and bearer tokens) and truncated; tests assert keys
  cannot leak through provider failures.
* **Self-contained HTML**: all CSS/JS/SVG are inlined; opening a report
  contacts no external service.

## 4. AI prompt/response schema

The adapter is provider-neutral (any OpenAI-compatible `/chat/completions`
endpoint) and temperature is fixed at 0.

Request: the system-style instructions plus
`{summary, findings[], recommendations[], limitations[], additional_measurements[]}`, where every
finding/recommendation is `{title, detail, evidence: [metric key], confidence: low|medium|high}`
and every recommendation additionally carries `validation` (how to test it in a reproducible
`tools/eval` run). The prompt embeds the list of allowed evidence keys (leaf paths enumerated
from `metrics.*`) and the aggregate JSON payload.

Validation (client side, `parse_and_validate_response`):

* response must parse as a single JSON object (markdown fences tolerated);
* `summary` non-empty string; `findings`/`recommendations` non-empty lists (<= 10 items);
* every item needs non-empty `evidence`, and **each cited key must resolve inside the report's
  `metrics` subtree** — invented keys fail the whole response;
* recommendations must carry a `validation` string;
* `confidence` restricted to `low|medium|high`; strings are length-bounded.

Any rejection degrades to `ai_analysis.status = "error"` with a redacted
message: deterministic reports are still produced and a warning is printed
(issue acceptance criteria 6 and 7).

## 5. Validation strategy

* **Unit tests** (`tests/`): sampling determinism; stats helpers; both HDF5
  layouts incl. corrupt/edge encodings; all profiler metric families incl.
  determinism, sample bounds, metric aliasing, invalid values, and the
  offsets/scan equivalence invariant; HTML self-containment and escaping; AI
  schema validation and failure modes; CLI exit codes and output handling.
* **Integration**: the CLI is exercised end to end against generated dense and
  sparse fixtures; `make_toy_datasets.py` reproduces the committed example
  reports byte-for-byte.
* **Manual spot checks**: rendered charts were opened locally and cross-checked
  against direct NumPy computations on the toy datasets (counts, zero ratios,
  duplicate groups).

## 6. Limitations

* Sampled metrics (pairwise, NN distances, anisotropy, clustering) are
  estimates: their report entries carry `scope: sample` and the sample size,
  and the HTML repeats this per section.
* Near-duplicate detection is a bounded *indicator* over the sample, not a
  full-dataset guarantee.
* The duplicate hash includes invalid bytes as-is (byte equality, not numeric
  equality). Dense one-byte unsigned input is numerically converted by HDF5 to
  signed `INT8` before hashing and numeric statistics, matching the evaluator's
  `ALPHA_I8` reads; values above 127 saturate at 127 rather than wrapping.
* Full norm quantiles/outliers, sparse nnz distributions, and exact duplicate
  counts are unavailable when their estimated per-row state exceeds the 256 MiB
  per-structure budget; the report marks these fields unavailable rather than
  returning partial results.
* k-means is an O(n·k) Lloyd loop over the 2-D projection; it is intended for
  structure hints, not production clustering.
* Sparse dimensionality is inferred as `max(id) + 1`; a file with id gaps at
  the top reports a smaller dim than its producer may have assumed (this is
  what the evaluation loader implies as well).
* The AI layer follows the repository's issue #2862 boundary: it explains
  measured metrics and does not inspect raw data, and its suggestions are
  explicitly hypotheses to validate with `tools/eval` benchmarks.
