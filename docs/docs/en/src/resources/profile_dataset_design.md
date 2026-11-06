# Dataset Profiling Tool: Technical Report

This report describes how [`profile_dataset`](profile_dataset.md) chooses and computes its
metrics, how sampling and reproducibility work, where the privacy boundary of the optional AI
layer lies, how the AI prompt and response are structured and validated, how the tool was
validated, and what it does not do.

## 1. Goals and non-goals

Goals:

- Measure the dataset properties that drive VSAG decisions (index type, metric, quantizer,
  transform, search parameters) from the HDF5 files that `eval_performance` already uses.
- Make every number reproducible: the same file, options and seed give bit-identical metrics for
  any thread count or I/O chunk size, and the same sampled rows on every platform.
- Keep memory bounded: stream the full split, never build an all-pairs `N x N` matrix, and bound
  every sample-based computation by explicit options.
- Explain results offline (HTML and rule-based observations) and, optionally, with an AI provider
  that only sees aggregate metrics and whose output is validated before it is shown.

Non-goals: training or fine-tuning models, modifying or building indexes, starting production
workloads, uploading raw vectors, or replacing reproducible VSAG benchmarks.

## 2. Architecture

```text
HDF5 file
  |  inspect_dataset (structure, attributes, eval_performance compatibility)
  v
per split (test first, so its sample can be searched against train in the same pass)
  pass 1: full streaming scan in 4096-row blocks
          invalid values, norms, per-dimension moments, zeros, duplicate hashes,
          capture of the sampled rows
  pass 2: full streaming scan
          value histogram, 6-sigma outliers, exact k-NN of sampled queries
          (and of the other split's sampled queries)
  sample analyses: pairwise distances, LID, TwoNN, relative contrast, hubness,
          PCA, k-means, 2-D projection, quantization simulation
  v
report JSON: input, config, splits, cross_split, ground_truth, labels,
             signals (rules), determinism digest, run
  |-- HTML renderer (inline SVG, CSP) -------------------------> report.html
  '-- optional AI: allow-listed evidence -> prompt -> provider -> validation
                   -> ai_analysis -> JSON and HTML rewritten
```

| File | Responsibility |
| --- | --- |
| `tools/profile_dataset/hdf5_source.cpp` | File inspection, streaming dense reader, sparse byte-stream parser |
| `tools/profile_dataset/dense_profiler.cpp`, `tools/profile_dataset/sparse_profiler.cpp` | Two-pass split profilers |
| `tools/profile_dataset/dataset_profiler.cpp`, `tools/profile_dataset/profile_signals.cpp` | Orchestration, cross-split, ground truth, labels, rules, digest |
| `tools/profile_dataset/neighbor_search.cpp` | Exact k-NN (dense tiles, sparse inverted index), LID, TwoNN |
| `tools/profile_dataset/statistics.cpp`, `tools/profile_dataset/deterministic_random.cpp` | Moments, histograms, quantiles; portable RNG and sampling |
| `tools/profile_dataset/linear_algebra.cpp`, `tools/profile_dataset/kmeans.cpp`, `tools/profile_dataset/quantization_simulation.cpp` | PCA, clustering, quantizer models |
| `tools/profile_dataset/html_report.cpp`, `tools/profile_dataset/svg_chart.cpp` | Self-contained HTML report |
| `tools/profile_dataset/ai_analysis.cpp`, `tools/profile_dataset/http_transport.cpp` | Evidence, prompt, validation, providers, transport |
| `tools/profile_dataset/profile_cli.cpp` | Command line, output ordering, exit codes |

The tool does not link `libvsag`; it shares the HDF5 layout, metric conventions and quantizer
rules with VSAG but computes everything itself, so it cannot change any index.

## 3. Metric selection

Each metric answers a question that a VSAG user faces; the rule-based signals encode the
thresholds.

| Question | Metrics | Background |
| --- | --- | --- |
| Is the data valid for the metric? | NaN/Inf counts from bit patterns, zero vectors, unsorted or repeated sparse ids | Undefined distances; cosine of a zero vector |
| Are `l2`, `ip` and `cosine` interchangeable? | Unit-norm ratio, norm CV | Normalized rows rank identically under all three; spread norms make `ip` a MIPS problem |
| Does the data waste memory? | Exact duplicates (full), near duplicates (k-NN), constant dimensions, sparsity | Duplicates crowd top-k results and graph edges |
| How hard is nearest-neighbor search? | Relative contrast, LID (MLE), TwoNN, distance concentration, hubness | He, Kumar and Chang, ICML 2012; Levina and Bickel, NeurIPS 2004; Facco et al., Sci. Rep. 2017; Aumuller and Ceccarello, 2019 (LID and ANN difficulty); Radovanovic et al., JMLR 2010 (hubness) |
| Which quantizer and transform fit? | Per-dimension ranges, 6-sigma outliers, kurtosis, PCA spectrum, simulated recall of fp16/bf16/sq8/sq4/sq8_uniform/sq4_uniform | Mirrors `ScalarQuantizationTrainer`: per-dimension truncated bounds (rate 0.001 for 8 bit, 0.05 for 4 bit), global min/max for `sq8_uniform`, narrowest truncated bounds for `sq4_uniform`, floor encoding |
| Will IVF-like partitions balance? | k-means size CV, largest/mean size, silhouette | Coarse proxy; IVF uses many more buckets |
| Do queries look like the base? | Norm KS, standardized mean shift, out-of-distribution ratio, 2-D projection overlay | Out-of-distribution queries need tuning on real queries |
| Is the ground truth trustworthy? | Id and tie-tolerant distance recall against recomputed neighbors, distance convention | Wrong ground truth silently corrupts every recall number |
| Sparse/SINDI sizing | `max_term_id` (for `term_id_limit`), id density (`remap_term_ids`), document frequency and Zipf/Gini, retained terms under SINDI's `doc_prune_ratio` mass rule, postings per query | Inverted-list lengths dominate SINDI work |

Distances follow VSAG conventions (`l2` squared, `ip` as `1 - <x, y>`, `cosine` as
`1 - cos`). Estimators that require a metric space (LID, TwoNN, relative contrast, near
duplicates) use Euclidean distance, or chord distance between normalized rows for `cosine`;
for `ip` an extra Euclidean neighbor list is maintained during the same scan.

## 4. Sampling design

- **Rows.** For each split, `--sample-size` row ids are drawn uniformly without replacement with
  Floyd's algorithm from a generator seeded with `derive_seed(seed, "sample/<split>")` and sorted,
  so the streaming pass can capture them. Sparse splits without offsets are counted first.
- **Order.** The sampled positions are shuffled with the stream `order/<split>`; rows that are
  invalid for distances (non-finite values, zero vectors under `cosine`) are skipped. Every
  analysis uses a *prefix* of this order: the first `--knn-queries` rows are the k-NN queries,
  the first `--hubness-sample` rows build the within-sample graph, and so on. A prefix of a
  random permutation of a uniform sample is itself a uniform sample, so each analysis is
  unbiased and the subsets are nested, which makes results comparable across options.
- **Independent streams.** Pairs (`pairs/<split>`) and k-means seeding (`kmeans/<split>`) use
  their own seeds, so changing one option does not change the rows another analysis uses.
- **k-NN against the full split.** Nearest-neighbor distances are searched in the full split,
  not in the sample: neighbors within a sample are systematically farther than true neighbors.
  The cost is a full streaming pass (`O(N * D * queries)` time) with `O(queries * k)` memory.
- **Labels.** Every sampled section records the method, population, sampled and valid rows and
  the rows used; the HTML shows them on each chart. Sampling error shrinks with the sample size;
  borderline sampled metrics can be confirmed with a larger `--sample-size` or `--knn-queries`.

## 5. Reproducibility engineering

| Risk | Measure |
| --- | --- |
| Library-dependent random distributions | Own xoshiro256** with SplitMix64 seeding, rejection-based bounded draws, Floyd sampling, Fisher-Yates shuffle (verified against an independent Python implementation) |
| Thread count and chunk size changing float sums | Fixed 4096-row blocks merged in block order with Pebay's moment formulas; per-pair kernels with a fixed association order |
| k-NN order depending on scheduling | Strict (distance, row id) order; the k smallest elements form a unique set |
| Compiler optimizations | The target is built with `-fno-fast-math -ffp-contract=off` (release builds otherwise use `-Ofast`); NaN/Inf are classified from bit patterns |
| Linear algebra libraries | Serial double-precision Householder + implicit QL eigen-decomposition; eigenvector signs fixed |
| Clustering | k-means++ seeded from the RNG, ties broken by the lowest centroid, centroid sums in row order |
| Proof | `determinism.metrics_sha256` over every section except `run` and `ai_analysis` |

Measured: identical digests for 1, 7 and 24 threads and for 1 MB and 64 MB chunks on the dense
and sparse fixtures, for 24 threads with 64 MB chunks against 3 threads with 8 MB chunks on the
three public datasets below, and in the unit tests. Release builds from GCC 11 and GCC 13 gave
identical metrics on those datasets. Sampled rows come from integer arithmetic and match on every
platform. Floating-point metrics also use `log`, `exp` and `hypot` from the C
library, so reports from different math libraries or architectures can differ in the last bits.

## 6. Complexity, memory and measured performance

| Stage | Time | Memory |
| --- | --- | --- |
| Pass 1 | `O(N * D)` | 9 bytes per dense row (norm, non-zero count, validity), 16 bytes per row of duplicate hashes (24 while grouping; budgeted by `--duplicate-memory-mb`), one I/O chunk |
| Pass 2 | `O(N * D)` plus `O(N * D * Q)` for exact k-NN (`Q` = own plus cross queries) | `O(Q * k)` neighbor lists, one I/O chunk |
| Pairwise distances | `O(P * D)` | `O(P)` |
| Hubness | `O(H^2 * D)` | `O(H * k)` |
| PCA | `O(S * D^2 + D^3)`; skipped above `--pca-max-dim` | `O(S * D + D^2)` |
| k-means and silhouette | `O(I * S * K * D)` and `O(2000^2 * D)` | `O(S * D)` |
| Quantization simulation | `O(schemes * queries * base * D)` | `O(base * D)` |

Default options (sample 10,000 rows, 1,000 k-NN queries per split with k = 20, 100,000 pairs),
GCC 13 release build, 24 threads (Intel Core Ultra 9 275HX, WSL2), public ann-benchmarks files. The
machine was busy with other builds (load average about 25), so wall times are upper bounds:

| Dataset | Train x dim | Wall time | Peak RSS |
| --- | --- | --- | --- |
| `fashion-mnist-784-euclidean` | 60,000 x 784 | 13.3 s | 191 MB |
| `sift-128-euclidean` | 1,000,000 x 128 | 17.2 s | 166 MB |
| `glove-100-angular` | 1,183,514 x 100 | 15.9 s | 178 MB |

The findings agree with what is known about these datasets: SIFT1M contains 14,538 exact
duplicate rows (1.45%), and GloVe has the lowest contrast and the highest LID of the three
(train split: median relative contrast 1.62, median LID 32.7), consistent with its reputation
as a hard dataset.

The two dimension estimators can disagree. On the SIFT1M train split TwoNN is 3.9 and the
median LID is 14.2. TwoNN reads only the first two neighbors of each query and LID reads the
first 20, so the report shows both. Ties and duplicates do not explain the gap: they exclude
2.4% of the queries, and dropping the 10% of queries with the closest first neighbors raises
TwoNN only to about 5.0.

## 7. Validation strategy

- **Unit tests** (`profile_dataset_test`): RNG outputs against an independent Python reference,
  FIPS 180-4 SHA-256 vectors, moments against direct formulas and across every merge split,
  histogram edges, quantiles against NumPy's definition, distance kernels against direct loops,
  half-precision rounding, top-k order independence, k-NN against brute force for every metric
  and any block split, LID/TwoNN on data of known dimension, eigen-decomposition residuals and
  orthonormality, PCA of planar data, k-means on separated clusters, quantizer bound rules.
- **Fixtures with planted defects**, generated deterministically at test time (no binary files):
  NaN/+Inf/-Inf rows, duplicates, a zero vector, norm outliers, a canary row, labels with an
  unmatched query label, sparse records with an empty and a duplicated record. Tests check the
  exact counts, the ground-truth convention, the label selectivity and the signals.
- **Reader robustness**: records straddling chunk boundaries for several chunk sizes, unsorted
  and repeated ids, truncated streams and corrupt lengths (rejected before allocating), wrong
  ranks, empty or mismatched datasets, unknown types, non-HDF5 files.
- **Determinism and bounds**: identical digests across threads, chunk sizes and runs; seed changes
  only sampled sections; a 100,000-row profile with configured sample sizes stays below 2 GB
  peak memory (an all-pairs matrix would need 80 GB).
- **HTML**: Content-Security-Policy present, no external `src`/`href`/`@import`/`url()`, exactly
  two script elements, escaped file names and model output, and every chart card carries its
  metric key, definition, scope and population or sample size.
- **AI layer**: requests contain no canary values, file name, per-row arrays or credentials;
  the key appears only in the header and never in reports, stdout or stderr; schema violations,
  unknown evidence keys, overclaiming and malformed envelopes are rejected; unreachable providers,
  HTTP 401, retries on 429/503, dry runs and an end-to-end run against a local mock provider
  (then with the provider stopped) all leave valid deterministic reports.
- **Independent reference**: `tools/profile_dataset/validate_profile.py` re-implements the
  sampling in Python, selects the same rows and recomputes full-split metrics and the exact k-NN
  metrics with NumPy. On the dense fixture (both splits), Fashion-MNIST, SIFT1M and GloVe all
  checks pass: sampled rows identical, k-NN distance quantiles, LID and TwoNN equal to about
  10 significant digits, per-dimension means and standard deviations within 1e-11 relative.
- **CI**: the test binary runs in the community and team pipelines next to the other tool tests,
  including the AddressSanitizer build.

## 8. Privacy boundaries

| Asset | Boundary |
| --- | --- |
| Raw vectors and sampled rows | Never leave the process. The AI request is built from an allow list of aggregate report fields; per-row data (projection coordinates, duplicate and outlier examples), per-dimension arrays, histograms, term ids and label values are excluded by construction. Evidence size does not depend on N. |
| File paths and names | Paths are never sent; the file name only with `--ai-include-file-name`. |
| Credentials | Read only from an environment variable, sent only in a request header, redacted from every message; URLs with embedded credentials are rejected; reports store the endpoint without its query string. |
| Untrusted model output | Validated against the request's evidence keys and a schema, HTML-escaped, and stored apart from the deterministic sections; it never changes metrics or signals. |
| Report files | Local artifacts that contain derived per-row data (projection, example row ids); the HTML page blocks all network requests with its CSP. Share them like the dataset. |
| Network | No network access unless an AI provider is configured explicitly; failures fall back to the offline reports. |

## 9. AI prompt and response schema

The request (prompt version `1`, temperature 0) has two messages:

- **System**: role; rules (use only EVIDENCE; cite EVIDENCE keys exactly; recommendations are
  hypotheses to validate with `eval_performance`, no "guaranteed", "always optimal", "best
  possible" or "definitely"; sampled and simulated results are uncertain; use VSAG terms for
  indexes, quantizers, `tq_chain`, SINDI and search parameters; cover characteristics, effects,
  suggestions, uncertainty and missing measurements; reply with one JSON object); for `zh`, write
  the text in Simplified Chinese; then the response schema.
- **User**: the evidence object, a flat map from report keys to values.

Response schema:

```json
{
  "summary": "string, at most 800 characters",
  "findings": [{"title": "", "severity": "info | notice | warning | critical", "detail": "",
                "evidence": ["EVIDENCE key"]}],
  "recommendations": [{"title": "", "category": "index | metric | quantization | transformation | evaluation | data_quality",
                       "hypothesis": "", "rationale": "", "validation": "", "confidence": "low | medium | high",
                       "evidence": ["EVIDENCE key"]}],
  "uncertainties": [""],
  "additional_measurements": [""]
}
```

Validation: the reply may be wrapped in a code fence or prose; the JSON object is extracted and
parsed (at most 256 KB). `summary` and the four arrays are required. Each item must have its
required fields within length limits and valid enumerations; evidence keys not present in the
request are removed and items left without evidence are rejected; recommendations containing
guarantee or optimality claims (English and Chinese phrasings) are rejected; at most 20 items
of each kind are kept. If nothing survives, the status is `invalid_response`. Rejected items
and reasons are kept in `ai_analysis.rejected_items` for review. Statuses: `ok`, `failed`
(configuration, transport or HTTP error), `invalid_response`, `dry_run`.

## 10. Limitations

- Multi-vector datasets are not profiled.
- Exact k-NN costs `O(N * D * queries)`; for very large splits reduce `--knn-queries`.
- Sampled metrics carry sampling error; LID and TwoNN assume local uniformity, and the two can
  disagree because TwoNN reads only the first two neighbors (`two_nn_excluded_ratio` reports how
  many queries it skips).
- The quantization section models VSAG's scalar quantizers (ranges, encoding, query encoding);
  it does not cover PQ, RaBitQ or transform chains and does not replace benchmarks.
- k-means with a small k is a coarse view of partition structure.
- Graph-specific hardness measures (for example Steiner-hardness) would need an index and are
  out of scope.
- The AI layer can still produce plausible but wrong text within the cited evidence; it is
  framed as hypotheses for that reason.
