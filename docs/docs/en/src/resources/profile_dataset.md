# Dataset Profiling Tool (`profile_dataset`)

`profile_dataset` measures the characteristics of a vector dataset that influence index
selection, quantization quality, recall and search speed, and explains them in three layers:

1. a **deterministic profile**: a versioned JSON report computed from the HDF5 file;
2. a **self-contained HTML report** that visualizes every metric and opens offline;
3. an **optional AI-assisted interpretation** that turns the aggregate metrics into findings
   and hypotheses to benchmark. It is off by default and never receives raw vectors.

The tool lives in `tools/profile_dataset/`. The design, sampling and privacy model are
described in the [technical report](profile_dataset_design.md).

## Build

Tools are not built by default. Enable them with `VSAG_ENABLE_TOOLS=ON` (CMake options
`ENABLE_TOOLS=ON` and `ENABLE_CXX11_ABI=ON`):

```bash
VSAG_ENABLE_TOOLS=ON make release
# Output: ./build-release/tools/profile_dataset/profile_dataset
```

HDF5 is built by the project, as for [`eval_performance`](eval.md). The binary does not link
`libvsag`, so it runs without `LD_LIBRARY_PATH`. HTTPS AI endpoints need OpenSSL 3 at build
time (`apt install libssl-dev`, already present where `libcurl4-openssl-dev` is installed).
Without it the tool still builds; AI requests are then limited to `http://` endpoints such as
a local gateway, and the tool says so if an `https://` endpoint is configured.

## Supported inputs

The tool reads the [HDF5 dataset format](dataset_format.md) used by VSAG's evaluation tooling.

| Layout | Selected by | Profiled data |
| --- | --- | --- |
| Dense | `type` attribute `dense` or missing | `/train` and `/test` as `(N, D)` matrices. `float32` and `int8` match `eval_performance`; `uint8`, `float64` and wider integers are read and flagged. |
| Sparse | `type` attribute `sparse` | `/train` and `/test` byte streams of `uint32 len, uint32 ids[len], float32 vals[len]` records; `/train_offsets` and `/test_offsets`, when present, give the record count; a stream with more or fewer records is rejected. |
| Multi-vector | `type` attribute `multi_vector` | Not supported; the tool exits with a clear error. |

`/neighbors` and `/distances` (ground truth) and `/train_labels`, `/test_labels` and
`/valid_ratios` are optional; when present they are checked as well. The metric comes from the
`distance` attribute (`euclidean` → `l2`, `ip` → `ip`, `angular` → `cosine`) unless `--metric`
overrides it. Sparse data supports `ip` and `cosine`. Differences from what `eval_performance`
accepts (for example missing ground truth, `uint8` storage or a missing `distance`
attribute) are listed under `input.eval_compatibility` and shown as warnings.

## Quick start (offline)

```bash
./build-release/tools/profile_dataset/profile_dataset \
    --datapath /tmp/data/sift-128-euclidean.hdf5
```

This profiles every present split (`train`, `test`) and writes
`sift-128-euclidean.profile.json` and `sift-128-euclidean.profile.html` in the current
directory. No network access is needed. Progress goes to stderr; the summary on stdout lists
the report paths, the metrics digest and the rule-based observations:

```text
profile_dataset: sift-128-euclidean.hdf5 (dense, metric l2)
  JSON report: sift-128-euclidean.profile.json
  HTML report: sift-128-euclidean.profile.html
  metrics SHA-256: 3c1f...
  observations: 0 critical 1 warning 3 notice 2 info
  AI analysis: disabled
```

Common options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--datapath`, `-d` | | HDF5 file to profile |
| `--output`, `-o` / `--html` | `<stem>.profile.json` / `<stem>.profile.html` | Report paths; `--no-html` skips the HTML report |
| `--splits` | all present | `train`, `test` or `train,test` |
| `--metric` | `auto` | `auto`, `l2`, `ip` or `cosine` |
| `--seed` | `42` | Seed of every sample |
| `--sample-size` | `10000` | Rows sampled per split for sample-based metrics |
| `--knn-queries`, `--knn-k` | `1000`, `20` | Sampled rows whose exact k-NN is searched in the full split |
| `--pair-count` | `100000` | Random sampled pairs for the pairwise-distance distribution |
| `--clusters` | `16` | k of the sampled k-means summary (`0` disables it) |
| `--pca-max-dim` | `2048` | PCA is skipped above this dimensionality |
| `--disable` | | Any of `duplicates,knn,pca,clustering,hubness,quantization,ground_truth,labels` |
| `--duplicate-memory-mb` | `4096` | Budget for exact duplicate detection (24 bytes per row); above it duplicates are checked on the sample |
| `--threads`, `--chunk-mb` | all cores, `64` | Execution only; results do not depend on them |
| `--from-profile` | | Re-render the HTML report or run the AI analysis from an existing JSON report |

Run `profile_dataset --help` for the full list, including the AI options below.

## Reproducible example

The public [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) datasets already use
this format (`scripts/download_annbench_datasets.sh` downloads several of them):

```bash
mkdir -p /tmp/data
wget http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5 -O /tmp/data/fashion-mnist-784-euclidean.hdf5
./build-release/tools/profile_dataset/profile_dataset \
    -d /tmp/data/fashion-mnist-784-euclidean.hdf5 --seed 42 -o run1.json --html run1.html
./build-release/tools/profile_dataset/profile_dataset \
    -d /tmp/data/fashion-mnist-784-euclidean.hdf5 --seed 42 -o run2.json --no-html --threads 3 --chunk-mb 8
```

Both runs print the same `metrics SHA-256`, and `run1.json` and `run2.json` differ only in
the `run` section (timestamps, duration, threads, peak memory). Changing `--seed` changes the
sample-based sections while full-split sections such as `norms`, `dimensions` and
`duplicates` stay identical.

## Report structure

| Section | Content |
| --- | --- |
| `schema`, `schema_version`, `generator` | `vsag.dataset_profile`, layout version `1.0`, VSAG version of the build |
| `input` | File name and size, vector type, `type`/`distance` attributes, HDF5 datasets, `eval_compatibility`, warnings |
| `config` | Every option that can change a metric value, the resolved metric and its source |
| `splits.<train/test>` | Per-split metric sections (below) |
| `cross_split` | Query (test) versus base (train) comparison |
| `ground_truth`, `labels` | Consistency of `/neighbors` and `/distances`; label and filter statistics |
| `signals` | Rule-based observations with fixed thresholds and the report keys they cite |
| `determinism` | `metrics_sha256` over every section except `run` and `ai_analysis` |
| `run` | Start time, duration, threads, chunk size, peak resident memory (not deterministic) |
| `ai_analysis` | Present only when the AI analysis was enabled (not deterministic) |

Every metric section carries `scope` (`full` for the whole split, `sample` for a sample,
`mixed` when it combines both, with a per-field scope) and a `definition`. Sampled sections
also carry `sample`: the method, the split size (`population`), the sampled and valid sampled
row counts and the rows the analysis actually used (`used_rows`). Histograms record their
bins and the values `below` and `above` the plotted range; summaries always use every value.

## Metric definitions

Distances use VSAG conventions (see [Metric Semantics](metric_semantics.md)): `l2` is squared
L2, `ip` is `1 - <x, y>` and `cosine` is `1 - cos(x, y)`. Estimators that need a true metric
(intrinsic dimensionality, relative contrast, near duplicates) use the *geometric* distance:
Euclidean distance, or chord distance between normalized rows for `cosine`.

### Dense splits

| Key | Scope | Definition | Why it matters |
| --- | --- | --- | --- |
| `basic` | full | Rows, dimensionality, element type | Memory and dtype choice |
| `invalid_values` | full | NaN, +Inf and -Inf coordinates from their IEEE-754 bit patterns; rows containing any | Undefined distances; such rows are excluded from norm and distance metrics |
| `norms` | full | L2 norm of every valid row: summary (min, p01-p99, max, mean, std), histogram, coefficient of variation, `unit_norm_ratio` (\|norm - 1\| <= 1e-3), zero-norm rows, Tukey far-out outliers | Normalized data makes `l2`, `ip` and `cosine` rank alike; spread norms bias `ip` |
| `values` | full | Every finite coordinate: min, max, mean, std, skewness, excess kurtosis, histogram, share beyond 6 standard deviations of its dimension | Scalar-quantization range use |
| `dimensions` | full | Per-dimension mean, std, min, max, zero ratio, skewness, excess kurtosis and 6-sigma share; summary of constant dimensions, std and range ratios, variance share of the top 10% dimensions, heavy-tailed share, mean-vector norm over the mean row norm | Dead dimensions, uneven ranges (per-dimension `sq8` versus global-range `sq8_uniform`) |
| `sparsity` | full | Exact zeros among finite coordinates, all-zero rows, non-zeros per row | Whether a sparse representation fits better |
| `duplicates` | full | Rows with identical canonical values (`-0.0 == 0.0`, one NaN), grouped by a 128-bit hash; examples list row ids | Wasted memory, duplicate-crowded top-k |
| `pairwise_distances` | sample | Metric and geometric distances of random pairs of distinct sampled rows; `concentration_ratio` = std / mean of geometric distances; pairwise cosine similarity | Distance concentration makes neighbors hard to separate |
| `nearest_neighbors` | sample vs full | Exact k-NN of sampled rows searched in the **full** split (the row itself excluded): first and k-th neighbor distances, exact-duplicate neighbors, near duplicates (geometric distance <= `near_duplicate_threshold` x median pairwise distance), isolated rows (Tukey far-out first-neighbor distance) | Range-search radii, duplicates, poorly connected rows |
| `intrinsic_dimensionality` | sample | Levina-Bickel maximum-likelihood local intrinsic dimensionality from the k geometric neighbor distances; TwoNN estimate from first/second neighbor ratios | High LID needs more candidates per query |
| `relative_contrast` | sample | Mean geometric distance to reference sampled rows divided by the first-neighbor distance (He, Kumar and Chang, 2012) | Low contrast means hard search: larger `ef_search` |
| `hubness` | sample | k-occurrence of each sampled row in the k-NN lists of the other sampled rows: skewness, anti-hub share, top-1% share (Radovanovic et al., 2010) | Hubs dominate graph traversals; anti-hubs are hard to reach |
| `anisotropy` | mixed | Mean pairwise cosine (sample), mean-vector norm ratio (full), top principal component share (sample) | Rows in a narrow cone compress cosine distances |
| `pca` | sample | Sample covariance spectrum: explained variance (top 32), cumulative curve, components for 50/80/90/95/99% variance, effective rank, participation ratio; skipped above `--pca-max-dim` | Candidates for a `pca` stage in a `tq_chain` |
| `clustering` | sample | k-means (k-means++ seeding): sorted sizes, size CV, largest over mean size, normalized entropy, silhouette | Coarse partition balance, for example IVF buckets |
| `projection` | sample | 2-D PCA coordinates of sampled rows with their cluster | HTML scatter plot only; never sent to an AI provider |
| `quantization` | sample | Simulated `fp16`, `bf16`, `sq8`, `sq4`, `sq8_uniform`, `sq4_uniform`: recall@k against exact search, mean relative reconstruction error, code size including metadata and alignment. Ranges, floor encoding and the 0.999 clamp follow `ScalarQuantizationTrainer` and the quantizers; as in VSAG, queries are encoded for fp16/bf16 and the uniform quantizers, whose ip/cosine scores use the stored sums of the original values | Which codes need `use_reorder`; a model to confirm with `eval_performance` |

### Sparse splits

| Key | Scope | Definition |
| --- | --- | --- |
| `basic` | full | Records, stored entries, stream bytes, `max_nonzeros` (the `dim` that `eval_performance` derives), offsets present, records with unsorted or repeated ids |
| `invalid_values`, `norms`, `values` | full | As for dense data, over stored values; `values` also counts negative values and explicit zeros |
| `sparsity` | full | Empty records, stored entries per record, density over the id space |
| `terms` | full | Distinct terms, `max_term_id`, `min_term_id_limit` (SINDI `term_id_limit` must be at least this), term-id density, postings, document-frequency summary and log10 histogram, Zipf exponent, Gini, posting share of the top 1% terms, most frequent terms, and `doc_prune_retained_term_fraction`: the mean share of terms SINDI keeps for `doc_prune_ratio` 0.1/0.2/0.3/0.5 under its mass rule |
| `duplicates` | full | Identical records (ids and values) |
| `pairwise_distances` | sample | Metric distance and cosine similarity of random sampled pairs; share of pairs without a shared term |
| `nearest_neighbors` | sample vs full | Exact k-NN through an inverted index over the sampled queries; records sharing no term have distance exactly 1 and take their place in the k-NN lists by row id (above records with a negative inner product); records with non-finite values are excluded; queries with fewer than k overlapping records |
| `hubness` | sample | As for dense data |

### Cross-split, ground truth and labels

| Key | Scope | Definition |
| --- | --- | --- |
| `cross_split.norm_ks_statistic` | full | Two-sample Kolmogorov-Smirnov statistic of train and test norms |
| `cross_split.standardized_mean_shift` | full | Per-dimension \|mean_test - mean_train\| / std_train (dense) |
| `cross_split.out_of_distribution_ratio` | sample | Share of sampled queries whose nearest-train distance exceeds the 95th percentile of train rows' own nearest-neighbor distances |
| `cross_split.query_term_coverage`, `postings_per_query`, `nonzeros_ks_statistic` | sample / full | Sparse only: query terms present in train, postings an exhaustive inverted-index scan touches, KS of entry counts |
| `ground_truth` | sample | Stored `/neighbors` and `/distances` of sampled queries against recomputed exact neighbors: id recall, distance recall (stored ids whose recomputed distance is within the exact k-th distance, so ties are tolerated), the distance convention that best matches the stored distances to the recomputed distances of the stored ids (`sqrt(L2Sqr)` versus squared for `euclidean`; `1 - similarity` versus similarity for `ip`/`angular`), out-of-range ids, unsorted rows |
| `labels` | full | Label counts, largest label share, normalized entropy, per-query filter selectivity, queries whose label no base row carries, `valid_ratios` summary |

### Rule-based observations

`signals` apply fixed thresholds to the metrics above, for example `invalid_values.total > 0`
(critical), `duplicates.duplicate_ratio > 0.01` (warning), `relative_contrast.summary.p50 < 2`,
`hubness.skewness > 1`, `pca.components_for_variance.90 <= 0.25 * dim`, simulated
`sq8` recall@k below 0.95, query shift (norm KS > 0.1 or out-of-distribution ratio > 0.1) and a
stored ground truth that disagrees with recomputed neighbors. Each signal records its rule and
the report keys it cites. They are available without any AI provider and are the reference
the AI output is checked against.

## Determinism, sampling and memory

- Every sample is drawn with an xoshiro256** generator whose seed is derived from `--seed`
  and a stream name (`sample/<split>`, `order/<split>`, `pairs/<split>`, `kmeans/<split>`).
  Rows are drawn uniformly without replacement with Floyd's algorithm, then put in a seeded
  random order; each analysis uses a prefix of that order, so every subset is itself a uniform
  sample. All draws are fully specified, so a seed gives the same rows with any compiler,
  standard library or platform.
- Full-split statistics are computed in fixed 4096-row blocks and merged in block order;
  k-NN lists use a strict (distance, row id) order; k-means and the eigen-decomposition are
  serial and double precision. The target is compiled with `-fno-fast-math -ffp-contract=off`
  and classifies NaN/Inf from bit patterns. Results therefore do not depend on `--threads` or
  `--chunk-mb`, which is checked by `determinism.metrics_sha256`.
- Memory is bounded: per row the tool keeps a norm, a non-zero count and a validity flag
  (13 bytes for dense data) plus 16 bytes of duplicate hashes (24 while grouping, limited by
  `--duplicate-memory-mb`). Sample-based work is bounded by the options: the sample is
  `sample_size x dim` floats, the k-NN state is `knn_queries x k`, and within-sample analyses
  use at most `sample_size` rows. No all-pairs `N x N` matrix is ever built for a split; the
  full split is only streamed, twice.

## HTML report

The HTML report is one file with inline CSS, inline SVG charts and a small inline script. Its
Content-Security-Policy (`default-src 'none'`) prevents any network request and lets only the
page's own script run (pinned by its SHA-256 hash), so it opens from disk without contacting a
service. It contains:

- an overview with headline numbers and the rule-based observations;
- one tab per split; dense and sparse splits use different charts (norms, values,
  per-dimension bands, non-zeros, pairwise and neighbor distances, LID, relative contrast,
  hubness, PCA spectrum, 2-D projection with test queries overlaid, cluster sizes, simulated
  quantization recall; or document frequency, top terms and the SINDI pruning curve);
- query-versus-base, ground-truth and label tabs, the AI analysis tab, and a configuration tab
  with a filterable index of every metric and a button that downloads the embedded JSON.

Every chart card shows its metric key, its definition, and whether it covers the full split
(`N = ...`) or a sample (`n = ... of ...`, seed). Histograms can switch to a log scale, and
evidence links jump to the chart that supports a finding.

## AI-assisted interpretation

The AI analysis is disabled unless a provider is configured explicitly:

| Option | Environment variable | Default | Meaning |
| --- | --- | --- | --- |
| `--ai-provider` | `VSAG_PROFILE_AI_PROVIDER` | `none` | `openai` (any OpenAI-compatible chat-completions API) or `anthropic` (Messages API) |
| `--ai-endpoint` | `VSAG_PROFILE_AI_ENDPOINT` | provider default | Base URL, e.g. `https://api.openai.com/v1`; a URL ending in `/chat/completions` is used as is |
| `--ai-model` | `VSAG_PROFILE_AI_MODEL` | | Model name (required) |
| `--ai-api-key-env` | | `VSAG_PROFILE_AI_API_KEY` | Name of the environment variable that holds the key |
| `--ai-auth-header` | | `Authorization` | Header for the key; `Authorization` sends `Bearer <key>`, other names (e.g. `api-key` for Azure OpenAI) send the raw key |
| `--ai-language` | `VSAG_PROFILE_AI_LANGUAGE` | `en` | Language of the generated text: `en` or `zh` |
| `--ai-timeout`, `--ai-retries` | | `60`, `1` | Per-request timeout in seconds; retries for transport errors, HTTP 429 and 5xx |
| `--ai-dry-run`, `--ai-request-out` | | | Build (and save) the exact request body without sending it |
| `--ai-output` | | | Also write the `ai_analysis` section to a separate file |
| `--ai-required` | | off | Exit with code 3 when the analysis does not succeed (the reports are still written) |

API keys are read only from the environment variable named by `--ai-api-key-env`; the tool
has no option that takes a key, so keys do not end up in shell history or process lists. Keys
are sent only in the request header and are redacted from every message, report and log.
`HTTPS_PROXY`, `HTTP_PROXY` and `NO_PROXY` are honored.

```bash
# OpenAI or any OpenAI-compatible service (DeepSeek, Qwen/DashScope compatible mode, vLLM, ...)
export VSAG_PROFILE_AI_API_KEY=...        # never pass the key as an argument
./build-release/tools/profile_dataset/profile_dataset -d /tmp/data/sift-128-euclidean.hdf5 \
    --ai-provider openai --ai-model gpt-4o-mini

# A local model served by Ollama, no key and no data leaving the machine
./build-release/tools/profile_dataset/profile_dataset -d /tmp/data/sift-128-euclidean.hdf5 \
    --ai-provider openai --ai-endpoint http://127.0.0.1:11434/v1 --ai-model qwen2.5:7b

# Review exactly what would be sent, without sending it
./build-release/tools/profile_dataset/profile_dataset -d /tmp/data/sift-128-euclidean.hdf5 \
    --ai-provider openai --ai-dry-run --ai-request-out request.json

# Interpret an existing report again, in Chinese, without re-profiling
./build-release/tools/profile_dataset/profile_dataset --from-profile sift-128-euclidean.profile.json \
    --ai-provider anthropic --ai-model claude-sonnet-4-5 --ai-language zh
```

### Privacy behavior

The request contains a flat evidence object of allow-listed **aggregate** metrics (summaries,
ratios, counts and short spectra), the rule-based observations, the resolved metric and the
sampling configuration. It never contains vectors, sampled rows, row ids, PCA coordinates,
per-dimension arrays, histograms, term ids, label values, file paths or credentials. The file
name is included only with `--ai-include-file-name`. The evidence size does not depend on the
number of rows. The report records `request_sha256`, the number of evidence keys and
`raw_vectors_sent: false`; `--ai-dry-run` shows the exact payload.

### Response contract and validation

The model must answer with one JSON object: `summary`, `findings` (title, severity, detail,
evidence), `recommendations` (title, category, hypothesis, rationale, validation, confidence,
evidence), `uncertainties` and `additional_measurements`. The tool rejects any item whose
evidence keys are missing or not part of the request, items with invalid fields, and
recommendations phrased as guaranteed or optimal settings; rejected items are listed in
`ai_analysis.rejected_items`. Malformed replies yield `status: invalid_response`, unreachable
providers yield `status: failed`. In every case the deterministic JSON and HTML reports were
already written before the request; the tool prints a warning and exits with code 0 unless
`--ai-required` is set. Model output is escaped in the HTML report.

### Validating AI recommendations

Treat every recommendation as a hypothesis:

1. Check the cited evidence in the HTML report (evidence links jump to the chart) and confirm
   that the observation is real and not a sampling artifact (increase `--sample-size` if a
   sampled metric is borderline).
2. Translate the hypothesis into an [`eval_performance`](eval.md) experiment with a baseline:
   for example, compare `base_quantization_type` `sq8` with and without `use_reorder`, or
   `sq8` against `"rom, sq8_uniform"` in a `tq_chain`, at the same recall target.
3. Tune search parameters (`ef_search`, `scan_buckets_count`) separately for each candidate
   and compare latency and QPS at a fixed recall, using the dataset's real queries.
4. Keep the recommendation only when the benchmark confirms it, and record the profile's
   `metrics_sha256`, the tool version and the benchmark configuration with the result.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Reports written (AI warnings, if any, are printed) |
| `1` | Invalid options or input file |
| `3` | `--ai-required` was set and the AI analysis did not succeed; reports were still written |

## Limitations

- Multi-vector datasets are not supported.
- Sample-based metrics carry sampling error; their `sample` fields state the sample sizes.
- LID and TwoNN assume locally uniform data; relative contrast and hubness depend on k.
- The quantization section models VSAG's quantizers; it does not call them and does not
  replace `eval_performance`.
- Duplicate detection is hash based; the false-positive probability is about `N^2 / 2^129`.
- Intrinsic-dimensionality, contrast, PCA and clustering metrics are not computed for sparse
  data.
- AI output can be wrong even after validation; it is advice to benchmark, not a decision.
