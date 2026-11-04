# Dataset Feature Visualization (`dataset_profiler`)

VSAG ships tools for evaluating index performance (`eval_performance`) and for visualizing the
serialized layout of an index (`visualize_index`). The **dataset profiler** completes the picture
from the other side: it analyzes and visualizes the *feature characteristics of an input vector
dataset* — the vector norms, per-dimension value distributions, sparsity, duplicates, sampled
distance distributions, outliers, and anisotropy that influence index selection, quantization
quality, recall, and search performance.

It is a pure-Python tool (`tools/dataset_profiler`) that understands the same dense and sparse
HDF5 files consumed by `eval_performance`, and it produces two artifacts:

- a **versioned JSON report** with every metric tagged by definition, scope (full or sampled),
  and sample size;
- a **self-contained interactive HTML report** (inline CSS/JS/SVG, opens offline);

plus an **optional AI-assisted interpretation** that explains the *aggregate metrics* — never
raw vectors — and phrases each suggestion as a hypothesis backed by cited metric keys.

## Installation

The profiler requires Python 3.10+ with NumPy and h5py; it is not part of the C++ build.

```bash
cd tools/dataset_profiler
pip install -r requirements.txt
```

## Quick start

```bash
# Profile the train split with the default deterministic settings.
python -m dataset_profiler /path/to/my_dataset.hdf5
```

By default this writes `my_dataset.profile.json` and `my_dataset.profile.html` next to the
input file. Use `--out-json` / `--out-html` to place them elsewhere (pass `-` to `--out-json`
to print the report to stdout instead).

A richer invocation:

```bash
python -m dataset_profiler my_dataset.hdf5 \
    --split train --seed 42 \
    --sample-size 512 --pairwise-sample-size 256 \
    --metric euclidean --clusters 8 \
    --out-json my_dataset.profile.json --out-html my_dataset.profile.html
```

`tools/dataset_profiler/examples/make_toy_datasets.py` regenerates the bundled demo datasets
and reports end to end, and the `tools/dataset_profiler/examples/output/` directory contains
the committed example JSON/HTML files.

## Supported input formats

The tool reads the layouts defined in [HDF5 Dataset Format](dataset_format.md):

- **Dense** (`type` absent or `"dense"`): `/<split>` is an `(N, D)` matrix of `INT8` or
  `FLOAT32` values.
- **Sparse** (`type="sparse"`): `/<split>` is a flat byte stream of
  `uint32 len | uint32 ids[len] | float32 vals[len]` records, optionally indexed by
  `/<split>_offsets`. Sparse dimensionality is inferred as `max(feature id) + 1`.

`multi_vector` files are rejected with an explanatory error. Data is streamed in bounded
chunks: the profiler never materializes a complete dataset, and pairwise work is capped by
the configured sample size, so no `O(N²)` matrix is ever allocated over the full dataset.

To profile arrays you only have in memory — for example the same arrays you pass to
`pyvsag` — first export them with h5py to the layout above; see
[tools/dataset_profiler/examples/README.md](https://github.com/antgroup/vsag/tree/main/tools/dataset_profiler/examples).

## Metrics

Every metric in the JSON report carries a human-readable `definition`, a `scope` of `full`,
`sample`, or `mixed`, and the `sample_size` used. The HTML report repeats this provenance per
section.

| Metric | Scope | Description |
| --- | --- | --- |
| Dataset summary | full | Count, dimensionality, element type, file attributes. |
| `norms` | full | L2 norm distribution of every vector. |
| `dimensions` | full | Per-coordinate mean, std, min, max, zero ratio; top-variance and top-zero dimensions listed separately; long series are bucket-averaged and flagged. |
| `sparsity` | full | Zero/non-zero counts and density; for sparse files also stored entries and nnz-per-vector distribution. |
| `invalid_values` | full | NaN/Inf counts (values and affected vectors). Invalid entries are excluded from numeric summaries and treated as zero elsewhere. |
| `outliers` | full | Robust norm outliers via modified z-score (median/MAD, IQR fallback). |
| `duplicates` | mixed | Exact duplicates (full, byte-identical) plus a near-duplicate indicator over sampled pairs (`distance ≤ epsilon × median sampled distance`); inner-product inputs use non-negative Euclidean distance for this indicator. |
| `pairwise_distances` | sample | Distance distribution between pairs of a bounded random sample. |
| `nearest_neighbor_distances` | sample | Per sampled vector, distance to the nearest other sampled vector. |
| `anisotropy` | sample | Centered-covariance eigenvalue concentration: top-1/top-10 share, effective rank, mean pairwise cosine. |
| `clustering` | sample | Deterministic PCA-2 projection (scatter) and optional seeded k-means summary (`--clusters K`). |

## Sampling and reproducibility

- All sampling uses an explicit `--seed` (default 42) with a dedicated PCG64 generator; the
  method string is recorded in the report.
- Sample indices are drawn with Floyd's algorithm (bounded by the sample size, streaming-safe);
  a seeded permutation selects the pairwise subset without favoring low-index rows.
- `--sample-size` bounds distribution/reduction work; `--pairwise-sample-size` bounds the
  distance matrix; both effective sizes are reported alongside the requested ones.
- Two runs with identical input, configuration, sample sizes, and seed produce byte-identical
  JSON reports (keys sorted, fixed indent). Sparse results are identical whether or not the
  file provides `*_offsets`.

## Privacy and offline behavior

- The deterministic JSON and HTML reports are generated entirely offline. The HTML inlines all
  assets and contacts no external service when opened.
- Raw vectors are never written into the report and never sent to an AI provider: the AI
  request contains only aggregate metrics, sampling configuration, warnings, and bounded
  non-sensitive metadata.
- AI credentials come only from environment variables, are transmitted exclusively over https
  in the `Authorization` header, and are never included in reports, command output, or error
  messages (errors are redacted). HTTP redirects are rejected so credentials cannot be forwarded.

## Optional AI-assisted interpretation

AI analysis is off by default. Enable it with `--ai` (or by setting
`VSAG_DATASET_PROFILER_AI=1`) and configure it through the environment:

```bash
export VSAG_AI_API_KEY=...      # bearer token (fallback: OPENAI_API_KEY)
export VSAG_AI_MODEL=...        # required model name
export VSAG_AI_BASE_URL=...     # optional https OpenAI-compatible endpoint
python -m dataset_profiler my_dataset.hdf5 --ai
```

The provider receives the aggregate profile and must return a structured analysis:
`summary`, `findings`, `recommendations` (each citing evidence keys that exist in the report
and phrased as hypotheses with a `validation` step), `limitations`, and
`additional_measurements`. The profiler validates the response before merging it into the
report as `ai_analysis`; malformed responses, unknown evidence references, and provider
failures never block report generation — they are surfaced as an `ai_analysis.status =
"error"` entry in the JSON and a visible warning box in the HTML, plus a terminal warning.

### Validating AI recommendations

Treat every recommendation as a hypothesis, not a decision:

1. Reproduce the measurement yourself — open the HTML report and check the cited metric keys
   and their scope/sample size.
2. Run the suggested experiment in `eval_performance` (see the [Evaluation Tool](eval.md)) with
   your real query set and ground truth; AI interpretation never replaces a reproducible
   benchmark.
3. Re-profile after any data transformation (normalization, rotation, pruning) and compare
   the before/after metric keys cited by the recommendation.

## Tests and reference

Run the tool's unit and integration tests with:

```bash
cd tools/dataset_profiler && python -m pytest tests -q
```

Design notes — metric selection, sampling internals, privacy boundaries, and the AI
prompt/response schema — are in
[`tools/dataset_profiler/TECHNICAL_REPORT.md`](https://github.com/antgroup/vsag/tree/main/tools/dataset_profiler/TECHNICAL_REPORT.md).

## Related

- [HDF5 Dataset Format](dataset_format.md)
- [Evaluation Tool](eval.md)
- [Index Analysis Tool](analyze_index.md)
- [FAQ: how data distribution affects vector search](faq.md)
