# VSAG Dataset Profiler (`dataset_profiler`)

A reproducible tool that profiles and visualizes the **feature characteristics of an input
vector dataset** — dimension distributions, value statistics, norms, sparsity, duplicates,
sampled pairwise distances, outliers, anisotropy, and optional clustering — for the dense and
sparse HDF5 formats already used by VSAG's evaluation tooling (and by the same arrays you hand
to `pyvsag`).

The canonical documentation lives in the website source tree.

- Local English docs:
  [docs/docs/en/src/resources/dataset_profiler.md](../../docs/docs/en/src/resources/dataset_profiler.md)
- Local Chinese docs:
  [docs/docs/zh/src/resources/dataset_profiler.md](../../docs/docs/zh/src/resources/dataset_profiler.md)
- HDF5 dataset format:
  [docs/docs/en/src/resources/dataset_format.md](../../docs/docs/en/src/resources/dataset_format.md)
- Website: <https://vsag.io/docs/resources/dataset_profiler.html>
- Chinese website: <https://vsag.io/docs/zh/resources/dataset_profiler.html>
- Design notes (metrics, sampling, privacy, AI schema):
  [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)
- Example datasets and reports: [examples/](examples/README.md)

## Quick start

Requires Python 3.10+ with `numpy` and `h5py` (see [requirements.txt](requirements.txt)). No
C++ build is needed.

```bash
cd tools/dataset_profiler
pip install -r requirements.txt

# Deterministic JSON + self-contained HTML, written next to the input file
python -m dataset_profiler /path/to/my_dataset.hdf5 --split train --seed 42

# Bounded sampling, explicit metric, k-means over the PCA-2 projection
python -m dataset_profiler my_dataset.hdf5 \
    --sample-size 512 --pairwise-sample-size 256 \
    --metric euclidean --clusters 8 \
    --out-json my_dataset.profile.json --out-html my_dataset.profile.html
```

Regenerate the bundled demo end to end with:

```bash
python examples/make_toy_datasets.py
```

## AI-assisted interpretation (optional)

The deterministic reports never require network access. The optional AI layer summarizes the
**aggregate metrics** (never raw vectors) and cites metric keys as evidence. Enable it
explicitly with `--ai` and configure it entirely through environment variables:

```bash
export VSAG_AI_API_KEY=...        # bearer token; never stored in reports or logs
export VSAG_AI_MODEL=...          # model name (OpenAI-compatible endpoint)
export VSAG_AI_BASE_URL=...       # optional, https only
python -m dataset_profiler my_dataset.hdf5 --ai
```

If the provider is unavailable the JSON/HTML reports are still generated and a clear warning
is printed. AI output is validated: every finding/recommendation must reference metric keys
that exist in the report, and recommendations are phrased as hypotheses to validate with
`tools/eval`. See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for the prompt/response schema.

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

## Repository checks

This tool is pure Python and does not affect the C++ library, so `make fmt`/`make lint`/
`make test` are not impacted by changes here; run the pytest suite above instead.
