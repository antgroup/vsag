# pyvsag single-query `knn_search` overhead benchmark

This directory contains a standalone harness for issue #3295. It measures the latency of one Python `index.knn_search(vector=..., k=..., parameters=...)` call at a time, including Python binding/conversion work and native HGraph search. It does **not** measure index build time, batch throughput, recall, or concurrent queries, and it does not modify the implementation in PR #3296.

Each invocation benchmarks one imported `pyvsag`. Run the pybind11 baseline and the cffi candidate in separate, otherwise equivalent environments on the same machine, then compare their JSON files.

## Requirements

- A platform supported by the selected `pyvsag` build (the published baseline wheel is commonly run on Linux x86-64).
- Python, NumPy, and the matching `pyvsag` package.
- For the cffi run, the `libvsag_c.so` and cffi-based Python package produced from the **same #3296 revision**. Do not point a pybind11 wheel at the C ABI library.

The harness uses a fixed seed (`3295`), HGraph with fp32/l2, contiguous `float32` vectors, `int64` IDs, and at most 256 deterministically generated query vectors. Index construction and warmup calls are excluded. Garbage collection is disabled only during the measured loop and restored afterward. The JSON output includes the environment, complete configuration, aggregate statistics, and raw per-call nanosecond samples.

## Run the pybind11 baseline

Create an isolated environment and install the published wheel (no VSAG source build is needed):

```bash
python3 -m venv .venv-pybind11
source .venv-pybind11/bin/activate
python -m pip install pyvsag==0.18.5 numpy

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchs/pyvsag_overhead/bench_knn_overhead.py \
  --label pybind11-0.18.5 \
  --num-vectors 5000 --iterations 100 \
  --json-out pybind11-small.json
```

The default benchmark uses `--num-vectors 20000 --dim 128 --k 10 --iterations 200 --warmup 20 --ef-search 100`. Omit `--library` so the benchmark imports the installed wheel unchanged.

## Run the #3296 cffi candidate

Obtain `libvsag_c.so` and its matching cffi `pyvsag` package by following #3296's build instructions in a separate checkout, or by downloading artifacts for that PR's exact revision. Make that cffi package importable in a clean environment, then run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchs/pyvsag_overhead/bench_knn_overhead.py \
  --library /path/to/libvsag_c.so \
  --label cffi-3296-REVISION \
  --num-vectors 5000 --iterations 100 \
  --json-out cffi-small.json
```

`--library` resolves the path and sets `PYVSAG_LIBRARY` **before** importing `pyvsag`. Replace `REVISION` in the label with the actual #3296 commit. Use the same non-binding arguments for both implementations. For acceptance runs, use at least 200 iterations (1,000 or more is preferable for p99) and repeat each side at least three times; keep the host otherwise idle and record power/CPU policies where relevant.

## Suggested acceptance thresholds

These are proposed initial review gates, **not measured results or an already adopted VSAG standard**. For matching runs, define `ratio = cffi / pybind11` and `delta = cffi - pybind11` independently for each statistic. Compare the median of at least three run-level results:

- `p50 ratio <= 1.5x`;
- `p99 ratio <= 2.0x`; and
- `p50 absolute delta <= 5 us`.

Report p90, mean, min, and all raw deltas as diagnostics. Investigate any repeated p99 breach rather than hiding it by averaging runs. A single pair of short runs is insufficient for acceptance because p99 is sensitive to scheduling and frequency changes.

Rationale: a single query is normally in the microsecond-to-millisecond range, while the fixed cost of one cffi call is expected to be sub-microsecond. The relative limits catch regressions when native search is cheap; the additional 5 us p50 budget bounds fixed binding, conversion, allocation, and error-handling overhead when native search is expensive.

Applicability boundaries:

- These limits are intended for the default shape (`N=20000`, `dim=128`, `k=10`, `ef_search=100`) and nearby single-query workloads on one unchanged host.
- Larger datasets or `ef_search` values increase native search time, making ratios less sensitive; retain the absolute p50 check and report the actual delta.
- Very small datasets or `ef_search` values can make timer and scheduler noise significant relative to total latency. Increase iterations/repetitions and inspect both ratio and absolute delta instead of extrapolating these gates.
- Different `dim` or `k` values change input/output marshalling costs and need their own baseline. These gates do not cover batch, concurrent, build, or serialization performance.

## Results in this PR

Only measurements actually executed for this change are entered here. No cffi-versus-pybind11 overhead claim is made without matching runs of both implementations.

| Implementation | Source/revision | Host | Configuration | Iterations/warmup | p50 (us) | p90 (us) | p99 (us) | mean (us) | min (us) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| pybind11 baseline | `pyvsag==0.18.5` wheel | Ubuntu 24.04 WSL2; AMD Ryzen 7 9700X; CPython 3.12.3; NumPy 2.5.4 | `N=5000`, `dim=128`, `k=10`, `ef_search=100` | 100/20 | 141.850 | 181.230 | 290.715 | 148.056 | 101.100 |
| #3296 cffi | not measured in this PR | not measured in this PR | not measured in this PR | not measured in this PR | not measured in this PR | not measured in this PR | not measured in this PR | not measured in this PR | not measured in this PR |

The baseline row is one small smoke measurement made on 2026-11-09 with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; it is not a three-run acceptance result. The exact harness arguments were `--label pybind11-0.18.5 --num-vectors 5000 --iterations 100` (all other options at their defaults). No cffi ratio or delta can be calculated from this single-implementation result.

The cffi row requires a compatible #3296 cffi Python package plus its exact `libvsag_c.so` on a supported machine; those artifacts were not built as part of this incremental benchmark-only change.

## Unit test

The percentile definition is linear interpolation over sorted samples (the same method as `numpy.percentile(..., method="linear")`). Its tests require only the Python standard library and pytest, not NumPy or pyvsag:

```bash
python -m pytest benchs/pyvsag_overhead/test_stats.py -q
python benchs/pyvsag_overhead/bench_knn_overhead.py --help
```

Dependency, index construction, query, and output-file failures produce explicit errors and a nonzero exit status.
