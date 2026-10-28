# VSAG Lite benchmark tools

- [English reproduction guide](../../docs/docs/en/src/development/lite_benchmarks.md) · [中文复现指南](../../docs/docs/zh/src/development/lite_benchmarks.md).
- [Published English docs](https://vsag.io/docs/development/lite_benchmarks.html) · [中文文档](https://vsag.io/docs/zh/development/lite_benchmarks.html) (available after this change is published).
- [Acceptance requirements and protocols](ACCEPTANCE.md) · [Detailed measured report](ACCEPTANCE_REPORT.md).
- [Historical cohorts and recovery](HISTORY.md) · [Original file/hash manifest](RAW_MANIFEST.json).

## Directory responsibilities

| Location / tools | Responsibility |
| --- | --- |
| `prepare_sift.py`, `prepare_gist.cpp`, `prepare_cohere.py` | Prepare the three real datasets and record their identities; source datasets are external |
| `main.cpp`, `dataset_main.cpp`, `mixed_cpu_acceptance.cpp`, `query_fixed_snapshot.cpp` | Optional Lite measurement consumers; `ENABLE_BENCHMARKS` defaults OFF |
| `full/`, `full_*main.cpp`, `build_fresh_full_reference.py` | Separately built native Full consumers and fresh-source build receipt |
| `run_*.py`, `verify_*.py`, `check_quality_target_oracle.py`, `quality_target_fp64.cpp` | Version-bound measurement and independent correctness/quality verification |
| `measure_runtime_package.py`, `verify_runtime_package.py`, `pack_mixed_cpu_acceptance.py` | Defined runtime-package measurement and lossless raw-evidence packaging |
| `test_*.py` | Tool/negative-path, build/documentation and current published-evidence regressions |
| `results/` | E10/E11/E12 reports, original raw archives and receipts; not generated build output |

Run from a clean checkout with new external build/output directories. Keep all
trials, source/configuration/input identities and adverse results. Historical
replay guards are intentional; do not remove them to claim latest-source results.
Read the guides before running Full/package/quality replay: they require explicit
matching source/dependency/archive prerequisites. Raw audit PASS is not a quality
threshold, a cold-I/O result, same-quality superiority or complete project acceptance.

The 2026-10-28 integration follows rebased #2904 `65d8912` and its mainline/CI.
See the guides for source boundaries; historical measurements and published raw
archives are unchanged, and no new private optimization candidate is included.
