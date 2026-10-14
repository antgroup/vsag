# VSAG Lite reproducible acceptance tools

[Start here: requirements, reports, commands and limitations](ACCEPTANCE.md).
[详细综合验收报告：三数据集、Full对照、CRUD、CPU、优化及限制](ACCEPTANCE_REPORT.md).

- Functionality/architecture/regressions: PR #2904, feature baseline `74a9420`.
- Benchmark and audit tools: this directory; `ENABLE_BENCHMARKS` defaults OFF.
- [Historical raw files and SHA256 manifest](RAW_MANIFEST.json): immutable snapshot `800a822`.
- [Preserved full historical runner guide](https://github.com/nw9663644-eng/vsag/blob/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark/README.md).

Historical replay commands may require their pinned feature/source checkout and
external prepared datasets. Explicit source/hash guards must not be bypassed to
claim current-version measurements. Main/current dataset targets retain the
`lite_benchmark` / `lite_dataset_benchmark` names; optional acceptance targets are
`lite_acceptance_mixed_cpu_acceptance` and `lite_acceptance_query_fixed_snapshot`.
Native Full consumers are independently built via `benchmark/full`, never
substituted by an old Lite baseline.

The latest tree excludes large raw outputs from the review diff; project history
and original server evidence remain intact. New performance output goes to a
fresh external directory and must record source/configuration/input identities.
The current integration does not retrospectively rerun or relabel old experiments.

## Maintained review scope

The current tree retains 29 source/script/test files, plus this guide, the
acceptance index, manifest and final-version evidence. The supported feature
contract is documented in [English](../../docs/docs/en/src/development/lite_first.md)
and [Chinese](../../docs/docs/zh/src/development/lite_first.md), owned by #2904.

- Data preparation: `prepare_sift.py`, `prepare_gist.cpp`, `prepare_cohere.py` and its regression fixture.
- Optional Lite/native Full consumers: the four Lite CMake targets and `full/`.
- Resident CRUD/CPU: `run_mixed_cpu_acceptance.py`, verifier, packer and tests.
- Pinned native Full: fresh build, comparison runner, verifier and regression test.
- Defined runtime bundle: package measurement, verifier and tests.
- Frozen quality/independent oracle: quality runner/verifier, FP64 checker and test.

32 exploration tools/redundant documents are excluded from this review tree;
their paths, hashes, sizes and immutable `c072a42` URLs are in the manifest's
`archived_tools` section. Replay those historical experiments from that pinned
checkout, not by mixing archived candidates into current production. No history
was rewritten and no original server dataset/raw result was deleted.

## Dependencies and scope validation

Cohere preparation requires NumPy, PyArrow and h5py (the shared SIFT utility
module imports h5py). For the validation environment use
`python -m pip install numpy==1.26.4 h5py==3.11.0 pyarrow==17.0.0` in an isolated
environment. Run `python lite/benchmark/test_prepare_cohere.py`; its synthetic
Parquet fixtures check normalization, FP64 truth/ties, original-ID mapping,
query offsets and pre-write rejection of invalid vectors/IDs/ranges. Without
optional dependencies unittest discovery explicitly skips this fixture; a skip
is not a successful data-preparation test. With dependencies it runs all checks
during import; the standalone command is the explicit execution receipt.

Prepare source `train.parquet` / `test.parquet` with
`python lite/benchmark/prepare_cohere.py /path/to/cohere-source /path/to/new-output --counts 10000 100000 --queries 100 --k 10`.
It refuses existing output, records source/output hashes and source-ID files,
normalizes with FP64 then stores FP32, and audits cosine-to-L2 rounding. Historical
10k/100k Cohere reports are linked from the comprehensive report; no new Cohere
performance matrix is claimed by restoring these tools.

SIFT preparation requires Python3, NumPy and h5py. A separate environment can use
`python -m pip install numpy==1.26.4 h5py==3.11.0`; these are the versions used by
the 2026-10-14 scope-validation smoke, not universal minimum versions. That smoke
prepared 1000 base vectors/two queries from the original SIFT HDF5 and checked
base/query prefixes against the existing data. It is not a new performance run.

`full/` uses `find_package(vsag CONFIG REQUIRED)` and therefore needs a complete
native Full development install, including `lib/cmake/vsag`, not just the copied
runtime library/headers used by some frozen measurement receipts. Configure with
`cmake -S lite/benchmark/full -B /path/to/build -DCMAKE_PREFIX_PATH=/path/to/full-sdk`.
Frozen Full comparison runners use their own explicitly recorded link paths.

After pruning: fresh Lite ReleaseON/benchmarks build and CTest5/5, OFF4/4,
three Full consumer builds against the known historical Full SDK, Python19pass/
2skip, eleven runner/verifier help entries, SIFT preparation smoke, all32 archived
tool hashes and final raw archive verification passed. Initial missing h5py/Full
CMake-package failures were environment mismatches, retained in server logs.
This validation is not new Full performance, sanitizer, coverage or hosted CI.

中文：[社区验收入口](ACCEPTANCE.md)将四项产出、功能、脚本、报告和证据对应起来。
原始数据通过固定提交及manifest提供，不在社区代码diff中逐行展开。
