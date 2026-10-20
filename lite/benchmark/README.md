# VSAG Lite reproducible acceptance tools

[Start here: requirements, reports, commands and limitations](ACCEPTANCE.md).
[详细综合验收报告：三数据集、Full对照、CRUD、CPU、优化及限制](ACCEPTANCE_REPORT.md).

- Functionality/architecture/regressions: PR #2904; this integration includes `cba3406`.
  Historical measurements retain their own recorded feature/source versions.
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

The latest tree excludes historical bulk raw outputs from the review diff, but
retains compact, version-bound raw archives and their receipts; project history
and original server evidence remain intact. New performance output goes to a
fresh external directory and must record source/configuration/input identities.
The current integration does not retrospectively rerun or relabel old experiments.

## Document roles

- This README: dependencies, complete reproduction recipes, outputs and verification.
- [ACCEPTANCE.md](ACCEPTANCE.md): four-output reviewer entry and source boundaries.
- [ACCEPTANCE_REPORT.md](ACCEPTANCE_REPORT.md): measured numbers, comparisons and explanations.
- [Historical evidence pointer](../PERFORMANCE_EVIDENCE.md): immutable promotion notes, not another latest-results table.

## Maintained review scope

The current tree retains 30 source/script/test files, plus this guide, the
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

## Reproduction recipes

Run commands from the repository root. The quick start in [ACCEPTANCE.md](ACCEPTANCE.md)
establishes `BENCH_ROOT`, `BENCH_WORK`, `BENCH_BUILD` and the Python environment.
All `/absolute/path/...` values below are explicit prerequisites to replace, not
server-specific defaults. Use new output/snapshot paths: the programs reject
overwriting existing results. Do not remove source/hash guards for convenience.

| Task | Required input/configuration | Output and success check |
| --- | --- | --- |
| Prepare SIFT/GIST/Cohere | Original source in the specified format, counts, query count, top-k | `scale-N/base.fvecs`, `queries.fvecs`, `groundtruth.ivecs`, preparation manifest; Cohere also preserves source IDs |
| Exact Lite/native Full reference | Same prepared data, FP32 squared-L2, distinct snapshot paths; Full SDK separately required | CSV stdout and snapshots; programs check pre/post-load results; not an ANN comparison |
| Mixed graph CRUD | Clean feature checkout/SHA, matching freshly built Lite library, CPU, trials, rounds, cycles | `identity.json`, raw per-call CSVs, `rows.json`, commands/resources, `manifest.json`; independent verifier prints `audit: PASS` |
| Pack and relocate raw evidence | Complete mixed-run output | `raw.tar.gz`, `archive.json`; safe extraction plus verifier must reproduce the same audit |
| Historical Full/package/quality replay | Exactly matching pinned sources, libraries, receipts, inputs and helper objects | Corresponding provenance/report, manifest and raw archive; see guarded recipes below |

### Prepare the three datasets

SIFT expects the ANN-Benchmarks `sift-128-euclidean.hdf5` with `train`/`test`
matrices and Euclidean metadata. GIST expects the recorded `maknee/gist1m` fbin
mirror (little-endian count/dimension header, FP32 rows), not an fvecs file renamed
to fbin. Cohere expects `train.parquet` and `test.parquet` with `id` and 768-value
`emb` columns. Source origins and hashes remain in preparation manifests and
[RAW_MANIFEST.json](RAW_MANIFEST.json); datasets themselves are not committed.

```bash
BENCH_SIFT_SOURCE=/absolute/path/to/sift-128-euclidean.hdf5
BENCH_GIST_BASE=/absolute/path/to/base-100k.fbin
BENCH_GIST_QUERY=/absolute/path/to/queries.fbin
BENCH_COHERE_SOURCE=/absolute/path/to/cohere-parquet-directory
python3 "$BENCH_ROOT/lite/benchmark/prepare_sift.py" \
  "$BENCH_SIFT_SOURCE" "$BENCH_WORK/sift" --counts 10000 100000 --queries 100 --k 10
c++ -std=c++17 -O3 -I"$BENCH_ROOT/src" \
  "$BENCH_ROOT/lite/benchmark/prepare_gist.cpp" -L"$BENCH_BUILD" \
  -Wl,-rpath,"$BENCH_BUILD" -lvsag-lite -o "$BENCH_WORK/prepare-gist"
"$BENCH_WORK/prepare-gist" "$BENCH_GIST_BASE" "$BENCH_GIST_QUERY" \
  "$BENCH_WORK/gist" 100000 100
python3 "$BENCH_ROOT/lite/benchmark/prepare_cohere.py" \
  "$BENCH_COHERE_SOURCE" "$BENCH_WORK/cohere" --counts 10000 100000 --queries 100 --k 10
```

Ground truth is recomputed for each selected base prefix, not copied from the
full dataset. SIFT preparation uses FP32 squared-L2 arithmetic; GIST uses its
retained distance helper. Cohere normalizes in FP64, stores FP32 and computes
FP64 squared-L2 truth with a cosine-rounding audit. These preparation protocols
are not interchangeable; the mixed CRUD probe independently recomputes FP64
truth for the changed data. Training/base queries are not independent held-out
queries merely because a program executes them repeatedly.

### Exact baseline and separately built native Full consumers

`lite_dataset_benchmark` and `full_dataset_benchmark` are FP32 **BruteForce**
consumers. This pair checks the exact baseline and persistence, not the graph
or RaBitQ results in the comprehensive report.

```bash
BENCH_DATASET="$BENCH_WORK/sift/scale-10000"
"$BENCH_BUILD/lite_dataset_benchmark" "$BENCH_DATASET" \
  "$BENCH_WORK/lite-exact.snapshot" > "$BENCH_WORK/lite-exact.csv"
BENCH_FULL_SDK=/absolute/path/to/complete-native-full-sdk
cmake -S "$BENCH_ROOT/lite/benchmark/full" -B "$BENCH_WORK/full-consumers" \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$BENCH_FULL_SDK"
cmake --build "$BENCH_WORK/full-consumers" -j2
"$BENCH_WORK/full-consumers/full_dataset_benchmark" "$BENCH_DATASET" \
  "$BENCH_WORK/full-exact.snapshot" > "$BENCH_WORK/full-exact.csv"
```

The Full SDK must include `lib/cmake/vsag`, headers and native Full libraries.
An old Lite library or a partial receipt-only install is not this SDK.
The separate `full_rabitq_dataset_benchmark DATASET SNAPSHOT MODE` supports
`fp32`, `rabitq1`, `rabitq3x5` and uses its own HGraph configuration; it is not
automatically equal-quality/configuration to Lite. Snapshot formats differ.
Use `/usr/bin/time -v -o RESOURCE_FILE` around a fresh process when collecting
CPU/peak RSS; preserve stdout and configuration too. Peak RSS includes input and
verification buffers, not just index-owned memory. These consumers' same-process
Load timings are not strict cold-start or filesystem-cache-controlled timings.

### Current-version graph CRUD and archive round trip

The feature checkout must be clean and the library must be freshly built from
that checkout. Do not pass a current SHA with a reused historical library.
`BENCH_ROOT` identifies the benchmark code; `BENCH_FEATURE` identifies the measured
feature implementation. They may differ, and both identities are recorded.

```bash
BENCH_FEATURE=/absolute/path/to/clean-feature-checkout
BENCH_FEATURE_SHA=$(git -C "$BENCH_FEATURE" rev-parse HEAD)
BENCH_FEATURE_BUILD="$BENCH_WORK/feature-build"
python3 "$BENCH_ROOT/lite/benchmark/run_mixed_cpu_acceptance.py" \
  --prepare-build --feature-repo "$BENCH_FEATURE" --feature-sha "$BENCH_FEATURE_SHA" \
  --build "$BENCH_FEATURE_BUILD"
BENCH_CPU=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
BENCH_RESULT="$BENCH_WORK/mixed-results"
python3 "$BENCH_ROOT/lite/benchmark/run_mixed_cpu_acceptance.py" \
  --feature-repo "$BENCH_FEATURE" --feature-sha "$BENCH_FEATURE_SHA" \
  --build "$BENCH_FEATURE_BUILD" \
  --dataset "SIFT=$BENCH_WORK/sift/scale-10000" \
  --dataset "GIST=$BENCH_WORK/gist/scale-10000" \
  --dataset "Cohere=$BENCH_WORK/cohere/scale-10000" \
  --trials 3 --rounds 20 --cycles 250 --cpu "$BENCH_CPU" --resident \
  --output "$BENCH_RESULT"
python3 "$BENCH_ROOT/lite/benchmark/verify_mixed_cpu_acceptance.py" "$BENCH_RESULT"
python3 "$BENCH_ROOT/lite/benchmark/pack_mixed_cpu_acceptance.py" "$BENCH_RESULT"
BENCH_UNPACKED="$BENCH_WORK/unpacked-mixed"
test ! -e "$BENCH_UNPACKED"
mkdir "$BENCH_UNPACKED"
PYTHONPATH="$BENCH_ROOT/lite/benchmark" python3 -c \
  'from pack_mixed_cpu_acceptance import unpack; from pathlib import Path; import sys; unpack(Path(sys.argv[1]), Path(sys.argv[2]))' \
  "$BENCH_RESULT/raw.tar.gz" "$BENCH_UNPACKED"
python3 "$BENCH_ROOT/lite/benchmark/verify_mixed_cpu_acceptance.py" "$BENCH_UNPACKED"
```

`--prepare-build` requires a nonexistent build directory and builds only the
standalone Release Lite library with RaBitQ enabled, tests/benchmarks disabled,
and compiler-cache launchers disabled. It preserves its log and writes
`build-source-receipt.json` only after a successful build and unchanged clean
source check. Existing builds are never cleaned or overwritten. A previous
manual build without this receipt must be rebuilt in a new directory, not relabeled.

The measurement checks the source revision and tracked Lite/SIMD/public-header
hashes, CMake source/build paths, production compilation entries, configuration
and library hash, then checks the probe's resolved Lite library. It repeats source
and library checks after measurement and copies the receipt into the raw manifest.
These are local accidental-mismatch guards, not signed or tamper-proof provenance;
system headers, compiler and OS remain environment dependencies. Historical raw
archives without receipts remain historical and are not promoted by this change.

Each trial runs FP32, FP16 and RaBitQ8 in rotated order, with fixed degree 16,
construction ef 128, query ef 512 and k=10. `--resident` keeps the original index
alive; loaded copies check persistence. Snapshots are temporary; sizes/hashes are
retained in audit records, not the snapshot binaries. `samples.csv` records per-API
wall/user/system CPU, `quality.csv` initial/final quality and `states.csv` memory
observations. A `PASS` audit proves completeness and consistency, not a Recall
target, zero fragmentation, concurrent safety or hours-long reliability.

For a small toolchain smoke, prepare SIFT with `--counts 128 256 --queries 5 --k 10`,
use dataset labels `SIFT128` and `SIFT256`, and `--trials 1 --rounds 2 --cycles 3`.
This exercises six processes and packing/unpacking; do not promote its timings
or recall to the formal 10k/100k acceptance tables. The full three-dataset command
above is an explicit new-run recipe, not a claim that this matrix has been run.

### Guarded historical Full, package and quality replay

The three runners check required input files, pinned identities, receipts, tools
and historical helper-source availability before creating an output directory.
Rejected prerequisites leave no new output directory; existing outputs are never
overwritten. Input-content validation, compilation and measurement can still fail
later and retain diagnostics. These checks do not relax historical version guards
or turn the scripts into current-version benchmarks.

The following are **historical replays**, not generic latest-source commands.
Replace paths with the matching recorded inputs. `BENCH_PINNED_FEATURE` must be a
clean `dbb2fe0bec685ae7a343b1d2060d0553a98e20d2` checkout; its Lite build must match.
`BENCH_HISTORY` must contain helper commit `a62e62e08db3f0d7da2e4439785feed6fcf90257`.
Source objects, original datasets and dependency archives are external prerequisites,
not implicitly supplied by this checkout. Missing receipts or hash mismatches are
failures to resolve, not reasons to edit expected hashes.

```bash
BENCH_PINNED_FEATURE=/absolute/path/to/clean-dbb2fe0-checkout
BENCH_PINNED_LITE_BUILD=/absolute/path/to/matching-lite-build
BENCH_HISTORY=/absolute/path/to/repository-with-pinned-helper-objects
BENCH_DEPS=/absolute/path/to/extracted-dependency-sources
BENCH_ANTLR_ARCHIVE=/absolute/path/to/matching-antlr4-4.13.2-archive
python3 "$BENCH_ROOT/lite/benchmark/build_fresh_full_reference.py" \
  --source "$BENCH_PINNED_FEATURE" --deps "$BENCH_DEPS" \
  --antlr-archive "$BENCH_ANTLR_ARCHIVE" --install "$BENCH_WORK/pinned-full" \
  --output "$BENCH_WORK/full-build-receipt"
python3 "$BENCH_ROOT/lite/benchmark/run_pinned_full_acceptance.py" \
  --feature "$BENCH_PINNED_FEATURE" --feature-build "$BENCH_PINNED_LITE_BUILD" \
  --source-repo "$BENCH_HISTORY" --source-ref a62e62e08db3f0d7da2e4439785feed6fcf90257 \
  --full-install "$BENCH_WORK/pinned-full" \
  --fresh-full-receipt "$BENCH_WORK/full-build-receipt/receipt.json" \
  --dataset "SIFT=$BENCH_WORK/sift/scale-10000" \
  --dataset "GIST=$BENCH_WORK/gist/scale-10000" --output "$BENCH_WORK/full-replay"
python3 "$BENCH_ROOT/lite/benchmark/verify_pinned_full_acceptance.py" \
  "$BENCH_WORK/full-replay" --dataset "SIFT=$BENCH_WORK/sift/scale-10000" \
  --dataset "GIST=$BENCH_WORK/gist/scale-10000"
```

The build helper saves `receipt.json`, commands, object/header/library identities
and matching runtime headers/library; it does not create the complete CMake SDK
used by `full/`. It uses temporary `/dev/shm` build space and cached dependency
sources, not cached objects; check RAM, dependency versions and system libraries
before starting. Comparison outputs include `provenance.json`, per-query results,
loader records, `manifest.json`, `raw.tar.gz` and `archive.json`. Equal ef is not
equal Recall; the full unit suite is not run by this build helper.

Package replay additionally requires the original recorded Lite/Full installs,
not the newly rebuilt Full library above. Its trusted consumer-source archive is
`results/equal-ef-native-full-20261012/raw.tar.gz` in immutable snapshot `800a822`,
with hash recorded in `RAW_MANIFEST.json`. The script checks exact archive/library
identities and requires Debian package-owner/license metadata, `ldd` and `strip`.

```bash
BENCH_RECORDED_LITE=/absolute/path/to/recorded-lite-install
BENCH_RECORDED_FULL=/absolute/path/to/recorded-full-install
BENCH_SOURCE_ARCHIVE=/absolute/path/to/recorded-equal-ef-raw.tar.gz
python3 "$BENCH_ROOT/lite/benchmark/measure_runtime_package.py" \
  --feature "$BENCH_PINNED_FEATURE" --lite-install "$BENCH_RECORDED_LITE" \
  --full-install "$BENCH_RECORDED_FULL" --source-archive "$BENCH_SOURCE_ARCHIVE" \
  --dataset "$BENCH_WORK/sift/scale-10000" --output "$BENCH_WORK/package-replay"
python3 "$BENCH_ROOT/lite/benchmark/verify_runtime_package.py" "$BENCH_WORK/package-replay"
```

`report.json` inventories the app/runtime/loader/launcher/notices, compressed sizes
and six relocated smokes. The temporary runnable bundles are not released SDKs,
wheels or containers; output retains accounting/verification receipts, not a
permanent binary distribution. Do not call a stripped ELF closure a complete SDK.

Quality replay requires the recorded raw GIST base/query fbin and prior prepared
100-query dataset, plus the matching Full receipt and helper objects:

```bash
BENCH_RAW_GIST=/absolute/path/to/recorded-raw-gist-directory
BENCH_PRIOR_GIST=/absolute/path/to/recorded-gist-10k-100-query-directory
python3 "$BENCH_ROOT/lite/benchmark/run_quality_target.py" \
  --feature "$BENCH_PINNED_FEATURE" --lite-build "$BENCH_PINNED_LITE_BUILD" \
  --full-install "$BENCH_WORK/pinned-full" \
  --full-receipt "$BENCH_WORK/full-build-receipt/receipt.json" \
  --source-repo "$BENCH_HISTORY" --source-ref a62e62e08db3f0d7da2e4439785feed6fcf90257 \
  --raw-gist "$BENCH_RAW_GIST" --prior-dataset "$BENCH_PRIOR_GIST" \
  --output "$BENCH_WORK/quality-replay"
python3 "$BENCH_ROOT/lite/benchmark/verify_quality_target.py" \
  "$BENCH_WORK/quality-replay" --dataset "$BENCH_WORK/quality-replay/dataset/scale-10000"
```

The protocol writes calibration/held-out rows, frozen selections, resource records
and archives; the verifier also writes `verification.json`. Audit success does not
mean the diagnostic Recall target was met. Do not tune again on already observed
queries and still call them unseen. Historical commands/identities remain in the
pinned reports linked by [ACCEPTANCE.md](ACCEPTANCE.md); no old performance numbers
are updated merely because documentation or the integration branch changes.

## Dependencies and scope validation

Cohere preparation requires NumPy, PyArrow and h5py (the shared SIFT utility
module imports h5py). For the validation environment use
`python -m pip install numpy==1.26.4 h5py==3.11.0 pyarrow==17.0.0` in an isolated
environment. Run `python lite/benchmark/test_prepare_cohere.py`; its synthetic
Parquet fixtures check normalization, FP64 truth/ties, original-ID mapping,
query offsets and pre-write rejection of invalid vectors/IDs/ranges. Without
optional dependencies unittest discovery explicitly skips this fixture; a skip
is not a successful data-preparation test. With dependencies it runs four normal
unittest cases, not import-time assertions. The RaBitQ-ON CI installs dependencies
and checks imports before running discovery, so missing packages fail that job.

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

`test_benchmark_contracts.py` compiles the actual percentile and record-reader
helpers without timing an index. It checks P0/P50/P99/P100, singleton/empty and
invalid fractions, input preservation, and malformed records/dimensions. Native
single-query results use one row with the neighbor count in `GetDim()`; switching
the count check to `GetNumElements()==k` would not follow the Full API contract.
The same test module guards the merged CI workflow: benchmark builds and required
Cohere dependency/import checks must remain alongside the Lite CI contracts and
the 90% scoped coverage gate.
The RaBitQ-ON job also exports `VSAG_MIXED_PROBE` to the built acceptance executable
and requires it to be executable before discovery, so the resident/reloaded CRUD
lifecycle and invalid-mode integration cases execute rather than silently skip.

`run_quality_target.py` intentionally replays a pinned historical protocol, not
an automatic latest-source benchmark. It requires `--full-receipt /path/to/receipt.json`
from a matching fresh Full build, plus matching feature/build/install/data paths.
The recorded receipt is preserved with the fresh Full report linked from ACCEPTANCE.md;
it is not implicitly available in this checkout. Missing receipt arguments/files
are rejected before output creation. Do not bypass source/library identity guards.
Example entry: `python3 lite/benchmark/run_quality_target.py --full-receipt /path/to/receipt.json --feature /path/to/pinned-feature --lite-build /path/to/matching-build --full-install /path/to/matching-full --output /path/to/new-results`.
Use `--help` for raw GIST and prepared dataset path overrides.

After pruning: fresh Lite ReleaseON/benchmarks build and CTest5/5, OFF4/4,
three Full consumer builds against the known historical Full SDK, Python19pass/
2skip, eleven runner/verifier help entries, SIFT preparation smoke, all32 archived
tool hashes and final raw archive verification passed. Initial missing h5py/Full
CMake-package failures were environment mismatches, retained in server logs.
This validation is not new Full performance, sanitizer, coverage or hosted CI.

中文：[社区验收入口](ACCEPTANCE.md)将四项产出、功能、脚本、报告和证据对应起来。
原始数据通过固定提交及manifest提供，不在社区代码diff中逐行展开。
