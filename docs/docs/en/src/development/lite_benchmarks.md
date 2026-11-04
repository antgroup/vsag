# VSAG Lite benchmark reproduction

Use the [acceptance protocol](../../../../../lite/benchmark/ACCEPTANCE.md),
[measured report](../../../../../lite/benchmark/ACCEPTANCE_REPORT.md) and
[evidence identities](../../../../../lite/benchmark/results/manifest.json) together.
Functionality and API are described by the [Lite guide](lite_first.md).
Measurements remain bound to their recorded source, library, compiler and inputs.
Building another integration commit does not relabel the recorded Full results.

## Build and test quick start

Run from a clean Linux x86_64 checkout with C++17, CMake, Ninja, Python3/pip and
the repository-pinned Catch2 source. Resource measurements require `/proc`,
`/dev/shm`, `taskset` and `/usr/bin/time`. Use external directories.

Follow the current repository's `AGENTS.md` and build guidance for tool versions.
The maintained formatting and lint tools are clang-format 20 and clang-tidy 20;
standalone Lite and its consumers remain C++17. Compiler/toolchain validation is
separate from the historical measurement environment recorded in the results.

```bash
BENCH_ROOT=$(pwd -P)
BENCH_WORK=$(mktemp -d /tmp/vsag-lite-accept.XXXXXX)
python3 -m venv "$BENCH_WORK/venv"
. "$BENCH_WORK/venv/bin/activate"
python3 -m pip install numpy==1.26.4 h5py==3.11.0 pyarrow==17.0.0
BENCH_BUILD="$BENCH_WORK/consumer-build"
cmake -S "$BENCH_ROOT/lite" -B "$BENCH_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DENABLE_TESTS=ON -DENABLE_RABITQ_LITE_BACKEND=ON -DENABLE_BENCHMARKS=ON
cmake --build "$BENCH_BUILD" -j2
ctest --test-dir "$BENCH_BUILD" --output-on-failure
export VSAG_MIXED_PROBE="$BENCH_BUILD/lite_acceptance_mixed_cpu_acceptance"
test -x "$VSAG_MIXED_PROBE"
python3 -c 'import numpy, h5py, pyarrow'
python3 -m unittest discover -s lite/benchmark -p 'test_*.py'
```

For offline Catch2 builds pass a matching `FETCHCONTENT_SOURCE_DIR_CATCH2`.
Missing dependencies or a missing compiled probe must not be counted as passing
by skipping tests. These targets default OFF outside an explicit benchmark build.

## Reproduction recipes

### Inputs and truth

Prepare legal copies of SIFT HDF5, GIST little-endian fbin and Cohere Parquet.
Set the source paths explicitly. A fvecs file cannot be renamed to fbin.

```bash
BENCH_SIFT_SOURCE=/absolute/path/to/sift.hdf5
BENCH_GIST_BASE=/absolute/path/to/base.fbin
BENCH_GIST_QUERY=/absolute/path/to/queries.fbin
BENCH_COHERE_SOURCE=/absolute/path/to/cohere-parquet
python3 lite/benchmark/prepare_sift.py "$BENCH_SIFT_SOURCE" "$BENCH_WORK/sift" \
  --counts 10000 --queries 1000 --k 10
c++ -std=c++17 -O3 -I"$BENCH_ROOT/src" lite/benchmark/prepare_gist.cpp \
  -L"$BENCH_BUILD" -Wl,-rpath,"$BENCH_BUILD" -lvsag-lite -o "$BENCH_WORK/prepare-gist"
"$BENCH_WORK/prepare-gist" "$BENCH_GIST_BASE" "$BENCH_GIST_QUERY" "$BENCH_WORK/gist" 10000 1000
python3 lite/benchmark/prepare_cohere.py "$BENCH_COHERE_SOURCE" "$BENCH_WORK/cohere" \
  --counts 10000 --queries 1000 --k 10
```

Keep each preparation manifest and file hash. The query cohort used in the report
must match its input identities; a new query cohort produces new measurements.
Recompute subset truth. The mixed probe uses current-state FP64 truth separately
from preparation-tool truth. Cohere is initially normalized in FP64 then stored
in FP32; modifications are not renormalized and are evaluated with squared L2.

### Source-bound resident CRUD

The source checkout must be clean at the supplied commit. The preparation step
creates a new Release build and `build-source-receipt.json`, recording actual
translation units, options, source hashes and the loaded library. It refuses to
clean or overwrite an existing build directory. Receipts are consistency records,
not signed provenance.

```bash
BENCH_FEATURE="$BENCH_ROOT"
BENCH_FEATURE_SHA=$(git -C "$BENCH_FEATURE" rev-parse HEAD)
BENCH_FEATURE_BUILD="$BENCH_WORK/source-bound-build"
BENCH_DATASET=/absolute/path/to/prepared/scale-10000
BENCH_RESULT="$BENCH_WORK/resident-result"
BENCH_CPU=0
python3 lite/benchmark/run_mixed_cpu_acceptance.py \
  --prepare-build --feature-repo "$BENCH_FEATURE" --feature-sha "$BENCH_FEATURE_SHA" \
  --build "$BENCH_FEATURE_BUILD"
python3 lite/benchmark/run_mixed_cpu_acceptance.py \
  --feature-repo "$BENCH_FEATURE" --feature-sha "$BENCH_FEATURE_SHA" \
  --build "$BENCH_FEATURE_BUILD" --dataset SIFT="$BENCH_DATASET" \
  --resident --trials 3 --rounds 20 --cycles 250 --cpu "$BENCH_CPU" --output "$BENCH_RESULT"
python3 lite/benchmark/verify_mixed_cpu_acceptance.py "$BENCH_RESULT"
python3 lite/benchmark/pack_mixed_cpu_acceptance.py "$BENCH_RESULT"
```

Use an available CPU and record it. Run each dataset separately with its own
label, data path and new output. For the fixed-live-count stability window use
GIST with `--rounds 100`; do not rename that run as hours-long endurance.
Optional `--checkpoint-every` and isolated `--mutation-mode` controls are
quality diagnostics: extra oracle/rebuild/fork work changes their timing scope.
All positive and adverse outcomes must remain in the result.

### Native Full and runtime package

Full consumers are independently built in `lite/benchmark/full`:
`full_main.cpp`, `full_dataset_main.cpp`, `full_rabitq_dataset_main.cpp`.
The comparison runner also resolves its exact helper-source reference before
creating outputs. Supply the helper repository/reference and recorded consumer
bundle explicitly; do not substitute unrelated helpers or a Lite baseline.

```bash
BENCH_FULL_DEPS=/absolute/path/to/matching/dependency-sources
BENCH_ANTLR=/absolute/path/to/antlr4-4.13.2.tar.gz
BENCH_FULL_INSTALL="$BENCH_WORK/full-install"
BENCH_FULL_RECORD="$BENCH_WORK/full-build-record"
BENCH_HELPER_REPO=/absolute/path/to/helper-source-repository
BENCH_HELPER_REF=exact-helper-source-commit
BENCH_CONSUMER_BUNDLE=/absolute/path/to/verified-consumer-source.tar.gz
python3 lite/benchmark/build_fresh_full_reference.py \
  --source "$BENCH_FEATURE" --source-sha "$BENCH_FEATURE_SHA" \
  --deps "$BENCH_FULL_DEPS" --antlr-archive "$BENCH_ANTLR" \
  --install "$BENCH_FULL_INSTALL" --output "$BENCH_FULL_RECORD" --strict-fp
python3 lite/benchmark/run_pinned_full_acceptance.py \
  --feature "$BENCH_FEATURE" --feature-sha "$BENCH_FEATURE_SHA" \
  --feature-build "$BENCH_FEATURE_BUILD" --source-repo "$BENCH_HELPER_REPO" \
  --source-ref "$BENCH_HELPER_REF" --full-install "$BENCH_FULL_INSTALL" \
  --fresh-full-receipt "$BENCH_FULL_RECORD/receipt.json" \
  --dataset SIFT="$BENCH_DATASET" --output "$BENCH_WORK/full-result"
python3 lite/benchmark/verify_pinned_full_acceptance.py "$BENCH_WORK/full-result" \
  --dataset SIFT="$BENCH_DATASET"
cmake --build "$BENCH_FEATURE_BUILD" --target vsag_lite_static -j2
BENCH_LITE_INSTALL="$BENCH_WORK/lite-install"
cmake --install "$BENCH_FEATURE_BUILD" --prefix "$BENCH_LITE_INSTALL"
python3 lite/benchmark/measure_runtime_package.py \
  --feature "$BENCH_FEATURE" --feature-sha "$BENCH_FEATURE_SHA" \
  --lite-build "$BENCH_FEATURE_BUILD" --lite-install "$BENCH_LITE_INSTALL" \
  --full-install "$BENCH_FULL_INSTALL" --fresh-full-receipt "$BENCH_FULL_RECORD/receipt.json" \
  --dataset "$BENCH_DATASET" --source-archive "$BENCH_CONSUMER_BUNDLE" \
  --output "$BENCH_WORK/runtime-package"
python3 lite/benchmark/verify_runtime_package.py "$BENCH_WORK/runtime-package"
```

The bundle digest is pinned by the package tool; preserve that identity instead
of disabling its check. The fixed-reference path uses `--reference-provenance`
when no fresh Full receipt is provided and still enforces the exact library and
header hashes. New-source comparisons require matching fresh Full/Lite receipts.
Record O3/Ofast semantics and actual selected SIMD paths; equal ef is not equal
quality. Package bytes cover the stated application/runtime/loader/notices only.
The fresh Full SDK retains CMake's actual versioned SONAME aliases rather than
assuming a fixed version. Bundle unpacking accepts regular files and same-directory
file-alias chains ending at an archived regular file. Parent traversal, file-valued
ancestors and dangling/cyclic aliases are rejected before any extraction.

### Evidence verification

The three maintained records are `resident-crud`, `full-comparison` and
`dynamic-stability`. `results/manifest.json` binds their archives to measured
source identities, sizes and SHA256. Verify outer members before unpacking
nested archives into new directories.

```bash
BENCH_EVIDENCE="$BENCH_WORK/evidence"
PYTHONPATH=lite/benchmark python3 -c \
  'from pathlib import Path; from pack_mixed_cpu_acceptance import unpack; import sys; unpack(Path(sys.argv[1]), Path(sys.argv[2]))' \
  lite/benchmark/results/resident-crud/raw.tar.gz "$BENCH_EVIDENCE"
for BENCH_NAME in SIFT GIST Cohere; do
  PYTHONPATH=lite/benchmark python3 -c \
    'from pathlib import Path; from pack_mixed_cpu_acceptance import unpack; import sys; unpack(Path(sys.argv[1]), Path(sys.argv[2]))' \
    "$BENCH_EVIDENCE/$BENCH_NAME/raw.tar.gz" "$BENCH_EVIDENCE/$BENCH_NAME/raw"
  python3 lite/benchmark/verify_mixed_cpu_acceptance.py "$BENCH_EVIDENCE/$BENCH_NAME/raw"
done
```

For dynamic stability unpack `measurements/raw.tar.gz` and run the mixed verifier.
For Full unpack `full/raw.tar.gz` and run the Full verifier with matching dataset
paths, then run the package verifier on the unpacked `package` directory.
`quality_target_fp64.cpp` / `check_quality_target_oracle.py` provide an independent
FP64 check outside timed runs. Fixed-snapshot quality calibration requires
explicit inputs and Full receipts; it is not a contractual acceptance threshold.
That calibration clears inherited dynamic-loader overrides and checks both Lite
and Full builders/query consumers against their pinned libraries before timing.
New records include manifest-bound linkage receipts for offline verification.

## Interpreting stability and resources

Functional consistency, quality, RSS and concurrency are separate observations.
Repeated use of a finite query cohort does not establish an independent tail
sample. Current-state truth, not initial or full-dataset truth, evaluates changed
vectors. Save/Load equality is explicit snapshot consistency, not per-operation
WAL/fsync/power-loss durability. Calls are externally serialized.

The report retains measured dynamic recall loss. A flat late-window RSS trace
cannot cancel it or prove zero fragmentation, independent index peak or unlimited
stability. Input/oracle/measurement buffers and allocator-retained pages remain
in process RSS; getrusage high water can include launcher history. Do not backfill
new VmHWM fields into existing records. Warm Load is not strict cold start.

For the synthetic `main.cpp` stability CSV, `resident_rss_kib` is sampled after
CRUD/search and before Save/Load; `peak_rss_kib` is sampled later and is the
process-lifetime high water. Subtracting two high waters is not a per-round peak.
These fields are retained for compatibility and are distinct from the real-data
resident protocol's resources. SIFT preparation uses one N-by-dimension buffer
per query; bounded subsets are required rather than unbounded dataset preparation.

## Tool responsibilities

| File | Responsibility |
| --- | --- |
| `prepare_sift.py` | Read SIFT HDF5; write bounded subsets, query/truth files and input identities. |
| `prepare_gist.cpp` | Read little-endian GIST fbin; prepare Lite inputs and subset truth. |
| `prepare_cohere.py` | Validate Cohere Parquet vectors/IDs, normalize initial inputs and write subsets/truth. |
| `main.cpp` | Synthetic Lite lifecycle, graph/stability and fresh-process load consumer. |
| `dataset_main.cpp` | Prepared real-data Lite lifecycle and query consumer. |
| `mixed_cpu_acceptance.cpp` | Resident or isolated CRUD, current-state FP64 quality, query and resource samples. |
| `run_mixed_cpu_acceptance.py` | Source-bound build receipts and repeated CRUD orchestration; explicit inputs/outputs. |
| `verify_mixed_cpu_acceptance.py` | Independently check complete runs, manifests, operations and quality records. |
| `pack_mixed_cpu_acceptance.py` | Verify and package original evidence; safely unpack into new destinations. |
| `full/CMakeLists.txt` | Build standalone Full consumers against an explicitly installed Full SDK. |
| `full_main.cpp` | Native Full BruteForce synthetic lifecycle baseline. |
| `full_dataset_main.cpp` | Native Full BruteForce prepared-data baseline. |
| `full_rabitq_dataset_main.cpp` | Native HGraph FP32/FP16/RaBitQ prepared-data consumer. |
| `build_fresh_full_reference.py` | Build a clean matching Full source and record compiler/library/header provenance. |
| `run_pinned_full_acceptance.py` | Repeated native Full/Lite comparison with explicit matching source/build receipts. |
| `verify_pinned_full_acceptance.py` | Recompute comparison summaries and validate linked inputs and artifacts. |
| `measure_runtime_package.py` | Measure the defined native runtime dependency closure and relocated execution. |
| `verify_runtime_package.py` | Validate package inventories, bytes, linkage, hashes and measurements. |
| `query_fixed_snapshot.cpp` | Query a fixed Lite/Full snapshot and emit ordered neighbors plus per-query timing. |
| `run_quality_target.py` | Optional fixed-source historical quality calibration; not the default current-source acceptance path. |
| `verify_quality_target.py` | Audit that calibration's query, truth, timing and provenance records. |
| `quality_target_fp64.cpp` | Independent FP64 exact-L2 oracle for fixed-snapshot quality checks. |
| `check_quality_target_oracle.py` | Run the independent oracle outside timed measurement and compare its results. |
| `test_prepare_cohere.py` | Preparation validation, normalization, IDs and malformed-input regressions. |
| `test_prepare_sift.py` | Finite SIFT subsets, duplicate counts, malformed shape and preflight-before-output regressions. |
| `test_mixed_cpu_acceptance.py` | Build identity, resource/lifecycle, isolated mutations and retained CRUD-evidence regressions. |
| `test_pack_mixed_cpu_acceptance.py` | Safe evidence packaging, member validation and malformed-archive regressions. |
| `test_pinned_full_acceptance.py` | Full/Lite source receipts, input preflight and retained comparison evidence. |
| `test_runtime_package.py` | Runtime closure, safe extraction, fresh-source preflight and package verification. |
| `test_quality_target.py` | Optional calibration/oracle audits, provenance checks and invalid-input regressions. |
| `test_benchmark_contracts.py` | Consumer behavior, required dependencies, build/CI integration, docs and evidence navigation. |
