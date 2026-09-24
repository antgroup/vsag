# SAQ reproducible evaluation

## Summary

The authoritative Release run is `results/formal-final-20260825`. It completed successfully on
SIFT1M and GIST1M with one million vectors encoded per method, equal complete record lengths, fixed
HGraph parameters, a shared `ef_search` sweep, exact environment capture, and all configured
ablations.

At four bits per dimension, SAQ improves the direct query-to-neighbor distance estimator over
VSAG's production fast multi-bit RaBitQ:

- SIFT1M: mean relative error is 26.3% lower and relative MSE is 42.1% lower.
- GIST1M: mean relative error is 85.3% lower and relative MSE is 96.2% lower (26.6x).
- SAQ encoding is 7.2% faster on SIFT1M and 4.1% faster on GIST1M than the production fast RaBitQ
  encoder. Against exact RaBitQ encoding, the speedups are 4.10x and 1.81x respectively.

The HGraph result is dataset-dependent but establishes a recall benefit at the same record length.
On GIST1M, SAQ reaches the configured Recall@10 target of 0.95 at `ef_search=400` (0.9531,
2,678.7 QPS), while RaBitQ reaches only 0.9100 at the largest tested `ef_search=1600`. At the common
reachable target of 0.91, SAQ uses `ef_search=200` and provides 2.92x the QPS of RaBitQ. On SIFT1M,
neither method reaches 0.95; SAQ reaches 0.88913 versus RaBitQ's 0.85321. At the stable common target
of 0.85, SAQ provides 1.91x the QPS.

Not every SAQ primitive is faster. The current query-to-code kernel is 18.9%
slower than RaBitQ on SIFT1M and 55.7% slower on GIST1M in the four-candidate microbenchmark, and
SAQ index construction is 2.36x and 1.69x slower. The end-to-end gain at a common recall comes from
the more accurate estimator requiring a much smaller search breadth, not from a faster individual
distance scan.

## Reproducibility and implementation evidence

| Area | Evidence | Status |
| --- | --- | --- |
| Native SAQ training, encoding and search | PCA, centering, segmentation, bit allocation, CAQ, deterministic rotation, persistence and distance kernels | Complete |
| VSAG integration | HGraph, IVF, Pyramid, flatten and bucket datacells; L2, inner product and cosine tests | Complete |
| Two reproducible datasets | Fixed SIFT1M/GIST1M Release run and one-command scripts | Complete |
| Equal-size comparison with RaBitQ | 76 bytes/vector on SIFT1M; 492 bytes/vector on GIST1M | Complete |
| Complete raw evidence | JSON, stdout, rendered YAML, CSV, environment and serialized indexes | Complete |
| Controlled ablations | PCA, deterministic random rotation, adjustment rounds, fixed one/two segments | Complete |
| Documentation | Design, benchmark protocol, bilingual user guides and tool references | Complete |

## Implementation corrections behind the measured result

The earlier negative experiment predates four independent implementation corrections and is not
comparable with the retained result:

1. `LAPACKE_ssyev` row-major eigenvectors were read with the wrong orientation. The PCA regression
   now uses a non-diagonal covariance matrix, so a transposed eigenbasis cannot pass silently.
2. L2 input and query vectors are mean-centered even when PCA is disabled. Earlier no-PCA
   measurements mixed the PCA effect with a centering defect.
3. The record layout now stores the two factors required by the estimator per segment and supports
   1--13-bit, byte-aligned bit planes. This allows the dynamic planner to select meaningful
   non-uniform segments within the same complete record budget.
4. Query-to-code scans use lookup/SIMD four-way batches, while code-to-code distance computes
   decoded moments directly from bit-plane intersections and population counts. This removes dense
   inverse transforms and scalar unpacking from the HGraph construction hot path.

Deserialization additionally validates dimensions, metric, training state, parameter ranges,
ordered/aligned segment plans, exact projection/mean sizes, record-budget arithmetic and fixed
segment count. Every segment rotation must also contain exactly the expected number of finite
matrix elements. Random rotation uses the fixed seed `20260825`, making the Release run
reproducible.

## Protocol

The run started at `2026-08-25T16:15:47+08:00` and finished with `run_exit_status=0` after 2,784
seconds. It used Ubuntu 24.04.4, GCC 13.3.0, CMake 3.28.3 and two Intel Xeon Platinum 8375C sockets.
The CMake build type was Release.

| Parameter | Value |
| --- | ---: |
| Dataset base vectors | 1,000,000 |
| Training vectors | 65,536 |
| Encoded vectors | 1,000,000 |
| Average payload | 4 bits/dimension |
| CAQ adjustment rounds | 6 |
| Random-rotation seed | 20260825 |
| HGraph `max_degree` | 32 |
| HGraph `ef_construction` | 300 |
| Build/search threads | 16 |
| Precise reorder | disabled |
| `ef_search` sweep | 40, 80, 120, 200, 400, 800, 1600 |
| Search `topk` | 10 |
| Target Recall@10 | 0.95 |

Distance error is evaluated on query-to-ground-truth-neighbor pairs, up to ten ranks per query:
100,000 pairs for SIFT1M and 10,000 pairs for GIST1M. Encoding and direct distance error use all
one million base vectors; reconstruction is an explicitly labelled auxiliary measurement capped
at 10,000 vectors. SIFT1M search uses 10,000 independent queries once per point. GIST1M contains
1,000 independent queries; the evaluator repeats that set ten times in round-robin order to collect
10,000 timed operations per point. Recall remains an equal-weight average over the standard query
set, while QPS and P50/P99 describe the repeated timed workload. Each index build and search point
was measured once in the authoritative run; the results do not include run-to-run variance or a
confidence interval.

The production comparison uses VSAG fast RaBitQ with six encoder-adjustment rounds. The direct
benchmark also reports exact RaBitQ, because that is closer to the expensive Extended RaBitQ
encoding baseline used by the SAQ paper. Both SAQ and RaBitQ occupy exactly the same complete record
length; this includes payload, factors and required alignment.

The run records the feature branch, tracked-change state and complete build/search artifacts. The
measured L2 implementation is now integrated into the feature branch; post-run hardening is
distinguished below from the performance snapshot.

Post-run hardening defines finite cosine distance for zero norms, makes query-to-code cosine use the
reconstructed candidate norm, and validates serialized rotation sizes/non-finite values. These
changes affect cosine template instantiations or load-time rejection only; the Release benchmark
uses L2 throughout, and its record budget, encoded L2 values, distance kernels, and measured numbers
are unchanged. The performance numbers are therefore reported as the retained 2026-08-25 run rather
than as a new measurement of the later hardening changes.

## Direct quantization results

| Dataset | Method | Bytes/vector | Train ms | Encode vector/s | Relative MSE | Mean relative error | Batch-4 scan/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SIFT1M | SAQ | 76 | 122.273 | 152,183 | 4.6466e-4 | 0.020094 | 13,157,886 |
| SIFT1M | RaBitQ fast | 76 | 4.251 | 141,995 | 8.0319e-4 | 0.027271 | 16,226,188 |
| SIFT1M | RaBitQ exact | 76 | 4.339 | 37,088 | 9.3705e-4 | 0.029132 | 16,158,375 |
| GIST1M | SAQ | 492 | 8,438.802 | 5,931 | 1.7485e-6 | 0.000909 | 1,614,442 |
| GIST1M | RaBitQ fast | 492 | 94.610 | 5,697 | 4.6437e-5 | 0.006192 | 3,646,315 |
| GIST1M | RaBitQ exact | 492 | 96.238 | 3,271 | 5.5301e-5 | 0.006744 | 3,639,013 |

SAQ's code-to-code moment kernel processes 3.45 million SIFT pairs/s and 391 thousand GIST
pairs/s. Relative to the earlier scalar-decode implementation retained in the historical run, this
is approximately 13.2x and 12.5x faster. It preserves decoded-vector distance semantics and has
regression coverage for dimensions not divisible by 64 and segment widths above eight bits.

The learned plans are:

- SIFT1M: 64 dimensions at 5 bits, then 64 dimensions at 2 bits.
- GIST1M: segment lengths 64/128/192/256/320 at 10/7/5/3/1 bits.

## HGraph end-to-end results

### Configured target and common reachable target

| Dataset | Method | Selected `ef` | Recall@10 | Target met | QPS | P50 ms | P99 ms | Build s | Index bytes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| SIFT1M | SAQ | 1600 | 0.88913 | no | 4,742.1 | 3.449 | 4.436 | 92.122 | 230,326,378 |
| SIFT1M | RaBitQ | 1600 | 0.85321 | no | 5,126.5 | 3.193 | 4.091 | 39.091 | 230,305,725 |
| GIST1M | SAQ | 400 | 0.95310 | yes | 2,678.7 | 6.186 | 7.330 | 547.617 | 650,503,142 |
| GIST1M | RaBitQ | 1600 | 0.91000 | no | 1,518.3 | 10.875 | 13.727 | 324.223 | 649,665,521 |

The configured 0.95 comparison does not produce a RaBitQ speedup denominator on GIST1M, because
RaBitQ never reaches the target. The valid conclusion is that SAQ reaches 0.95 under the fixed
sweep and RaBitQ does not.

For a same-target throughput comparison, `reported_common_recall_results.csv` is generated from the
declared 0.85 SIFT1M and 0.91 GIST1M targets:

| Dataset | Method | Target | `ef` | Recall@10 | QPS | P50 ms | P99 ms | Build peak memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SIFT1M | SAQ | 0.85 | 40 | 0.85834 | 87,866.9 | 0.181 | 0.230 | 324.52 MB |
| SIFT1M | RaBitQ | 0.85 | 120 | 0.85013 | 46,050.2 | 0.353 | 0.481 | 116.22 MB |
| GIST1M | SAQ | 0.91 | 200 | 0.91700 | 4,427.0 | 3.725 | 4.372 | 716.67 MB |
| GIST1M | RaBitQ | 0.91 | 1600 | 0.91000 | 1,518.3 | 10.875 | 13.727 | 456.28 MB |

SAQ is 1.91x faster on SIFT1M and 2.92x faster on GIST1M at these declared minimum-recall
requirements. The corresponding build-memory costs are 2.79x and 1.57x.

`common_recall_results.csv` also selects the exact maximum common SIFT recall of 0.85321. Because
RaBitQ has plateaued and only crosses that numerical value at `ef=1600`, the resulting 17.1x ratio
is mathematically reproducible but too sensitive to the plateau for the primary comparison. The
0.85 comparison above is the conservative interpretation.

### Complete recall/QPS curve

| Dataset | `ef` | SAQ recall | SAQ QPS | RaBitQ recall | RaBitQ QPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIFT1M | 40 | 0.85834 | 87,866.9 | 0.82598 | 102,210.4 |
| SIFT1M | 80 | 0.88118 | 55,407.4 | 0.84611 | 64,552.9 |
| SIFT1M | 120 | 0.88591 | 39,718.9 | 0.85013 | 46,050.2 |
| SIFT1M | 200 | 0.88824 | 27,557.7 | 0.85220 | 30,815.6 |
| SIFT1M | 400 | 0.88901 | 15,514.8 | 0.85309 | 17,504.6 |
| SIFT1M | 800 | 0.88912 | 8,600.0 | 0.85320 | 9,933.4 |
| SIFT1M | 1600 | 0.88913 | 4,742.1 | 0.85321 | 5,126.5 |
| GIST1M | 40 | 0.70780 | 10,769.9 | 0.69200 | 17,590.9 |
| GIST1M | 80 | 0.82090 | 7,888.6 | 0.79040 | 12,923.7 |
| GIST1M | 120 | 0.86870 | 6,189.8 | 0.83170 | 9,964.8 |
| GIST1M | 200 | 0.91700 | 4,427.0 | 0.86830 | 7,604.5 |
| GIST1M | 400 | 0.95310 | 2,678.7 | 0.89530 | 4,716.6 |
| GIST1M | 800 | 0.96930 | 1,586.3 | 0.90470 | 2,594.7 |
| GIST1M | 1600 | 0.97780 | 934.4 | 0.91000 | 1,518.3 |

At the same `ef`, SAQ is slower but more accurate. At a common recall, the accuracy improvement can
more than repay the scan cost by allowing a smaller `ef`. This distinction is essential when
answering whether SAQ improves end-to-end performance.

## Controlled ablations

Relative MSE is used because it measures the estimator directly. Every row keeps the same complete
record length as the full method.

| Variant | SIFT1M relative MSE | GIST1M relative MSE |
| --- | ---: | ---: |
| Full: PCA + rotation + 6 rounds + dynamic plan | 4.6466e-4 | 1.7485e-6 |
| No PCA, rotation retained | 7.6203e-4 | 4.6656e-5 |
| No PCA, no rotation | 9.2907e-4 | 1.7842e-4 |
| PCA, no rotation, 6 rounds | 1.2706e-3 | 2.5109e-6 |
| PCA, no rotation, 0 rounds | 1.3357e-3 | 2.7619e-6 |
| PCA, no rotation, fixed 1 segment | 8.7769e-3 | 4.8434e-2 |
| PCA, no rotation, fixed 2 segments | 1.2706e-3 | 2.9202e-5 |

The ablations identify the source of the measured gain:

- Segmentation is decisive. In the controlled no-rotation rows, GIST1M's dynamic five-segment plan
  has 19,290x lower relative MSE than one fixed segment and 11.6x lower than two fixed segments.
  SIFT1M's dynamic two-segment plan has 6.91x lower relative MSE than one fixed segment; the fixed
  two-segment planner reproduces the same 64/64, 5/2-bit plan and therefore the same error.
- PCA is essential on GIST1M. In the no-rotation control, PCA lowers relative MSE by 98.6% (71.1x)
  versus no PCA. PCA interacts with SIFT's short, highly structured dimensions and the selected
  plan; it is not independently beneficial in that controlled SIFT pair, while the complete
  PCA+rotation+dynamic configuration is the best tested SIFT variant.
- Random rotation is beneficial after PCA and dynamic segmentation: disabling it raises relative
  MSE by 173.5% on SIFT1M and 43.6% on GIST1M. The fixed seed makes this a reproducible one-seed
  result, not a statement about the distribution over seeds.
- Adjustment rounds provide a smaller controlled contribution. Comparing the two no-rotation rows,
  six rounds reduce relative MSE by 4.9% on SIFT1M and 9.1% on GIST1M.

These controls also explain the earlier apparent lack of improvement: the PCA layout defect and
record-layout constraints collapsed both planners to a single segment, so the implementation being
measured did not exercise SAQ's main mechanism.

## Reference-implementation adaptation audit

The current code was checked mechanism by mechanism against the authors' implementation, not only by
comparing benchmark output:

| Mechanism | VSAG adaptation and evidence |
| --- | --- |
| CAQ encoder | Uses the same symmetric range, midpoint levels, `1e-8` adjustment tolerance, default six rounds, cosine-improving coordinate moves and post-adjustment rescale. Tests cover zero vectors, adjustment, 1--13-bit plans and reconstruction. |
| Segmentation | Uses PCA-ordered variance, 64-dimension boundaries, variance/`2^bits` dynamic-programming cost and per-segment factor cost. VSAG additionally charges byte-aligned planes and fixes the total complete-record budget to the production RaBitQ record size. |
| Transforms | Applies full orthogonal PCA globally and deterministic orthogonal rotation per segment. L2 subtracts and restores the training mean; IP/cosine retain inner-product invariance. Non-diagonal PCA and inverse-transform regressions cover the earlier layout defect. |
| Accurate query estimate | Computes each segment as query norm plus stored original norm minus twice the corrected query/code inner product, then sums segments. Scalar and SIMD four-candidate paths are exactly cross-checked. |
| Persistence/index access | Stores a self-describing plan and trained transforms, validates dimensions/budget/layout and rotation size/finiteness on load, and plugs into VSAG's independently addressable datacell records. A real HGraph build/search/serialize/restore/search test covers index persistence; HGraph/IVF/Pyramid parameter tests and flatten/bucket tests cover the remaining integration seams. |
| Paper fast path | Not equivalent: m-bound/error-factor pruning and 32-candidate transposed FastScan are absent. The current path uses a safe L2 segment threshold and four-candidate lookup batching. This is an explicit performance-integration boundary, not a hidden accuracy substitution. |

No remaining formula or production-interface defect was found in this audit. The known difference is
the specialized scan/storage architecture in the last row; the performance analysis below keeps
that boundary explicit.

## Relation to the SAQ paper

The [SAQ paper](https://arxiv.org/html/2509.12086v2) reports up to 80% lower quantization error, more than 80x encoding speedup over Extended
RaBitQ in selected settings, and up to 12.5x ANNS QPS at 95% recall. Its Table 3 reports GIST B4
mean relative error of 0.05% for SAQ and a 5.4x RaBitQ error blowup. The measured VSAG result agrees
with that direction: its GIST values are 0.0909% for SAQ and 0.619% for production fast RaBitQ, a
6.81x blowup. The VSAG absolute SAQ error is 1.82x the paper value, so it is the same order of
magnitude but not a numerical reproduction of the paper implementation.

The speed numbers are not directly interchangeable:

- The paper's encoding baseline is Extended RaBitQ. At GIST B4, its Table 4 reports 16.9 seconds
  for RaBitQ and 3.5 seconds for SAQ (4.8x); the above-80x maximum occurs at GIST B9. VSAG
  production RaBitQ already uses a fast six-round encoder. SAQ is only 4.1--7.2% faster than that
  production path and 1.81--4.10x faster than the exact path included here.
- The paper evaluates an IVF-4096, `nprobe=200`, top-100, probabilistic multi-stage scan with
  AVX-512 and 24 pinned cores. This project evaluates VSAG HGraph, top-10, and a safe segment-level
  threshold. The paper explicitly leaves proximity-graph integration as future work.
- This VSAG implementation has byte lookup/SIMD four-candidate batching but does not implement the
  paper's probabilistic m-bound and bit-level FastScan pipeline. The reference FastScan path
  transposes codes in blocks of 32 candidates, while its independently addressable single-vector
  quantizer explicitly disables FastScan. VSAG graph traversal currently reads independent
  datacell records, so reproducing that path requires a block-transposed persisted layout and a
  matching graph scan/reorder interface rather than a quantizer flag. The missing specialized
  layout is consistent with the slower raw query-to-code scan observed here; it does not change the
  CAQ adjustment or segmented bit-allocation formulas validated above.

The paper's Table 5 is also nuanced: at 95% recall on GIST B5 it reports 4,073 QPS for SAQ versus
4,018 for RaBitQ, while at GIST B2/B3 only SAQ reaches the target. The current GIST B4 HGraph result
has the same qualitative pattern--SAQ reaches 0.95 while RaBitQ does not--but is not a Table 5
replication because the index, top-k and distance pipeline differ.

The authors' reference code is available at <https://github.com/howarlii/saq>.

Therefore the measured alignment is: the quantization-error result is reproduced strongly on
GIST1M and moderately on SIFT1M; common-recall HGraph throughput improves because recall rises; the
paper's maximum encoding and IVF QPS multipliers are neither reproduced nor contradicted by this
different production baseline and index architecture.

## Validation

The current source was rebuilt with coverage instrumentation in the repository's official OpenBLAS
CI image (`vsaglib/vsag:ci-x86-openblas`). The complete unit suite and both non-daily functional
partitions passed:

```text
make DEBUG_BUILD_DIR=./build-coverage COMPILE_JOBS=4 cov
unit:                 85,350,390 assertions in 814 test cases, passed
functional non-HGraph: 1,384,194 assertions in 191 test cases, passed
functional HGraph:                      95 test cases, passed
```

The HGraph partition is unusually expensive under gcov instrumentation. It was reconciled against
the binary's 95-case registration list and executed as completed Catch2 shards plus exact-name
reruns for the two interrupted shards; every registered non-daily HGraph case has a successful
final execution. The longest single case, `HGraph Duplicate Build`, took 7,472.789 seconds.
Existing recall diagnostics remained Catch2 warnings and did not fail a case.

Focused evidence includes 222 assertions in 13 SAQQuantizer cases, 9,926,588 assertions in two PCA
cases, 11,173,824 assertions in three random-orthogonal-transform cases, and 228 assertions in six
SAQ index/datacell integration cases. These explicitly cover zero-vector cosine, decoded-vector
query/code equivalence, trained 9--13-bit plans, malformed rotation state, and a real HGraph index
round trip. The repository CI `clang-tidy-15` changed-source filter passed all ten changed production
`src/**/*.cpp` files with zero errors. Static checks also cover `clang-format-15 --dry-run
--Werror` on every changed C/C++ file, `git diff --check`, Bash syntax, Python byte compilation and
JSON/CSV parsing. The latest-source Release target `saq_quantization_benchmark` links and its
`--help` invocation exits successfully.

The same source also passed the combined SAQ/PCA/rotation suite in independently configured
Debug sanitizer builds:

```text
ASan + UBSan: 21,100,634 assertions in 18 test cases
TSan:         21,100,634 assertions in 18 test cases
```

Earlier monitor and mock-implementation smoke tests also passed 34 assertions in six cases. The
`[daily]` suite is not part of the standard functional-test entry point and is not included in
the reported validation scope.

The host's default virtual-address layout conflicts with the GCC ThreadSanitizer runtime, so the
TSan command was run under `setarch x86_64 -R`; no ThreadSanitizer diagnostic was emitted.

Coverage was collected with lcov v2.3 and the repository's source exclusions. The resulting line
coverage is 85.0% for the full filtered repository (33,287/39,151), 88.2% across the ten changed
production source files (4,097/4,646), and 98.8% for the four files in
`src/quantization/saq_quantization` (742/751); the SAQ directory has 91.3% function coverage
(94/103). The repository's standalone `scripts/coverage/check_cov.sh` threshold is 90%, so the
full-repository result does not meet that optional local threshold. The published Coverage workflow
collects and uploads the trace but does not invoke this threshold script. lcov v2.3 also requires
ignoring one strict consistency diagnostic in `/usr/include/c++/11/chrono` during capture and two
existing consistency diagnostics while reading the filtered trace; none points to an SAQ source
file.

## Artifact map and reproduction

From the repository root:

```bash
SAQ_BUILD_JOBS=2 \
SAQ_TRAIN_COUNT=65536 \
SAQ_ENCODE_COUNT=1000000 \
SAQ_EXACT_RABITQ=1 \
SAQ_TARGET_RECALL=0.95 \
SAQ_SIFT_COMMON_RECALL=0.85 \
SAQ_GIST_COMMON_RECALL=0.91 \
./benchs/saq/run.sh \
  /tmp/vsag-saq-data \
  ./benchs/saq/results/<run-name>
```

The authoritative directory contains:

- `environment.txt`: complete parameters, dirty-state manifest, hardware/software details,
  start/finish times and exit status;
- `{sift1m,gist1m}/quantization.json`: direct metrics and all ablations;
- `{sift1m,gist1m}/eval.json`: unmodified build/search measurements;
- `{sift1m,gist1m}/*.stdout.txt` and `eval.yaml`: complete console output and rendered configs;
- `all_results.csv`, `target_recall_results.csv`, `common_recall_results.csv`,
  `reported_common_recall_results.csv` and `ablation_results.csv`: machine-readable summaries;
- `{sift1m,gist1m}/indexes/*.index`: serialized indexes used by the evaluation.

Superseded runs are retained locally under the workspace-level
`experiments/results/historical/` directory and are not part of the retained benchmark. They must
not be combined with or cited as the current result.
