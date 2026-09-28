# SAQ reproducible benchmark

This directory contains the benchmark contract and audit trail for the VSAG SAQ implementation.
The authoritative result is `results/formal-final-20260825`; `REPORT.md` explains the measurements
and their relationship to the SAQ paper.

## Fixed Release experiment

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

The script downloads SIFT1M and GIST1M from ann-benchmarks when absent, configures a Release tools
build, runs the direct quantizer/ablation benchmark, builds both HGraph indexes, evaluates the same
search sweep, and generates CSV summaries. Its defaults are the values above except for the output
directory and compiler parallelism.

The fixed comparison uses:

- four configured bits per dimension and an asserted equal complete record length;
- 65,536 deterministic training vectors and all 1,000,000 base vectors for encoding;
- query-to-ground-truth-neighbor distance error (up to 100,000 SIFT and 10,000 GIST pairs);
- HGraph `max_degree=32`, `ef_construction=300`, 16 build/search threads, no reorder store;
- `ef_search=40,80,120,200,400,800,1600` and Top-10; SIFT1M executes its 10,000 independent
  queries once, while GIST1M repeats its 1,000 independent queries ten times for 10,000 timed
  operations;
- SAQ automatic segmentation, six adjustment rounds, PCA, and deterministic segment rotation seed
  `20260825`;
- production fast multi-bit RaBitQ for end-to-end comparison and an additional exact-RaBitQ direct
  encoding control.

## Direct metrics and ablations

`saq_quantization_benchmark` records training and encoding wall time, code bytes, mean relative
distance error, relative MSE, query preparation, scalar and batch-four query-to-code throughput,
code-to-code throughput, and reconstruction on the first 10,000 encoded vectors. Encoding and
distance evaluation still use the configured full base count; the reconstruction cap is an
explicit auxiliary-measurement field, not a reduction of the measured workload.

Every SAQ variant is required to have the same complete code length:

- full dynamic SAQ;
- PCA disabled with rotation retained;
- PCA and rotation disabled;
- rotation disabled with PCA retained;
- zero adjustment rounds with PCA and rotation-disabled control;
- fixed one-segment and fixed two-segment plans with the same no-rotation control.

The fixed one-segment row isolates the value of segmentation. The fixed two-segment row tests a
coarser plan against the learned plan. Because boundaries and bit widths are coupled under a fixed
metadata budget, fixed segmentation is intentionally reported as a plan-level ablation rather than
as a boundary-only change.

## Retained files

Each dataset directory contains:

- `quantization.json` and `quantization.stdout.txt`: complete direct measurements;
- `eval.yaml`: the exact rendered end-to-end configuration;
- `eval.json` and `eval.stdout.txt`: complete HGraph build/search measurements;
- `indexes/`: local serialized indexes (ignored by Git because of size).

At the run root:

- `environment.txt` records start/finish status, elapsed time, Git branch and tracked changes,
  CPU, memory, kernel, compiler/CMake versions, fixed parameters, and linked libraries;
- `all_results.csv` is the full end-to-end curve;
- `target_recall_results.csv` selects the lowest-P50 row reaching 0.95 or, if unavailable, retains
  the highest measured recall with `target_met=false`;
- `common_recall_results.csv` selects the lowest-P50 rows at the highest recall reachable by both
  measured curves;
- `reported_common_recall_results.csv` selects the lowest-P50 rows at the declared 0.85 SIFT1M and
  0.91 GIST1M comparison targets;
- `ablation_results.csv` preserves all direct SAQ factors and learned plans.

Do not compare QPS values at different recall as if they were the same operating point. Use the
full curve or an explicitly stated common target. Historical result directories are retained for
diagnosis but do not support the current result; in particular, runs predating the PCA layout,
mean-centering, metadata-budget, and bit-plane fixes are not valid SAQ comparisons.

See `DESIGN.md` for the algorithm and record layout and `REPORT.md` for the measured results.
