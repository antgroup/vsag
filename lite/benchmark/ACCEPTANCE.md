# VSAG Lite acceptance protocol

The feature PR owns standalone Lite, architecture, API, supported backends and regressions.
This directory supplies measurements, independent verification and the [measured report](ACCEPTANCE_REPORT.md).
See [evidence identities](results/manifest.json) and the [reproduction guide](../../docs/docs/en/src/development/lite_benchmarks.md).

## Requirements

| Output | Evidence |
| --- | --- |
| Reviewable standalone/configurable Lite, CRUD/Search/SaveLoad, tests and usage | Feature PR and bilingual Lite guides |
| Retained/cut modules, index/layout/memory/API design | Canonical Lite and RaBitQ design guides |
| At least two datasets/scales; package/snapshot/load/peak/CRUD/P50/P99/recall | Three-dataset resident CRUD, native Full comparison and dynamic stability |
| At least one resource improvement, functional correctness and stable retrieval | Bounded runtime-package gain and explicit dynamic quality limitations |

## Resident protocol

Use SIFT128, GIST960 and Cohere768, each with 10000 live vectors and 1000 query rows.
GIST has 998 distinct queries. Set degree16/build128/query512/k10 and one available CPU/thread.
Run `--resident --trials 3 --rounds 20 --cycles 250` for each dataset into new output directories.
Update uses FP32(.875*current+.125*original donor); Remove/Add keeps the live count fixed.
Keep the original index resident; Load copies verify snapshots without resetting it.
Calculate initial/final current-state FP64 exact truth. Call counts are not independent queries.
API time excludes oracle/consistency work; process resources include them.
For the stability window use GIST with 100 rounds and preserve every repetition.

## Full and runtime package

Require explicit feature/source identities, clean source-bound Lite builds and matching Full receipts.
Use the same inputs and record compiler profiles. Equal ef is not equal recall: report both.
The bounded package includes a native application, recursive ELF dependencies, loader, launcher and notices.
Exclude vectors, snapshots, SDK headers/static archives, Python and OS/container layers.
Verify relocated execution and mapped libraries; result-evidence archive size is not product size.

## Stability and data limitations

Live IDs/count/order/finite distances and Save/Load equality establish functional consistency.
They do not establish a recall lower bound, internal concurrency, zero fragmentation or unlimited stability.
Keep measured recall decline and the exact input/change recipe. No numeric quality threshold is invented.
RSS includes input/measurement buffers and allocator-retained pages. getrusage peak can inherit launcher high water;
current-process VmHWM is separate and must not be backfilled into existing records.
Snapshot consistency is not WAL, atomic replacement, fsync or power-loss durability.
Local verification, hosted CI and retrieval quality are independent conclusions.
