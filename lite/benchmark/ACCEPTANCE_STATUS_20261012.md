# OSPP Lite acceptance evidence status, 2026-10-12

Latest Full addition: [equal-ef current Lite / pinned native Full comparison](results/equal-ef-native-full-20261012/README.md),
36builders108loaders, bothquery512. Total loadedRSS6/6lower andwarmload5/6faster,
but incrementalloadedRSS6/6higher, allSave/snapshotsworse andGISTqualitylower.
DT_NEEDEDclosure93.754%smaller is not a complete package. Full is the pinned installed
d18c82a incremental rebuild, not latestfresh; same-quality/selectedISA andfreshFull gates remain.

Latest addition: [20-round resident-index mixed CRUD/CPU matrix](results/resident-mixed-cpu-pr2904-20261012/README.md).
The original index is retained across all rounds;18fresh processes each15k mutation
calls and15k queries,270k each cumulatively. Per-round persistence, current-state truth,
sampled thread counts, raw archive/hash verification and late-window RSS are recorded.
This closes the earlier reload-resetting-object weakness for this fixed-round workload,
but neither hours-long endurance nor zero fragmentation is established. GIST RaBitQ
recall .970→.963 and a264KiB late RSS step are retained as limitations.

Code, supported contracts, tests and architecture live in [PR2904](https://github.com/antgroup/vsag/pull/2904).
Benchmark scripts, protocols, raw results and acceptance analysis belong to
[PR2926](https://github.com/antgroup/vsag/pull/2926), branch `experiment/lite-baseline-v01`.

| Deliverable | Available evidence | Remaining acceptance gap |
| --- | --- | --- |
| Reviewable configurable Lite with Add/Remove/Update/Search/Save/Load | PR2904, currently measured dbb2fe0; Release5/5 rerun for this matrix; prior ON/OFF/SAN/coverage/install receipts | Hosted checks/review must be verified at final delivery; passing local tests is not approval |
| Architecture: features/cuts/index/layout/memory/API | PR2904 bilingual canonical Lite and RaBitQ docs | Keep aligned with final code; independent API/format and externally serialized contract are explicit limitations |
| Reproducible >=2 datasets/scales, complete metric set | [Latest two-dataset mixed CPU matrix](results/mixed-cpu-pr2904-20261012/README.md), with scripts/config/source/input hashes and per-call/raw resource data | Full deployment package, larger independent P99 cohort, additional mixed ratios and resident endurance remain |
| Clear improvement vs actual native Full in package/load/memory with correctness/quality stable | [Historical native Full matrix, pinned personal source](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/current-full-10k600-20261011/README.md) | Historical Full installed build and unequal query budgets must not be promoted to fresh same-quality acceptance; perform final aligned comparison |
| Report/limits/performance analysis | Latest matrix records CPU versus wall-time, quantization tradeoffs, negative results, memory/load limits | GIST100k maintenance quality is unresolved; no project-complete declaration |

The original `FINAL_REPORT.md` and older results describe their measured versions.
They are not automatically the final report for dbb2fe0. Likewise, historical
89–91% CRUD or ~73% FP16 Save improvements compare different source cohorts and
must not be merged into an invented latest Full/Lite scorecard.

Next order: current-feature native Full matching -> continuously resident
changed-value endurance -> independent-query quality/100k -> evidence-based
production optimization in PR2904 -> rerun affected cells here -> final report.
