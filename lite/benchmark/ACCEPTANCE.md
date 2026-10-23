# VSAG Lite acceptance: reviewer entry point

[详细综合报告 / comprehensive evidence report](ACCEPTANCE_REPORT.md) covers the
four outputs, three datasets, Lite/native Full comparisons, real CRUD, CPU/memory,
100k experiments, measured optimizations and failed quality gates. The prior E9
cohort includes Cohere; historical Full comparisons remain separate evidence.

## Current CRUD isolation (E14)

[Update-only and Remove/Add controls](results/crud-isolation-20261022/README.md) preserve all nine processes, fixed-budget curves and separate final ef=N diagnostics.
No production optimization or #2904 change is claimed.

## Prior CRUD diagnosis (E13)

[GIST quality curve and same-data rebuild control](results/crud-degradation-curve-20261022/README.md)
adds 33 checkpoints without changing #2904 or the earlier results. One trial/storage;
not an algorithm optimization, a durability guarantee or a completed stability verdict.

## Scope and source identity

PR #2904 owns the supported functionality, architecture, API, regressions and CI.
The current integration depends on #2904 feature commit
`cba340660f2ea479b316d7b73a27f34186b8a1d9`; its production sources and public
headers match that feature dependency. The earlier integration
`74a9420c771e75dea5e56324775d8db5d1472459` remains a historical measurement
identity, not the current integration baseline. No rejected graph
policy is enabled by this benchmark integration. `ENABLE_BENCHMARKS` defaults OFF.
Older evidence explicitly measures `dbb2fe0` or its report-specific source, not
automatically the latest merged implementation. E9 measures `a3a76a8`, before the
review fixes; its original records remain unchanged. A new measurement of the
review-fixed feature follows the same resident protocol below, with a fresh build
receipt and the exact same input hashes. It is a version revalidation, not an
algorithm speedup or a new Full comparison.

The previous experiment snapshot remains publicly recoverable at
[`800a82232da3a91dfc90229c3a8fdbf4fa9b428a`](https://github.com/nw9663644-eng/vsag/tree/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark).
Raw data and older exploratory probes are retained there, not deleted from project
history or server originals. The latest PR diff excludes bulky raw result files;
see `RAW_MANIFEST.json` for their path, byte size, SHA256 and pinned URL. Positive
and negative experiments are preserved. This reduces review noise, not evidence.

## Output requirements and evidence

| Required output | Review location | State / boundaries |
| --- | --- | --- |
| At least one reviewable standalone/configurable Lite PR, CRUD/SaveLoad/config/tests | #2904; bilingual canonical Lite guides and `src/lite/*_test.cpp` | Implemented; community checks and maintainer approval are separate |
| Architecture: retained/cut modules, graph/layout/memory/API compatibility | #2904 `docs/docs/{en,zh}/src/development/lite_first.md` and `lite_rabitq_design.md` | Separate Lite API/format, fixed RaBitQ model, external serialization |
| Reproducible >=2 datasets/scales; package/snapshot/load/peak/CRUD/P50/P99/recall | Pinned mixed CPU, resident CRUD, fresh Full and runtime-package reports below; replay scripts and verifiers here | Metrics have workload/version boundaries; hours-long endurance and full unified latest-source matrix are not claimed |
| At least one clear improvement vs actual Full, correctness/stable retrieval | Fresh native Full and defined runtime-package profiles | Read negative quality/Save/peak results too; diagnostic .990 is not an OSPP contractual threshold; no universal speedup |

## Start with these reports

The [GIST 100-round resident extension (E12)](results/extended-resident-gist-20261020/README.md)
retains three trials/storage mode and all CPU/memory/quality data; it is not hours-long or Full.

The [current-source native Full/O3 and runtime-package cohort (E11)](results/current-full-package-20261020/README.md)
contains 54 builders, 162 fresh warm loaders, all three datasets/storage modes and
six relocated package smokes. It uses clean cba3406 on both sides, not historical
Full binaries. Lower Lite total RSS/latency does not erase lower GIST/Cohere recall,
larger snapshots or incremental-RSS costs. This is not same-quality or endurance.

The [review-fixed three-dataset cohort (E10)](results/review-fixed-resident-20261020/README.md)
measures `cba3406` with a fresh source-bound build and the same input hashes and
27-process protocol as E9. It is the current Lite resident matrix, not a current
Full/package measurement, a speedup attribution or an hours-long endurance test.

The [prior three-dataset resident cohort](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/resident-three-datasets-20261019/README.md)
uses 1000 query rows on SIFT/GIST/Cohere, a source-bound `a3a76a8` Lite build and
27 fresh processes. It documents that measured version, not a new Full or
runtime-package comparison. Quality declines and the pending quality criterion
remain separate from raw-audit success.

For historical source closure and the earlier verdict, use the [2026-10-14 final-version audit](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/final-version-20261014/README.md)
for the four-output verdict, fresh resident CRUD matrix and compiler dependency
closure/library-hash binding to historical Lite measurements. It preserves the
GIST/RaBitQ quality limits and outstanding community gates; it does not relabel
historical Full/package measurements as new runs.

The following historical links use immutable snapshot800a822; follow each report's own
measured source SHA, input hashes, commands, configuration and raw verification.

- [Mixed CPU/CRUD](https://github.com/nw9663644-eng/vsag/blob/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark/results/mixed-cpu-pr2904-20261012/README.md): SIFT/GIST10k, three storages, CPU vs wall and permanent changes, per-call data.
- [Resident mixed CRUD](https://github.com/nw9663644-eng/vsag/blob/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark/results/resident-mixed-cpu-pr2904-20261012/README.md): original index remains live, per-round truth/persistence/RSS; seconds-long processes, not hours.
- [Fresh same-source native Full](https://github.com/nw9663644-eng/vsag/blob/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark/results/full-fresh-equal-r1-20261012/README.md): actual Full, complete negative cells; equal ef does not imply equal quality.
- [Runtime-package profile](https://github.com/nw9663644-eng/vsag/blob/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark/results/runtime-package-r1-20261012/README.md): app/runtime/loader/notices scope, relocated smokes and measured archive sizes; not universal SDK/wheel/container.
- [Frozen quality validation](https://github.com/nw9663644-eng/vsag/blob/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark/results/quality-target-r2-20261012/README.md): preserve failed Lite quality target, not an equal-quality victory.
- [Strict compiler-aligned candidate](https://github.com/nw9663644-eng/vsag/blob/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark/results/aligned-fp32-full-r2-20261012/README.md): candidate results are not production #2904 default acceptance; mutation quality remains a limit.

## Reproduce and verify

Follow the [English build/test quick start](../../docs/docs/en/src/development/lite_benchmarks.md#build-and-test-quick-start)
or [中文复现指南](../../docs/docs/zh/src/development/lite_benchmarks.md).
Both require a compiled probe and all Python preparation dependencies; a skip is not success.

For a new run, pass a clean feature checkout and its actual SHA, a fresh build,
prepared datasets and a new output directory. Do not silently reuse historical
binaries. Report-specific frozen Full/quality runners intentionally enforce their
historical feature identity: use the matching pinned checkout to replay those
protocols, rather than removing guards or claiming they measured this new head.

Only the formal acceptance runners and their dependency/test closure remain in
the current review tree. 32 historical exploration tools/redundant documents are
preserved at immutable `c072a42`; `RAW_MANIFEST.json` records their paths/hashes/URLs.
Historical exploratory replay must use that checkout. Negative experiment reports
remain linked above, so pruning review code does not hide failed candidates.
The default production API is solely #2904's; obsolete SQ8/codec probes and graph
candidates are neither current supported backend options nor acceptance defaults.

## Current-version resident protocol

This is a new Lite-only cohort, separate from the historical E1 results and from
the tiny reproduction smoke. Freeze this protocol before measuring; retain all
cells, not the fastest repetition. The feature baseline is the clean #2904
dependency identified above; the measured build must have a valid source receipt.

| Item | Fixed value / interpretation |
| --- | --- |
| Inputs | SIFT128, GIST960, normalized Cohere768; first 10000 base vectors and first 1000 query rows from the recorded real sources |
| Query identity | 1000 supplied query rows per dataset, not 15000 independent queries; record exact input hashes and duplicate-row count |
| Modes | FP32, FP16, RaBitQ8; degree16, construction ef128, query ef512, k10 |
| Repeats | Three fresh processes per dataset/mode; mode order rotated by trial, fixed dataset order SIFT/GIST/Cohere |
| Dynamic work | `--resident --trials 3 --rounds 20 --cycles 250`; each process has 5000 Update, 5000 Remove, 5000 Add and 15000 interleaved queries |
| Mutation rule | Existing FP32 `.875 * current + .125 * original donor`; original resident index is not replaced by a loaded copy |
| Correctness | Live IDs, count, finite/ordered/unique neighbors; 20 Save/Load checks over all 1000 queries; initial/final current-state FP64 exact truth |
| Aggregation | Retain all three process values; median per-process P50/P99/throughput, min/max and per-trial quality; do not pool percentiles or cherry-pick trials |
| CPU/memory | CPU0 if available, otherwise the explicitly recorded available CPU; one process/thread, OMP/OpenBLAS/MKL1; record API user/system CPU and external process resources |
| Persistence/peak | Warm uncontrolled-cache same-process Save/Load; peak includes input/truth/loaded-copy buffers, not index-only RSS |
| Quality verdict | Report Recall@10 and initial/final difference separately from raw audit PASS; an agreed contractual threshold is not invented; .990 remains only a diagnostic target |
| Full comparison | Not part of this cohort; fresh current Full, equal-parameter and separately frozen same-quality protocols remain distinct follow-ups |

Run the existing source-bound mixed runner once per dataset, with all three
trials inside each command, then run its independent verifier and packer. This
preserves mode rotation while limiting temporary snapshot occupancy. Do not run
compilation, tests or dataset preparation concurrently with the timed cohort.
Dataset preparation, FP64 truth, round-trip comparison searches and bookkeeping
are outside API throughput. Duration is measured, not labeled hours-long endurance.
Input preparation/source hashes, pre-measurement protocol hash, exact commands,
build receipt, all raw rows and archive verification accompany the resulting report.

## Community review and remaining acceptance

### Current-source native Full protocol

The next Full cohort pins the clean feature `cba340660f2ea479b316d7b73a27f34186b8a1d9`
and the same three E10 input hashes, 10k base vectors and 1000 query rows.
Use the existing fresh root-CMake builder with explicit `--source-sha`; no old
objects or installed Full binary may be substituted. Build the isolated recorded
O3 overlay (`--strict-fp`) for compiler-semantics alignment with Lite Release O3;
this is native Full code, not a Lite reference, and not the default Ofast profile.
The overlay and selected compilation records must accompany the result.

Keep degree16/construction ef128/query ef512/k10, CPU0 and one query thread.
Retain all three alternating Lite/Full pairs per dataset/storage and all three
fresh-process warm loads per builder. Recompute hits and quantiles independently;
retain inferior cells and all repeats. Equal parameters or O3 flags do not imply
equal recall or identical executed ISA. Runtime-dispatch evidence is a separate
check and must not be inferred merely from CPU capabilities or compiled ISA files.
This static initial-index cohort is not a Full CRUD or hours-long endurance result.
Source-bound Lite receipt and fresh Full source/library/header receipts are mandatory.

Historical commands below retain their old defaults and cannot silently become
current-source commands. A new source explicitly opts in and requires fresh receipts.

Review production separately in #2904; review reusable runners/oracles/verifiers,
their tests, this index and raw manifest in #2926. Since #2904 is not merged into
main, this main-target PR still displays the dependency's feature files. Its
feature ancestry is now aligned, but that dependency cannot be hidden by a normal
commit. After #2904 merges, re-evaluate the base diff and rerun affected checks.
No force push or automatic merge is performed.

Remaining work includes GIST100k maintenance quality, long-duration permanent
changes, selected-ISA/configuration controls and a unified latest-source matrix.
Current replay/build validation is not a new full performance study.

中文：社区首先评审 #2904 功能代码，#2926 评审可复用复现工具、真值/审计、测试及验收索引。
原始数据保留在不可变800a822提交，manifest可逐项验证，不要求维护者逐行检查几万行CSV。
当前代码已对齐最新功能，历史实验仍按各自被测SHA解释，不伪称重跑，不宣称整个项目验收完成。

Historical E1/E9 files are preserved byte-for-byte; see [the archive inventory and recovery procedure](HISTORY.md). Current E10/E11/E12 raw archives and all measurement tools stay local.
