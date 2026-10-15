# VSAG Lite promoted implementation: evidence and acceptance limits

Promotion date: 2026-10-12. Production reference: personal commit
`a62e62e08db3f0d7da2e4439785feed6fcf90257`.
This PR incorporates its supported production core, not the experimental branch's
entire history, rejected diversity switches or raw dataset artifacts.

## Supported functionality

Standalone shared/static builds; BruteForce and FP32/FP16 graphs; opt-in RaBitQ8
(default OFF); Add/Remove/Update/Search; Save/Load; filters; local incoming-edge
repair; allocation-failure transactional rollback. Calls to one instance must
be externally serialized. Snapshot wire formats remain separate from native
Full VSAG. Update replaces values; durable atomic file replacement is not promised.

Promoted improvements include 32-bit incoming adjacency and local mutation
journals, shared native SIMD supplementary-code kernels, cached query sums and
coarse inner products, bounded query heaps, size-specialized visited bitmaps,
owned graph construction, streamed snapshot I/O, and direct FP16 serialization.
The current public RaBitQ adapter enables reverse adjacency and CACHED incoming
protection with safe link pruning: `CACHED, false, false, false, true`.
Update protection is active; optional Add/Remove/displaced-target guards remain
off. Historical NONE/reverse-disabled cohorts are not the current configuration.

## Historical measurements, not a composite current-version matrix

All source versions, commands, configs, verifiers and raw-result archives remain
in the [pinned benchmark repository](https://github.com/nw9663644-eng/vsag/tree/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark).
Report links below identify the actual workload and measured source provenance.
Do not combine different-stage best numbers into one latest-version result.

| Change / workload | Observed benefit | Qualification / remaining cost |
| --- | --- | --- |
| [Native SIMD reuse, SIFT/GIST/Cohere 10k](https://github.com/nw9663644-eng/vsag/tree/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/shared-rabitq-kernels-20261009) | RaBitQ query P50 about -18%/-51%/-44%, unchanged recall and ordered IDs | Compared with previous Lite RaBitQ, not Full; still slower than floating modes in this cohort |
| [Production RaBitQ local CRUD, GIST/Cohere 100k](https://github.com/nw9663644-eng/vsag/tree/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-production-100k-crud-20261011) | 90k mutation API calls: 329487 to 35222 ms / 337975 to 30193 ms, about -89%/-91% | One paired run per dataset; final GIST recall .853, Cohere .953; quality gap not solved |
| [SIFT 100k persistent changed CRUD](https://github.com/nw9663644-eng/vsag/tree/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/sift-production-100k-crud-20261011) | FP32/FP16/RaBitQ: 8332/9636/7176 mutation API calls/s; final recall .989/.989/.983 | One fresh run/mode, serialized initial/final queries, not concurrent reads or hours-long acceptance |
| [FP16 direct Save, three datasets 100k](https://github.com/nw9663644-eng/vsag/tree/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/fp16-direct-save-20261011) | Warm loaded-state Save about -73%, unchanged snapshot bytes | Seven paired runs; not cold I/O or fsync durability |
| [Streamed RaBitQ Save, GIST/Cohere 100k](https://github.com/nw9663644-eng/vsag/tree/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-stream-save-20261011) | Warm Save 52.819 to 45.774 / 44.151 to 37.441 ms; after-Save HWM about 16 MiB lower | Seven pairs, loaded-state Save; prior build/query regressions remain separate |
| [Visited workspace](https://github.com/nw9663644-eng/vsag/tree/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/query-visited-specialized-20261011) | 100k visited payload 100000 to 12504 bytes; cumulative ordinary allocation bytes -51.42% | Not process peak/RSS; this stage had about +1.17% GIST query P50 cost |

Latest routing specialization is a small optimization, not another SIMD-sized
gain: the repeated GIST100k window improved mean wall latency about 1.87% versus
its immediate parent, with identical ordered IDs/distances and recall .859.
It reused 100 distinct historical queries ten times, not 1000 independent queries.
The shared library grew from 395648 to 412512 bytes (+4.26%).

## Native Full comparison boundary

The pinned [expanded 600-query study](https://github.com/nw9663644-eng/vsag/tree/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/acceptance-full-expanded-r1-20261009)
uses actual installed native HGraph, not old Lite. Lite query budget 512 versus
Full 128 is not identical-budget or exact quality-matched acceptance.
Observed Lite queries were slower; warm-load and total loaded RSS had advantages.
The measured stripped recursive ELF dependency closure is not a complete SDK,
wheel or container package. No claim of superiority on every metric is made.

## Fresh promotion verification (2026-10-12)

Standalone Release: 5/5 CTest suites passed. RaBitQ default OFF: 4/4.
ASan/UBSan with leak detection and halt-on-UB: 5/5. Instrumented Debug: 5/5.
Fresh gcov JSON union for production `src/lite` sources and inline headers:
3072/3233 executable lines, 95.0201%. This is scoped Lite line coverage,
not entire Full VSAG or branch coverage. No new CPU/performance matrix was run
as part of this promotion; historical performance evidence remains versioned.

clang-format 15, focused clang-tidy 15 on four core/test translation units and
an installed-header/library RaBitQ CRUD/filter/Save/Load consumer passed.
Dependency-header warnings were suppressed by clang-tidy's default scope;
this does not claim a warning-free whole-repository lint.

## Remaining acceptance boundaries

Current-version CPU time, utilization and hardware counters are not supplied by
wall latency. Cold load, complete deployment package size, matched-quality native
Full measurements and hours-long interleaved permanent changed-vector CRUD remain
open. GIST maintenance recall needs further work. Historical tests are not a
substitute for this promotion's fresh Release/OFF/sanitizer/coverage verification.
The project is not declared fully complete.

中文说明：本次将已验证生产代码、能力和回归测试移入 #2904，RaBitQ 仍默认关闭。
表格每行属于各自被测版本，不能拼成“最新全部指标”。CRUD 吞吐是 API 调用数，
不是完整事务数；当前支持调用方串行化，不承诺内部并发读写。已有空间、加载、
CRUD 和查询优化收益，但 GIST 召回、同质量 Full 对照、冷加载、完整部署包、
最新 CPU 资源和小时级长期交错真实改值测试仍有缺口。
