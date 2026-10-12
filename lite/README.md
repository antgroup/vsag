# VSAG Lite

Standalone shared/static library with exact FP32 BruteForce, explicit FP32/FP16
graphs and opt-in RaBitQ8. Supports Add, changed Update, physical Remove, Search,
ID filtering, per-query budgets and Save/Load. Same-instance calls must be
externally serialized; this is not Full Index ABI or snapshot compatibility.

```bash
cmake -S lite -B build-lite -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON
cmake --build build-lite -j2
ctest --test-dir build-lite --output-on-failure
```

RaBitQ defaults OFF. Add `-DENABLE_RABITQ_LITE_BACKEND=ON` and explicitly call
`BuildGraph(VectorStorage::RABITQ8, degree, ef)` to use it. Its current public
adapter enables incoming adjacency, cached Update protection and safe link
pruning; extra Add/Remove/displaced-target guards are off. It uses a fixed trained
model and returns quantized estimates, not exact original-vector distances.

Public API: `include/vsag/lite/index.h`; implementation/tests: `src/lite`;
installed shared/static examples: `lite/example`. Floating snapshots are
VSAGLT01 v1/v2/v3; RaBitQ uses independent VSAGLQ01 v1. No checksum, fsync,
atomic file replacement, SQ8, mmap or same-instance concurrent access is promised.

- [Current English function/source/test/requirement mapping](../docs/docs/en/src/development/lite_first.md)
- [当前中文功能、源码、测试及产出要求对照](../docs/docs/zh/src/development/lite_first.md)
- [RaBitQ implementation and limits](../docs/docs/en/src/development/lite_rabitq_design.md)
- [Versioned performance evidence and acceptance gaps](PERFORMANCE_EVIDENCE.md)

`tools/check_tidy.py` checks configured Lite translation units and private/public
headers using clang-tidy15. `tools/collect_coverage.py` unions production Lite
source/header executable lines across instrumented tests and enforces 90%; it
does not measure Full or branch coverage. See `.github/workflows/lite.yml` for
default-OFF, ON, sanitizer and coverage CI matrices. Replay scripts/configs/raw
benchmark results are pinned by the evidence report; they are not fictitious
current-version results. Functional tests do not establish final quality,
performance or hours-long stability acceptance.
