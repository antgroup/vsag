# VSAG Lite acceptance: reviewer entry point

## Scope and source identity

PR #2904 owns the supported functionality, architecture, API, regressions and CI.
This evidence branch now incorporates feature commit
`74a9420c771e75dea5e56324775d8db5d1472459` through a normal merge; its production
`src/lite` and public Lite headers match that feature version. No rejected graph
policy is enabled by this benchmark integration. `ENABLE_BENCHMARKS` defaults OFF.
Older evidence explicitly measures `dbb2fe0` or its report-specific source, not
automatically the latest merged implementation.

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

Start with the [2026-10-14 final-version audit](results/final-version-20261014/README.md)
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

```bash
cmake -S lite -B build-lite -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=ON -DENABLE_RABITQ_LITE_BACKEND=ON -DENABLE_BENCHMARKS=ON
cmake --build build-lite -j2
ctest --test-dir build-lite --output-on-failure
python3 -m unittest discover -s lite/benchmark -p 'test_*.py'
python3 lite/benchmark/run_mixed_cpu_acceptance.py --help
```

For a new run, pass a clean feature checkout and its actual SHA, a fresh build,
prepared datasets and a new output directory. Do not silently reuse historical
binaries. Report-specific frozen Full/quality runners intentionally enforce their
historical feature identity: use the matching pinned checkout to replay those
protocols, rather than removing guards or claiming they measured this new head.

Historical dataset preparation, comparison and diagnostic runners are retained
for provenance. The default production API is solely #2904's; obsolete SQ8/codec
probes are not exposed as current supported backend options.

## Community review and remaining acceptance

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
