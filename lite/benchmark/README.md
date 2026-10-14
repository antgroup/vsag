# VSAG Lite reproducible acceptance tools

[Start here: requirements, reports, commands and limitations](ACCEPTANCE.md).

- Functionality/architecture/regressions: PR #2904, feature baseline `8e82a2a`.
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

The latest tree excludes large raw outputs from the review diff; project history
and original server evidence remain intact. New performance output goes to a
fresh external directory and must record source/configuration/input identities.
The current integration does not retrospectively rerun or relabel old experiments.

中文：[社区验收入口](ACCEPTANCE.md)将四项产出、功能、脚本、报告和证据对应起来。
原始数据通过固定提交及manifest提供，不在社区代码diff中逐行展开。
