# VSAG Lite benchmark and acceptance tools

Start with the [English reproduction guide](../../docs/docs/en/src/development/lite_benchmarks.md)
or [中文复现指南](../../docs/docs/zh/src/development/lite_benchmarks.md).
The published guide URLs are [English](https://vsag.io/docs/development/lite_benchmarks.html)
and [中文](https://vsag.io/docs/zh/development/lite_benchmarks.html); these become available
when the documentation is published. Repository links are the source of truth here.

- [Acceptance protocol](ACCEPTANCE.md): inputs, scope and measurement rules.
- [Measured report](ACCEPTANCE_REPORT.md): results, interpretation and limitations.
- [Evidence manifest](results/manifest.json): measured source, original sizes and hashes.

The core Lite implementation is provided by merged PR #2904. This directory is
for optional measurement consumers, reproduction, independent verification and
acceptance evidence, not a second production implementation. `ENABLE_BENCHMARKS`
defaults OFF. Historical measurements retain their recorded source identity;
compiling current sources does not make those measurements current-source results.

## Directory map

```text
benchmark/
  full/                       installed Full SDK consumer build
  results/resident-crud/      three-dataset, 20-round CRUD evidence
  results/full-comparison/    native Full and bounded runtime-package evidence
  results/dynamic-stability/  100-round GIST evidence, including recall decline
  results/manifest.json       identities of the three original evidence archives
```

Each result directory contains a concise `README.md`, unchanged `raw.tar.gz`
and unchanged `archive.json`. Generated builds, datasets and new measurements
belong in fresh external directories, not in these recorded evidence directories.

See [every file's responsibility](../../docs/docs/en/src/development/lite_benchmarks.md#tool-responsibilities)
or [逐文件职责](../../docs/docs/zh/src/development/lite_benchmarks.md#工具职责) for the code map.
Use fresh external builds/outputs, preserve source guards and all repetitions,
and keep adverse outcomes. Verification PASS is not a quality, concurrency or
power-loss durability guarantee. Source-pinned diagnostics are optional.
