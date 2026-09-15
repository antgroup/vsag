# VSAG Lite v0.1 (experimental)

Independent BruteForce-only build; does not link the Full VSAG library.
Implementation: `src/lite/`. Public API: `include/vsag/lite/index.h`.

- [English guide](../docs/docs/en/src/development/lite_first.md)
- [中文说明](../docs/docs/zh/src/development/lite_first.md)

This prototype is not the earlier `ENABLE_LITE` pruning variant. It does not
include graph search, quantization, mmap, or a Full-compatible snapshot format.

## Installed example and measurement

The [independent consumer](example/main.cpp) uses `find_package(vsag-lite CONFIG REQUIRED)` and links only `vsag::lite`. The English and Chinese guides above
show its build commands, complete CRUD/Save/Load output, and a concise
shared-library-size/process-RSS comparison. The complete seven-run experiment
is in [benchmark PR #2926](https://github.com/antgroup/vsag/pull/2926).
