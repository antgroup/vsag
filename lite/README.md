# VSAG Lite v0.1 (experimental)

Independent BruteForce-only build; does not link the Full VSAG library.
Implementation: `src/lite/`. Public API: `include/vsag/lite/index.h`.

- [English guide](../docs/docs/en/src/development/lite_first.md)
- [中文说明](../docs/docs/zh/src/development/lite_first.md)

This prototype is not the earlier `ENABLE_LITE` pruning variant. It does not
include graph search, quantization, mmap, or a Full-compatible snapshot format.
