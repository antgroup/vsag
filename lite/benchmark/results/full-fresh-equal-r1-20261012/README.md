# Fresh native Full / current Lite, equal query budget, 2026-10-12

Both libraries use source commit `dbb2fe0bec685ae7a343b1d2060d0553a98e20d2`.
Full was compiled from a new empty root-CMake build directory, with ccache OFF,
243 newly compiled core objects, Release, target `vsag`, two build jobs, and
cached dependency **sources** plus system OpenBLAS/fmt. No old object files or
installed VSAG library were reused. This is a fresh native Full core, not a fresh
build of every system dependency, complete SDK, or full upstream test suite.
[Build commands, logs, object hashes and compiler configuration](../full-fresh-build-r4-20261012/)
record configure/build success; build wall time was 525.018 s. The retained copy
was stripped; the unstripped RAM build was not modified and was then discarded.
Full library SHA-256: `8740d2199518703883021d0d49b427e0e6a01be32a6866f561cfd86012f9c888`.
Lite SHA-256: `3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330`.

## Protocol and complete primary results

SIFT-128 and GIST-960: 10,000 base vectors, 100 independent query vectors and
prefix ground truth; degree 16, build ef 128, query ef 512, top-10. CPU 0,
single build thread and BLAS/OpenMP thread limits 1. Three fresh builder pairs
per cell, alternating side order, three fresh loader processes per builder:
36 builders, 108 loaders, 3,600 queries. Tables show medians, **Lite / Full**.
Query P50/P99 are medians of three per-process nearest-rank quantiles, not pooled
quantiles. Loads are warm-page-cache fresh-process loads, not cold boot.

| Dataset/storage | Recall@10 | Query P50 us | Query P99 us | Load ms | Loaded process RSS KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIFT FP32 | .993 / .987 | 168.475 / 274.703 | 196.215 / 440.798 | 3.830 / 5.549 | 12592 / 32860 |
| SIFT FP16 | .993 / .987 | 165.645 / 274.803 | 188.634 / 442.968 | 3.005 / 3.515 | 10128 / 30352 |
| SIFT RaBitQ8 | .987 / .979 | 201.824 / 335.451 | 354.951 / 383.320 | 3.979 / 2.709 | 8540 / 29824 |
| GIST FP32 | .971 / .992 | 446.936 / 529.685 | 604.513 / 704.161 | 17.722 / 29.785 | 45156 / 66112 |
| GIST FP16 | .971 / .992 | 357.390 / 434.567 | 442.128 / 517.226 | 10.903 / 16.019 | 26368 / 46928 |
| GIST RaBitQ8 | .970 / .989 | 439.427 / 520.206 | 695.172 / 648.962 | 7.125 / 9.536 | 16700 / 38184 |

| Dataset/storage | Build ms | Save ms | Snapshot bytes | Builder peak RSS KiB |
| --- | ---: | ---: | ---: | ---: |
| SIFT FP32 | 546.987 / 791.412 | 3.759 / 3.220 | 6560064 / 5766017 | 27724 / 42340 |
| SIFT FP16 | 535.058 / 811.327 | 3.269 / 2.499 | 4000064 / 3144565 | 22684 / 37072 |
| SIFT RaBitQ8 | 605.148 / 930.795 | 2.667 / 1.989 | 2960672 / 2203866 | 21620 / 38256 |
| GIST FP32 | 1330.220 / 1528.252 | 16.428 / 14.110 | 39840064 / 39808753 | 125344 / 108884 |
| GIST FP16 | 1321.432 / 1494.708 | 10.661 / 8.465 | 20640064 / 20148057 | 103840 / 85404 |
| GIST RaBitQ8 | 1607.813 / 1384.545 | 4.891 / 4.671 | 11284416 / 10690982 | 103856 / 102252 |

Loaded process RSS is lower in all six cells; loading is faster in five cells.
Query P50 is lower in six cells, P99 in five. However, incremental loaded RSS
(loaded minus before-load RSS) is higher in all six Lite cells. Save time and
snapshot size are worse in all six, and GIST builder peaks are higher in all
three. SIFT RaBitQ loading is slower; GIST RaBitQ P99 is about 7.12% higher.
GIST recall is lower in all three Lite modes. Full RaBitQ recall varies across
fresh builds: SIFT .976-.988, GIST .987-.990; the median must not hide that range.
Within Lite, RaBitQ saves RSS versus FP16 but has slower P50 and worse P99.

The stripped recursive ELF shared-library closure is 6,147,776 / 98,428,088 bytes
(Lite 93.754% smaller). This excludes consumers, launchers, notices and aliases:
it is **not** the complete runtime package. The earlier runnable-package report
uses a different historical Full library and remains separately versioned.

## Evidence, reproduction and remaining gates

[Raw archive](raw.tar.gz): 489003 bytes, 702 members, SHA-256
`13f08607e3efb79a1e752f460085795a85f0ebb56eeb23aba5e89b14ed019b81`.
The verifier passed: manifests, fresh Full receipt, matching source/library/header
identities, equal ef, all result IDs/ranks/distances and ground-truth hits, raw
quantiles, and 108 loader counts/resources. Original server records are preserved.
Use `build_fresh_full_reference.py` with the source/cache/archive/install/output
arguments from `../full-fresh-build-r4-20261012/commands.json`; then use
`run_pinned_full_acceptance.py --fresh-full-receipt <build-output>/receipt.json`
with arguments recorded in `provenance.json`, and run
`verify_pinned_full_acceptance.py <output>` (optional dataset path overrides).

Native Full Release uses root `-Ofast/-g`; Lite uses `-O3`. This is equal search
budget, **not equal recall or identical compiler/selected SIMD execution**.
The Full test suite, larger independent P99 cohort, same-quality tuning, latest
fresh-Full runtime-package rerun, CPU counters and hours-long CRUD remain open.
CRUD/current-truth evidence is the separately versioned resident mixed matrix;
this initial-index experiment does not establish new CRUD/endurance results.
