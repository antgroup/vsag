# Relocated native C++ runtime package measurement, 2026-10-12

This extends the earlier shared-library-only closure measurement to a **defined,
runnable application-bundle profile**, not a universal VSAG SDK/package claim.
Current Lite feature is PR2904 `dbb2fe0bec685ae7a343b1d2060d0553a98e20d2`, installed
library SHA256 `3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330`.
Native Full is the known installed incremental `d18c82a` reference, library SHA256
`9aeaa550703f116bef2959769d4f2f03da254b07c906ba941d8186224cc70c72`, **not a newly built
Full library**. Consumer sources come from the pinned equal-ef raw archive.

## Exact package scope

Included for both sides:

- One fresh C++17/O3 consumer that builds the selected graph, queries, saves and
  loads; Full and Lite use their respective APIs with degree16/build128/query512.
- Recursively resolved dynamic ELF dependencies, including the dynamic loader.
- Relative aliases required by ELF dependency names, and a shell launcher that
  invokes the packaged loader with the packaged library directory/cache disabled.
- VSAG Apache2 license, owning distro-package copyright texts for system libraries,
  and the same common-license text set. License receipts are recorded.
- Identical `strip --strip-unneeded` applied to copies of every ELF, not originals.

Excluded: SDK headers/static archives, Python bindings, containers, OS/kernel/shell,
input datasets, snapshots and dlopen/plugin inventory. A compatible Linuxx86_64
host kernel and POSIX shell are assumed. Licensing/redistribution review is **not
certified**: these are measurement prototypes, not official release artifacts.
Vendored/static third-party notices and broader distribution obligations need a
separate release review; the copied distro notices do not settle those questions.

The Full installed tree is approximately1.8GiB and Lite1.2MiB, but that includes
different static/developer artifacts. Those directory totals are **not** used as
the comparable runtime-package result below.

## Package bytes

Sizes are decimal bytes; payload sums regular files including consumer/loader/
licenses/launcher. Symlink aliases are listed separately. Compressed size is the
actual gzip9tar archive, with normalized tar metadata/gzip timestamp. Different
compiler/OS/library versions or compression choices can change the result.

| Metric | Lite | Native Full | Lite reduction |
| --- | ---: | ---: | ---: |
| Regular-file payload bytes | 6,689,348 | 99,246,762 | 93.260% |
| Actual tar.gz bytes | 2,664,623 | 26,932,509 | 90.106% |
| ELF files including consumer/loader | 7 | 17 | — |
| Inventory entries including aliases/notices | 30 | 57 | — |
| Distro copyright documents | 3 | 11 | — |

Unlike the previous6.15MB/98.43MB shared-library closure, this profile counts the
application executable and packaging components too. It still must be called
**this native C++ application bundle**, not a complete SDK, wheel, container or
proof of performance/portability for every VSAG deployment.

## Actual relocation checks

Each tar.gz was created, unpacked into a different fresh directory, and compared
file-for-file/hash-for-hash/alias-for-alias with the built bundle. Both relocated
launchers ran SIFT10k/100queries with FP32, FP16 and RaBitQ: six successful executions.
Each consumer's native ordered Save/Load check passed, and snapshots were hashed.

| Storage | Lite smoke Recall@10 | Full smoke Recall@10 |
| --- | ---: | ---: |
| FP32 | .993 | .987 |
| FP16 | .993 | .987 |
| RaBitQ3+5 | .987 | .982 |

These are functional smoke results, not repeated quality/latency statistics. Full
RaBitQ training is randomized; a different smoke run can produce a different result.
A `.95` smoke floor is a fixture guard, **not an official OSPP numerical requirement**.

Actual `/proc/PID/maps` captured7/17ELF paths, all under the relocated package,
including the consumer and packaged loader. Loader `--list` receipts also resolve
dependencies inside that package. Therefore the test did not silently use the
original VSAG install or host shared-library directories for these dependencies.
It was not an isolated container/chroot test and does not prove that no external
kernel/shell/configuration/data files are consulted.

## Raw evidence and replay

[report.json](report.json) contains compile commands/source hashes, every original
ELF path/hash/size and stripped package file/hash/size, license receipts, full
inventories/aliases, archive sizes/hashes and relocation argv/maps/metrics/snapshot
hashes. Loader listings and per-storage stdout/stderr are retained; [manifest.json](manifest.json)
binds measured raw files. Independent accounting/map/hash audit and three dependency,
deterministic-pack/relocation, unsafe-member tests passed.

Runtime binaries were **not published** as release packages. They were created
and checked in this run's own RAM scratch, then automatically removed; receipts
remain and the packages can be rebuilt. Old sources, datasets, installations and
raw results were not deleted. Reports/scripts belong to PR2926; PR2904 is unchanged.

```bash
python3 lite/benchmark/measure_runtime_package.py \
  --feature /path/to/dbb2fe0-feature \
  --lite-install /path/to/matching-lite-install \
  --full-install /path/to/matching-known-native-full-install \
  --dataset /path/to/sift/scale-10000 \
  --source-archive /path/to/equal-ef-native-full-20261012/raw.tar.gz \
  --output /path/to/new-results
python3 lite/benchmark/verify_runtime_package.py /path/to/new-results
python3 -m unittest discover -s lite/benchmark -p test_runtime_package.py -v
```

Requires GNUstrip and the compatible ELF loader, ldd, dpkg-query/installed
copyright texts/common licenses, Linux `/proc`, C++17, Python and writable `/dev/shm`.
Only the trusted pinned source archive is accepted. Exact recorded install hashes
are required; fresh Full elsewhere needs a separately pinned baseline. Source paths
in inventories are provenance, not instructions. No source ELF is modified.

## Acceptance conclusion

The binary-size advantage is established for this **runnable, relocatable native
application profile** under the recorded environment. This fills the former
executable/launcher/license omission in the shared-library-only measurement, while
keeping the scope explicit. It does not establish a certified final release package.

Still open: fresh Full all-object build/config/selectedISA and same-quality operating
points; broader SDK/wheel/container packaging if chosen for delivery; release
license review; larger independent queries,100k/Cohere and longer elapsed resident
quality/CPU acceptance. Whole project acceptance is not complete.
