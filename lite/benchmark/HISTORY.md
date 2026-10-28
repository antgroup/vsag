# Historical acceptance evidence

Current formal results remain local: [E10 Lite CRUD/CPU](results/review-fixed-resident-20261020/README.md),
[E11 native Full/O3 and runtime package](results/current-full-package-20261020/README.md),
and [E12 extended GIST churn](results/extended-resident-gist-20261020/README.md).
All current raw archives and their receipts remain unchanged.

E1 and E9 below are historical source-specific cohorts, not current performance.
Their complete reports, original raw archives, summaries, source receipts and logs
are preserved at immutable commit `2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b` and in the personal fork's
`archive/lite-ospp-acceptance-20261022` branch. Nothing is discarded for being slower
or for showing lower recall. The comprehensive report retains the historical
numbers and limitations, with immutable links replacing removed local paths.

## Complete archived file inventory

Every size and SHA256 is in `RAW_MANIFEST.json` under `archived_cohorts`.
The earlier `files` and `archived_tools` entries retain their original source pins.
The following links select the measured snapshot, not a moving branch.

| Preserved file | Bytes | Fixed source |
| --- | ---: | --- |
| `results/final-version-20261014/README.md` | 6694 | [original](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/final-version-20261014/README.md) |
| `results/final-version-20261014/archive.json` | 136 | [original](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/final-version-20261014/archive.json) |
| `results/final-version-20261014/benchmark-tests.log` | 138 | [original](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/final-version-20261014/benchmark-tests.log) |
| `results/final-version-20261014/merge-base-tests.log` | 102 | [original](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/final-version-20261014/merge-base-tests.log) |
| `results/final-version-20261014/raw-verification.log` | 179 | [original](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/final-version-20261014/raw-verification.log) |
| `results/final-version-20261014/raw.tar.gz` | 4483066 | [original](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/final-version-20261014/raw.tar.gz) |
| `results/final-version-20261014/release-off.log` | 590 | [original](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/final-version-20261014/release-off.log) |
| `results/final-version-20261014/release-on.log` | 688 | [original](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/final-version-20261014/release-on.log) |
| `results/final-version-20261014/simd-coverage-tests.log` | 124 | [original](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/final-version-20261014/simd-coverage-tests.log) |
| `results/final-version-20261014/source-closure.json` | 9405 | [original](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/final-version-20261014/source-closure.json) |
| `results/final-version-20261014/summary.json` | 12593 | [original](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/final-version-20261014/summary.json) |
| `results/resident-three-datasets-20261019/README.md` | 12512 | [original](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/resident-three-datasets-20261019/README.md) |
| `results/resident-three-datasets-20261019/archive.json` | 209 | [original](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/resident-three-datasets-20261019/archive.json) |
| `results/resident-three-datasets-20261019/raw.tar.gz` | 7420407 | [original](https://github.com/nw9663644-eng/vsag/blob/2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b/lite/benchmark/results/resident-three-datasets-20261019/raw.tar.gz) |

## Replay historical validation

Use a separate checkout; do not combine historical expectations with current
source or silently remove source/hash checks. Resolve the archived checkout's
documented Python, compiler and dataset prerequisites first. Original vector
datasets and binaries are not included in the compact result archives; input
origins, preparation parameters, identities and per-report recipes remain required.

```bash
git fetch https://github.com/nw9663644-eng/vsag.git \
  refs/heads/archive/lite-ospp-acceptance-20261022
git worktree add --detach /absolute/path/to/new-history-checkout \
  2ce4cb3445fb49f131a5f6b1dd43b4a2221d764b
```

Follow that checkout's `lite/benchmark/ACCEPTANCE.md` quick start. It builds the
compiled lifecycle probe, installs the preparation dependencies and executes the
complete original test discovery, including `ResidentPublishedEvidenceTests`.
That historical report test is preserved there, not disabled or replaced by a
network download during normal current-source CI. Current tool/negative-path
regressions and E10/E11/E12 report checks stay in the default suite.

To inspect one file without changing a worktree, use
`git show <immutable-source>:<path>` and compare bytes/SHA256 with the manifest.
Recover the entire cohort rather than copying just its fastest row. Historical
raw audit success is not a current-version quality verdict or hosted CI success.
