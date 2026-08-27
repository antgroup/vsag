# SAQ benchmark raw results

`run.sh` creates a timestamped or user-selected result directory here. Each run contains the
rendered configuration, unmodified JSON/stdout output, serialized indexes, environment metadata,
`all_results.csv`, and the target-recall comparison in `target_recall_results.csv`.

Large generated indexes and machine-specific ad-hoc runs are intentionally ignored by Git. Results
used in a published benchmark or report must retain their raw measurements and provenance; never replace
missing measurements with estimated numbers.

`formal-final-20260825` is the authoritative complete Release SIFT1M/GIST1M benchmark run. It
contains the corrected PCA/centering/layout implementation, deterministic rotation, bit-plane
code-pair kernel, complete `ef_search=40..1600` curve and all controlled ablations.

Superseded and interrupted diagnostics are preserved outside the Git worktree at
`experiments/results/historical/` relative to the workspace root. They remain available for
root-cause auditing but do not support the current benchmark conclusions.
