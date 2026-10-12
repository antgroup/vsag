# Post-hoc calibration and mutation-quality diagnosis

This is development/post-mutation calibration, **not independent validation**.
Three processes per budget, CPU0/thread1, immutable retained snapshots from the
strict-FP comparison. No mutation performed here. Raw verifierPASS24runs/2400queries.

| Source | Budget | Recall | P50 us | P99 us |
|---|---:|---:|---:|---:|
| SIFT native Full |256|.968|144.366|161.106|
| SIFT native Full |384|.987|203.034|331.602|
| SIFT native Full |512|.987|260.902|408.139|
| SIFT native Full |1024|.996|493.277|748.018|
| GIST candidate after3whole-index mutation passes |256|.973|332.660|404.980|
| GIST candidate after3whole-index mutation passes |384|.981|464.467|560.404|
| GIST candidate after3whole-index mutation passes |512|.989|600.353|708.191|
| GIST candidate after3whole-index mutation passes |1024|.997|1046.171|1192.296|

First tested qualifying SIFT Full budget1024 is post-hoc; do not relabel it a
frozen choice on fresh queries. GIST recovery with more search effort suggests
routing-efficiency degradation, not proof of disconnected components or missing
data. It does not prove every correct neighbor remains efficiently reachable.
Only100queries are used per process. No Full post-mutation quality comparison.

Snapshot/query/binary/library hashes and raw ordered neighbors are retained.
Reproduce via run_aligned_followup.py with the saved artifacts from the previous
run; retained snapshots currently live in /home/ubuntu/project/aligned-fp32-followup-20261012.
verify_aligned_followup.py checks all rows/hashes/statistics.
Raw archive102members168424bytes SHA256
`121c22d176d6c5caa92044830a9f13e64840480f52b8def849afec60909dc122`.
