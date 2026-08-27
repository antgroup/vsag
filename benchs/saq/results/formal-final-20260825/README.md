# Authoritative SAQ Release experiment

This directory contains the fixed-parameter SIFT1M/GIST1M Release result analyzed in
`../../REPORT.md`.

- Started: `2026-08-25T16:15:47+08:00`
- Finished: `2026-08-25T17:02:05+08:00`
- Exit status: `0`
- Elapsed: `2784` seconds

The run encodes all one million base vectors per method, trains on 65,536 vectors, uses four bits
per dimension and equal complete record lengths, and searches the same HGraph configurations at
`ef_search=40,80,120,200,400,800,1600`.

SIFT1M uses 10,000 independent queries. GIST1M uses its 1,000 independent queries in ten
round-robin repetitions, producing 10,000 timed operations per search point. The declared
minimum-recall comparisons (0.85 for SIFT1M and 0.91 for GIST1M) are generated in
`reported_common_recall_results.csv`.

Read `environment.txt` before interpreting results. `quantization.json` and `eval.json` are the
authoritative raw metrics; the CSV files are derived summaries. Serialized indexes are retained so
the measured search can be inspected without rebuilding.

The four serialized indexes remain ignored by Git because of size. They are retained locally and
can be regenerated from the rendered configuration and public datasets.

Interrupted local diagnostics, if present beside this directory, are not part of this result.
