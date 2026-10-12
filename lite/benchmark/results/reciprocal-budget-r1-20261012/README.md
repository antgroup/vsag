# Reciprocal candidate query-budget calibration, 2026-10-12

This extends the isolated FP32 candidate in `../reciprocal-diversity-r1-20261012`.
PR2904 source at `dbb2fe0bec685ae7a343b1d2060d0553a98e20d2` is unchanged.
No FP16/RaBitQ performance claim or native Full comparison is made here.

## Protocol

CPU0, one thread, Release, RaBitQ backend compiled ON but FP32 storage selected;
degree16/build budget128/top10/10,000 vectors. Candidate budgets128/192/256/384
and production budget512. Three fresh processes per cell; cyclic rotation of
the five cells. Each process builds its own snapshot; queries run afterward.
SIFT100 and GIST1000 queries are existing **development data**, not fresh holdout.
No builds or performance workloads run concurrently.

## Results

Medians of three runs. Budgets differ: this measures structure plus budget
tuning, not same-parameter code speedup. Production itself has not yet been
calibrated at these lower budgets; this does not prove superiority over a fully
tuned production baseline or native Full.

| Dataset | Side | Query budget | Recall@10 | P50 us | P99 us | Build ms |
|---|---|---:|---:|---:|---:|---:|
| SIFT | Production | 512 | .9930 | 165.516 | 191.016 | 541.368 |
| SIFT | Candidate | 128 | 1.0000 | 55.839 | 69.947 | 686.007 |
| SIFT | Candidate | 192 | 1.0000 | 79.797 | 98.187 | 685.627 |
| SIFT | Candidate | 256 | 1.0000 | 100.338 | 120.576 | 685.950 |
| SIFT | Candidate | 384 | 1.0000 | 140.316 | 162.035 | 686.119 |
| GIST | Production | 512 | .9646 | 457.467 | 609.393 | 1297.364 |
| GIST | Candidate | 128 | .9833 | 196.735 | 249.123 | 2141.319 |
| GIST | Candidate | 192 | .9905 | 273.112 | 347.600 | 2137.468 |
| GIST | Candidate | 256 | .9938 | 342.771 | 429.937 | 2152.420 |
| GIST | Candidate | 384 | .9966 | 482.485 | 596.002 | 2140.590 |

Candidate128 SIFT P50 decreases66.26% and P99 63.38% versus production512.
Candidate192 GIST P50 decreases40.30% and P99 42.96%, recall+2.59 percentage
points. GIST .990 is a developer diagnostic target, not an agreed user
acceptance threshold. Candidate256 leaves more observed recall margin but is
slower. Building remains more expensive: SIFT+26.72%, GIST192+64.75%.
Snapshot bytes unchanged: SIFT6,560,064; GIST39,840,064. No package/load/memory,
long-term CRUD or CPU-resource optimization conclusion from this query pilot.

## Reproduction and evidence

`python3 lite/benchmark/run_reciprocal_budget.py --output NEW_OUTPUT`
uses pinned host paths, library/source identity and input hashes; fails on an
existing output directory. Paths are recorded by scripts and command receipts.
Verify with `python3 lite/benchmark/verify_reciprocal_budget.py OUTPUT --gist
GIST_DIRECTORY --sift SIFT_DIRECTORY` (on one command line).
Verifier checks30runs/16,500queries, raw archive members and hashes, configuration,
per-query IDs, distances, ground-truth hits and P50/P99.
`raw.tar.gz`:166members/1,692,648bytes, SHA256
`cbe76274e3e42bcd7b66ec909edcfe81411d76d82175d60f66a1d98fdb908dd3`.
The source patch matches the previous pilot's Release5/5 functional validation;
this measured rebuild is recorded separately in `identity.json`.

Next gates: calibrate production and native Full fairly; freeze selected budgets
before fresh holdout validation; sanitizer and long-term true-value CRUD; complete
package/load/memory/CPU acceptance table. Do not promote candidate to PR2904 yet.
