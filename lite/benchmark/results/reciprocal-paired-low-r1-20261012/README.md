# Same-budget FP32 graph comparison, 2026-10-12

This closes the low-budget production-control gap from the preceding pilot.
It is not native Full benchmarking or final acceptance. PR2904 remains unchanged.
Candidate patch matches the previous isolated reciprocal-occlusion pilot.

## Protocol

CPU0, one thread, Release, FP32, degree16/build128/top10, SIFT/GIST10k.
Both sides at query budgets128/192/256/384, three fresh processes each.
Budget order rotates by trial; side order reverses in the middle trial. No
concurrent build or measurement. Each process builds before measuring queries.
Existing SIFT100/GIST1000 queries are developer data, not unseen validation.
Medians of per-run metrics; not pooled P99 or 26,400 distinct queries.

| Dataset | Budget | Production recall | Candidate recall | Production P50 us | Candidate P50 us | Production P99 us | Candidate P99 us |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIFT | 128 | .9850 | 1.0000 | 50.799 | 55.778 | 63.788 | 70.168 |
| SIFT | 192 | .9890 | 1.0000 | 71.098 | 78.817 | 87.389 | 99.488 |
| SIFT | 256 | .9890 | 1.0000 | 90.767 | 99.717 | 109.277 | 120.726 |
| SIFT | 384 | .9910 | 1.0000 | 128.647 | 140.045 | 154.566 | 162.475 |
| GIST | 128 | .8979 | .9833 | 143.926 | 197.684 | 198.925 | 256.084 |
| GIST | 192 | .9243 | .9905 | 195.104 | 273.773 | 264.993 | 349.450 |
| GIST | 256 | .9397 | .9938 | 251.062 | 345.768 | 341.490 | 434.187 |
| GIST | 384 | .9563 | .9966 | 349.050 | 484.277 | 472.027 | 603.803 |

At the same budget, candidate increases recall but is slower: SIFT P50 about
9-11%, GIST about37-40%. Do not advertise this as same-parameter speedup.
For a developer-only .990 minimum-recall diagnostic, the smallest tested SIFT
passing budget is production384 vs candidate128: P50 128.647 vs55.778us
(-56.64%), recalls .991 vs1.000. This is threshold-qualified, not identical-recall
or independent-validation evidence. GIST production fails .990 throughout this
low grid; candidate first passes at192. No qualified GIST equal-quality comparison
is available from this grid. Do not infer original can never pass at higher budgets.
Build remains more expensive, roughly SIFT27%/GIST64-66%; snapshot bytes unchanged.
No package/load/memory/CPU-resource/CRUD/SAN or FP16/RaBitQ gain claimed here.

## Evidence and reproduction

`run_reciprocal_budget.py --paired-low --output NEW_DIRECTORY` reproduces this
mode using pinned remote paths and identities. `verify_reciprocal_budget.py`
accepts the same OUTPUT/--gist/--sift arguments as the preceding pilot and retains
legacy-mode verification. Raw verifier PASS48runs/26,400recorded queries,
configuration, input hashes, archive contents, IDs/hits and P50/P99 checked.
`raw.tar.gz`:257members/2,701,546bytes, SHA256
`16c44d5db1563e5ae0c1bc51db989995f1eacce91241b8af4ff59398391e93c6`.
Next: native Full matched runtime/SIMD and compiler semantics, frozen budgets on
fresh queries, then candidate sanitizers and long-term true-value CRUD before
considering promotion. Keep the complete current-implementation acceptance table
separate from experimental candidate claims.
