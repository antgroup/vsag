# FP32 runtime SIMD dispatch evidence, 2026-10-12

**Dispatch audit PASS, not a performance benchmark.** No new Full query latency
or equal-quality result is produced here. Neither production source is modified.

On CPU0 of VM-0-10-ubuntu (AMD EPYC9K65), actual exported dispatch function
pointers were compared with exported AVX512 kernel functions in fresh processes:

| Library | Runtime pointer | Equals AVX512 function |
|---|---|---|
| Production Lite | select_fp32_distance() | Yes |
| Native Full | FP32ComputeL2Sqr | Yes |
| Native Full | FP32ComputeL2SqrBatch4 | Yes |

This is stronger than CPU flags or compiled symbol availability. It verifies
default FP32 dispatch, not call counts, every graph phase, FP16 or RaBitQ dispatch.
The candidate uses unchanged dispatch source, but its measured library is no
longer retained; this probe does not directly inspect that binary.
Full batch4 is a native capability, not a claim that Lite has equivalent batching.

Pinned library hashes, probe source hash and commands/logs are in receipt.json
and commands.json. Reproduce with run_fp32_dispatch_probe.py --output NEW_DIR.
The probe consumers use -O3; linked Full remains native -Ofast/-g and Lite -O3.
Matching runtime SIMD does not remove this compiler-semantics boundary.
Next: same-semantics Full/Lite build comparison, fresh holdout query cohorts,
quality-qualified timings, then sanitizer/long-term CRUD candidate acceptance.
The r1 failure was a missing consumer declaration macro; r2 successfully ran
both probes but parser rejected native initialization logs. Those failed logs
are preserved separately; r3 parses exactly one JSON record and passes.
