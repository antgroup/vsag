# VSAG Lite baseline experiment

This experimental tool records a deterministic standalone Lite BruteForce baseline. It does not claim a performance improvement and is intentionally kept outside PR #2904 while its API and build boundary are reviewed.

Build it with `-DENABLE_BENCHMARKS=ON`, then run:

```bash
lite/benchmark/run_baseline.sh build-lite-baseline/lite_benchmark baseline-results
```

The runner executes fixed 10k x 128 and 100k x 128 cases. Each case writes one CSV row, a Lite snapshot, and `/usr/bin/time -v` process measurements. It also records the commit, CPU, compiler, CMake version, OS, and benchmark binary checksum in `environment.txt`. The CSV includes build, CRUD, search, save/warm-load, snapshot-size, process peak-RSS, top-1 self-query recall, and a result checksum. The fixed seed makes generated vectors reproducible.

The same runner also executes a 20-round CRUD stability case. Every round performs 500 Update-Remove-Add cycles while preserving 10,000 live vectors, verifies exact self-query results before and after Save-Load, and records per-round latency, resident/peak RSS, snapshot size, recall, and checksum in `crud-stability.csv`. The snapshot is deliberately overwritten inside a newly created output directory so the experiment retains one verifiable final snapshot instead of twenty identical-size files.

The stability command can also be run directly:

```bash
lite_benchmark stability COUNT DIM ROUNDS CRUD_OPS QUERIES K SEED SNAPSHOT_DIRECTORY
```

`warm_load_ms` is measured in the same process immediately after saving and must not be reported as cold-start latency. A later cold-load experiment must use a separate process and explicitly control the operating-system page cache.

The benchmark refuses to overwrite an existing output directory or snapshot. Run it on an otherwise idle machine and retain the compiler, commit SHA, CPU, and raw output with any report.
