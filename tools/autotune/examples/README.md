# AutoTune Examples

The SIFT example expects the ann-benchmarks dataset at
`/tmp/sift-128-euclidean.hdf5`. Edit `data_path` if the dataset is stored elsewhere.

From the VSAG repository root, run:

```bash
./build-release/tools/autotune/autotune \
  tools/autotune/examples/sift_hgraph_autotune_request.json
```

The command prints a compact summary. The complete report is written to
`/tmp/vsag_autotune_sift/report.json`.

This example expands six HGraph build configurations. For each build, AutoTune doubles
`ef_search` from `40` until recall passes, then binary-searches that interval for the smallest
passing value, capped at `1000`. Every evaluated point uses all queries. Add `$choices` or stepped
`$range` expressions to other parameter leaves to change the tuning space. Bare JSON arrays are
passed through as native index parameter values.
