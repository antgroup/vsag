# VSAG Dataset Profiling Tool

`profile_dataset` profiles a VSAG-compatible HDF5 dataset (dense or sparse) into a versioned JSON
report and a self-contained HTML report, with optional AI-assisted interpretation of the aggregate
metrics. The canonical documentation lives in the website source tree.

```bash
VSAG_ENABLE_TOOLS=ON make release
./build-release/tools/profile_dataset/profile_dataset --datapath /tmp/data/sift-128-euclidean.hdf5
```

- Local English docs: [docs/docs/en/src/resources/profile_dataset.md](../../docs/docs/en/src/resources/profile_dataset.md)
- Local Chinese docs: [docs/docs/zh/src/resources/profile_dataset.md](../../docs/docs/zh/src/resources/profile_dataset.md)
- Technical report: [docs/docs/en/src/resources/profile_dataset_design.md](../../docs/docs/en/src/resources/profile_dataset_design.md)
- Website: <https://vsag.io/docs/resources/profile_dataset.html>
- Chinese website: <https://vsag.io/docs/zh/resources/profile_dataset.html>

Tests: build with `ENABLE_TESTS=ON` and run `./build/tools/profile_dataset/profile_dataset_test`.
