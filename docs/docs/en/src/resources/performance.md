# Reference Performance

This page is the entry point and explanation for official performance numbers. For concrete
figures, use the latest GitHub releases and reproduce with the
[performance evaluation tool](eval.md) in your target environment.

## Reference Hardware

Official benchmarks typically run on hardware in the following class (concrete SKUs vary per
release):

- **CPU**: mainstream x86_64 server CPUs (with AVX2 / AVX-512)
- **Memory**: enough DDR4/DDR5 to cover the index plus OS page cache
- **Disk**: NVMe SSD (for DiskANN scenarios)
- **OS**: Ubuntu 20.04 / 22.04 or CentOS 7 / 8
- **Build**: `make release`, with Intel MKL enabled when available

## Reference Datasets

Official comparisons use HDF5 datasets compatible with
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks):

| Dataset | Dim | Metric | Size |
|---------|-----|--------|------|
| SIFT-1M | 128 | L2 | 1,000,000 |
| GIST-1M | 960 | L2 | 1,000,000 |
| Deep-10M | 96 | L2 | 10,000,000 |
| Text-to-Image-1M | 200 | IP | 1,000,000 |

## Key Metrics

- QPS (single- and multi-threaded)
- Average recall (Recall@k)
- P50 / P95 / P99 latency
- Peak memory and index size
- Build time

## Reproduction

```bash
make release
./build-release/tools/eval/eval_performance --config tools/eval/eval_template.yaml
```

Compare the resulting JSON / Markdown output against the official figures to catch performance
regressions or quantization degradations.

## Contributing Numbers

Pull requests that extend this page with "results on additional hardware" sections are welcome.
Please include:

- Detailed CPU / memory / disk information.
- The VSAG version (`git rev-parse HEAD`).
- The `eval_performance` output (JSON and Markdown are both helpful).
- The exact build command and environment variables (e.g. `VSAG_ENABLE_INTEL_MKL`).
