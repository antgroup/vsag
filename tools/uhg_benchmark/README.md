# UHG benchmark

See the [canonical benchmark documentation](../../docs/docs/en/src/indexes/uhg.md#ms-marco-comparison-benchmark) for dataset format, build commands, measurement scope, and offline construction details.

Generate tables and plots after a completed run:

```sh
python3 -m pip install -r tools/uhg_benchmark/requirements.txt
python3 tools/uhg_benchmark/report.py /data/results/uhg-msmarco-full
```

The independent small-subset oracle compares top-100 membership and allows FP32 rounding differences in near-tied ordering:

```sh
python3 tools/uhg_benchmark/verify_gold.py \
  /data/datasets/msmarco-1024-8m-ip.hdf5 /data/results/uhg-msmarco-pilot 10000 32
```

Compare original and optimized query runs with the same cached graph:

```sh
python3 tools/uhg_benchmark/compare.py \
  /data/results/uhg-msmarco-full /data/results/uhg-msmarco-optimized
```

Use a separate output directory containing links to the original guide/ground-truth caches and a hard link or copy of the original UHG archive. The benchmark upgrades the archive atomically with forward vectors, retaining the graph and guides; the original archive remains unchanged. New vector storage increases memory and disk usage.
