# SAQ quantization benchmark

`saq_quantization_benchmark` compares SAQ and multi-bit RaBitQ directly on an ann-benchmarks HDF5
dataset. It measures model training, encoding, query preparation plus scalar distance, batch-four
distance, code-pair distance, and reconstruction error. It writes one machine-readable JSON file;
progress messages go to standard error.

For the complete reproducible SIFT1M/GIST1M workflow, including Release configuration, HGraph
build/search, environment capture, checksums, and summary CSVs, use the
[`benchs/saq` workflow](https://github.com/antgroup/vsag/blob/main/benchs/saq/README.md).

## Build

Configure the repository with tools enabled, then build the target:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_TOOLS=ON
cmake --build build-release --target saq_quantization_benchmark -j2
```

## Usage

```text
saq_quantization_benchmark DATASET.hdf5 OUTPUT.json \
    [AVG_BITS=4] [TRAIN_COUNT=10000] [ENCODE_COUNT=100000] \
    [--ablations] [--exact-rabitq]
```

| Argument | Default | Constraint | Meaning |
| --- | --- | --- | --- |
| `DATASET.hdf5` | required | dense FP32, Euclidean ann-benchmarks layout | Input base vectors, queries, and ground truth. |
| `OUTPUT.json` | required | writable path | Complete structured result. |
| `AVG_BITS` | `4` | integer `[1, 8]` | Equal complete-record comparison point for SAQ and RaBitQ. |
| `TRAIN_COUNT` | `10000` | positive; capped by base count | Prefix used to train each quantizer. SAQ internally samples at most 65,536 inputs. |
| `ENCODE_COUNT` | `100000` | positive; capped by base count | Prefix encoded and used by the direct measurements. |
| `--ablations` | off | optional flag | Add PCA, rotation, adjustment-round, and fixed-segment SAQ variants. |
| `--exact-rabitq` | off | optional flag | Add exact RaBitQ encoding as a diagnostic control; the normal RaBitQ row uses its production fast encoder. |

Example:

```bash
./build-release/tools/saq_quantization_benchmark/saq_quantization_benchmark \
  /data/sift-128-euclidean.hdf5 /tmp/sift1m-saq.json \
  4 65536 1000000 --ablations --exact-rabitq
```

The benchmark currently evaluates L2 only. Direct throughput rows are microbenchmarks and do not
replace end-to-end index measurements at matched recall. Do not compare QPS values taken at
different recall levels.
