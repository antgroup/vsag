# Dataset profiler examples

Reproducible demo datasets and reports for `vsag-dataset-profiler`.

## Contents

| File | Description |
| --- | --- |
| `make_toy_datasets.py` | Generates the toy datasets below and profiles them with a fixed seed. |
| `output/toy_dense.hdf5` | 120 clustered `FLOAT32` vectors (16-D) with duplicates, an all-zero vector, an outlier, and one NaN (generated locally; not committed). |
| `output/toy_sparse.hdf5` | 82 sparse records with `train_offsets`, including a duplicate and an empty record (generated locally; not committed). |
| `output/toy_dense.profile.json` | Versioned deterministic JSON report for the dense file. |
| `output/toy_dense.profile.html` | Self-contained interactive HTML report for the dense file. |
| `output/toy_sparse.profile.json` / `output/toy_sparse.profile.html` | The same pair for the sparse file. |

## Reproduce

```bash
cd tools/dataset_profiler
pip install -r requirements.txt
python examples/make_toy_datasets.py            # writes into examples/output/
```

The generation seed and the profiling seed are fixed in the script, so re-running it on the
same NumPy build reproduces the committed reports byte-for-byte (verified with NumPy 2.2 /
Python 3.13; SVD sign conventions may differ across BLAS builds).

## Profile your own dataset

```bash
python -m dataset_profiler /path/to/your.hdf5 --split train --seed 42 \
    --out-json your.profile.json --out-html your.profile.html
```

To profile arrays that you currently only have in memory (for example the same arrays you
pass to `pyvsag`), first export them to the HDF5 layout described in
[`dataset_format.md`](../../../docs/docs/en/src/resources/dataset_format.md):

```python
import h5py, numpy as np

vectors = np.load("train.npy").astype("float32")   # your (N, D) array
with h5py.File("my_dataset.hdf5", "w") as f:
    f.attrs["distance"] = "euclidean"
    f.create_dataset("train", data=vectors)
```
