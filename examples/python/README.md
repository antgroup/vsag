# VSAG Python Examples

Short Python programs that exercise `pyvsag`, VSAG's official Python
binding. Each example builds a small index from random vectors and
runs a k-NN search; pick one as a template and swap in your data.

## Prerequisites

- Python 3.8+ (see the [PyPI page](https://pypi.org/project/pyvsag/)
  for the exact set of supported interpreters per release).
- NumPy.
- `pyvsag` itself:
  ```bash
  pip install pyvsag
  ```

For local development against an in-tree build of VSAG, follow the
[Python bindings build instructions](../../python_bindings/) instead
of installing from PyPI.

## Run

```bash
pip install pyvsag numpy
python examples/python/101_index_hnsw.py
```

## File naming convention

Numbered examples mirror the numbering in
[`../cpp/`](../cpp/README.md) so a given prefix means the same topic
across languages. The legacy `example_*.py` files predate that scheme
and are kept for backward compatibility — prefer the numbered ones for
new code.

## Examples

| File | Index | Notes |
| --- | --- | --- |
| [`101_index_hnsw.py`](101_index_hnsw.py) | HNSW | Shortest "build + KNN" round-trip; start here. |
| [`102_index_diskann.py`](102_index_diskann.py) | DiskANN | Disk-resident graph index. |
| [`103_index_hgraph.py`](103_index_hgraph.py) | HGraph | VSAG's headline in-memory graph index. |
| [`105_index_brute_force.py`](105_index_brute_force.py) | BruteForce | Exact reference for recall calibration. |
| [`106_index_ivf.py`](106_index_ivf.py) | IVF | Inverted file, tuned for large-`k` / batch queries. |
| [`109_index_sindi.py`](109_index_sindi.py) | SINDI | Sparse vector index (text / inverted retrieval). |
| [`example_hnsw.py`](example_hnsw.py) | HNSW (legacy) | Kept for backward compatibility; prefer `101_*.py`. |
| [`example_diskann.py`](example_diskann.py) | DiskANN (legacy) | Kept for backward compatibility; prefer `102_*.py`. |

## Where to go next

- The C++ examples in [`../cpp/`](../cpp/README.md) cover topics not yet
  available in Python (custom allocator / thread pool, range search,
  filtered search, serialization, `Tune` optimizer, extra info,
  persistent streaming, …). Contributions to add Python counterparts
  are welcome — see
  [`CONTRIBUTING.md`](../../CONTRIBUTING.md).
- For deeper conceptual material, see the user guide at
  [https://vsag.io/docs](https://vsag.io/docs).
