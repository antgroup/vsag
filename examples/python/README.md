# VSAG Python Examples

Short Python programs that exercise `pyvsag`, VSAG's official Python
binding. Each example builds a small index from random vectors and
runs a k-NN search; pick one as a template and swap in your data.

## Prerequisites

- Linux (`x86_64` or `aarch64`, glibc 2.28+) and CPython 3.10+. See the
  [pyvsag guide](../../docs/docs/en/src/guide/pyvsag.md) for the full requirements.
- `pyvsag` itself, which installs NumPy as a dependency:
  ```bash
  pip install pyvsag
  ```

For local development against an in-tree build of VSAG, build the wheel
with `make pyvsag` and install it, or load the in-tree library directly;
both are described in the
[pyvsag guide](../../docs/docs/en/src/guide/pyvsag.md#build-from-source).

## Run

```bash
pip install pyvsag
python examples/python/103_index_hgraph.py
```

## File naming convention

Numbered examples mirror the numbering in
[`../cpp/`](../cpp/README.md) so a given prefix means the same topic
across languages. Use the numbered examples for new code.

## Examples

| File | Index | Notes |
| --- | --- | --- |
| [`103_index_hgraph.py`](103_index_hgraph.py) | HGraph | VSAG's headline in-memory graph index; start here. |
| [`104_index_hgraph_fp16_bf16.py`](104_index_hgraph_fp16_bf16.py) | HGraph (FP16 / BF16) | Half-precision vectors: `numpy.float16`, and `numpy.uint16` bit patterns for bfloat16. |
| [`105_index_brute_force.py`](105_index_brute_force.py) | BruteForce | Exact reference for recall calibration. |
| [`106_index_ivf.py`](106_index_ivf.py) | IVF | Inverted file, tuned for large-`k` / batch queries. |
| [`109_index_sindi.py`](109_index_sindi.py) | SINDI | Sparse vector index (text / inverted retrieval); converts readable term-weight mappings to CSR. |
| [`110_index_sindi_sq8_dmq_reorder.py`](110_index_sindi_sq8_dmq_reorder.py) | SINDI SQ8 + DMQ8 | Sparse vectors with SQ8 quantization + DMQ reranking. |

## Where to go next

- The C++ examples in [`../cpp/`](../cpp/README.md) cover topics not yet
  available in Python (custom allocator / thread pool, filtered search,
  `Tune` optimizer, extra info, persistent streaming, …). Contributions to add Python counterparts
  are welcome — see
  [`CONTRIBUTING.md`](../../CONTRIBUTING.md).
- For deeper conceptual material, see the user guide at
  [https://vsag.io/docs](https://vsag.io/docs).
