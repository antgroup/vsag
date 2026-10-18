# Design

## Repository placement

The kit lives at `embeddinggemma2_kit/` in the vsag repository as a
self-contained Python package. It does not touch `python/` (the pyvsag package),
`python_bindings/`, or the C++ build, so it cannot affect how vsag itself
builds or ships.

```
vsag/
├── python/                  # pyvsag package (unmodified)
├── python_bindings/         # pybind11 bindings (unmodified)
├── src/algorithm/hgraph/    # HGraph C++ (unmodified)
└── embeddinggemma2_kit/     # this package
```

## Decoupling from pyvsag

"Not coupled to pyvsag" is a testable property here, not a slogan:

1. **Static.** Exactly one file imports pyvsag:
   `src/embeddinggemma2_kit/index/backends/pyvsag_hgraph.py`. Enforced by
   `tests/test_decoupling.py::test_only_one_module_imports_pyvsag`.
2. **Runtime.** `import embeddinggemma2_kit` succeeds with pyvsag absent, and
   the numpy backend remains fully usable. Enforced by a subprocess test that
   blocks the import.
3. **No silent downgrade.** With pyvsag missing, `AutoIndex(dim=768)` raises
   `IndexBackendUnavailable`. Only an explicit `fallback=True` selects the exact
   numpy backend, and it logs a warning. The two backends have different recall
   semantics (approximate vs exact), so substituting one for the other without
   the caller's knowledge would be a correctness hazard, not a convenience.
4. **Install.** `pip install embeddinggemma2-kit` never pulls or builds vsag.

## Module layout

| Module | Responsibility |
|---|---|
| `model/embedder.py` | Unified encoding for every modality; enforces the output contract |
| `model/loader.py` | Checkpoint loading and strict verification |
| `model/prompts.py` | Maps the public `role` onto the checkpoint's prompt names |
| `model/preprocess/` | Media decoding into model-input dicts |
| `index/auto.py` | Backend selection, parameter tuning, recall measurement |
| `index/backends/` | `pyvsag_hgraph` (approximate) and `numpy_flat` (exact) |
| `search/pipeline.py` | Corpus build plus text↔image search |
| `store/manifest.py` | Declarative corpus description |
| `store/catalog.py` | Vectors + metadata + provenance persistence |
| `finetune/trainer.py` | Adapter interfaces and config validation |

## Output contract

Every embedding call returns contiguous `float32` of shape `(N, 768)`, with
finite, L2-normalised rows. This is asserted in `_finalize`, not merely intended,
because downstream code (and the HGraph cosine metric) assumes unit norm.

The parameter count is asserted at load time against **744,371,488**. A
text-only or partially downloaded checkpoint would still emit plausible 768-d
vectors, which would silently invalidate every comparison against the recorded
baselines.

## Query vs document roles

EmbeddingGemma 2 is an asymmetric retrieval model. Queries and documents carry
different task prefixes, and using the wrong one produces *worse neighbours
without raising an error*. The kit therefore exposes only `role="query"` and
`role="document"` and never accepts a raw prompt name.

This failure mode is directly detectable with the Flickr30k gate: a prefix swap
moves hit@1 well outside the expected confidence interval.

## Measured tuning results

These numbers come from runs on the development host and drive the defaults in
the code.

### Threading

Encoding one image on a 192-core shared host:

| Configuration | Threads spawned | Time |
|---|---|---|
| torch defaults only (192 intra-op) | 335 | 4.97 s |
| `torch.set_num_threads(8)` | 335 | 4.97 s |
| \+ `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` | 209 | **1.84 s** |

Setting torch's thread count alone is insufficient: torchvision's native image
kernels honour OpenMP, not torch's pool. The kit sets both.

### HGraph recall at 768 dimensions

recall@10 against exact search, 20k unit-norm 768-d vectors:

| max_degree | ef_construction | ef_search | recall@10 |
|---|---|---|---|
| 32 | 200 | 100 | 0.324 |
| 32 | 200 | 400 | 0.745 |
| 32 | 200 | 1000 | 0.956 |
| 32 | 200 | 2000 | 0.992 |
| **64** | **400** | **1000** | **1.000** |

Two lessons:

* The low `ef_search` values used in vsag's 128-d examples (e.g. 100) are far
  too small at 768-d — they yield 0.32 recall.
* `max_degree`/`ef_construction` dominate more than `ef_search`. Building the
  graph denser (64/400) reaches perfect recall at a *lower* `ef_search` than a
  sparse graph needs, so it is also the faster configuration.

### pyvsag return-value semantics

`knn_search` with `metric_type="cosine"` returns **distances** ordered
ascending (a vector matched against itself gives ~1.2e-07). The kit converts
these to similarities via `1 - distance`, which for unit-norm vectors is the
cosine similarity. Returning raw distances would have inverted the ranking.

`knn_search` also accepts a single vector and returns `(ids, dists)`; there is
no batch entry point, so batching happens in the adapter.

## pyvsag environment

pyvsag needs **Python 3.10** and **numpy < 2**.

The wheel bundled on the development host (`pyvsag-1.1.0.dev159`, cp310) was
built against numpy 1.x. Importing it after numpy 2 is imported in the same
process **segfaults** inside `pyvsag/__init__.py`, before any kit code runs:

```
import numpy; import pyvsag   # SIGSEGV
import pyvsag                # ok
```

So the index environment is kept separate from the inference environment:

```bash
uv venv --python 3.10 ~/vsag-py310
uv pip install --python ~/vsag-py310/bin/python 'numpy<2' pyvsag
```

`~/vsag-py310` intentionally holds no torch. Embeddings are handed over as
`.npy` files, which also means index building can be deployed independently of
the (large) inference stack.

## Provenance

Every saved catalog records the model revision, parameter count, dtype and
Python version. Embeddings are only comparable when produced by the same model
at the same precision, and recording this makes an accidental cross-model
comparison detectable after the fact rather than mysterious.

## Known constraints

* Audio and video items are encoded one at a time. The verified encoder used
  `batch_size=1` for media; batching changes padding and therefore the vectors.
* The model does not support FP16 activations. Use `bfloat16` (verified CPU
  default) or `float32`.
* Video frame selection happens in the kit, never inside the processor.
  `frames_indices` is asserted to match exactly, because a mismatch misaligns the
  embedding silently.