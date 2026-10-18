# EmbeddingGemma 2 Kit

Multimodal embedding, vector indexing and cross-modal retrieval for
[EmbeddingGemma 2](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/)
(744M, text + image + audio + video), packaged as a plain importable Python
module and designed to live inside the [vsag](https://github.com/antgroup/vsag)
repository without coupling to the pyvsag build.

```python
from embeddinggemma2_kit import Embedder, AutoIndex

emb = Embedder(dtype="bfloat16")

# One shared 768-d space for every modality.
vectors = emb.embed_text(["a dog on grass"], role="query")
images = emb.embed_images(["dog.jpg"])
audio = emb.embed_audio(["clip.wav"])
video = emb.embed_video(["clip.mp4"], frames=8)

# Build a searchable index (HGraph when pyvsag is available).
index = AutoIndex(expected_size=len(images))
index.build(images, ids)
hits = index.search(vectors, topk=5)
```

Cross-modal retrieval, both directions over one corpus:

```python
from embeddinggemma2_kit.search import CrossModalSearch
from embeddinggemma2_kit.store.manifest import Manifest

search = CrossModalSearch(embedder=emb)
search.build_from_manifest(Manifest.from_jsonl("corpus.jsonl"))

search.search_text("a dog on grass", topk=5)   # text -> image
search.search_image("query.jpg", topk=5)       # image -> text
search.save("out/index")                        # persist; .load() restores it
```

## Install

The package itself needs only CPU PyTorch and the usual multimodal decoders.

```bash
pip install -e embeddinggemma2_kit            # core (text + image)
pip install -e 'embeddinggemma2_kit[media]'   # + audio/video decoding (av, soundfile, soxr)
pip install -e 'embeddinggemma2_kit[finetune]'# + PEFT, for adapter interfaces
```

`pyvsag` is deliberately **not** a dependency (see
[docs/design.md](docs/design.md#decoupling-from-pyvsag)).

## Provision the weights

Model weights are not distributed with this repository. Link a checkpoint you
have provisioned locally:

```bash
python scripts/fetch_weights.py --src /path/to/embeddinggemma-2
```

The script symlinks (never copies) and refuses to overwrite a real directory.
`weights/` is git-ignored along with all common model file patterns.

## Verify the installation

Two independent checks, neither requiring a GPU:

```bash
# 1. Retrieval quality, against the recorded Flickr30k baselines.
#    Gate: text->image hit@1 in [0.8506, 0.8752], image->text hit@1 in [0.9300, 0.9580]
python scripts/verify_flickr30k.py

# 2. HGraph recall vs exact search (needs the env that has pyvsag).
~/vsag-py310/bin/python scripts/verify_hgraph.py

# 3. End-to-end demo on a small subset.
python scripts/e2e_demo.py
```

Unit tests:

```bash
python -m pytest tests/ -q
```

## Examples

`examples/` runs in order, from raw encoding to end-to-end retrieval:

| Example | Shows |
|---|---|
| `01_embed_text.py` | Query vs document roles, and why they differ |
| `02_embed_image.py` | Image encoding, text→image agreement |
| `03_embed_audio_video.py` | Audio and video (with audio track) encoding |
| `04_text_to_image_e2e.py` | **Full pipeline**: images → index → text query → ranked images, then persistence |
| `05_image_to_text.py` | Image query → captions |

`04_text_to_image_e2e.py` is the one to read first: it ingests a corpus from
disk, builds the index, searches it with text, checks each hit against the
ground-truth image, and reloads the saved index to confirm the results are
unchanged.

The examples default to `fallback=True` so they run on a host without pyvsag.
Drop that argument to use the HGraph backend.

## Choosing a device

`Embedder` defaults to `device="cpu"`, which is the safe choice everywhere. On
Apple Silicon, `device="mps"` uses the integrated GPU and is much faster — but
the default `dtype="bfloat16"` is a trap on arm64, because there is no bf16
hardware path and it falls back to emulation. Measured on an M5 Pro (one image,
warm):

| Device | dtype | ms/image | vs best |
|---|---|---|---|
| mps | bfloat16 | **128** | 1.0× |
| mps | float32 | 262 | 2.0× |
| cpu | float32 | 1000 | 7.8× |
| cpu | bfloat16 | 12144 | **95×** |

So on a Mac the two settings that matter are `device="mps"` and, if you stay on
CPU, `dtype="float32"`. The default `bfloat16` is tuned for x86 servers with
AVX512-BF16, where it is the fast path; on arm64 it is the slow one.

Accuracy is unaffected either way: against the float32 CPU reference the minimum
cosine is 0.99994 for bf16 and 1.000000 for fp32, and a 10-image text-to-image
check scores 10/10 on both `cpu/float32` and `mps/bfloat16`.

```python
Embedder(device="mps", dtype="bfloat16")   # fastest on Apple Silicon
Embedder(device="cpu", dtype="float32")    # fastest on Apple Silicon CPUs
```

## Environment notes

* **CPU inference** runs on the repo's Python 3.14 environment
  (`torch 2.14.1+cpu`, `sentence-transformers 6.1.0`, pinned `transformers`).
  `sentence-transformers >= 6.1.0` is a hard requirement: older versions ignore
  multimodal input ordering.
* **pyvsag** needs Python 3.10 and an older numpy (`<2`); the wheel is built
  against numpy 1.x and importing it after numpy 2 segfaults. See
  [docs/design.md](docs/design.md#pyvsag-environment).
* **Threading**: the kit bounds torch/OMP/MKL threads on CPU. Without this, one
  image encode spawned 335 threads on a 192-core shared host and took 4.97s;
  bounded, it takes 1.84s. Raise `num_threads=` if you own the machine.

## Documentation

| Document | Contents |
|---|---|
| [docs/design.md](docs/design.md) | Architecture, decoupling contract, measured tuning results |
| [docs/api.md](docs/api.md) | API reference and the query/document role rules |
| [docs/finetune.md](docs/finetune.md) | Fine-tuning interfaces and GPU constraints |

## License

The kit's own code is Apache-2.0, matching the vsag repository. Model weights
are **not** covered by that license; they are governed by the Gemma terms of
use and must be provisioned separately.