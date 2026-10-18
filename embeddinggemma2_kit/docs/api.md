# API reference

## `Embedder`

```python
Embedder(
    model_path=None,          # default: <kit>/weights/embeddinggemma-2
    device="cpu",
    dtype="bfloat16",         # or "float32"; float16 is unsupported
    text_batch_size=32,
    video_frames=8,
    video_include_audio=False,
    verify=True,
    num_threads=None,         # None -> bounded default (8); raise on a private host
)
```

The checkpoint is loaded lazily on first use, so constructing an `Embedder` is
cheap.

All methods return contiguous `float32`, shape `(N, 768)`, L2-normalised.

| Method | Input | Notes |
|---|---|---|
| `embed_text(texts, role=...)` | list of str | `role` selects the task prefix |
| `embed_images(images)` | paths or PIL images | no role: images are symmetric |
| `embed_audio(paths)` | paths | mono 16 kHz, one forward pass each |
| `embed_video(paths, frames=, with_audio=)` | paths | even frame sampling |
| `embed(items, text_role=...)` | list of mixed dicts | mixes modalities in one call |
| `facts` | — | checkpoint verification facts |

### The `role` argument

`role="query"` for the thing you search **with**; `role="document"` for the
thing being searched **for**. These map to different task prefixes in the
checkpoint. Using the wrong one degrades retrieval *silently*, so raw prompt
names are not accepted.

The default is `document`, because most calls encode corpus items.

## `AutoIndex`

```python
AutoIndex(
    dim=768,
    metric="cosine",
    backend="auto",        # "auto" | "hgraph" | "numpy"
    fallback=False,        # see below
    expected_size=None,    # tunes HGraph parameters
    ef_search=None,        # default: size-aware (1000 for <=100k vectors)
    index_params=None,
)
```

| Method | Purpose |
|---|---|
| `build(vectors, ids)` | create the index |
| `add(vectors, ids)` | append (rejects duplicate ids) |
| `search(queries, topk)` | returns `list[list[SearchHit]]` |
| `search_one(query, topk)` | single query convenience |
| `save(dir)` / `load(dir)` | persistence |
| `backend` | which backend was actually selected |

`SearchHit` carries `id`, `score` (similarity, **higher is better**) and `rank`.

### Fallback policy

With pyvsag unavailable:

```python
AutoIndex(dim=768)                  # raises IndexBackendUnavailable
AutoIndex(dim=768, fallback=True)   # uses exact numpy search, logs a warning
```

The refusal is deliberate. HGraph is approximate and numpy is exact; swapping
them without the caller's knowledge changes recall and latency characteristics.
The exception message includes installation instructions.

## Cross-modal retrieval

```python
from embeddinggemma2_kit.search import CrossModalSearch
from embeddinggemma2_kit.store.manifest import Manifest

search = CrossModalSearch(embedder=emb, fallback=True)
search.build_from_manifest(manifest)
search.search_text("query", topk=10)
search.search_image("q.jpg", topk=10)
search.save("out/")
```

One corpus encoding serves both directions: which side is the query is decided
by the call, not by how the corpus was built. Internally the corpus is encoded
with the document prefix and queries with the query prefix.

### Filtering by modality — read this

`search_text(..., modality="image")` filters **after** searching. If the corpus
also contains text items (captions), those are text too and legitimately rank
above images for a text query. With a small `topk` the filter can therefore
return nothing.

Measured on a 48-item corpus containing 8 images and their 40 captions, for a
query that is one of those captions:

```
rank 0-4 : the 5 captions of the target image
rank 5   : the target image          <- first image result
```

So request a wide window and filter afterwards:

```python
hits = search.search_text(query, topk=len(search.catalog))
images = [h for h in hits if h.metadata.get("kind") == "image"][:10]
```

Alternatively, index images only (a corpus of just images) so no filtering is
needed.

## Manifest

One JSON object per line:

```json
{"id": "img-1", "image": "/data/1.jpg", "metadata": {"kind": "image"}}
{"id": "cap-1", "text": "a dog on grass", "metadata": {"kind": "caption", "image_id": "img-1"}}
```

Rules:

* `id` is required and unique.
* At least one of `text`/`image`/`audio`/`video` must be present.
* Unknown fields are rejected, so a typo like `img` fails loudly.
* `primary_modality` prefers media over text, so an image with a caption is an
  image item.
* Empty or whitespace-only text is rejected: it would produce a valid-looking
  embedding of nothing.

## Errors

All deliberate errors derive from `KitError`.

| Error | Raised when |
|---|---|
| `ConfigError` | invalid configuration or manifest |
| `ModelLoadError` | checkpoint missing, incomplete, or wrong size |
| `MediaDecodeError` | media unreadable or malformed |
| `MissingDependencyError` | an optional decoder/extra is absent |
| `IndexBackendUnavailable` | backend missing and `fallback=False` |
| `IndexStateError` | invalid index operation or shape mismatch |
| `FinetuneError` | fine-tuning misuse or missing GPU |