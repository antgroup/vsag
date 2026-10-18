"""End-to-end text-to-image search: build an index from images, query with text.

This is the full pipeline in one file -- corpus ingestion, embedding, index
construction, retrieval, and persistence -- rather than a synthetic-vector demo:

    images on disk -> embed -> index -> text query -> ranked images

With pyvsag installed this uses HGraph. On a host without it, pass
``fallback=True`` to select the exact numpy backend explicitly; the kit refuses
to substitute an exact index for an approximate one silently.

    python examples/04_text_to_image_e2e.py
"""

import json
from pathlib import Path

from embeddinggemma2_kit import Embedder
from embeddinggemma2_kit.search import CrossModalSearch
from embeddinggemma2_kit.store.manifest import CorpusItem, Manifest

DATA = Path("/home/zhuangye/embeddinggemma2-cpu-20261014/data")
INDEX_DIR = Path("/tmp/eg2_text_to_image_index")
N_IMAGES = 20

raw = json.loads((DATA / "manifest.json").read_text())[:N_IMAGES]

# Corpus side: images only. Each item carries its ground-truth id so a hit can
# be checked against what the query actually describes.
manifest = Manifest([
    CorpusItem(
        id=entry["id"],
        image=entry["image"],
        metadata={"kind": "image", "caption": entry["captions"][0]},
    )
    for entry in raw
])
print(f"corpus: {len(manifest)} images")

# --------------------------------------------------------------- build index
# Images are encoded with the *document* prompt and text queries with the
# *query* prompt. CrossModalSearch owns that asymmetry so it cannot be inverted.
search = CrossModalSearch(
    embedder=Embedder(dtype="bfloat16", num_threads=16),
    fallback=True,  # drop this once pyvsag is installed, to use HGraph
)
search.build_from_manifest(manifest, progress=False)
print("backend:", search.index.backend)

# -------------------------------------------------------------------- query
# Each image's own caption is the text query, so that image is the expected
# result. Searching a wide topk and filtering to images keeps this honest: a
# text query can legitimately rank captions above images.
correct = 0
for entry in raw[:5]:
    query = entry["captions"][0]
    hits = [h for h in search.search_text(query, topk=len(manifest))
            if h.metadata.get("kind") == "image"]

    top1 = hits[0]
    hit = top1.id == entry["id"]
    correct += hit

    print(f"\nquery: {query[:64]!r}")
    for h in hits[:3]:
        marker = "  <-- expected" if h.id == entry["id"] else ""
        print(f"   [{h.score:.4f}] {Path(h.image_path).name}{marker}")
    print(f"   top-1 {'HIT' if hit else 'MISS'} (expected {entry['id']})")

print(f"\ntext -> image top-1 accuracy: {correct}/5")

# ----------------------------------------------------------------- persist
search.save(INDEX_DIR)

reloaded = CrossModalSearch(embedder=search.embedder).load(INDEX_DIR)
query = raw[0]["captions"][0]
before = [h.id for h in search.search_text(query, topk=3)]
after = [h.id for h in reloaded.search_text(query, topk=3)]
print("saved to:", INDEX_DIR)
print("top-3 identical after reload:", before == after)