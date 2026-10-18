"""Image-to-text search: find captions for a query image."""

import json
from pathlib import Path

from _common import corpus_dir, make_embedder
from embeddinggemma2_kit.search import CrossModalSearch
from embeddinggemma2_kit.store.manifest import CorpusItem, Manifest

DATA = corpus_dir()   # EG2_DATA overrides; see docs/platforms.md
raw = json.loads((DATA / "manifest.json").read_text())[:20]

# Captions as corpus items, each remembering which image it belongs to.
items = []
for entry in raw:
    for j, caption in enumerate(entry["captions"]):
        items.append(CorpusItem(
            id=f"{entry['id']}::c{j}",
            text=caption,
            metadata={"kind": "caption", "image_id": entry["id"], "text": caption},
        ))
manifest = Manifest(items)

search = CrossModalSearch(
    embedder=make_embedder(num_threads=16), fallback=True
)
search.build_from_manifest(manifest, progress=False)

probe = raw[0]
hits = search.search_image(probe["image"], topk=5)
print("query image:", Path(probe["image"]).name)
for h in hits:
    print(f"   [{h.score:.4f}] {h.metadata['image_id']}  {h.text[:60]!r}")
print("top1 belongs to the query image:",
      hits[0].metadata.get("image_id") == probe["id"])
