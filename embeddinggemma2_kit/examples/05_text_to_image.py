"""Text-to-image search over a real corpus."""

import json
from pathlib import Path

from embeddinggemma2_kit import Embedder
from embeddinggemma2_kit.search import CrossModalSearch
from embeddinggemma2_kit.store.manifest import CorpusItem, Manifest

DATA = Path("/home/zhuangye/embeddinggemma2-cpu-20261014/data")
raw = json.loads((DATA / "manifest.json").read_text())[:20]

# A corpus of images only, so no post-filtering is needed.
manifest = Manifest([
    CorpusItem(id=e["id"], image=e["image"], metadata={"kind": "image"})
    for e in raw
])

search = CrossModalSearch(
    embedder=Embedder(dtype="bfloat16", num_threads=16), fallback=True
)
search.build_from_manifest(manifest, progress=False)

for entry in raw[:3]:
    query = entry["captions"][0]
    hits = search.search_text(query, topk=3)
    print(f"query: {query[:60]!r}")
    for h in hits:
        print(f"   [{h.score:.4f}] {Path(h.image_path).name}")
    print(f"   expected {entry['id']} -> top1 "
          f"{'HIT' if hits[0].id == entry['id'] else 'MISS'}")
