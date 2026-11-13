"""Encode images and measure text-to-image agreement."""

import json
from pathlib import Path

from _common import corpus_dir, make_embedder

DATA = corpus_dir()   # EG2_DATA overrides; see docs/platforms.md
manifest = json.loads((DATA / "manifest.json").read_text())[:5]

emb = make_embedder(num_threads=16)
paths = [m["image"] for m in manifest]
images = emb.embed_images(paths)
print("image vectors:", images.shape)

# Use each image's own caption as the query and check it retrieves that image.
captions = emb.embed_text([m["captions"][0] for m in manifest], role="query")
scores = captions @ images.T
for i, m in enumerate(manifest):
    best = int(scores[i].argmax())
    print(f"{Path(m['image']).name}: top1={'HIT' if best == i else 'MISS'} "
          f"(score {scores[i][best]:.4f})")
