"""Encode images and measure text-to-image agreement."""

import json
from pathlib import Path

from embeddinggemma2_kit import Embedder

DATA = Path("/home/zhuangye/embeddinggemma2-cpu-20261014/data")
manifest = json.loads((DATA / "manifest.json").read_text())[:5]

emb = Embedder(dtype="bfloat16", num_threads=16)
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
