"""Build and persist an index, then reload it.

With pyvsag available this uses HGraph; otherwise it needs fallback=True, since
the kit refuses to silently substitute an exact index for an approximate one.
"""

import numpy as np

from embeddinggemma2_kit import AutoIndex

rng = np.random.default_rng(0)
vectors = rng.standard_normal((5000, 768), dtype=np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
ids = [f"doc-{i}" for i in range(len(vectors))]

# fallback=True is only needed on a host without pyvsag.
index = AutoIndex(dim=768, expected_size=len(vectors), fallback=True)
print("backend:", index.backend)

index.build(vectors, ids)
hits = index.search(vectors[:1], topk=3)
print("nearest:", [(h.id, round(h.score, 4)) for h in hits[0]])

index.save("/tmp/eg2_example_index")
reloaded = AutoIndex(dim=768, fallback=True)
reloaded.load("/tmp/eg2_example_index")
print("after reload:", [(h.id, round(h.score, 4))
                        for h in reloaded.search(vectors[:1], topk=3)[0]])
