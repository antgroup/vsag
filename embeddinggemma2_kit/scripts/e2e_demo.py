"""End-to-end check: build a real index from Flickr30k and search both ways.

Run with the inference environment (3.14) that has torch:
    ~/embeddinggemma2-av-cpu-20261015/.venv/bin/python scripts/e2e_demo.py

Uses a small subset so it finishes quickly; the full-corpus correctness gate is
scripts/verify_flickr30k.py.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

DATA = Path("/home/zhuangye/embeddinggemma2-cpu-20261014/data")

# Queries are real captions taken from the corpus below, so ground truth is
# exact. (Paraphrased queries are a fine stress test but have no defined
# ground-truth image.)
QUERY_SOURCE_INDEX = [0, 1, 2]


def pick_queries(raw):
    return [raw[i]["captions"][0] for i in QUERY_SOURCE_INDEX if i < len(raw)]


def main() -> int:
    from embeddinggemma2_kit import Embedder
    from embeddinggemma2_kit.search.pipeline import CrossModalSearch
    from embeddinggemma2_kit.store.manifest import CorpusItem, Manifest

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    raw = json.loads((DATA / "manifest.json").read_text())[:limit]

    # Corpus holds the images and their captions. Which side is the query is
    # decided by the search call, not by how the corpus was built.
    items = []
    for entry in raw:
        items.append(
            CorpusItem(
                id=entry["id"],
                image=entry["image"],
                metadata={"kind": "image", "text": entry["captions"][0]},
            )
        )
        for j, caption in enumerate(entry["captions"]):
            items.append(
                CorpusItem(
                    id="{}::c{}".format(entry["id"], j),
                    text=caption,
                    metadata={
                        "kind": "caption",
                        "image_id": entry["id"],
                        "text": caption,
                    },
                )
            )
    manifest = Manifest(items)
    print("corpus: {} items ({} images, {} captions)".format(
        len(manifest), len(raw), len(manifest) - len(raw)))

    embedder = Embedder(dtype="bfloat16", num_threads=16)
    search = CrossModalSearch(embedder=embedder, fallback=False)
    started = time.perf_counter()
    search.build_from_manifest(manifest, progress=False)
    print("built index in {:.1f}s backend={} n={}".format(
        time.perf_counter() - started, search.index.backend, len(search.catalog)))

    # ---------------- text -> image ----------------
    print("\n=== TEXT -> IMAGE ===")
    t2i_hits = 0
    t2i_total = 0
    queries = pick_queries(raw)
    for query_pos, query in enumerate(queries):
        # Ask for a wide window, then filter: for a text query the corpus's own
        # captions are also text and legitimately rank above the images, so a
        # small topk would filter everything away. See docs/api.md.
        hits = search.search_text(query, topk=len(search.catalog))
        hits = [h for h in hits if h.metadata.get("kind") == "image"][:3]
        print("query: {!r}".format(query))
        for h in hits:
            name = Path(h.image_path).name if h.image_path else None
            print("   [{:.4f}] {}  img={}".format(h.score, h.id, name))
        # Ground truth: the image that owns this caption. Captions in the
        # Flickr30k manifest are not verbatim-identical, so match by whether any
        # caption of an image is close enough to be the source of this query.
        truth = raw[QUERY_SOURCE_INDEX[query_pos]]["id"]
        if truth and hits:
            hit = hits[0].id == truth
            t2i_hits += int(hit)
            t2i_total += 1
            print("   ground-truth {} -> top1 {}".format(truth, "HIT" if hit else "MISS"))
        elif truth:
            print("   ground-truth {} -> no image returned".format(truth))

    # ---------------- image -> text ----------------
    print("\n=== IMAGE -> TEXT ===")
    probe_entry = raw[0]
    hits = search.search_image(probe_entry["image"], topk=5)
    hits = [h for h in hits if h.metadata.get("kind") == "caption"]
    print("query image: {}  captions={}".format(
        Path(probe_entry["image"]).name, probe_entry["captions"][:2]))
    for h in hits:
        print("   [{:.4f}] {}  {!r}".format(h.score, h.id, (h.text or "")[:60]))
    i2t_ok = hits[0].metadata.get("image_id") == probe_entry["id"]
    print("   top1 caption belongs to the query image: {}".format(i2t_ok))

    # ---------------- persistence ----------------
    print("\n=== PERSISTENCE ===")
    with tempfile.TemporaryDirectory() as tmp:
        search.save(tmp)
        reloaded = CrossModalSearch(embedder=embedder)
        reloaded.load(tmp)
        before = [h.id for h in search.search_text(queries[0], topk=3)]
        after = [h.id for h in reloaded.search_text(queries[0], topk=3)]
        print("reloaded {} items, identical results: {}".format(
            len(reloaded.catalog), before == after))
        persist_ok = before == after

    # ---------------- cross-backend agreement ----------------
    # The same vectors through the exact backend should agree with HGraph.
    print("\n=== BACKEND AGREEMENT ===")
    exact = CrossModalSearch(embedder=embedder, backend="numpy")
    exact.catalog = search.catalog
    from embeddinggemma2_kit import AutoIndex

    exact.index = AutoIndex(dim=768, backend="numpy")
    exact.index.build(search.catalog.vectors, search.catalog.ids)
    a = [h.id for h in search.search_text(queries[0], topk=5)]
    b = [h.id for h in exact.search_text(queries[0], topk=5)]
    print("hgraph top5: {}".format(a))
    print("exact  top5: {}".format(b))
    print("identical: {}".format(a == b))

    print("\nSUMMARY: t2i {}/{} hits, i2t {}, persistence {}".format(
        t2i_hits, t2i_total, i2t_ok, persist_ok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())