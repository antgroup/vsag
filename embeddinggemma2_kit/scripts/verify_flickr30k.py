#!/usr/bin/env python3
"""Reproduce the Flickr30k text->image / image->text baselines with the new kit.

This is the correctness gate described in the development plan (phase 3.5).
It encodes the same 1000 images and 5000 captions the earlier experiments used,
then compares against their recorded embeddings and retrieval metrics.

Reference metrics (results/bfloat16-t4/quality.json, 768-d):
    text->image hit@1 = 0.8632   (5000 queries / 1000 corpus)
    image->text hit@1 = 0.9440   (1000 queries / 5000 corpus)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path("/home/zhuangye/embeddinggemma2-cpu-20261014/data"))
    ap.add_argument("--baseline", type=Path,
                    default=Path("/home/zhuangye/embeddinggemma2-cpu-20261014/results/bfloat16-t4"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/eg2_verify"))
    ap.add_argument("--limit-images", type=int, default=0, help="0 = all 1000")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    from embeddinggemma2_kit import Embedder

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.data / "manifest.json")
    if args.limit_images:
        manifest = manifest[: args.limit_images]
    dev_manifest = manifest

    image_ids = [m["id"] for m in dev_manifest]
    image_paths = [m["image"] for m in dev_manifest]
    # One row per caption, in the same order the baseline used: manifest order,
    # captions in their listed order.
    caption_ids = [f"{m['id']}::caption:{j}" for m in dev_manifest
                   for j in range(len(m["captions"]))]
    captions = [c for m in dev_manifest for c in m["captions"]]

    print(f"images={len(image_ids)} captions={len(caption_ids)}", flush=True)
    if os.environ.get("OMP_NUM_THREADS") is None:
        # Without this the image path oversubscribes a shared host badly; the
        # package sets it on model load, but keep it visible for long runs.
        print("note: OMP_NUM_THREADS not preset; the kit will bound it on load", flush=True)

    emb = Embedder(dtype=args.dtype, num_threads=args.threads)
    print(f"model facts: {emb.facts}", flush=True)

    def encode_in_chunks(fn, items, chunk=25, label=""):
        """Encode with periodic progress so long runs are observable."""
        out = []
        t0 = time.perf_counter()
        for start in range(0, len(items), chunk):
            part = items[start : start + chunk]
            out.append(fn(part))
            done = min(start + chunk, len(items))
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(items) - done) / rate if rate > 0 else 0
            print(
                f"  [{label}] {done}/{len(items)} "
                f"({rate:.2f}/s, elapsed {elapsed:.0f}s, eta {eta:.0f}s)",
                flush=True,
            )
        return np.concatenate(out, axis=0) if out else np.zeros((0, 768), np.float32)

    # ---- images (document side, no role) ----
    img_vecs = encode_in_chunks(emb.embed_images, image_paths, chunk=25, label="images")
    print(f"encoded images shape={img_vecs.shape}", flush=True)

    # ---- captions as documents (corpus for image->text) ----
    doc_vecs = encode_in_chunks(
        lambda part: emb.embed_text(part, role="document"),
        captions, chunk=250, label="captions/doc",
    )
    print(f"encoded caption documents shape={doc_vecs.shape}", flush=True)

    # ---- captions as queries (for text->image) ----
    qry_vecs = encode_in_chunks(
        lambda part: emb.embed_text(part, role="query"),
        captions, chunk=250, label="captions/query",
    )
    print(f"encoded caption queries shape={qry_vecs.shape}", flush=True)

    np.save(args.out / "images.npy", img_vecs)
    np.save(args.out / "text_documents.npy", doc_vecs)
    np.save(args.out / "text_queries.npy", qry_vecs)

    # ---------------- numeric equivalence vs baseline ----------------
    report = {"dtype": args.dtype, "images": len(image_ids), "captions": len(caption_ids)}
    for name, mine in (("images", img_vecs), ("text_documents", doc_vecs),
                       ("text_queries", qry_vecs)):
        ref_path = args.baseline / f"{name}.npy"
        if not ref_path.exists():
            report[name] = {"status": "no_baseline"}
            continue
        ref = np.load(ref_path)
        if args.limit_images and ref.shape != mine.shape:
            report[name] = {"status": "skipped_shape", "ref": list(ref.shape), "mine": list(mine.shape)}
            continue
        diff = np.abs(ref - mine)
        cos = (ref * mine).sum(axis=1)
        report[name] = {
            "status": "ok",
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "min_cosine": float(cos.min()),
            "allclose_1e-5": bool(np.allclose(ref, mine, atol=1e-5, rtol=1e-4)),
            "rows": int(mine.shape[0]),
        }

    # ---------------- retrieval metrics ----------------
    def metrics(query, corpus, qids, cids, positive_fn, k=10):
        # Clamp k to the corpus size, otherwise argpartition raises for small
        # corpora (e.g. a reduced smoke run with fewer items than topk).
        k = min(k, corpus.shape[0])
        scores = query @ corpus.T
        idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        chosen = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-chosen, axis=1, kind="stable")
        top = np.take_along_axis(idx, order, axis=1)
        hits = {1: 0, 5: 0, 10: 0}
        rr = {1: 0.0, 5: 0.0, 10: 0.0}
        for row in range(query.shape[0]):
            ranked = [cids[i] for i in top[row]]
            for kk in hits:
                ranked_k = ranked[:kk]
                if any(positive_fn(qids[row], c) for c in ranked_k):
                    hits[kk] += 1
            for kk in rr:
                for rank, c in enumerate(ranked[:kk], 1):
                    if positive_fn(qids[row], c):
                        rr[kk] += 1.0 / rank
                        break
        n = query.shape[0]
        return {f"hit@{kk}": hits[kk] / n for kk in hits} | {
            f"mrr@{kk}": rr[kk] / n for kk in rr
        }

    # text->image: every caption of image X should retrieve image X
    t2i = metrics(qry_vecs, img_vecs, caption_ids, image_ids,
                  lambda q, d: q.split("::caption:")[0] == d, k=args.topk)
    # image->text: image X should retrieve any caption of image X
    i2t = metrics(img_vecs, doc_vecs, image_ids, caption_ids,
                  lambda q, d: d.split("::caption:")[0] == q, k=args.topk)
    report["text_to_image"] = t2i
    report["image_to_text"] = i2t

    (args.out / "verify.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # ---------------- gate ----------------
    ok = True
    exp = {"text_to_image": (0.8506, 0.8752), "image_to_text": (0.9300, 0.9580)}
    if not args.limit_images:
        for direction, (lo, hi) in exp.items():
            got = report[direction]["hit@1"]
            passed = lo <= got <= hi
            ok &= passed
            print(f"[{'PASS' if passed else 'FAIL'}] {direction} hit@1={got:.4f} expected in [{lo},{hi}]")
    else:
        print(f"partial run ({args.limit_images} images): gate not enforced")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())