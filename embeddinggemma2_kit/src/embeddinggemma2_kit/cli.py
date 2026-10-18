"""Command line interface.

Thin wrapper over the library so the kit is usable without writing Python.

    kit-embed text --role query "a dog on grass"
    kit-embed image photo.jpg
    kit-index build corpus.jsonl --out out/index
    kit-search text "a dog on grass" --index out/index --topk 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_embedder(args) -> object:
    from embeddinggemma2_kit.model.embedder import Embedder

    return Embedder(
        model_path=args.model,
        dtype=args.dtype,
        num_threads=args.threads,
    )


def _cmd_embed(args) -> int:
    emb = _build_embedder(args)
    import numpy as np

    if args.modality == "text":
        vectors = emb.embed_text(args.input, role=args.role)
    elif args.modality == "image":
        vectors = emb.embed_images(args.input)
    elif args.modality == "audio":
        vectors = emb.embed_audio(args.input)
    elif args.modality == "video":
        vectors = emb.embed_video(args.input, frames=args.frames)
    else:
        print(f"unknown modality {args.modality}", file=sys.stderr)
        return 2

    if args.out:
        np.save(args.out, vectors)
        print(f"wrote {vectors.shape} to {args.out}")
    else:
        print(json.dumps({
            "shape": list(vectors.shape),
            "norm": float(np.linalg.norm(vectors[0])),
            "head": [round(float(v), 6) for v in vectors[0][:8]],
        }, indent=2))
    return 0


def _cmd_index(args) -> int:
    from embeddinggemma2_kit.search.pipeline import CrossModalSearch
    from embeddinggemma2_kit.store.manifest import Manifest

    search = CrossModalSearch(
        embedder=_build_embedder(args),
        backend=args.backend,
        fallback=args.fallback,
    )
    manifest = Manifest.from_jsonl(args.manifest)
    print(f"corpus: {len(manifest)} items", file=sys.stderr)
    search.build_from_manifest(manifest)
    search.save(args.out)
    print(f"saved index ({search.index.backend}, {len(search.catalog)} items) to {args.out}")
    return 0


def _cmd_search(args) -> int:
    from embeddinggemma2_kit.search.pipeline import CrossModalSearch

    search = CrossModalSearch(embedder=_build_embedder(args), fallback=args.fallback)
    search.load(args.index)

    if args.query_type == "text":
        results = search.search_text(args.query, topk=args.topk)
    else:
        results = search.search_image(args.query, topk=args.topk)

    payload = [
        {
            "rank": hit.rank,
            "id": hit.id,
            "score": round(hit.score, 6),
            "metadata": hit.metadata,
        }
        for hit in results
    ]
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for entry in payload:
            print(f"{entry['rank']:3d}  {entry['score']:.4f}  {entry['id']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kit-embed", description=__doc__)
    parser.add_argument("--model", default=None, help="checkpoint directory")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    sub = parser.add_subparsers(dest="command", required=True)

    embed = sub.add_parser("embed", help="encode inputs to vectors")
    embed.add_argument("modality", choices=["text", "image", "audio", "video"])
    embed.add_argument("input", nargs="+")
    embed.add_argument("--role", default="document", choices=["query", "document"])
    embed.add_argument("--frames", type=int, default=8)
    embed.add_argument("--out", default=None, help="write .npy here")
    embed.set_defaults(func=_cmd_embed)

    index = sub.add_parser("index", help="build an index from a manifest")
    index.add_argument("manifest")
    index.add_argument("--out", required=True)
    index.add_argument("--backend", default="auto", choices=["auto", "hgraph", "numpy"])
    index.add_argument("--fallback", action="store_true")
    index.set_defaults(func=_cmd_index)

    search = sub.add_parser("search", help="query a saved index")
    search.add_argument("query_type", choices=["text", "image"])
    search.add_argument("query")
    search.add_argument("--index", required=True)
    search.add_argument("--topk", type=int, default=10)
    search.add_argument("--fallback", action="store_true")
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=_cmd_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())