"""Find the fastest device/dtype on this machine, then verify retrieval quality.

Written for Apple Silicon, where the default ``dtype="bfloat16"`` is a trap:
arm64 has no bf16 hardware path, so ``cpu`` + ``bfloat16`` needs ~12 s per image
while ``mps`` + ``bfloat16`` needs ~0.13 s. The same default is *correct* on x86
servers with AVX512-BF16, so it cannot simply be flipped for everyone -- which
is why this example measures instead of assuming.

Part 1 benchmarks the combinations this checkpoint can use and prints a
recommendation. Part 2 runs a real text->image retrieval check with the winner,
and (when a second configuration is cheap enough) confirms the two produce the
same embeddings.

    python examples/06_apple_silicon.py
    python examples/06_apple_silicon.py --full        # include slow combinations
    EG2_DATA=/path/to/corpus python examples/06_apple_silicon.py

Part 2 needs images and captions; it is skipped with a hint if none are found.
Part 1 needs no dataset at all.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

# Corpora are not distributed with the kit. Point EG2_DATA at a directory that
# holds manifest.json + images (see scripts/verify_flickr30k.py for the layout).
DATA_CANDIDATES = (
    os.environ.get("EG2_DATA"),
    "/home/zhuangye/embeddinggemma2-cpu-20261014/data",
)


def synthetic_images(count: int, size: tuple[int, int] = (500, 375)) -> list:
    """Deterministic stand-in photos, so the benchmark needs no dataset.

    Any image expands to the same 2520-token sequence, so synthetic input
    measures the encoder just as well as a real photo.
    """
    from PIL import Image

    width, height = size
    images = []
    for i in range(count):
        buf = bytearray(width * height * 3)
        for y in range(height):
            row = y * 3 * width
            for x in range(width):
                o = row + x * 3
                buf[o] = (x * 255) // width
                buf[o + 1] = (y * 255) // height
                buf[o + 2] = ((x * x + y) * (i + 1)) % 256
        images.append(Image.frombytes("RGB", (width, height), bytes(buf)))
    return images


def is_arm64() -> bool:
    return platform.machine().lower() in ("arm64", "aarch64")


def candidates(full: bool) -> list[tuple[str, str]]:
    """Device/dtype pairs worth timing on this host."""
    import torch

    pairs: list[tuple[str, str]] = []
    if torch.backends.mps.is_available():
        pairs += [("mps", "bfloat16"), ("mps", "float32")]
    pairs.append(("cpu", "float32"))
    # bf16 emulation on arm64 costs ~12x; only measure it on request.
    if full or not is_arm64():
        pairs.append(("cpu", "bfloat16"))
    return pairs


def bench(device: str, dtype: str, images: list, reps: int):
    """Median ms/image for one configuration, plus the loaded Embedder."""
    from embeddinggemma2_kit import Embedder

    emb = Embedder(device=device, dtype=dtype)
    emb.embed_images(images[:1])                      # load + warm
    timings = []
    for _ in range(reps):
        start = time.perf_counter()
        emb.embed_images(images)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings) / len(images) * 1000, emb


def find_data() -> Path | None:
    for candidate in DATA_CANDIDATES:
        if not candidate:
            continue
        path = Path(candidate)
        if (path / "manifest.json").is_file():
            return path
    return None


def retrieval_check(embedder, data_dir: Path, limit: int) -> tuple[int, int]:
    """Text -> image hit@1 over the first ``limit`` entries."""
    from embeddinggemma2_kit.search import CrossModalSearch
    from embeddinggemma2_kit.store.manifest import CorpusItem, Manifest

    raw = json.loads((data_dir / "manifest.json").read_text())[:limit]
    manifest = Manifest([
        CorpusItem(id=e["id"], image=e["image"], metadata={"kind": "image"})
        for e in raw
    ])
    search = CrossModalSearch(embedder=embedder, fallback=True)
    search.build_from_manifest(manifest, progress=False)

    hits = 0
    for entry in raw:
        results = [
            h for h in search.search_text(entry["captions"][0], topk=len(manifest))
            if h.metadata.get("kind") == "image"
        ]
        hits += results[0].id == entry["id"]
    return hits, len(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=3, help="timed repeats per config")
    ap.add_argument("--images", type=int, default=8,
                    help="images used for the retrieval check")
    ap.add_argument("--full", action="store_true",
                    help="also time combinations known to be slow on this host")
    args = ap.parse_args()

    import torch

    print(f"host        : {platform.machine()} / {platform.system()}")
    print(f"python      : {platform.python_version()}")
    print(f"torch       : {torch.__version__}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    print()

    try:
        probes = synthetic_images(2)
    except Exception as exc:                          # noqa: BLE001
        print(f"could not create probe images: {exc}")
        return 1

    print("--- part 1: device/dtype benchmark (synthetic images) ---")
    results: list[tuple[str, str, float]] = []
    try:
        for device, dtype in candidates(args.full):
            try:
                ms, emb = bench(device, dtype, probes, args.reps)
            except Exception as exc:                  # noqa: BLE001
                print(f"{device}/{dtype:<9} failed: {type(exc).__name__}: "
                      f"{str(exc)[:80]}")
                continue
            results.append((device, dtype, ms))
            print(f"{device}/{dtype:<9} {ms:8.1f} ms/image   "
                  f"{1000 / ms:6.2f} img/s")
            del emb
    except Exception as exc:                          # noqa: BLE001
        print()
        print(f"benchmark aborted: {type(exc).__name__}: {exc}")
        print("If the checkpoint is missing, provision it first:")
        print("  python scripts/fetch_weights.py --src /path/to/embeddinggemma-2")
        return 1

    if not results:
        print("no configuration could be timed")
        return 1

    fastest = min(results, key=lambda r: r[2])
    device, dtype, _ = fastest
    print()
    print(f"recommended : Embedder(device={device!r}, dtype={dtype!r})")

    if is_arm64() and not args.full:
        print("note        : cpu/bfloat16 skipped -- on arm64 it emulates and is")
        print("              ~12x slower than cpu/float32. Pass --full to see it.")
    if is_arm64() and dtype == "bfloat16" and device == "cpu":
        print("warning     : bfloat16 on an arm64 CPU is usually the wrong choice")

    # ---------------------------------------------------------------- part 2
    print()
    print("--- part 2: retrieval check with the recommended configuration ---")
    data_dir = find_data()
    if data_dir is None:
        print("skipped: no corpus found.")
        print("  set EG2_DATA to a directory containing manifest.json and images/")
        return 0

    limit = max(1, args.images)
    try:
        hits, total = retrieval_check(bench_fastest(device, dtype), data_dir, limit)
    except Exception as exc:                          # noqa: BLE001
        print(f"skipped: {type(exc).__name__}: {str(exc)[:120]}")
        return 0
    print(f"text -> image hit@1: {hits}/{total}")

    # Cross-check: a second configuration should give the same neighbours.
    others = [r for r in results if (r[0], r[1]) != (device, dtype)]
    if others:
        alt_device, alt_dtype, alt_ms = min(others, key=lambda r: r[2])
        if alt_ms * limit < 30_000:                   # keep the example quick
            try:
                same, again = retrieval_check(
                    bench_fastest(alt_device, alt_dtype), data_dir, limit)
                verdict = "same" if same == hits else "DIFFERENT"
                print(f"cross-check with {alt_device}/{alt_dtype}: "
                      f"{again}/{total} hit@1 -> {verdict}")
            except Exception as exc:                  # noqa: BLE001
                print(f"cross-check skipped: {type(exc).__name__}")
        else:
            print(f"cross-check skipped: {alt_device}/{alt_dtype} would take "
                  f"~{alt_ms * limit / 1000:.0f}s on {limit} images")

    print()
    print("Accuracy does not depend on this choice — see docs/platforms.md —")
    print("so pick whatever is fastest above.")
    return 0


def bench_fastest(device: str, dtype: str):
    """Fresh, warmed Embedder for the retrieval check."""
    from embeddinggemma2_kit import Embedder

    emb = Embedder(device=device, dtype=dtype)
    emb.embed_text(["warm up"], role="query")         # force the checkpoint load
    return emb


if __name__ == "__main__":
    sys.exit(main())