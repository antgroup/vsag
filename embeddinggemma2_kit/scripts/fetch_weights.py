#!/usr/bin/env python3
"""Link a locally provisioned EmbeddingGemma 2 checkpoint into ``weights/``.

Why a symlink instead of a copy
-------------------------------
The checkpoint is ~2.9 GB. This host runs close to its disk quota, and a copy
would create a second full replica of weights that are already verified and
present on disk. A symlink costs nothing, keeps a single source of truth, and
makes it obvious that the weights live outside this repository.

Why this script does not download anything
------------------------------------------
This kit targets local, offline development and inference. Fetching weights is
the deploying party's responsibility:

* It keeps internal artifact hosts, mirrors, and credentials out of a public
  repository.
* The weights are governed by the Gemma terms of use, not by this repo's
  license, so redistributing or auto-downloading them here would be wrong.
* An explicit local source path makes runs reproducible and auditable.

Usage
-----
    python scripts/fetch_weights.py
    python scripts/fetch_weights.py --src /path/to/embeddinggemma-2
    python scripts/fetch_weights.py --dest weights/embeddinggemma-2 --force

Behaviour
---------
* If the destination is already a symlink to an existing directory, this is a
  no-op (idempotent) unless ``--force`` is given.
* If the destination exists as a real directory or a regular file, the script
  REFUSES to touch it and exits non-zero. It never deletes user data.
* After linking, it verifies the checkpoint looks complete (see REQUIRED_FILES)
  and prints a short summary.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# A checkpoint verified by the earlier CPU evaluation experiments on this host.
# Override with --src when provisioning elsewhere.
DEFAULT_SOURCE = "/home/zhuangye/embeddinggemma2-cpu-20261014/models/embeddinggemma-2"

# Files that must be present for the model to load in every modality.
# text/image need config.json + model.safetensors + tokenizer*;
# audio/video additionally need processor_config.json and preprocessor_config.json.
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "config_sentence_transformers.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Symlink a local EmbeddingGemma 2 checkpoint into weights/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(os.environ.get("EMBEDDINGGEMMA2_SRC", DEFAULT_SOURCE)),
        help=f"checkpoint directory (default: {DEFAULT_SOURCE}, or $EMBEDDINGGEMMA2_SRC)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="symlink to create (default: <kit root>/weights/embeddinggemma-2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing symlink (still refuses to touch real directories)",
    )
    return parser.parse_args(argv)


def verify_checkpoint(src: Path) -> list[str]:
    """Return the list of expected files that are missing from ``src``."""
    return [name for name in REQUIRED_FILES if not (src / name).exists()]


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kit_root = Path(__file__).resolve().parents[1]
    dest: Path = args.dest or (kit_root / "weights" / "embeddinggemma-2")
    src: Path = args.src.resolve()

    if not src.is_dir():
        return fail(f"source checkpoint directory not found: {src}")

    missing = verify_checkpoint(src)
    if missing:
        return fail(
            f"source checkpoint at {src} is incomplete; missing: {', '.join(missing)}"
        )

    if dest.is_symlink():
        current = os.readlink(dest)
        if Path(current) == src and not args.force:
            print(f"ok: {dest} already -> {src} (nothing to do)")
            return 0
        if not args.force:
            return fail(
                f"{dest} already points to {current}; pass --force to repoint it to {src}"
            )
        # Only a symlink is being removed here, never real data.
        dest.unlink()
    elif dest.exists():
        # Refuse to clobber a real directory or file: that could be a user's copy.
        kind = "directory" if dest.is_dir() else "file"
        return fail(
            f"{dest} exists as a real {kind}; refusing to replace it. "
            "Move or remove it manually if you really want a symlink here."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(src, target_is_directory=True)

    size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
    print(f"ok: {dest} -> {src}")
    print(f"    files={len(REQUIRED_FILES)} verified, total size={size / 1e9:.2f} GB")
    print("    weights are git-ignored and will not be committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())