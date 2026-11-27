#!/usr/bin/env python3
"""Provision the EmbeddingGemma 2 checkpoint into ``weights/``.

The weights are not distributed with this repository (see NOTICE.md). This
script obtains them from one of three places and symlinks the result into
``weights/embeddinggemma-2`` so the package finds it without configuration.

Choosing the source
-------------------
One environment variable selects the source; the default is the public upstream
repository.

    EG2_WEIGHTS_SOURCE=huggingface          # default: google/embeddinggemma-2
    EG2_WEIGHTS_SOURCE=local                # symlink a checkpoint you already have
    EG2_WEIGHTS_SOURCE=https://HOST/PREFIX  # any static file host (internal OSS, S3)

`local` still needs a path, either ``--src`` or ``$EMBEDDINGGEMMA2_SRC``.
Everything else can be overridden per-run with ``--source`` / ``--src``.

For an internal OSS bucket the intended workflow is:

    # once, on a machine that has the checkpoint
    python scripts/fetch_weights.py --src /path/to/checkpoint --write-manifest
    ossutil cp -r /path/to/checkpoint oss://bucket/models/embeddinggemma-2/v1.0/

    # then everywhere else
    EG2_WEIGHTS_SOURCE=https://bucket.oss-cn-hangzhou.aliyuncs.com/models/embeddinggemma-2/v1.0 \
        python scripts/fetch_weights.py

The manifest (``weights.sha256``) doubles as the file list and the integrity
check, so a URL source is verified by default. Without a manifest the script
downloads the known file set and says so.

Private sources
---------------
Nothing is stored here. Hugging Face honours its own ``HF_TOKEN``; for a
private bucket pass a pre-signed URL, or download with your own tooling and use
``EG2_WEIGHTS_SOURCE=local``.

Safety
------
* ``weights/`` is git-ignored; nothing is ever committed.
* An existing symlink is only repointed with ``--force``.
* A real directory or file at the destination is never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

SOURCE_ENV = "EG2_WEIGHTS_SOURCE"
LOCAL_ENV = "EMBEDDINGGEMMA2_SRC"
DEFAULT_SOURCE_KIND = "huggingface"
DEFAULT_REPO_ID = "google/embeddinggemma-2"

# A checkpoint verified by the earlier CPU evaluation experiments on this host.
# Only used when the source kind is "local" and no path is given.
DEFAULT_LOCAL_SOURCE = "/home/zhuangye/embeddinggemma2-cpu-20261014/models/embeddinggemma-2"

MANIFEST_NAME = "weights.sha256"

# Files the model needs to load and to support every modality. All three groups
# below were determined by removing each file from a working checkpoint and
# observing the failure:
#   * without sentence_bert_config.json the model silently degrades to
#     text-only ("Modality 'image' is not supported");
#   * without chat_template.jinja the text processor fails
#     ("does not have a chat template");
#   * without 1_Pooling/config.json loading fails outright
#     ("Pooling.__init__() missing ... 'embedding_dimension'").
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "chat_template.jinja",
    "1_Pooling/config.json",
)

# Nice to have, or used only on some code paths. Missing ones are skipped.
OPTIONAL_FILES = (
    "2_Normalize/config.json",   # Normalize falls back to its defaults
    "tokenizer.model",           # only needed by SentencePiece-only tooling
    "README.md",                 # upstream model card
)


# --------------------------------------------------------------------- helpers

def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source(args: argparse.Namespace) -> tuple[str, str]:
    """Work out (kind, value) from CLI flags and ``EG2_WEIGHTS_SOURCE``.

    Precedence: ``--src`` beats ``--source``, which beats the environment.
    """
    if args.src is not None:
        return "local", str(args.src)

    raw = args.source or os.environ.get(SOURCE_ENV) or DEFAULT_SOURCE_KIND
    raw = raw.strip()
    lowered = raw.lower()

    if lowered in ("huggingface", "hf"):
        return "huggingface", args.repo or DEFAULT_REPO_ID
    if lowered in ("local", "symlink", "path"):
        return "local", os.environ.get(LOCAL_ENV) or DEFAULT_LOCAL_SOURCE
    if lowered.startswith(("http://", "https://", "file://")):
        return "url", raw if raw.endswith("/") else raw + "/"

    raise ValueError(
        f"{SOURCE_ENV}={raw!r} is not understood. Use 'huggingface', 'local', "
        "or an http(s) base URL."
    )


# ------------------------------------------------------------------- manifests

def write_manifest(directory: Path) -> Path:
    """Write ``weights.sha256`` (sha256sum format) for a checkpoint directory."""
    lines: list[str] = []
    for name in (*REQUIRED_FILES, *OPTIONAL_FILES):
        path = directory / name
        if path.is_file():
            lines.append(f"{sha256_of(path)}  {name}")
    manifest = directory / MANIFEST_NAME
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def read_manifest(path: Path) -> dict[str, str]:
    """Parse a ``sha256sum``-style manifest into ``{relative path: hash}``."""
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        if not digest or not name:
            raise ValueError(f"malformed manifest line: {line!r}")
        entries[name.strip()] = digest.strip().lower()
    return entries


def verify_checkpoint(directory: Path) -> list[str]:
    """Return the required files missing from ``directory``."""
    return [name for name in REQUIRED_FILES if not (directory / name).is_file()]


def verify_manifest(directory: Path, entries: dict[str, str]) -> list[str]:
    """Return human-readable problems with ``directory`` against ``entries``."""
    problems: list[str] = []
    for name, expected in entries.items():
        path = directory / name
        if not path.is_file():
            problems.append(f"missing: {name}")
            continue
        actual = sha256_of(path)
        if actual != expected:
            problems.append(f"checksum mismatch: {name} (expected {expected[:12]}…, got {actual[:12]}…)")
    return problems


# ------------------------------------------------------------------ provisioning

def provision_huggingface(repo_id: str, revision: str | None, cache: Path) -> Path:
    """Download the checkpoint from the Hugging Face hub and return its path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:                       # pragma: no cover - env issue
        raise RuntimeError(
            "huggingface_hub is required for the default source. It normally "
            "arrives with transformers; install it with "
            "'pip install huggingface-hub', or set "
            f"{SOURCE_ENV}=local to symlink an existing checkpoint."
        ) from exc

    cache.mkdir(parents=True, exist_ok=True)
    # allow_patterns mirrors the files the kit actually reads, so the download
    # does not drag in every alternative weight format the repo carries.
    patterns = [*REQUIRED_FILES, *OPTIONAL_FILES]
    print(f"downloading {repo_id}"
          + (f"@{revision}" if revision else "")
          + f" -> {cache}", flush=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(cache),
            allow_patterns=patterns,
        )
    except Exception as exc:                         # noqa: BLE001 - re-raised
        raise RuntimeError(
            f"could not download {repo_id}: {type(exc).__name__}: {exc}\n"
            "hint: on a host without access to the Hugging Face hub, point "
            f"{SOURCE_ENV} at an internal mirror instead, e.g.\n"
            f"  {SOURCE_ENV}=https://your-bucket/models/embeddinggemma-2/v1 "
            "scripts/fetch_weights.py\n"
            "or provision the checkpoint yourself and use "
            f"{SOURCE_ENV}=local."
        ) from exc
    return cache


def _download_one(url: str, target: Path, *, timeout: float = 60.0) -> None:
    """Fetch one file, resuming a partial download when the server allows it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    have = partial.stat().st_size if partial.exists() else 0

    request = urllib.request.Request(url)
    if have:
        request.add_header("Range", f"bytes={have}-")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            mode = "ab" if have and status == 206 else "wb"
            if mode == "wb":
                have = 0
            with partial.open(mode) as handle:
                shutil.copyfileobj(response, handle, length=1 << 20)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GET {url} failed: HTTP {exc.code}") from exc
    partial.replace(target)


def provision_url(base: str, cache: Path, *, project_root: Path | None = None) -> Path:
    """Download a checkpoint from a static file host rooted at ``base``."""
    del project_root  # kept for symmetry with the other provisioners
    cache.mkdir(parents=True, exist_ok=True)

    manifest_url = base + MANIFEST_NAME
    entries: dict[str, str] = {}
    try:
        local_manifest = cache / MANIFEST_NAME
        _download_one(manifest_url, local_manifest)
        entries = read_manifest(local_manifest)
        print(f"manifest: {manifest_url} ({len(entries)} files)")
    except RuntimeError:
        print(f"note: no {MANIFEST_NAME} at {base}; downloading the known file "
              "set without checksum verification")
        entries = {name: "" for name in (*REQUIRED_FILES, *OPTIONAL_FILES)}

    for name in entries:
        target = cache / name
        if target.is_file() and entries[name] and sha256_of(target) == entries[name]:
            print(f"  cached  {name}")
            continue
        try:
            _download_one(base + name, target)
        except RuntimeError as exc:
            if name in OPTIONAL_FILES:
                print(f"  skipped {name} (optional)")
                continue
            raise RuntimeError(f"required file unavailable: {name}: {exc}") from exc
        print(f"  fetched {name}")

    return cache


def provision_local(source: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError(f"source checkpoint directory not found: {source}")
    return source


# -------------------------------------------------------------------- linking

def link_destination(dest: Path, src: Path, *, force: bool) -> int:
    """Point ``dest`` at ``src`` without ever deleting real user data."""
    if dest.is_symlink():
        current = os.readlink(dest)
        if Path(current) == src and not force:
            print(f"ok: {dest} already -> {src} (nothing to do)")
            return 0
        if not force:
            return fail(
                f"{dest} already points to {current}; pass --force to repoint it to {src}"
            )
        dest.unlink()                                # only the link, never the data
    elif dest.exists():
        kind = "directory" if dest.is_dir() else "file"
        return fail(
            f"{dest} exists as a real {kind}; refusing to replace it. "
            "Move or remove it manually if you really want a symlink here."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(src, target_is_directory=True)
    return 0


# ------------------------------------------------------------------------ cli

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Provision the EmbeddingGemma 2 checkpoint into weights/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment:\n"
            f"  {SOURCE_ENV}   huggingface (default) | local | http(s) base URL\n"
            f"  {LOCAL_ENV}   checkpoint path used by the 'local' source\n"
            "  HF_TOKEN          honoured by huggingface_hub for private repos\n"
            "  HF_ENDPOINT       hub mirror, e.g. https://hf-mirror.com\n"
        ),
    )
    parser.add_argument("--source", default=None,
                        help=f"override {SOURCE_ENV}")
    parser.add_argument("--src", type=Path, default=None,
                        help="local checkpoint directory (implies --source local)")
    parser.add_argument("--repo", default=None,
                        help=f"Hugging Face repo id (default {DEFAULT_REPO_ID})")
    parser.add_argument("--revision", default=None, help="Hugging Face revision/tag")
    parser.add_argument("--dest", type=Path, default=None,
                        help="symlink to create (default <kit>/weights/embeddinggemma-2)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="where downloads land (default <kit>/weights/.cache)")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing symlink (never a real directory)")
    parser.add_argument("--write-manifest", action="store_true",
                        help="write weights.sha256 next to the source and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kit_root = Path(__file__).resolve().parents[1]
    dest: Path = args.dest or (kit_root / "weights" / "embeddinggemma-2")
    cache: Path = args.cache_dir or (kit_root / "weights" / ".cache" / "embeddinggemma-2")

    # `--write-manifest` is a maintenance helper for people who will host the
    # checkpoint themselves; it never downloads.
    if args.write_manifest:
        source = args.src or Path(args.source or os.environ.get(LOCAL_ENV)
                                  or DEFAULT_LOCAL_SOURCE)
        source = source.expanduser().resolve()
        if not source.is_dir():
            return fail(f"source checkpoint directory not found: {source}")
        missing = verify_checkpoint(source)
        if missing:
            print(f"warning: source is missing required files: {', '.join(missing)}",
                  file=sys.stderr)
        manifest = write_manifest(source)
        print(f"ok: wrote {manifest}")
        print("    upload the directory, manifest included, then point "
              f"{SOURCE_ENV} at its URL.")
        return 0

    try:
        kind, value = resolve_source(args)
    except ValueError as exc:
        return fail(str(exc))

    try:
        if kind == "local":
            source = provision_local(Path(value))
        elif kind == "huggingface":
            source = provision_huggingface(value, args.revision, cache)
        else:
            source = provision_url(value, cache)
    except (RuntimeError, OSError) as exc:
        return fail(str(exc))

    missing = verify_checkpoint(source)
    if missing:
        return fail(
            f"checkpoint at {source} is incomplete; missing: {', '.join(missing)}"
        )

    manifest = source / MANIFEST_NAME
    if manifest.is_file():
        problems = verify_manifest(source, read_manifest(manifest))
        if problems:
            return fail("checkpoint failed verification:\n  " + "\n  ".join(problems))
        print(f"verified {len(read_manifest(manifest))} files against {MANIFEST_NAME}")

    code = link_destination(dest, source, force=args.force)
    if code:
        return code

    known = [source / name for name in (*REQUIRED_FILES, *OPTIONAL_FILES)]
    used = sum(f.stat().st_size for f in known if f.is_file())
    everything = sum(f.stat().st_size for f in source.rglob("*") if f.is_file())
    print(f"ok: {dest} -> {source}")
    print(f"    source={kind}, {len(REQUIRED_FILES)} required files verified, "
          f"checkpoint size={used / 1e9:.2f} GB")
    if everything > used * 1.05:
        # A chunked download can leave assembled parts behind and double the
        # apparent size; they are ignored by the kit but worth flagging.
        print(f"    note: {source} also holds {(everything - used) / 1e9:.2f} GB "
              "of unlisted files (ignored, safe to delete)")
    print("    weights are git-ignored and will not be committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())