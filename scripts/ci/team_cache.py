#!/usr/bin/env python3
"""Validate immutable, deployment-authorized ccache snapshots; restore private copies.

This utility does not authenticate publishers. Deployment must authorize publication,
protect the manifest and snapshot together, and prevent concurrent mutation. A failed
restore is an explicit cache miss: the caller may start a cold build in a fresh directory.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time

MAX_HISTORY_BYTES = 4 * 1024 * 1024

NAMESPACE_FIELDS = {"image", "compiler", "arch", "sanitizer", "build_options", "path_scheme"}


def namespace_id(namespace):
    if not isinstance(namespace, dict) or set(namespace) != NAMESPACE_FIELDS:
        raise ValueError("Namespace must contain exactly: " + ", ".join(sorted(NAMESPACE_FIELDS)))
    if any(not isinstance(value, str) or not value.strip() for value in namespace.values()):
        raise ValueError("Namespace values must be nonempty strings")
    if namespace["sanitizer"] != "asan":
        raise ValueError("Only the asan cache namespace is supported")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", namespace["image"]):
        raise ValueError("Namespace image must be pinned by sha256 digest")
    encoded = json.dumps(namespace, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def checked_path(value):
    if ".." in Path(value).parts:
        raise ValueError("Parent traversal is forbidden")
    path = Path(os.path.abspath(value))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Symlink path is forbidden: {part}")
    return path


def overlap(left, right):
    return left == right or left in right.parents or right in left.parents


def digest(path):
    checksum = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"Not a regular file: {path}")
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            checksum.update(chunk)
    return size, checksum.hexdigest()


def inventory(source):
    if not source.is_dir():
        raise ValueError(f"Snapshot directory missing: {source}")
    files = []
    def walk_error(error):
        raise error

    for root, dirs, names in os.walk(source, followlinks=False, onerror=walk_error):
        for name in dirs + names:
            path = Path(root) / name
            mode = path.lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError(f"Only regular files and directories allowed: {path}")
        for name in names:
            path = Path(root) / name
            size, checksum = digest(path)
            files.append({"path": path.relative_to(source).as_posix(),
                          "size": size, "sha256": checksum})
    if not files:
        raise ValueError("Empty snapshot")
    return sorted(files, key=lambda entry: entry["path"])


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def validate_manifest(manifest, namespace):
    expected = {"schema", "namespace", "namespace_id", "publisher_main_sha", "files"}
    if not isinstance(manifest, dict) or set(manifest) not in (expected, expected | {"auxiliary"}) or type(manifest["schema"]) is not int or manifest["schema"] != 1:
        raise ValueError("Invalid manifest schema")
    if manifest["namespace"] != namespace or manifest["namespace_id"] != namespace_id(namespace):
        raise ValueError("Cache namespace mismatch")
    publisher = manifest["publisher_main_sha"]
    if publisher is not None and not re.fullmatch(r"[0-9a-f]{40}", str(publisher)):
        raise ValueError("Invalid publisher main SHA")
    if "auxiliary" in manifest:
        auxiliary = manifest["auxiliary"]
        if not isinstance(auxiliary, dict) or set(auxiliary) != {"test_history"}:
            raise ValueError("Invalid auxiliary manifest")
        entry = auxiliary["test_history"]
        if not isinstance(entry, dict) or set(entry) != {"size", "sha256"}:
            raise ValueError("Invalid history manifest entry")
        if type(entry["size"]) is not int or not 0 < entry["size"] <= MAX_HISTORY_BYTES:
            raise ValueError("Invalid history size")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("Invalid history digest")
    entries = manifest["files"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Missing manifest file inventory")
    paths = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise ValueError("Invalid manifest file entry")
        name = entry["path"]
        if not isinstance(name, str) or not name or "\\" in name:
            raise ValueError("Invalid manifest path")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name or name == ".":
            raise ValueError("Noncanonical or escaping manifest path")
        if name in paths:
            raise ValueError("Duplicate manifest path")
        paths.add(name)
        if type(entry["size"]) is not int or entry["size"] < 0:
            raise ValueError("Invalid manifest size")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("Invalid manifest digest")
    return sorted(entries, key=lambda entry: entry["path"])


def read_history(history_path):
    path = checked_path(history_path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 0 < info.st_size <= MAX_HISTORY_BYTES:
            raise ValueError("Invalid history file")
        raw = stream.read(MAX_HISTORY_BYTES + 1)
    if len(raw) != info.st_size:
        raise ValueError("History changed or exceeds size limit")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate history key")
            value[key] = item
        return value

    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict) or set(value) != {"schema", "source_main_sha", "inventory_digest", "duration_history"} or type(value["schema"]) is not int or value["schema"] != 1:
        raise ValueError("Invalid history schema")
    for key, size in (("source_main_sha", 40), ("inventory_digest", 64)):
        if not isinstance(value[key], str) or not re.fullmatch(r"[0-9a-f]{" + str(size) + r"}", value[key]):
            raise ValueError("Invalid history provenance")
    durations = value["duration_history"]
    if not isinstance(durations, dict) or not 0 < len(durations) <= 100000:
        raise ValueError("Invalid duration history")
    for identity, seconds in durations.items():
        if not re.fullmatch(r"[0-9a-f]{64}", identity) or type(seconds) not in (int, float) or seconds <= 0 or seconds > 1e12 or not math.isfinite(seconds):
            raise ValueError("Invalid duration hint")
    return raw, value


def verify_history(history_path, manifest_path, namespace):
    # A hint never authenticates a test result or changes the inventory. The
    # mounted manifest/history are jointly protected by deployment authorization.
    manifest = read_json(checked_path(manifest_path))
    validate_manifest(manifest, namespace)
    entry = manifest.get("auxiliary", {}).get("test_history")
    if entry is None:
        raise ValueError("No published test history")
    raw, value = read_history(history_path)
    if len(raw) != entry["size"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("History size or digest mismatch")
    if value["source_main_sha"] != manifest["publisher_main_sha"]:
        raise ValueError("History publisher mismatch")
    return {"cases": len(value["duration_history"]), "sha256": entry["sha256"],
            "inventory_digest": value["inventory_digest"]}


def generate(source, manifest_path, namespace, publisher_main_sha=None, history_path=None):
    source, manifest_path = checked_path(source), checked_path(manifest_path)
    if overlap(source, manifest_path):
        raise ValueError("Manifest must be outside the immutable source tree")
    if manifest_path.exists():
        raise ValueError("Refusing to replace an existing manifest")
    manifest = {"schema": 1, "namespace": namespace, "namespace_id": namespace_id(namespace),
                "publisher_main_sha": publisher_main_sha, "files": inventory(source)}
    if history_path is not None:
        history_path = checked_path(history_path)
        if overlap(source, history_path) or history_path == manifest_path:
            raise ValueError("History must be outside the cache entry tree")
        raw, history = read_history(history_path)
        if history["source_main_sha"] != publisher_main_sha:
            raise ValueError("History publisher mismatch")
        manifest["auxiliary"] = {"test_history": {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
    validate_manifest(manifest, namespace)
    # Exclusive creation also prevents replacing another publisher's manifest.
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return {"namespace_id": manifest["namespace_id"], "files": len(manifest["files"])}


def verify(source, manifest_path, namespace):
    source, manifest_path = checked_path(source), checked_path(manifest_path)
    if overlap(source, manifest_path):
        raise ValueError("Manifest must be outside the immutable source tree")
    entries = validate_manifest(read_json(manifest_path), namespace)
    if inventory(source) != entries:
        raise ValueError("Snapshot inventory or digest mismatch")
    return entries


def restore(source, manifest_path, destination, namespace):
    started = time.monotonic()
    source, manifest_path, destination = map(checked_path, (source, manifest_path, destination))
    if overlap(source, destination) or overlap(manifest_path, destination):
        raise ValueError("Source, manifest and destination must not overlap")
    if destination.exists():
        raise ValueError("Destination must not exist; use a new private directory")
    entries = verify(source, manifest_path, namespace)
    verified = time.monotonic()
    staging = Path(tempfile.mkdtemp(prefix=".team-cache-", dir=destination.parent))
    try:
        for entry in entries:
            target = staging / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            with os.fdopen(os.open(source / entry["path"], flags), "rb") as src:
                if not stat.S_ISREG(os.fstat(src.fileno()).st_mode):
                    raise ValueError("Source changed during restore")
                with target.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
            if digest(target) != (entry["size"], entry["sha256"]):
                raise ValueError("Source changed or copy corrupted during restore")
        # Recheck source and manifest before making the private copy visible.
        if verify(source, manifest_path, namespace) != entries:
            raise ValueError("Manifest changed during restore")
        if checked_path(destination).exists():
            raise ValueError("Destination appeared during restore")
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    finished = time.monotonic()
    return {"namespace_id": namespace_id(namespace), "files": len(entries),
            "bytes": sum(entry["size"] for entry in entries),
            "verify_seconds": verified - started, "copy_seconds": finished - verified,
            "total_seconds": finished - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("manifest", "verify", "restore", "verify-history"))
    parser.add_argument("--source")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--namespace-file", required=True,
                        help="JSON with image/compiler/arch/sanitizer/build_options/path_scheme")
    parser.add_argument("--publisher-main-sha")
    parser.add_argument("--destination")
    parser.add_argument("--history", help="Published auxiliary history, outside ccache entries")
    args = parser.parse_args()
    try:
        namespace = read_json(checked_path(args.namespace_file))
        if args.command != "verify-history" and args.source is None:
            raise ValueError("This command requires --source")
        if args.command == "verify-history":
            if args.history is None:
                raise ValueError("verify-history requires --history")
            result = verify_history(args.history, args.manifest, namespace)
        elif args.command == "manifest":
            result = generate(args.source, args.manifest, namespace, args.publisher_main_sha, args.history)
        elif args.command == "verify":
            entries = verify(args.source, args.manifest, namespace)
            result = {"namespace_id": namespace_id(namespace), "files": len(entries)}
        else:
            if args.destination is None:
                raise ValueError("restore requires --destination")
            result = restore(args.source, args.manifest, args.destination, namespace)
        print(json.dumps({"ok": True, **result}, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
