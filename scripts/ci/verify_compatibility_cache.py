#!/usr/bin/env python3
"""Verify a complete compatibility release snapshot without writing to it.

Deployment must protect the publisher's manifest and assets as trusted read-only
inputs. Digests establish consistency with that manifest, not publisher identity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time


def canonical_path(value):
    if ".." in Path(value).parts:
        raise ValueError("Parent traversal is forbidden")
    path = Path(os.path.abspath(value))
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"Symlink path is forbidden: {component}")
    return path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def positive_integer(value):
    return type(value) is int and value > 0


def verify(directory, manifest):
    started = time.monotonic()
    directory, manifest = canonical_path(directory), canonical_path(manifest)
    if not directory.is_dir() or manifest.parent != directory or manifest.name != "manifest.json":
        raise ValueError("Manifest must be directory/manifest.json in an existing directory")
    if not manifest.is_file():
        raise ValueError("Missing manifest.json")
    with manifest.open() as stream:
        document = json.load(stream, object_pairs_hook=unique_object)
    fields = {"schema_version", "repository", "release_tag", "release_id", "assets"}
    if not isinstance(document, dict) or set(document) != fields:
        raise ValueError("Invalid manifest fields")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("Unsupported schema_version")
    if document["repository"] != "antgroup/vsag" or document["release_tag"] != "compatibility-indexes":
        raise ValueError("Wrong compatibility release identity")
    if not positive_integer(document["release_id"]):
        raise ValueError("Invalid release_id")
    assets = document["assets"]
    if not isinstance(assets, list) or not assets:
        raise ValueError("Asset inventory must be nonempty")
    names, identifiers = set(), set()
    for asset in assets:
        if not isinstance(asset, dict) or set(asset) != {"name", "id", "size", "sha256"}:
            raise ValueError("Invalid asset fields")
        name = asset["name"]
        if (not isinstance(name, str) or not name or name in (".", "..", "manifest.json")
                or "/" in name or "\\" in name or "\0" in name):
            raise ValueError("Asset names must be plain filenames, excluding manifest.json")
        if name in names or asset["id"] in identifiers:
            raise ValueError("Duplicate asset name or id")
        if not positive_integer(asset["id"]) or not positive_integer(asset["size"]):
            raise ValueError("Asset id and size must be positive integers")
        if not isinstance(asset["sha256"], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", asset["sha256"]):
            raise ValueError("Invalid asset sha256")
        names.add(name)
        identifiers.add(asset["id"])
    actual = set()
    for path in directory.iterdir():
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"Only regular files are allowed: {path.name}")
        actual.add(path.name)
    if actual != names | {"manifest.json"}:
        raise ValueError("Asset inventory mismatch: missing=" + repr(sorted(names - actual))
                         + ", extra=" + repr(sorted(actual - names - {"manifest.json"})))
    total_bytes = 0
    for asset in assets:
        checksum, size = hashlib.sha256(), 0
        path = directory / asset["name"]
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError(f"Not a regular asset: {asset['name']}")
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                checksum.update(chunk)
        if size != asset["size"] or checksum.hexdigest() != asset["sha256"].lower():
            raise ValueError(f"Asset size or digest mismatch: {asset['name']}")
        total_bytes += size
    return {"release_id": document["release_id"], "assets": len(assets), "bytes": total_bytes,
            "verify_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    try:
        result = verify(args.directory, args.manifest)
        print(json.dumps({"ok": True, **result}, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
