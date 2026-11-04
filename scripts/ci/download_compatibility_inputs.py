#!/usr/bin/env python3
"""Download and validate the complete published compatibility input set."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

REPOSITORY = "antgroup/vsag"
TAG = "compatibility-indexes"


def api(path, paginated=False):
    command = ["gh", "api", f"repos/{REPOSITORY}/{path}"]
    if paginated:
        command += ["--paginate", "--slurp"]
    return json.loads(subprocess.check_output(command, text=True))


def inventory():
    release = api(f"releases/tags/{TAG}")
    release_id = release["id"]
    if type(release_id) is not int or release_id <= 0 or release.get("tag_name") != TAG:
        raise ValueError("Invalid compatibility release identity")
    assets = {}
    identifiers = set()
    for page in api(f"releases/{release_id}/assets?per_page=100", paginated=True):
        for asset in page:
            name = asset["name"]
            if not name.endswith((".index", ".bin", ".json")):
                continue
            if (not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in assets
                    or asset["id"] in identifiers or type(asset["size"]) is not int
                    or asset["size"] <= 0 or asset.get("state") != "uploaded"):
                raise ValueError("Invalid or duplicate compatibility asset")
            digest = asset.get("digest")
            if digest is not None and not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
                raise ValueError("Unsupported compatibility asset digest")
            assets[name] = {key: asset.get(key) for key in ("id", "size", "digest", "updated_at")}
            identifiers.add(asset["id"])
    if not assets or not any(name.endswith(".index") for name in assets):
        raise ValueError("Empty compatibility index inventory")
    return {"release_id": release_id, "assets": assets}


def verify(directory, expected):
    actual = {path.name for path in directory.iterdir()}
    if actual != set(expected["assets"]):
        raise ValueError("Downloaded compatibility inventory differs from release")
    for name, asset in expected["assets"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != asset["size"]:
            raise ValueError(f"Missing, incomplete or non-regular compatibility asset: {name}")
        if asset["digest"]:
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
            if "sha256:" + checksum.hexdigest() != asset["digest"].lower():
                raise ValueError(f"Compatibility digest mismatch: {name}")


def download(directory):
    directory.mkdir(parents=True, exist_ok=False)
    expected = inventory()
    subprocess.run(["gh", "release", "download", TAG, "--repo", REPOSITORY,
                    "--dir", str(directory), "--pattern", "*.index", "--pattern", "*.bin",
                    "--pattern", "*.json"], check=True)
    verify(directory, expected)
    if inventory() != expected:
        raise ValueError("Compatibility release changed during download")
    print(f"Verified {len(expected['assets'])} assets from release {expected['release_id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    download(args.directory)


if __name__ == "__main__":
    main()
