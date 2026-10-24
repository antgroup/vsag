"""Publish exact-commit source for consumers of one sealed Team CI run.

The controller authenticates the admitted run and seals its volume. This helper
checks publication consistency; a PR-supplied checksum is not authentication.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile

MAX_FILES = 100000
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
CONTEXT_KEYS = {"repository_id", "run_id", "run_attempt", "head_sha", "test_sha", "image"}
RESERVED_ROOTS = {"build", "ci-logs", ".ci-downloads"}


def checked_name(name):
    if not isinstance(name, str):
        raise ValueError("Unsafe source path")
    parts = PurePosixPath(name).parts
    if (not parts or name != "/".join(parts)
            or name.startswith("/") or "\\" in name
            or any(ord(char) < 32 or ord(char) == 127 for char in name)
            or any(part in (".", "..") or part.lower() == ".git" for part in parts)
            or parts[0] in RESERVED_ROOTS):
        raise ValueError("Unsafe or reserved source path")
    return name


def publish(repository, destination, manifest_path, context):
    repository = Path(repository).resolve(strict=True)
    destination, manifest_path = Path(destination), Path(manifest_path)
    if (set(context) != CONTEXT_KEYS or not all(isinstance(value, str) and value for value in context.values())
            or not re.fullmatch(r"[0-9a-f]{40}", context["test_sha"])):
        raise ValueError("Invalid source execution identity")
    if destination.exists() or destination.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
        raise ValueError("Refusing to replace a source publication")
    if destination.parent.resolve() != destination.parent or manifest_path.parent.resolve() != manifest_path.parent:
        raise ValueError("Source publication parent must be a canonical directory")
    if destination == manifest_path or destination in manifest_path.parents:
        raise ValueError("Manifest must be outside source directory")
    git = ["git", "-C", str(repository)]
    head = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    if head != context["test_sha"]:
        raise ValueError("Checkout does not match tested SHA")
    tree = subprocess.check_output(git + ["ls-tree", "-r", "-z", "--full-tree", head])
    entries = []
    for record in tree.split(b"\0"):
        if not record:
            continue
        header, raw_name = record.split(b"\t", 1)
        mode, kind, oid = header.decode("ascii").split()
        name = checked_name(raw_name.decode("utf-8"))
        if mode not in ("100644", "100755") or kind != "blob" or not re.fullmatch(r"[0-9a-f]{40}", oid):
            raise ValueError("Only regular tracked source files are supported")
        entries.append((name, mode, oid))
    if not entries or len(entries) > MAX_FILES or len({row[0] for row in entries}) != len(entries):
        raise ValueError("Invalid tracked source inventory")
    staging = Path(tempfile.mkdtemp(prefix=".team-source-", dir=destination.parent))
    files, total = {}, 0
    try:
        with subprocess.Popen(git + ["cat-file", "--batch"], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as reader:
            try:
                for name, mode, oid in entries:
                    reader.stdin.write((oid + "\n").encode("ascii"))
                    reader.stdin.flush()
                    fields = reader.stdout.readline(256).decode("ascii").split()
                    if len(fields) != 3 or fields[:2] != [oid, "blob"] or not fields[2].isdigit():
                        raise ValueError("Invalid Git blob response")
                    size = int(fields[2])
                    total += size
                    if size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                        raise ValueError("Source publication exceeds size bound")
                    path = staging / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    remaining = size
                    with path.open("xb") as output:
                        while remaining:
                            chunk = reader.stdout.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError("Incomplete Git blob response")
                            remaining -= len(chunk)
                            digest.update(chunk)
                            output.write(chunk)
                    if reader.stdout.read(1) != b"\n":
                        raise ValueError("Invalid Git blob terminator")
                    permissions = 0o755 if mode == "100755" else 0o644
                    path.chmod(permissions)
                    files[name] = {"mode": permissions, "size": size, "sha256": digest.hexdigest()}
                reader.stdin.close()
                if reader.wait(timeout=10) != 0:
                    raise ValueError("Git blob reader failed")
            finally:
                if reader.poll() is None:
                    reader.kill()
                    reader.wait()
        manifest = {"schema": 1, "context": context, "files": files}
        content = json.dumps(manifest, sort_keys=True) + "\n"
        if len(content.encode("utf-8")) > 16 * 1024 * 1024:
            raise ValueError("Source manifest exceeds size bound")
        # The workflow publishes complete.json only after source, build and plan
        # are ready. Interrupted exports can never be selected by consumers.
        staging.rename(destination)
        with manifest_path.open("x") as output:
            output.write(content)
        return {"files": len(files), "bytes": total, "test_sha": head}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    args = parser.parse_args()
    result = publish(args.repository, args.destination, args.manifest, json.loads(args.context.read_text()))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
