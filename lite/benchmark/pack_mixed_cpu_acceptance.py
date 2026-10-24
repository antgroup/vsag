#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Preserve large raw CSV cohorts compactly, verify unpacking, never delete originals."""
import argparse
import json
from pathlib import Path
import tarfile
import tempfile

from run_mixed_cpu_acceptance import sha, write_json
from verify_mixed_cpu_acceptance import verify


def unpack(archive, destination):
    root = destination.resolve()
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream.getmembers():
            path = (root / member.name).resolve()
            if not member.isfile() or member.name.startswith('/') or root not in path.parents:
                raise ValueError('Unsafe archive member: ' + member.name)
            stream.extract(member, root)


def pack(output):
    audit = verify(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    archive = output / 'raw.tar.gz'
    receipt = output / 'archive.json'
    assert not archive.exists() and not receipt.exists(), 'Archive must not be overwritten'
    with tarfile.open(archive, 'w:gz') as stream:
        for name in sorted(list(manifest) + ['manifest.json']):
            path = (output / name).resolve()
            assert output.resolve() in path.parents and path.is_file()
            stream.add(path, arcname=name, recursive=False)
    with tempfile.TemporaryDirectory(prefix='vsag-unpack-audit-', dir='/dev/shm') as directory:
        destination = Path(directory)
        unpack(archive, destination)
        assert verify(destination) == audit
    write_json(receipt, dict(sha256=sha(archive), bytes=archive.stat().st_size,
        members=len(manifest) + 1, unpacked_audit=audit,
        originals_preserved=True, format='gzip tar of measured manifest files plus manifest'))
    return json.loads(receipt.read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(pack(args.output), indent=2))
