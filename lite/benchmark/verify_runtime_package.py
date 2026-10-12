#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Recompute runtime-bundle accounting and validate mapped-library smoke receipts."""
import argparse
import json
from pathlib import Path

from run_mixed_cpu_acceptance import sha


def verify(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, expected in manifest.items():
        assert sha(output / name) == expected, name
    report = json.loads((output / 'report.json').read_text())
    expected_libraries = {'lite':'3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330',
                          'full':'9aeaa550703f116bef2959769d4f2f03da254b07c906ba941d8186224cc70c72'}
    for side, package in report['packages'].items():
        contents = package['inventory']
        assert {'run','bin/probe','licenses/VSAG-Apache-2.0'} <= set(contents)
        assert sum(v.get('bytes',0) for v in contents.values()) == package['payload_bytes']
        assert 0 < package['tar_gzip_bytes'] < package['payload_bytes']
        for name, value in contents.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            if value['kind'] == 'symlink':
                assert not Path(value['target']).is_absolute() and '..' not in Path(value['target']).parts
                assert str(Path(name).parent / value['target']) in contents
            else:
                assert value['kind'] == 'file' and value['bytes'] > 0 and len(value['sha256']) == 64
        for source, record in package['files'].items():
            value = contents[record['packaged_path']]
            assert value['sha256'] == record['stripped_sha256'] and value['bytes'] == record['stripped_bytes']
            if 'vsag' in Path(source).name:
                assert record['source_sha256'] == expected_libraries[side]
        assert len(package['relocation_checks']) == 3
        for check in package['relocation_checks']:
            assert check['exit'] == 0 and int(check['metrics']['base_count']) == 10000
            assert float(check['metrics']['recall_at_k']) >= .95 and check['snapshot_bytes'] > 0
            root = Path(check['argv'][0]).parent
            assert len(check['mapped_elf_paths']) == len(package['files'])
            for name in check['mapped_elf_paths']:
                assert root in Path(name).parents
                assert str(Path(name).relative_to(root)) in contents
    for key, field in [('payload_reduction_percent','payload_bytes'),
                       ('compressed_reduction_percent','tar_gzip_bytes')]:
        computed = (1-report['packages']['lite'][field]/report['packages']['full'][field])*100
        assert abs(report[key]-computed) < 1e-10
    return dict(audit='PASS', payload_reduction_percent=report['payload_reduction_percent'],
                compressed_reduction_percent=report['compressed_reduction_percent'],
                relocated_smokes=6, profile=report['profile'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output),indent=2))
