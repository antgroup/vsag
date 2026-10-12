#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Recompute all measurements from raw CSV and verify completeness and hashes."""
import argparse
import csv
import json
from pathlib import Path

from run_mixed_cpu_acceptance import sha, verify_run


def verify(output):
    identity = json.loads((output / 'identity.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest
    for name, expected in manifest.items():
        assert sha(output / name) == expected, name
    assert sha(output / 'source.cpp') == identity['benchmark_source_sha256']
    config = identity['configuration']
    rows = json.loads((output / 'rows.json').read_text())
    modes = ['fp32', 'fp16', 'rabitq8']
    expected_keys = {(dataset, mode, trial) for dataset in identity['inputs']
                     for mode in modes for trial in range(config['trials'])}
    actual_keys = {(row['dataset'], row['storage'], row['trial']) for row in rows}
    assert len(rows) == len(expected_keys) and actual_keys == expected_keys
    fingerprints = {}
    for row in rows:
        key = row['dataset'], row['storage'], row['trial']
        folder = output / f'{key[0]}-{key[1]}-{key[2]}'
        with (folder / 'quality.csv').open() as stream:
            query_rows = list(csv.DictReader(stream))
        queries = len(query_rows) // 2
        raw = verify_run(folder, config['rounds'], config['cycles'], queries)
        expected = {k: v for k, v in row.items()
                    if k not in ['dataset', 'storage', 'trial', 'snapshots']}
        assert raw == expected, key
        assert json.loads((folder / 'audit.json').read_text()) == row
        assert len(row['snapshots']) == config['rounds']
        assert all(value['bytes'] > 0 and len(value['sha256']) == 64
                   for value in row['snapshots'].values())
        fingerprint = (query_rows, row['snapshots'])
        pair = key[:2]
        if pair in fingerprints:
            assert fingerprint == fingerprints[pair], ('nondeterministic quality/snapshot', pair)
        fingerprints[pair] = fingerprint
    return dict(audit='PASS', processes=len(rows), datasets=len(identity['inputs']),
                mutation_calls=len(rows) * config['rounds'] * config['cycles'] * 3,
                interleaved_queries=len(rows) * config['rounds'] * config['cycles'] * 3,
                source_sha=identity['feature_sha'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2))
