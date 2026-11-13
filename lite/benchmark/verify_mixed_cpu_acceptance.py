#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Recompute all measurements from raw CSV and verify completeness and hashes."""
import argparse
import csv
import json
from pathlib import Path

from run_mixed_cpu_acceptance import sha, verify_run, verify_checkpoints, verify_ceiling, mutation_factor


def verify(output):
    identity = json.loads((output / 'identity.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest
    for name, expected in manifest.items():
        assert sha(output / name) == expected, name
    assert sha(output / 'source.cpp') == identity['benchmark_source_sha256']
    # Historical archives predate build-source receipts; do not relabel them.
    if 'build_source_receipt_sha256' in identity:
        receipt_path = output / 'build-source-receipt.json'
        assert sha(receipt_path) == identity['build_source_receipt_sha256']
        receipt = json.loads(receipt_path.read_text())
        assert receipt['schema'] == 1 and receipt['build'] == identity['build']
        assert receipt['source']['feature_sha'] == identity['feature_sha']
        assert receipt['library_sha256'] == identity['library_sha256']
        assert receipt['metadata']['cache_sha256'] == sha(output / 'CMakeCache.txt')
        assert receipt['metadata']['compile_commands_sha256'] == sha(output / 'compile_commands.json')
        assert all(receipt['source']['source_sha256'][name] == digest
                   for name, digest in identity['feature_sources'].items())
    config = identity['configuration']
    mutation_mode = config.get('mutation_mode', 'all')
    factor = mutation_factor(mutation_mode)
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
        raw = verify_run(folder, config['rounds'], config['cycles'], queries, mutation_mode)
        every = config.get('checkpoint_every', 0)
        if every:
            assert config.get('resident', False)
            raw['checkpoints'] = verify_checkpoints(folder, config['rounds'], config['cycles'], queries, every, mutation_mode)
        if mutation_mode != 'all':
            assert every and config.get('calls_per_cycle') == factor
            assert config.get('final_diagnostic_ef') == 'N'
            raw['ceiling'] = verify_ceiling(folder, config['rounds'], queries)
        command = json.loads((output / (folder.name + '.command.json')).read_text())['argv']
        if config.get('resident', False):
            suffix = ['resident'] + ([str(every)] if every else [])
            if mutation_mode != 'all':
                suffix.append(mutation_mode)
            assert command[-len(suffix):] == suffix, ('wrong resident invocation', key)
            assert all(int(state['threads']) == 1 for state in raw['states']), key
            assert all(int(state['rss_before_persistence_kib']) > 0 for state in raw['states']), key
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
                mutation_calls=len(rows) * config['rounds'] * config['cycles'] * factor,
                interleaved_queries=len(rows) * config['rounds'] * config['cycles'] * factor,
                source_sha=identity['feature_sha'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2))
