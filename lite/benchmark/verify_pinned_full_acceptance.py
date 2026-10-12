#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Independent raw quantile, groundtruth-hit, loader and manifest verification."""
import argparse
import csv
import json
import math
from pathlib import Path
import struct

from run_mixed_cpu_acceptance import percentile, sha


def verify(output, overrides=None):
    provenance = json.loads((output / 'provenance.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, digest in manifest.items():
        assert sha(output / name) == digest, name
    identity = json.loads((output / 'measurements/identity.json').read_text())
    assert identity['arguments']['ef'] == identity['arguments']['full_ef'] == 512
    if provenance.get('fresh_full_core'):
        receipt_path = output / 'fresh-full-receipt.json'
        assert sha(receipt_path) == provenance['fresh_full_receipt_sha256']
        receipt = json.loads(receipt_path.read_text())
        assert receipt['fresh_core'] is True and receipt['ccache'] is False
        assert receipt['prior_object_count'] == 0 and receipt['core_object_count'] > 50
        assert receipt['full_source_head'] == provenance['full_source_sha'] == provenance['feature_sha']
        assert receipt['library_sha256'] == provenance['full_library_sha256']
        assert receipt['full_headers_sha256'] == provenance['full_headers']
        assert receipt['full_build_boundary'] == provenance['full_boundary']
        assert 'fresh-full-receipt.json' in manifest
        assert provenance['full_suite'] == receipt['full_suite']
    assert identity['reference_library_sha256'] == provenance['full_library_sha256']
    assert identity['lite_library_sha256'] == provenance['lite_library_sha256']
    rows = json.loads((output / 'measurements/rows.json').read_text())
    expected = {(dataset.split('=', 1)[0], mode, trial, side)
        for dataset in identity['arguments']['dataset'] for mode in ['fp32', 'fp16', 'rabitq8']
        for trial in range(3) for side in ['lite', 'full']}
    assert len(rows) == len(expected)
    assert {(r['dataset'], r['storage'], r['trial'], r['name']) for r in rows} == expected
    truths = {}
    for dataset in identity['arguments']['dataset']:
        name, directory = dataset.split('=', 1)
        directory = (overrides or {}).get(name, directory)
        receipts = json.loads((output / 'measurements' / (name + '.dataset.json')).read_text())
        for filename, info in receipts.items():
            assert sha(Path(directory) / filename) == info['sha256']
        with (Path(directory) / 'groundtruth.ivecs').open('rb') as stream:
            truth = []
            while True:
                width = stream.read(4)
                if not width:
                    break
                k = struct.unpack('<i', width)[0]
                assert k == 10
                expected_ids = set(struct.unpack('<10i', stream.read(40)))
                assert len(expected_ids) == 10
                truth.append(expected_ids)
        truths[name] = truth
    loaders = 0
    for row in rows:
        stem = f"{row['dataset']}-{row['storage']}-{row['trial']}-{row['name']}"
        suffix = '.latencies.csv' if row['name'] == 'full' else '.queries.csv.latencies.csv'
        with (output / 'measurements' / (stem + suffix)).open() as stream:
            timings = list(csv.DictReader(stream))
        values = [float(r['latency_us']) for r in timings]
        assert len(values) == len(truths[row['dataset']])
        for fraction, metric in [(.5, 'search_p50_us'), (.99, 'search_p99_us')]:
            assert math.isclose(percentile(values, fraction), float(row['build'][metric]),
                                rel_tol=0, abs_tol=.000002), (stem, metric)
        suffix = '.neighbors.csv' if row['name'] == 'full' else '.queries.csv.neighbors.csv'
        with (output / 'measurements' / (stem + suffix)).open() as stream:
            neighbors = list(csv.DictReader(stream))
        assert len(neighbors) == len(values) * 10
        hits = 0
        for query, expected_ids in enumerate(truths[row['dataset']]):
            selected = neighbors[query * 10:(query + 1) * 10]
            assert all(int(r['query']) == query for r in selected)
            assert [int(r['rank']) for r in selected] == list(range(10))
            ids = {int(r['id']) for r in selected}
            assert len(ids) == 10 and all(0 <= i < 10000 for i in ids)
            assert all(math.isfinite(float.fromhex(r['distance'])) for r in selected)
            hits += len(ids & expected_ids)
        assert math.isclose(hits / (len(values) * 10), float(row['build']['recall_at_k']),
                            rel_tol=0, abs_tol=.0000006), stem
        assert len(row['loads']) == 3
        assert int(row['build']['snapshot_bytes']) > 0
        for loader in row['loads']:
            assert int(loader['count']) == 10000 and float(loader['load_ms']) > 0
            assert int(loader['loaded_rss_kib']) >= int(loader['before_load_rss_kib'])
            loaders += 1
    return dict(audit='PASS', builders=len(rows), fresh_loaders=loaders,
                query_results=sum(len(truths[r['dataset']]) for r in rows),
                native_full_reference=provenance['full_source_sha'], equal_query_ef=512)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--dataset', action='append', default=[], help='Override name=path for replay')
    args = parser.parse_args()
    overrides = dict(value.split('=', 1) for value in args.dataset)
    print(json.dumps(verify(args.output, overrides), indent=2))
