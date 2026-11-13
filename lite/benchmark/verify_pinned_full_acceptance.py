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
from run_pinned_full_acceptance import (
    HISTORICAL_FEATURE_SHA, HISTORICAL_FULL_LIBRARY_SHA256, HISTORICAL_LITE_LIBRARY_SHA256,
)


LOADER_COLUMNS = (
    'dim', 'count', 'snapshot_bytes', 'load_ms', 'before_create_rss_kib',
    'before_load_rss_kib', 'loaded_rss_kib', 'process_peak_rss_kib', 'page_cache_control',
)


def verify_loader(output, manifest, stem, summary, dimension, snapshot_bytes):
    stdout = f'measurements/{stem}.stdout'
    peak = f'measurements/{stem}.peak_kib'
    if stdout not in manifest or peak not in manifest:
        raise ValueError(f'Missing raw loader evidence in manifest: {stem}')
    lines = (output / stdout).read_text().splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith('dim,count,')]
    if len(headers) != 1:
        raise ValueError(f'Expected one loader CSV header: {stem}')
    header = headers[0]
    records = csv.DictReader(lines[header:header + 2])
    raw = next(records, None)
    if tuple(records.fieldnames) != LOADER_COLUMNS or raw is None or None in raw or None in raw.values():
        raise ValueError(f'Invalid loader CSV record: {stem}')
    raw['external_peak_rss_kib'] = int((output / peak).read_text())
    if raw != summary:
        fields = sorted(key for key in raw.keys() | summary.keys() if raw.get(key) != summary.get(key))
        raise ValueError(f"Loader summary differs from raw evidence: {stem} ({', '.join(fields)})")
    if (int(raw['dim']), int(raw['count']), int(raw['snapshot_bytes'])) != (dimension, 10000, snapshot_bytes):
        raise ValueError(f'Loader identity differs from dataset or snapshot: {stem}')
    load_ms = float(raw['load_ms'])
    if (not math.isfinite(load_ms) or load_ms <= 0
            or not 0 <= int(raw['before_load_rss_kib']) <= int(raw['loaded_rss_kib'])):
        raise ValueError(f'Invalid loader metrics: {stem}')


def verify(output, overrides=None):
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, digest in manifest.items():
        if sha(output / name) != digest:
            raise ValueError(f'Manifest SHA-256 mismatch: {name}')
    for name in ('provenance.json', 'measurements/identity.json', 'measurements/rows.json'):
        if name not in manifest:
            raise ValueError(f'Missing required evidence in manifest: {name}')
    provenance = json.loads((output / 'provenance.json').read_text())
    identity = json.loads((output / 'measurements/identity.json').read_text())
    assert identity['arguments']['ef'] == identity['arguments']['full_ef'] == 512

    feature_sha = provenance['feature_sha']
    historical = feature_sha == HISTORICAL_FEATURE_SHA
    fresh_full = provenance.get('fresh_full_core') is True
    if not historical and not fresh_full:
        raise ValueError('Current-source comparison requires a fresh Full receipt')
    if fresh_full:
        name = 'fresh-full-receipt.json'
        if name not in manifest or manifest[name] != provenance.get('fresh_full_receipt_sha256'):
            raise ValueError('Missing or mismatched fresh Full receipt in manifest')
        receipt = json.loads((output / name).read_text())
        if (receipt['fresh_core'] is not True or receipt['ccache'] is not False
                or receipt['prior_object_count'] != 0 or receipt['core_object_count'] <= 50):
            raise ValueError('Full receipt does not record a fresh core build')
        if receipt['full_source_head'] != feature_sha or provenance['full_source_sha'] != feature_sha:
            raise ValueError('Full receipt does not match the measured feature commit')
        for receipt_key, provenance_key in (
                ('library_sha256', 'full_library_sha256'),
                ('full_headers_sha256', 'full_headers'),
                ('full_build_boundary', 'full_boundary'),
                ('full_suite', 'full_suite')):
            if receipt[receipt_key] != provenance[provenance_key]:
                raise ValueError(f'Full receipt disagrees with provenance: {receipt_key}')
    elif provenance['full_library_sha256'] != HISTORICAL_FULL_LIBRARY_SHA256:
        raise ValueError('Historical Full library SHA-256 mismatch')

    if (identity['reference_library_sha256'] != provenance['full_library_sha256']
            or identity['lite_library_sha256'] != provenance['lite_library_sha256']):
        raise ValueError('Measured library digests disagree with provenance')
    if historical:
        if provenance['lite_library_sha256'] != HISTORICAL_LITE_LIBRARY_SHA256:
            raise ValueError('Historical Lite library SHA-256 mismatch')
    else:
        if provenance.get('source_bound_current_comparison') is not True:
            raise ValueError('Current-source comparison requires a Lite build receipt')
        name = 'lite-build-source-receipt.json'
        if name not in manifest or manifest[name] != provenance.get('lite_build_source_receipt_sha256'):
            raise ValueError('Missing or mismatched Lite build receipt in manifest')
        receipt = json.loads((output / name).read_text())
        if (receipt['source']['feature_sha'] != feature_sha
                or receipt['library_sha256'] != provenance['lite_library_sha256']):
            raise ValueError('Lite build receipt disagrees with provenance')
    rows = json.loads((output / 'measurements/rows.json').read_text())
    expected = {(dataset.split('=', 1)[0], mode, trial, side)
        for dataset in identity['arguments']['dataset'] for mode in ['fp32', 'fp16', 'rabitq8']
        for trial in range(3) for side in ['lite', 'full']}
    actual = {(r['dataset'], r['storage'], r['trial'], r['name']) for r in rows}
    if not expected or len(rows) != len(expected) or actual != expected:
        raise ValueError('Comparison rows do not match the expected measurement matrix')
    truths = {}
    dimensions = {}
    for dataset in identity['arguments']['dataset']:
        name, directory = dataset.split('=', 1)
        directory = (overrides or {}).get(name, directory)
        record = f'measurements/{name}.dataset.json'
        if record not in manifest:
            raise ValueError(f'Missing dataset receipt in manifest: {name}')
        receipts = json.loads((output / record).read_text())
        for filename in ('base.fvecs', 'queries.fvecs', 'groundtruth.ivecs'):
            if sha(Path(directory) / filename) != receipts[filename]['sha256']:
                raise ValueError(f'Dataset SHA-256 mismatch: {name}/{filename}')
        with (Path(directory) / 'base.fvecs').open('rb') as stream:
            dimensions[name] = struct.unpack('<i', stream.read(4))[0]
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
        snapshot_bytes = int(row['build']['snapshot_bytes'])
        if len(row['loads']) != 3 or snapshot_bytes <= 0:
            raise ValueError(f'Invalid loader sample count or snapshot size: {stem}')
        for sample, loader in enumerate(row['loads']):
            verify_loader(output, manifest, f'{stem}-load-{sample}', loader,
                          dimensions[row['dataset']], snapshot_bytes)
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
