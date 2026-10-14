#!/usr/bin/env python3
"""Prepare normalized Cohere subsets and audit cosine-to-L2 rounding."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from prepare_sift import sha256, write_neighbors, write_vectors


def read_vectors(path):
    table = pq.read_table(path, columns=['id', 'emb'])
    ids = table['id'].combine_chunks().to_numpy()
    vectors = table['emb'].combine_chunks()
    if table.num_rows == 0 or not np.all(np.diff(vectors.offsets.to_numpy()) == 768):
        raise ValueError('expected nonempty 768-dimensional vectors')
    if vectors.null_count or vectors.values.null_count or table['id'].null_count:
        raise ValueError('null IDs or vectors')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate source IDs')
    return np.asarray(ids, dtype='<i8'), vectors.values.to_numpy().reshape(-1, 768)


def normalize(vectors):
    values = np.asarray(vectors, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError('nonfinite vector')
    norms = np.sqrt(np.einsum('ij,ij->i', values, values))
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError('invalid vector norm')
    return np.asarray(values / norms[:, None], dtype='<f4'), norms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--counts', type=int, nargs='+', default=[10000, 100000])
    parser.add_argument('--queries', type=int, default=100)
    parser.add_argument('--query-offset', type=int, default=0)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output directory already exists')
    if min(args.counts + [args.queries, args.k]) <= 0 or args.query_offset < 0:
        parser.error('positive counts/queries/k and nonnegative offset required')
    if len(set(args.counts)) != len(args.counts) or min(args.counts) < args.k:
        parser.error('unique counts >= k required')
    train_ids, raw_base = read_vectors(args.source / 'train.parquet')
    test_ids, raw_queries = read_vectors(args.source / 'test.parquet')
    end = args.query_offset + args.queries
    if max(args.counts) > len(raw_base) or end > len(raw_queries):
        parser.error('requested subset exceeds source data')
    raw_base = raw_base[:max(args.counts)]
    raw_queries = raw_queries[args.query_offset:end]
    base, base_norms = normalize(raw_base)
    queries, query_norms = normalize(raw_queries)
    source_hashes = {name: sha256(args.source / name) for name in ['train.parquet', 'test.parquet']}
    args.output.mkdir(parents=True)
    for count in args.counts:
        directory = args.output / f'scale-{count}'
        directory.mkdir()
        truth = np.empty((len(queries), args.k), dtype='<i4')
        ids = np.arange(count, dtype=np.int32)
        exact_cosine_matches = 0
        overlaps = []
        maximum_error = 0.0
        for row, query in enumerate(queries):
            distances = np.empty(count, dtype=np.float64)
            cosine_distances = np.empty(count, dtype=np.float64)
            raw_query = raw_queries[row].astype(np.float64)
            for start in range(0, count, 8192):
                stop = min(count, start + 8192)
                difference = base[start:stop].astype(np.float64) - query.astype(np.float64)
                distances[start:stop] = np.einsum('ij,ij->i', difference, difference)
                raw = raw_base[start:stop].astype(np.float64)
                cosine_distances[start:stop] = 1 - (raw @ raw_query) / base_norms[start:stop] / query_norms[row]
            truth[row] = np.lexsort((ids, distances))[:args.k]
            cosine_truth = np.lexsort((ids, cosine_distances))[:args.k]
            exact_cosine_matches += int(np.array_equal(truth[row], cosine_truth))
            overlaps.append(len(set(truth[row]) & set(cosine_truth)) / args.k)
            maximum_error = max(maximum_error, float(np.max(np.abs(distances - 2 * cosine_distances))))
        write_vectors(directory / 'base.fvecs', base[:count])
        write_vectors(directory / 'queries.fvecs', queries)
        write_neighbors(directory / 'groundtruth.ivecs', truth)
        train_ids[:count].tofile(directory / 'source-base-ids.i64')
        test_ids[args.query_offset:end].tofile(directory / 'source-query-ids.i64')
        manifest = {
            'source_url': 'https://assets.zilliz.com.cn/benchmark/cohere_small_100k/',
            'source_sha256': source_hashes, 'source_metric': 'cosine',
            'stored_metric': 'squared-L2', 'normalization': 'FP64 row norm/division, stored FP32',
            'dimension': 768, 'base_prefix': count, 'query_rows': [args.query_offset, end],
            'k': args.k, 'index_ids': 'zero-based base row ordinals; source IDs preserved in i64 files',
            'groundtruth': 'exhaustive FP64 squared-L2 on stored FP32 normalized vectors, ties by ID',
            'rounding_audit': {'queries': len(queries), 'ordered_topk_exact_cosine_matches': exact_cosine_matches,
                               'mean_topk_set_overlap': float(np.mean(overlaps)),
                               'minimum_topk_set_overlap': min(overlaps),
                               'max_absolute_distance_vs_twice_cosine_error': maximum_error},
            'files_sha256': {p.name: sha256(p) for p in directory.iterdir() if p.is_file()},
        }
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({'directory': str(directory), 'rounding_audit': manifest['rounding_audit']}), flush=True)


if __name__ == '__main__':
    main()
