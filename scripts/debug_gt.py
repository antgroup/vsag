#!/usr/bin/env python3
"""Compare Python GT and SINDI output for a single query to find discrepancy."""

import argparse
import struct
from collections import defaultdict

import h5py
import numpy as np


def deserialize_sparse(blob):
    """Deserialize the binary blob back to list of (ids, vals)."""
    vectors = []
    ptr = 0
    data = bytes(blob)
    while ptr < len(data):
        if ptr + 4 > len(data):
            break
        n = struct.unpack('<I', data[ptr:ptr + 4])[0]
        ptr += 4
        if n == 0:
            vectors.append((np.array([], dtype=np.uint32), np.array([], dtype=np.float32)))
            continue
        ids_size = n * 4
        vals_size = n * 4
        ids = np.frombuffer(data[ptr:ptr + ids_size], dtype=np.uint32).copy()
        ptr += ids_size
        vals = np.frombuffer(data[ptr:ptr + vals_size], dtype=np.float32).copy()
        ptr += vals_size
        # Sort
        order = np.argsort(ids)
        vectors.append((ids[order], vals[order]))
    return vectors


def deserialize_token_seqs(blob):
    """Deserialize token sequences."""
    seqs = []
    ptr = 0
    data = bytes(blob)
    while ptr < len(data):
        if ptr + 4 > len(data):
            break
        n = struct.unpack('<I', data[ptr:ptr + 4])[0]
        ptr += 4
        if n == 0:
            seqs.append(np.array([], dtype=np.uint32))
            continue
        size = n * 4
        seq = np.frombuffer(data[ptr:ptr + size], dtype=np.uint32).copy()
        ptr += size
        seqs.append(seq)
    return seqs


def compute_ip(q_ids, q_vals, d_ids, d_vals):
    qi, di = 0, 0
    ip = 0.0
    while qi < len(q_ids) and di < len(d_ids):
        if q_ids[qi] == d_ids[di]:
            ip += float(q_vals[qi]) * float(d_vals[di])
            qi += 1
            di += 1
        elif q_ids[qi] < d_ids[di]:
            qi += 1
        else:
            di += 1
    return ip


def compute_proximity(q_ids, d_token_seq, ordered=False):
    pos_map = defaultdict(list)
    for pos, tok in enumerate(d_token_seq):
        pos_map[int(tok)].append(pos)

    present = []
    for qid in q_ids:
        positions = pos_map.get(int(qid), [])
        if positions:
            present.append(positions)

    if len(present) < 2:
        return 0.0

    raw = 0.0
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            min_d = float('inf')
            for pa in present[i]:
                for pb in present[j]:
                    if not ordered:
                        d = abs(pa - pb)
                    else:
                        d = (pb - pa) if pa <= pb else (pa - pb) * 2
                    min_d = min(min_d, d)
            raw += 1.0 / (min_d + 1)
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5', required=True)
    parser.add_argument('--query_id', type=int, default=0)
    parser.add_argument('--proximity_weight', type=float, default=0.3)
    parser.add_argument('--topk', type=int, default=10)
    args = parser.parse_args()

    with h5py.File(args.hdf5, 'r') as f:
        train_blob = f['train'][:]
        test_blob = f['test'][:]
        gt_neighbors = f['neighbors'][:]
        gt_distances = f['distances'][:]
        train_seqs_blob = f['train_token_sequences'][:]

    base = deserialize_sparse(train_blob)
    queries = deserialize_sparse(test_blob)
    train_seqs = deserialize_token_seqs(train_seqs_blob)

    print(f"#docs={len(base)}, #queries={len(queries)}")
    print(f"Query {args.query_id}:")
    q_ids, q_vals = queries[args.query_id]
    print(f"  ids={q_ids.tolist()[:20]}... ({len(q_ids)} terms)")
    print(f"  vals={q_vals.tolist()[:20]}...")

    pair_count = len(q_ids) * (len(q_ids) - 1) / 2.0

    # Compute scores for all docs
    scores = []
    for di, ((d_ids, d_vals), seq) in enumerate(zip(base, train_seqs)):
        ip = compute_ip(q_ids, q_vals, d_ids, d_vals)
        if ip == 0:
            continue
        raw = compute_proximity(q_ids, seq) if args.proximity_weight > 0 else 0.0
        norm = raw / pair_count if pair_count > 0 else 0.0
        boosted = ip * (1 + args.proximity_weight * norm)
        scores.append((1.0 - boosted, di, ip, raw, norm))

    scores.sort()
    print(f"\nTop-{args.topk} by Python GT (with proximity_weight={args.proximity_weight}):")
    for rank, (dist, di, ip, raw, norm) in enumerate(scores[:args.topk]):
        print(f"  rank{rank}: doc={di} dist={dist:.6f} ip={ip:.6f} raw_boost={raw:.4f} norm_boost={norm:.4f}")

    print(f"\nGT from HDF5 for query {args.query_id}:")
    for j in range(args.topk):
        print(f"  rank{j}: doc={gt_neighbors[args.query_id][j]} dist={gt_distances[args.query_id][j]:.6f}")


if __name__ == '__main__':
    main()
