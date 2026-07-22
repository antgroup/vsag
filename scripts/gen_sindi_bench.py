#!/usr/bin/env python3
"""
Generate SINDI benchmark HDF5 dataset with proximity-aware ground truth.

Usage:
    python gen_sindi_bench.py --output sindi_bench.hdf5 \
        --num_base 10000 --num_query 100 --doc_len 512 \
        --vocab_size 5000 --max_terms 200 \
        --proximity_weight 0.3 --topk 10

Output HDF5 structure:
    /train       - serialized sparse vectors (binary blob)
    /test        - serialized query sparse vectors (binary blob)
    /neighbors   - ground truth neighbor IDs (int64, Q x K)
    /distances   - ground truth distances (float32, Q x K)
    attrs["distance"] = "ip"
"""

import argparse
import struct

import h5py
import numpy as np
from collections import defaultdict


def generate_sparse_doc(rng, vocab_size, max_terms, doc_len):
    """Generate a random sparse document with token sequence."""
    # Token sequence: random tokens from vocab
    token_seq = rng.integers(0, vocab_size, size=doc_len, dtype=np.uint32)

    # Sparse vector: unique terms with random weights
    unique_terms = np.unique(token_seq)
    if len(unique_terms) > max_terms:
        # Keep top max_terms by frequency
        counts = np.bincount(token_seq, minlength=vocab_size)
        top_terms = np.argsort(counts)[::-1][:max_terms]
        unique_terms = np.sort(top_terms[counts[top_terms] > 0])

    weights = rng.uniform(0.1, 1.0, size=len(unique_terms)).astype(np.float32)
    return unique_terms.astype(np.uint32), weights, token_seq


def serialize_sparse_vectors(vectors):
    """Serialize sparse vectors to binary blob for HDF5.

    Each vector: [len(uint32), ids(uint32 x len), vals(float32 x len)]
    """
    parts = []
    for ids, vals, _token_seq in vectors:
        parts.append(struct.pack('<I', len(ids)))
        parts.append(ids.tobytes())
        parts.append(vals.tobytes())
    return np.frombuffer(b''.join(parts), dtype=np.int8)


def serialize_token_sequences(vectors):
    """Serialize token sequences: [seq_len(uint32), token_ids(uint32 x seq_len), ...]

    Returns (blob, offsets) where offsets follows the [0, ..., total] contract:
    length N+1, offsets[i] is the byte start of record i, offsets[N] is the
    total byte length. The loader requires this companion dataset whenever
    train/test_token_sequences is present.
    """
    parts = []
    offsets = [0]
    total = 0
    for _, _, token_seq in vectors:
        if token_seq is None or len(token_seq) == 0:
            record = struct.pack('<I', 0)
        else:
            record = struct.pack('<I', len(token_seq)) + token_seq.astype(np.uint32).tobytes()
        parts.append(record)
        total += len(record)
        offsets.append(total)
    blob = np.frombuffer(b''.join(parts), dtype=np.int8)
    return blob, np.array(offsets, dtype=np.uint64)


def compute_ip(q_ids, q_vals, d_ids, d_vals):
    """Compute inner product between two sparse vectors."""
    qi, di = 0, 0
    ip = 0.0
    while qi < len(q_ids) and di < len(d_ids):
        if q_ids[qi] == d_ids[di]:
            ip += q_vals[qi] * d_vals[di]
            qi += 1
            di += 1
        elif q_ids[qi] < d_ids[di]:
            qi += 1
        else:
            di += 1
    return ip


def compute_pairwise_proximity(q_ids, d_ids, d_token_seq, ordered=False):
    """Compute normalized pairwise proximity boost."""
    # Build position map for doc terms
    pos_map = defaultdict(list)
    for pos, token in enumerate(d_token_seq):
        pos_map[token].append(pos)

    # Collect positions for query terms present in doc
    present_positions = []
    for qid in q_ids:
        positions = pos_map.get(int(qid), [])
        if positions:
            present_positions.append(positions)

    n_present = len(present_positions)
    if n_present < 2:
        return 0.0

    # Pairwise sloppyFreq
    raw_boost = 0.0
    for i in range(n_present):
        for j in range(i + 1, n_present):
            min_dist = float('inf')
            for pa in present_positions[i]:
                for pb in present_positions[j]:
                    if not ordered:
                        dist = abs(pa - pb)
                    else:
                        if pa <= pb:
                            dist = pb - pa
                        else:
                            dist = (pa - pb) * 2
                    min_dist = min(min_dist, dist)
            raw_boost += 1.0 / (min_dist + 1)

    # Normalize by C(Q, 2) where Q = total query terms
    pair_count = len(q_ids) * (len(q_ids) - 1) / 2.0
    if pair_count == 0:
        return 0.0
    return raw_boost / pair_count


def compute_distance(q_ids, q_vals, d_ids, d_vals, d_token_seq,
                     proximity_weight=0.0, proximity_ordered=False,
                     multiplicative=True):
    """Compute final distance = 1 - boosted_ip."""
    ip = compute_ip(q_ids, q_vals, d_ids, d_vals)

    if proximity_weight > 0 and d_token_seq is not None:
        boost = compute_pairwise_proximity(q_ids, d_ids, d_token_seq, proximity_ordered)
        if multiplicative:
            ip = ip * (1 + proximity_weight * boost)
        else:
            ip = ip + proximity_weight * boost

    return 1.0 - ip


def main():
    parser = argparse.ArgumentParser(description='Generate SINDI benchmark HDF5')
    parser.add_argument('--output', default='sindi_bench.hdf5')
    parser.add_argument('--num_base', type=int, default=10000)
    parser.add_argument('--num_query', type=int, default=100)
    parser.add_argument('--doc_len', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=5000)
    parser.add_argument('--max_terms', type=int, default=200)
    parser.add_argument('--proximity_weight', type=float, default=0.0,
                        help='0.0 for pure IP ground truth, >0 for proximity-boosted GT')
    parser.add_argument('--proximity_ordered', action='store_true')
    parser.add_argument('--multiplicative', action='store_true', default=True)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Generating {args.num_base} docs, {args.num_query} queries, "
          f"vocab={args.vocab_size}, doc_len={args.doc_len}")

    # Generate base docs
    base_docs = []
    for i in range(args.num_base):
        ids, vals, token_seq = generate_sparse_doc(
            rng, args.vocab_size, args.max_terms, args.doc_len)
        base_docs.append((ids, vals, token_seq))
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{args.num_base} docs")

    # Generate queries (sample from base docs, use their terms as query)
    query_indices = rng.choice(args.num_base, size=args.num_query, replace=False)
    queries = []
    for idx in query_indices:
        ids, vals, _ = base_docs[idx]
        # Use a subset of terms as query
        n_query_terms = min(len(ids), 10)
        sel = rng.choice(len(ids), size=n_query_terms, replace=False)
        sel.sort()
        q_ids = ids[sel]
        q_vals = vals[sel]
        queries.append((q_ids, q_vals, None))

    # Compute brute-force ground truth
    print(f"Computing ground truth (proximity_weight={args.proximity_weight})...")
    neighbors = np.zeros((args.num_query, args.topk), dtype=np.int64)
    distances = np.zeros((args.num_query, args.topk), dtype=np.float32)

    for qi in range(args.num_query):
        q_ids, q_vals, _ = queries[qi]
        dists = []
        for di in range(args.num_base):
            d_ids, d_vals, d_token_seq = base_docs[di]
            d = compute_distance(
                q_ids, q_vals, d_ids, d_vals,
                d_token_seq if args.proximity_weight > 0 else None,
                args.proximity_weight, args.proximity_ordered, args.multiplicative)
            dists.append((d, di))
        dists.sort()
        for k in range(args.topk):
            distances[qi, k] = dists[k][0]
            neighbors[qi, k] = dists[k][1]
        if (qi + 1) % 10 == 0:
            print(f"  Query {qi + 1}/{args.num_query}")

    # Serialize and write HDF5
    print(f"Writing {args.output}...")
    train_blob = serialize_sparse_vectors(base_docs)
    test_blob = serialize_sparse_vectors(
        [(q[0], q[1], np.array([], dtype=np.uint32)) for q in queries])
    train_seq_blob, train_seq_offsets = serialize_token_sequences(base_docs)

    with h5py.File(args.output, 'w') as f:
        f.create_dataset('train', data=train_blob)
        f.create_dataset('test', data=test_blob)
        f.create_dataset('train_token_sequences', data=train_seq_blob)
        f.create_dataset('train_token_sequences_offsets', data=train_seq_offsets)
        f.create_dataset('neighbors', data=neighbors)
        f.create_dataset('distances', data=distances)
        f.attrs['distance'] = 'ip'
        # Set type attribute for sparse vectors (required by eval_dataset.cpp)
        f.attrs['type'] = 'sparse'

    print(f"Done. {args.output}: {args.num_base} docs, {args.num_query} queries, "
          f"top-{args.topk} GT")


if __name__ == '__main__':
    main()
