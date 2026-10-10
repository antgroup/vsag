#!/usr/bin/env python3
"""Prepare SIFT subsets with exact ground truth for the selected base prefix."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def write_vectors(path, values):
    dim = values.shape[1]
    records = np.empty(values.shape[0], dtype=[("dim", "<i4"), ("vector", "<f4", (dim,))])
    records["dim"] = dim
    records["vector"] = values
    records.tofile(path)


def write_neighbors(path, values):
    k = values.shape[1]
    records = np.empty(values.shape[0], dtype=[("count", "<i4"), ("ids", "<i4", (k,))])
    records["count"] = k
    records["ids"] = values
    records.tofile(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--counts", type=int, nargs="+", default=[10000, 100000])
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists")
    if min(args.counts + [args.queries, args.k]) <= 0:
        parser.error("counts, queries and k must be positive")
    with h5py.File(args.source, "r") as dataset:
        if dataset.attrs.get("distance") != "euclidean":
            parser.error("expected a Euclidean dataset")
        train, test = dataset["train"], dataset["test"]
        if train.shape[1] != 128 or test.shape[1] != 128:
            parser.error("expected 128-dimensional SIFT data")
        if max(args.counts) > train.shape[0] or args.queries > test.shape[0]:
            parser.error("requested subset exceeds source data")
        if any(count < args.k for count in args.counts):
            parser.error("k exceeds base count")
        queries = np.asarray(test[: args.queries], dtype="<f4")
        args.output.mkdir(parents=True)
        source_hash = sha256(args.source)
        for count in args.counts:
            base = np.asarray(train[:count], dtype="<f4")
            directory = args.output / f"scale-{count}"
            directory.mkdir()
            base_file = directory / "base.fvecs"
            query_file = directory / "queries.fvecs"
            truth_file = directory / "groundtruth.ivecs"
            write_vectors(base_file, base)
            write_vectors(query_file, queries)
            ids = np.arange(count, dtype=np.int32)
            truth = np.empty((len(queries), args.k), dtype=np.int32)
            for i, query in enumerate(queries):
                difference = base - query
                distances = np.einsum("ij,ij->i", difference, difference)
                truth[i] = np.lexsort((ids, distances))[: args.k]
            write_neighbors(truth_file, truth)
            manifest = {
                "source_url": "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
                "source_sha256": source_hash,
                "base_prefix": count,
                "queries_prefix": len(queries),
                "dimension": 128,
                "k": args.k,
                "groundtruth": "recomputed squared-L2, ties by ascending ID on selected prefix",
                "files_sha256": {
                    path.name: sha256(path) for path in [base_file, query_file, truth_file]
                },
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps({"directory": str(directory), **manifest}), flush=True)


if __name__ == "__main__":
    main()
