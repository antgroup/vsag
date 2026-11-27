#  Copyright 2024-present the vsag project
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Recall/QPS benchmark of binary indexes on ann-benchmarks Hamming datasets.

Builds VSAG ``brute_force`` (exact) and ``hgraph`` binary indexes on a Hamming dataset from
https://ann-benchmarks.com (``sift-256-hamming``, ``word2bits-800-hamming``) and, when the
``faiss`` package is installed, FAISS ``IndexBinaryFlat`` and ``IndexBinaryHNSW`` with the same
graph degree for reference. Recall@k counts a returned id as correct when its Hamming distance is
not larger than the true k-th distance, so ties at the k-th distance are not penalized; missing
neighbors (id -1) count as misses, and a VSAG search that fails stops the script with the error
of the library instead of being reported as a result.

Each setting runs the queries ``--repeat`` times on one thread and reports the best pass. Pin the
process to idle physical cores of one NUMA node (for example with ``taskset``) for stable numbers.

Example:
    taskset -c 0-15 python3 bench_index.py --data sift-256-hamming.hdf5 --threads 16
"""

import argparse
import json
import time

import h5py
import numpy as np
import pyvsag


def load_dataset(path, max_base, max_queries):
    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"])
        test = np.asarray(f["test"])
        distances = np.asarray(f["distances"]) if "distances" in f else None
    # ann-benchmarks stores one bit per element (0/1); pack MSB-first into bytes.
    if train.dtype == bool or train.max() <= 1:
        dim = train.shape[1]
        train = np.packbits(train.astype(bool), axis=1)
        test = np.packbits(test.astype(bool), axis=1)
    else:
        dim = train.shape[1] * 8
    train_count = len(train)
    train = np.ascontiguousarray(train[:max_base], dtype=np.uint8)
    test = np.ascontiguousarray(test[:max_queries], dtype=np.uint8)
    if distances is not None and max_base < train_count:
        distances = None  # the stored ground truth refers to the full base set
    return dim, train, test, distances


def exact_kth_distances(train, test, dim, k):
    """The k-th smallest Hamming distance of every query, by brute force with numpy."""
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    kth = np.empty(len(test), dtype=np.float32)
    for q, query in enumerate(test):
        dist = table[np.bitwise_xor(train, query)].sum(axis=1)
        kth[q] = np.partition(dist, k - 1)[k - 1]
    return kth


def tie_aware_recall(result_ids, result_dists, kth):
    """A returned neighbor counts when its distance is not larger than the true k-th distance.
    Missing neighbors (id -1, left by a failed or short search) count as misses."""
    found = (result_ids >= 0) & (result_dists >= 0)
    return float(np.mean(found & (result_dists <= kth[:, None])))


def run_queries(search, test, k, repeat):
    """Runs all queries `repeat` times; returns the ids, the distances and the best queries per
    second. `search` returns the (ids, distances) of one query, with -1 in missing slots."""
    ids = np.empty((len(test), k), dtype=np.int64)
    dists = np.empty((len(test), k), dtype=np.float32)
    best_qps = 0.0
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        for q, query in enumerate(test):
            ids[q], dists[q] = search(query)
        best_qps = max(best_qps, len(test) / (time.perf_counter() - start))
    return ids, dists, best_qps


def check_vsag_searches(name, ef, result_ids, explain):
    """pyvsag.Index.knn_search reports a failed search only as -1 ids and distances. A k-NN
    search on a non-empty index always finds a neighbor, so a row without one is a failure:
    stop with the library's error, which `explain(query_number)` raises, instead of reporting
    failed searches as results."""
    failed = np.flatnonzero((result_ids < 0).all(axis=1))
    if len(failed) == 0:
        return
    reason = "no neighbor returned"
    try:
        explain(int(failed[0]))
    except RuntimeError as error:
        reason = str(error)
    raise SystemExit(f"{name} (ef_search {ef}): {len(failed)} of {len(result_ids)} searches "
                     f"failed: {reason}")


def bench_vsag(dim, train, test, kth, args):
    rows = []
    ids = np.arange(len(train), dtype=np.int64)
    base = {"dtype": "binary", "metric_type": "hamming", "dim": dim}

    hgraph_params = dict(
        base,
        index_param={
            "max_degree": args.max_degree,
            "ef_construction": args.ef_construction,
            "build_thread_count": args.threads,
        },
    )
    index = pyvsag.Index("hgraph", json.dumps(hgraph_params))
    start = time.perf_counter()
    index.build(train, ids, len(train), dim)
    build = time.perf_counter() - start
    for ef in args.ef:
        params = json.dumps({"hgraph": {"ef_search": ef}})

        def search(query, ef_params=params):
            return index.knn_search(query, args.k, ef_params)

        def explain(q, ef_params=params):
            index.knn_search_with_statistics(test[q], args.k, ef_params)

        found, dists, qps = run_queries(search, test, args.k, args.repeat)
        check_vsag_searches("vsag hgraph", ef, found, explain)
        rows.append(("vsag hgraph", ef, tie_aware_recall(found, dists, kth), qps, build))

    if len(train) <= args.exact_limit:
        exact = pyvsag.Index("brute_force", json.dumps(base))
        start = time.perf_counter()
        exact.build(train, ids, len(train), dim)
        build = time.perf_counter() - start
        subset = test[: args.exact_queries]
        found, dists, qps = run_queries(lambda q: exact.knn_search(q, args.k, "{}"), subset,
                                        args.k, args.repeat)
        check_vsag_searches("vsag brute_force", "-", found,
                            lambda q: exact.knn_search_with_statistics(subset[q], args.k, "{}"))
        rows.append(("vsag brute_force", "-", tie_aware_recall(found, dists, kth[: len(subset)]),
                     qps, build))
    return rows


def bench_faiss(dim, train, test, kth, args):
    try:
        import faiss
    except ImportError:
        return []
    faiss.omp_set_num_threads(args.threads)
    rows = []
    # IndexBinaryHNSW(M) keeps 2 * M neighbors on the base layer, like max_degree in HGraph.
    hnsw = faiss.IndexBinaryHNSW(dim, args.max_degree // 2)
    hnsw.hnsw.efConstruction = args.ef_construction
    start = time.perf_counter()
    hnsw.add(train)
    build = time.perf_counter() - start
    faiss.omp_set_num_threads(1)
    def searcher(index):
        # FAISS returns (distances, labels) of shape (1, k), with label -1 in missing slots.
        def search(query):
            distances, labels = index.search(query[None, :], args.k)
            return labels[0], distances[0]

        return search

    for ef in args.ef:
        hnsw.hnsw.efSearch = ef
        found, dists, qps = run_queries(searcher(hnsw), test, args.k, args.repeat)
        rows.append(("faiss IndexBinaryHNSW", ef, tie_aware_recall(found, dists, kth), qps, build))
    if len(train) <= args.exact_limit:
        flat = faiss.IndexBinaryFlat(dim)
        flat.add(train)
        subset = test[: args.exact_queries]
        found, dists, qps = run_queries(searcher(flat), subset, args.k, args.repeat)
        rows.append(("faiss IndexBinaryFlat", "-",
                     tie_aware_recall(found, dists, kth[: len(subset)]), qps, 0.0))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="ann-benchmarks Hamming HDF5 file")
    parser.add_argument("--base", type=int, default=10**9, help="use at most this many vectors")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    parser.add_argument("--max-degree", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--threads", type=int, default=16, help="build threads")
    parser.add_argument("--exact-limit", type=int, default=2 * 10**6)
    parser.add_argument("--exact-queries", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=3,
                        help="passes over the queries per setting; the best QPS is reported")
    parser.add_argument("--json", help="also write the rows to this JSON file")
    args = parser.parse_args()

    dim, train, test, distances = load_dataset(args.data, args.base, args.queries)
    if distances is not None:
        kth = np.asarray(distances[: len(test), args.k - 1], dtype=np.float64)
        if kth.max() <= 1.0:
            # ann-benchmarks stores the Hamming distance divided by the number of bits.
            kth = np.rint(kth * dim)
        kth = kth.astype(np.float32)
    else:
        kth = exact_kth_distances(train, test, dim, args.k)
    print(f"dataset {args.data}: {len(train)} x {dim} bits, {len(test)} queries, k={args.k}")

    rows = bench_vsag(dim, train, test, kth, args) + bench_faiss(dim, train, test, kth, args)
    print("| index | ef_search | recall@%d | QPS (1 thread) | build s |" % args.k)
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, ef, recall, qps, build in rows:
        print(f"| {name} | {ef} | {recall:.4f} | {qps:,.0f} | {build:.1f} |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump([dict(index=r[0], ef=r[1], recall=r[2], qps=r[3], build_s=r[4])
                       for r in rows], f, indent=2)


if __name__ == "__main__":
    main()
