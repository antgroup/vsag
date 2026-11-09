#!/usr/bin/env python3
# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent NumPy cross-check of a dense profile_dataset report.

Re-implements the sampling (SplitMix64-seeded xoshiro256**, Floyd's algorithm and the seeded
shuffle) to select the same rows as the C++ tool, then recomputes full-split metrics and the
exact k-NN based metrics with NumPy and compares them with the JSON report.

Usage: python3 validate_profile.py DATA.hdf5 REPORT.json [--split train]
Requires numpy and h5py. Exits with 1 when a metric differs beyond its tolerance.
"""

import argparse
import json
import sys

import h5py
import numpy as np

MASK = (1 << 64) - 1


def splitmix(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return state, z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self.s = []
        state = seed
        for _ in range(4):
            state, value = splitmix(state)
            self.s.append(value)

    def next(self):
        s = self.s
        result = (((((s[1] * 5) & MASK) << 7 | ((s[1] * 5) & MASK) >> 57) & MASK) * 9) & MASK
        t = (s[1] << 17) & MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = ((s[3] << 45) | (s[3] >> 19)) & MASK
        return result

    def uniform(self, bound):
        threshold = ((1 << 64) - bound) % bound
        while True:
            value = self.next()
            if value >= threshold:
                return value % bound


def derive_seed(seed, stream):
    h = 0xCBF29CE484222325
    for byte in stream.encode():
        h = ((h ^ byte) * 0x100000001B3) & MASK
    return splitmix(seed ^ h)[1]


def floyd(population, count, rng):
    if count >= population:
        return list(range(population))
    chosen = set()
    for j in range(population - count, population):
        candidate = rng.uniform(j + 1)
        chosen.add(j if candidate in chosen else candidate)
    return sorted(chosen)


def shuffle(values, rng):
    for i in range(len(values), 1, -1):
        j = rng.uniform(i)
        values[i - 1], values[j] = values[j], values[i - 1]


def quantile(sorted_values, p):
    return float(np.quantile(sorted_values, p))  # NumPy's default "linear" method


def lid_mle(distances):
    positive = distances[distances > 0]
    if positive.size < 2:
        return np.nan
    total = np.sum(np.log(positive[-1] / positive[:-1]))
    return np.nan if total <= 0 else (positive.size - 1) / total


class Checker:
    def __init__(self):
        self.failures = 0
        self.checks = 0

    def close(self, name, expected, actual, rtol):
        self.checks += 1
        ok = np.isclose(expected, actual, rtol=rtol, atol=rtol * 1e-3)
        print(f"{'ok  ' if ok else 'FAIL'} {name}: numpy={expected:.10g} report={actual:.10g}")
        self.failures += 0 if ok else 1

    def below(self, name, difference, limit):
        self.checks += 1
        ok = difference <= limit
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {difference:.3g} (limit {limit:g})")
        self.failures += 0 if ok else 1

    def equal(self, name, expected, actual):
        self.checks += 1
        ok = expected == actual
        print(f"{'ok  ' if ok else 'FAIL'} {name}: numpy={expected} report={actual}")
        self.failures += 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data")
    parser.add_argument("report")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    report = json.load(open(args.report))
    config = report["config"]
    split = report["splits"][args.split]
    metric = config["metric"]
    with h5py.File(args.data, "r") as f:
        data = np.asarray(f[args.split], dtype=np.float32)
    rows, dim = data.shape
    check = Checker()

    finite_rows = np.isfinite(data).all(axis=1)
    check.equal("invalid_values.total", int((~np.isfinite(data)).sum()), split["invalid_values"]["total"])
    check.equal("invalid_values.rows_with_invalid", int((~finite_rows).sum()), split["invalid_values"]["rows_with_invalid"])
    valid = data[finite_rows].astype(np.float64)
    norms = np.sort(np.sqrt((valid ** 2).sum(axis=1)))
    for key, p in (("p01", 0.01), ("p50", 0.5), ("p99", 0.99)):
        check.close(f"norms.summary.{key}", quantile(norms, p), split["norms"]["summary"][key], 1e-6)
    check.close("norms.summary.mean", norms.mean(), split["norms"]["summary"]["mean"], 1e-9)
    finite_values = np.where(np.isfinite(data), data, np.nan).astype(np.float64)
    means = np.nanmean(finite_values, axis=0)
    stds = np.nanstd(finite_values, axis=0)
    per_dim = split["dimensions"]["per_dimension"]
    scale = max(1.0, float(np.max(np.abs(means))))
    check.below("dimensions.mean max abs diff / scale", float(np.max(np.abs(means - per_dim["mean"]))) / scale, 1e-9)
    check.below("dimensions.std max rel diff",
                float(np.max(np.abs(stds - per_dim["std"]) / np.maximum(stds, 1e-30))), 1e-9)
    check.close("values.mean", float(np.nanmean(finite_values)), split["values"]["mean"], 1e-9)
    check.close("sparsity.zero_ratio", float(np.mean(valid == 0)) if (~finite_rows).sum() == 0
                else float(np.sum(finite_values == 0) / np.sum(np.isfinite(finite_values))),
                split["sparsity"]["zero_ratio"], 1e-9)
    if "duplicates" in split and split["duplicates"]["scope"] == "full":
        canonical = np.where(data == 0, np.float32(0), data)
        unique = np.unique(canonical.view(np.uint32), axis=0).shape[0]
        check.equal("duplicates.duplicate_rows", rows - unique, split["duplicates"]["duplicate_rows"])

    # Reproduce the sample and the k-NN queries.
    seed = config["seed"]
    sample = floyd(rows, config["sample_size"], Rng(derive_seed(seed, "sample/" + args.split)))
    order = list(range(len(sample)))
    shuffle(order, Rng(derive_seed(seed, "order/" + args.split)))
    sample_norms = np.sqrt((data[sample].astype(np.float64) ** 2).sum(axis=1))
    usable = [p for p in order if finite_rows[sample[p]] and (metric != "cosine" or sample_norms[p] > 0)]
    check.equal("sample.valid_sampled_rows", len(usable), split["pairwise_distances"]["sample"]["valid_sampled_rows"])
    if "nearest_neighbors" not in split:
        return report_result(check)
    queries = [sample[p] for p in usable[: split["nearest_neighbors"]["sample"]["used_rows"]]]
    k = split["nearest_neighbors"]["k"]
    base = data.astype(np.float64)
    base[~finite_rows] = np.nan
    base_norm2 = np.nansum(base ** 2, axis=1)
    first, kth, lids, ratios = [], [], [], []
    for start in range(0, len(queries), 64):
        block = queries[start:start + 64]
        q = data[block].astype(np.float64)
        ip = np.nan_to_num(q @ np.nan_to_num(base).T)
        if metric == "l2":
            distance = (q ** 2).sum(1)[:, None] + base_norm2[None, :] - 2 * ip
            geometric_of = np.sqrt
        elif metric == "ip":
            distance = 1.0 - ip
            geometric_of = None
        else:
            distance = 1.0 - ip / (np.sqrt((q ** 2).sum(1))[:, None] * np.sqrt(base_norm2)[None, :])
            geometric_of = lambda d: np.sqrt(np.maximum(0.0, 2.0 * d))
        distance[:, ~finite_rows] = np.inf
        if metric == "cosine":
            distance[:, base_norm2 == 0] = np.inf
        for i, row in enumerate(block):
            distance[i, row] = np.inf
            nearest = np.argsort(distance[i], kind="stable")[:k]
            # Recompute the selected distances exactly (the expansion above only ranks).
            exact = np.sum((base[nearest] - q[i]) ** 2, axis=1) if metric == "l2" else distance[i, nearest]
            first.append(exact[0])
            kth.append(exact[-1])
            if geometric_of is not None:
                geometric = np.sort(geometric_of(exact))
                lids.append(lid_mle(geometric))
                if geometric[0] > 0:
                    ratios.append(geometric[1] / geometric[0])
    first = np.sort(first)
    nn = split["nearest_neighbors"]
    check.close("nearest_neighbors.first_neighbor_distance.p50", quantile(first, 0.5),
                nn["first_neighbor_distance"]["summary"]["p50"], 1e-6)
    check.close("nearest_neighbors.kth_neighbor_distance.mean", float(np.mean(kth)),
                nn["kth_neighbor_distance"]["summary"]["mean"], 1e-6)
    if lids:
        lid_values = np.sort(np.array(lids)[np.isfinite(lids)])
        intrinsic = split["intrinsic_dimensionality"]
        check.close("intrinsic_dimensionality.lid_mle.p50", quantile(lid_values, 0.5),
                    intrinsic["lid_mle"]["summary"]["p50"], 1e-5)
        ratios = np.array(ratios)
        ratios = ratios[ratios > 1]
        check.close("intrinsic_dimensionality.two_nn", ratios.size / np.sum(np.log(ratios)), intrinsic["two_nn"], 1e-5)
    return report_result(check)


def report_result(check):
    print(f"{check.checks - check.failures}/{check.checks} checks passed")
    return 1 if check.failures else 0


if __name__ == "__main__":
    sys.exit(main())
