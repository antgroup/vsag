# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import statistics
import struct
import subprocess
import tempfile


STORAGES = ("bruteforce", "fp32", "fp16", "rabitq8")
VARIANTS = ("memory", "snapshot", "journal")
REPEATS = 3


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def csv_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def percentile(values, p):
    return sorted(values)[math.ceil(p * len(values)) - 1]


def active_builds():
    rows = subprocess.check_output(["ps", "-eo", "pid=,psr=,comm="], text=True)
    watched = {"cc1plus", "ninja", "make", "unittests", "functests", "clang-tidy-20"}
    return [
        {"cpu": int(cpu), "command": command}
        for _, cpu, command in (line.split(maxsplit=2) for line in rows.splitlines())
        if command in watched
    ]


def neighbor_ids(path):
    data = path.read_bytes()
    assert len(data) == 100 * 10 * 12
    ids = [identifier for identifier, _ in struct.iter_unpack("=qf", data)]
    return [ids[start : start + 10] for start in range(0, len(ids), 10)]


def same_recovery(runs):
    assert len({run["snapshot_sha256"] for run in runs}) == 1
    assert len({run["neighbors_sha256"] for run in runs}) == 1


class Experiment:
    def __init__(self, args):
        self.output = args.output.resolve()
        self.output.mkdir()
        self.benchmark = args.benchmark.resolve()
        self.library = args.library.resolve()
        links = subprocess.check_output(["ldd", str(self.benchmark)], text=True)
        matches = [row.split()[2] for row in links.splitlines() if "libvsag-lite.so =>" in row]
        if len(matches) != 1 or Path(matches[0]).resolve() != self.library:
            raise ValueError("benchmark executable resolves a different Lite shared library")
        self.prefix = [] if args.cpu is None else ["taskset", "-c", str(args.cpu)]
        self.cpu = args.cpu
        self.inputs = {"SIFT": args.sift.resolve(), "GIST": args.gist.resolve()}

    def command(self, *argv):
        subprocess.run([*self.prefix, str(self.benchmark), *map(str, argv)], check=True)

    def run(self, dataset, count, dim, mode, variant, adds, repeat, seed, scratch):
        with tempfile.TemporaryDirectory(prefix=f"{variant}-", dir=scratch) as folder:
            directory = Path(folder)
            data = self.inputs[dataset]
            self.command("write", variant, seed, directory, data / "base.fvecs", adds)
            self.command("read", variant, directory, data / "queries.fvecs")
            samples = csv_rows(directory / "adds.csv")
            queries = csv_rows(directory / "queries.csv")
            recovery, = csv_rows(directory / "recovery.csv")
            assert [int(row["id"]) for row in samples] == list(range(count, count + adds))
            assert [int(row["row"]) for row in queries] == list(range(100))
            assert int(recovery["count"]) == count + adds
            bytes_written = [int(row["bytes"]) for row in samples]
            if variant == "journal":
                assert bytes_written == [12 + 4 * dim] * adds
                assert (directory / "store" / "add.log").stat().st_size == 32 + sum(bytes_written)
                assert digest(directory / "store" / "checkpoint.vsag") == digest(seed)
            elif variant == "memory":
                assert bytes_written == [0] * adds
            else:
                assert all(size > 0 for size in bytes_written)
            run = {
                "dataset": dataset,
                "count": count,
                "dimension": dim,
                "mode": mode,
                "variant": variant,
                "adds": adds,
                "repeat": repeat,
                "add_us": [float(row["wall_us"]) for row in samples],
                "add_bytes": bytes_written,
                "query_us": [float(row["wall_us"]) for row in queries],
                "recovery_us": float(recovery["load_us"]),
                "recovery_peak_rss_kib": int(recovery["peak_rss_kib"]),
                "seed_sha256": digest(seed),
                "snapshot_sha256": digest(directory / "state.vsag"),
                "neighbors_sha256": digest(directory / "neighbors.bin"),
            }
            print(dataset, count, mode, variant, adds, repeat, flush=True)
            return run, neighbor_ids(directory / "neighbors.bin")

    def paired(self):
        raw, ordered = [], {}
        for dataset, count, dim in (("SIFT", 1000, 128), ("SIFT", 10000, 128), ("GIST", 10000, 960)):
            for mode in STORAGES:
                with tempfile.TemporaryDirectory(prefix="paired-", dir=self.output) as folder:
                    scratch = Path(folder)
                    seed = scratch / "seed.vsag"
                    self.command("seed", self.inputs[dataset] / "base.fvecs", count, seed, mode)
                    cell = []
                    for repeat in range(REPEATS):
                        order = VARIANTS[repeat:] + VARIANTS[:repeat]
                        for variant in order:
                            run, ids = self.run(dataset, count, dim, mode, variant, 32, repeat, seed, scratch)
                            cell.append(run)
                    same_recovery(cell)
                    ordered[f"{dataset}-{count}-{mode}"] = ids
                    raw.extend(cell)
        return raw, ordered

    def replay_scaling(self):
        raw = []
        for mode in ("fp32", "rabitq8"):
            with tempfile.TemporaryDirectory(prefix="replay-", dir=self.output) as folder:
                scratch = Path(folder)
                seed = scratch / "seed.vsag"
                self.command("seed", self.inputs["SIFT"] / "base.fvecs", 1000, seed, mode)
                for adds in (0, 128, 1024):
                    cell = []
                    for repeat in range(REPEATS):
                        for variant in ("journal", "memory"):
                            run, _ = self.run("SIFT", 1000, 128, mode, variant, adds, repeat, seed, scratch)
                            cell.append(run)
                    same_recovery(cell)
                    raw.extend(cell)
        return raw


def summarize(rows):
    output = {}
    key_names = ("dataset", "count", "mode", "adds", "variant")
    groups = {}
    for row in rows:
        key = "/".join(str(row[name]) for name in key_names)
        groups.setdefault(key, []).append(row)
    for key, runs in groups.items():
        assert len(runs) == REPEATS
        add_rows = [row["add_us"] for row in runs if row["add_us"]]
        cell = {
            "write_bytes": statistics.median(sum(row["add_bytes"]) for row in runs),
            "recovery_us": statistics.median(row["recovery_us"] for row in runs),
            "recovery_peak_rss_kib": statistics.median(row["recovery_peak_rss_kib"] for row in runs),
            "query_p50_us": statistics.median(statistics.median(row["query_us"]) for row in runs),
            "query_p99_us": statistics.median(percentile(row["query_us"], .99) for row in runs),
        }
        if add_rows:
            cell.update(
                add_p50_us=statistics.median(statistics.median(row) for row in add_rows),
                add_p99_us=statistics.median(percentile(row, .99) for row in add_rows),
                add_per_second=statistics.median(len(row) * 1e6 / sum(row) for row in add_rows),
            )
        output[key] = cell
    return output


def recalls(ordered):
    result = {}
    for cell, ids in ordered.items():
        prefix, _ = cell.rsplit("-", 1)
        truth = ordered[f"{prefix}-bruteforce"]
        result[cell] = sum(len(set(a) & set(b)) for a, b in zip(ids, truth)) / 1000
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--sift", type=Path, required=True)
    parser.add_argument("--gist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()

    run = Experiment(args)
    before = active_builds()
    paired, ordered = run.paired()
    scaling = run.replay_scaling()
    assert len(paired) == 108 and len(scaling) == 36
    metadata = {
        "platform": platform.platform(),
        "cpu_affinity": run.cpu,
        "filesystem": subprocess.check_output(
            ["findmnt", "-no", "FSTYPE", "-T", str(run.output)], text=True
        ).strip(),
        "background_start": before,
        "background_end": active_builds(),
        "library_sha256": digest(run.library),
        "benchmark_sha256": digest(run.benchmark),
        "inputs": {
            dataset: {name: digest(directory / name) for name in ("base.fvecs", "queries.fvecs")}
            for dataset, directory in run.inputs.items()
        },
    }
    summary = {
        "status": "PASS",
        "metadata": metadata,
        "paired_runs": len(paired),
        "scaling_runs": len(scaling),
        "recall_at_10": recalls(ordered),
        "paired": summarize(paired),
        "replay_scaling": summarize(scaling),
    }
    (run.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    raw = json.dumps({"paired": paired, "scaling": scaling, "ordered_top10": ordered}, separators=(",", ":"))
    (run.output / "raw.json.gz").write_bytes(gzip.compress(raw.encode(), mtime=0))
    print(f"PASS {len(paired)} paired runs and {len(scaling)} replay-scaling runs", flush=True)


if __name__ == "__main__":
    main()
