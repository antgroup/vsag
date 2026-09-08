#!/usr/bin/env python3
"""Controlled MCI mutation threshold experiments; see the canonical MCI report."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import re
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
STAGES = (
    "initial_100pct", "delete_10pct", "delete_20pct",
    "add_back_10pct", "add_back_20pct",
)
BASELINE = {
    "join_ratio": 0.6, "added_mct": 3, "clique_max": 50,
    "delete_size": 3, "delete_mct": 3,
}
FLAGS = {
    "join_ratio": "--mci-incremental-join-ratio-threshold",
    "added_mct": "--mci-incremental-added-mct",
    "clique_max": "--mci-incremental-clique-max",
    "delete_size": "--mci-delete-clique-size-threshold",
    "delete_mct": "--mci-delete-node-mct-threshold",
}
PARAM_COLUMNS = dict(zip(BASELINE, (
    "incremental_join_ratio", "incremental_added_mct", "incremental_clique_max",
    "delete_clique_size_threshold", "delete_node_mct_threshold",
)))


def numbers(value, convert, low, high=None):
    values = list(dict.fromkeys(convert(token.strip()) for token in value.split(",")))
    if not values or any(not math.isfinite(x) or x < low or
                         (high is not None and x > high) for x in values):
        raise ValueError(f"invalid numeric list: {value}")
    return values


def configurations(design, ranges):
    candidates = [dict(BASELINE)]
    if design == "oat":
        for key, values in ranges.items():
            candidates.extend({**BASELINE, key: value} for value in values)
    else:
        candidates.extend(dict(zip(ranges, values))
                          for values in itertools.product(*ranges.values()))
    result, seen = [], set()
    for config in candidates:
        signature = tuple(config[key] for key in BASELINE)
        if signature in seen:
            continue
        seen.add(signature)
        changes = [(k, v) for k, v in config.items() if v != BASELINE[k]]
        name = "baseline" if not changes else "__".join(f"{k}-{v:g}" for k, v in changes)
        result.append({"id": name, "parameters": config})
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path,
                        default=REPO / "build-release/tools/eval/mci_mutation_benchmark")
    parser.add_argument("--dataset", type=Path,
                        default=Path("/root/data/codefilter-10k-384-angular-f32.hdf5"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--design", choices=("oat", "grid"), default="oat")
    parser.add_argument("--join-ratios", default="0.4,0.6,0.8")
    parser.add_argument("--added-mcts", default="1,3,6")
    parser.add_argument("--clique-maxes", default="25,50,100")
    parser.add_argument("--delete-sizes", default="3,4,5,6")
    parser.add_argument("--delete-mcts", default="3,5,8")
    parser.add_argument("--only", help="Comma-separated configuration IDs from --dry-run")
    parser.add_argument("--remove-mode", choices=("force", "mark"), default="force")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-runs", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=20260907)
    parser.add_argument("--ef-search-values", default="40,80,160,320")
    parser.add_argument("--target-recalls", default="0.95,0.98,0.99")
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--search-count", type=int, default=10000)
    parser.add_argument("--mutation-batch-size", type=int, default=1000)
    parser.add_argument("--build-threads", type=int, default=1)
    parser.add_argument("--search-threads", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--flush-after-mutation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.ranges = {
            "join_ratio": numbers(args.join_ratios, float, 0, 1),
            "added_mct": numbers(args.added_mcts, int, 1),
            "clique_max": numbers(args.clique_maxes, int, 2),
            "delete_size": numbers(args.delete_sizes, int, 1),
            "delete_mct": numbers(args.delete_mcts, int, 1),
        }
        args.efs = numbers(args.ef_search_values, int, 1)
        args.targets = numbers(args.target_recalls, float, 0, 1)
        for name in ("repeats", "max_runs", "search_count", "mutation_batch_size",
                     "build_threads", "search_threads"):
            if getattr(args, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if args.query_count < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("query_count must be nonnegative and timeout finite/positive")
        if args.random_seed < 0 or args.random_seed + args.repeats >= 2**64:
            raise ValueError("random seeds must fit uint64")
    except ValueError as error:
        parser.error(str(error))
    return args


def command(args, config, repeat, output):
    result = [str(args.binary.resolve()), "--dataset", str(args.dataset.resolve()),
              "--output", str(output), "--ef-search-values", ",".join(map(str, args.efs)),
              "--query-count", str(args.query_count), "--search-count", str(args.search_count),
              "--mutation-batch-size", str(args.mutation_batch_size),
              "--build-threads", str(args.build_threads),
              "--search-threads", str(args.search_threads),
              "--random-seed", str(args.random_seed + repeat)]
    for key, flag in FLAGS.items():
        result.extend([flag, str(config["parameters"][key])])
    if args.remove_mode == "force":
        result.append("--force-remove")
    if args.flush_after_mutation:
        result.append("--flush-after-mutation")
    return result


def validate_rows(rows, efs, mode, parameters):
    by_key = {(r["stage"], int(r["ef_search"])): r for r in rows}
    expected = set(itertools.product(STAGES, efs))
    if len(rows) != len(expected) or set(by_key) != expected:
        raise ValueError("incomplete, duplicate, or unexpected stage/ef rows")
    first = [by_key[stage, efs[0]] for stage in STAGES]
    counts = [int(r["active_vectors"]) for r in first]
    n, deleted = counts[0], counts[0] - counts[1]
    if deleted <= 0 or counts != [n, n-deleted, n-2*deleted, n-deleted, n]:
        raise ValueError("invalid five-stage live counts")
    for stage_index, stage in enumerate(STAGES):
        live = counts[stage_index]
        slots = live if mode == "force" else n + max(0, stage_index-2)*deleted
        for ef in efs:
            row = by_key[stage, ef]
            if row["remove_mode"] != mode:
                raise ValueError("wrong removal mode")
            for key, value in (("active_vectors", live), ("index_elements", live),
                               ("mci_total_nodes", slots), ("mci_covered_nodes", live),
                               ("inactive_nodes", slots-live)):
                if int(row[key]) != value:
                    raise ValueError(f"{stage}: invalid {key}")
            for key in ("qps", "recall_at_k", "avg_hops", "avg_dist_cmp", "avg_seed_count",
                        "mci_avg_clique_size", "mci_total_memberships", "index_memory_bytes"):
                if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                    raise ValueError(f"invalid metric: {key}")
            if float(row["qps"]) <= 0 or float(row["recall_at_k"]) > 1:
                raise ValueError("invalid QPS/recall")
            if any(float(row[key]) != 1 for key in ("mci_route_ratio", "mci_raw_float_ratio")):
                raise ValueError("mixed/fallback routes invalidate this MCI comparison")
            for key, column in PARAM_COLUMNS.items():
                if not math.isclose(float(row[column]), parameters[key], abs_tol=1e-6):
                    raise ValueError(f"benchmark parameter mismatch: {key}")
    return rows


def read_rows(path, args, config):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    return validate_rows(rows, args.efs, args.remove_mode, config["parameters"])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(path, digest=False):
    path = path.resolve()
    stat = path.stat()
    value = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if digest:
        sha = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024*1024), b""):
                sha.update(block)
        value["sha256"] = sha.hexdigest()
    return value


def run_trial(args, root, config, repeat):
    trial = root / config["id"] / f"repeat-{repeat+1}"
    trial.mkdir(parents=True, exist_ok=True)
    completed = trial / "completed.json"
    if completed.exists():
        attempt = json.loads(completed.read_text())["attempt"]
        # Do not trust a resume marker without validating the stored measurements.
        return read_rows(trial / attempt / "curve.csv", args, config)
    attempt_number = 1
    while (trial / f"attempt-{attempt_number}").exists():
        attempt_number += 1
    attempt = trial / f"attempt-{attempt_number}"
    attempt.mkdir()
    cmd = command(args, config, repeat, attempt / "curve.csv")
    write_json(attempt / "command.json", cmd)
    start = time.monotonic()
    try:
        with (attempt / "benchmark.log").open("w") as log:
            process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=args.timeout, check=False)
        if process.returncode:
            raise RuntimeError(f"benchmark exited {process.returncode}; see {attempt}")
        rows = read_rows(attempt / "curve.csv", args, config)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        write_json(attempt / "status.json", {"ok": False, "error": str(error),
                   "elapsed_seconds": time.monotonic()-start})
        raise RuntimeError(str(error)) from error
    write_json(attempt / "status.json", {"ok": True,
               "elapsed_seconds": time.monotonic()-start})
    write_json(completed, {"attempt": attempt.name})
    return rows


def aggregate(runs, targets, repeats):
    raw, groups, target_groups = [], defaultdict(list), defaultdict(list)
    for config_id, repeat, rows in runs:
        baseline = {int(r["ef_search"]): float(r["recall_at_k"])
                    for r in rows if r["stage"] == STAGES[0]}
        for row in rows:
            tagged = {"config_id": config_id, "repeat": repeat+1, **row}
            tagged["recall_delta_pp"] = 100*(float(row["recall_at_k"])-
                                             baseline[int(row["ef_search"])])
            tagged["memberships_per_live_node"] = (float(row["mci_total_memberships"])/
                                                    int(row["active_vectors"]))
            raw.append(tagged)
            groups[config_id, row["stage"], int(row["ef_search"])].append(tagged)
        for stage in STAGES:
            stage_rows = [r for r in rows if r["stage"] == stage]
            for target in targets:
                passing = [r for r in stage_rows if float(r["recall_at_k"]) >= target]
                best = max(passing, key=lambda r: float(r["qps"])) if passing else None
                target_groups[config_id, stage, target].append(best)
    summary = []
    for (config_id, stage, ef), values in sorted(groups.items()):
        item = {"config_id": config_id, "stage": stage, "ef_search": ef,
                "runs": len(values), "planned_runs": repeats}
        for key in ("recall_at_k", "recall_delta_pp", "qps", "avg_dist_cmp", "avg_hops",
                    "avg_seed_count", "mci_avg_clique_size", "memberships_per_live_node",
                    "index_memory_bytes", "mutation_seconds", "mci_delta_cliques",
                    "mci_delta_extra_memberships"):
            sample = [float(r[key]) for r in values]
            item[f"{key}_median"] = statistics.median(sample)
            if key in ("recall_at_k", "recall_delta_pp", "qps"):
                item[f"{key}_min"], item[f"{key}_max"] = min(sample), max(sample)
        summary.append(item)
    target_summary = []
    for (config_id, stage, target), values in sorted(target_groups.items()):
        passing = [r for r in values if r is not None]
        target_summary.append({
            "config_id": config_id, "stage": stage, "target_recall": target,
            "runs": len(values), "planned_runs": repeats, "reached_runs": len(passing),
            "qps_median_if_all_reached": statistics.median(float(r["qps"]) for r in passing)
            if len(passing) == repeats else "",
            "selected_efs": ",".join(str(r["ef_search"]) if r else "unreached" for r in values),
        })
    return raw, summary, target_summary


def main(argv=None):
    args = parse_args(argv)
    configs = configurations(args.design, args.ranges)
    if args.only:
        selected = set(args.only.split(","))
        known = {c["id"] for c in configs}
        if selected - known:
            raise ValueError(f"unknown configurations: {sorted(selected-known)}")
        configs = [c for c in configs if c["id"] in selected]
    total = len(configs)*args.repeats
    if args.dry_run:
        print(json.dumps({"configurations": configs, "runs": total,
                          "curve_points": total*len(STAGES)*len(args.efs)}, indent=2))
        return 0
    if total > args.max_runs:
        raise ValueError(f"{total} runs exceed --max-runs={args.max_runs}; inspect --dry-run first")
    if args.resume and args.output_dir is None:
        raise ValueError("--resume requires --output-dir")
    help_result = subprocess.run([str(args.binary.resolve()), "--help"], capture_output=True,
                                 text=True, timeout=30, check=False)
    if any(flag not in help_result.stdout + help_result.stderr for flag in FLAGS.values()):
        raise ValueError("benchmark binary lacks threshold options; rebuild it first")
    linkage = subprocess.run(["ldd", str(args.binary.resolve())], capture_output=True,
                             text=True, timeout=30, check=False)
    libraries = re.findall(r"libvsag\S*\s+=>\s+(\S+)", linkage.stdout)
    manifest = {
        "schema": 1, "binary": fingerprint(args.binary, True),
        "vsag_libraries": [fingerprint(Path(p), True) for p in libraries],
        "dataset": fingerprint(args.dataset), "configurations": configs,
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                       if k not in ("output_dir", "resume", "dry_run", "max_runs", "timeout")},
    }
    root = args.output_dir.resolve() if args.output_dir else Path(
        tempfile.mkdtemp(prefix="mci-threshold-sweep-"))
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("resume manifest differs: parameters, data, or binary changed")
    elif any(root.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output directory: {root}")
    else:
        write_json(manifest_path, manifest)
    runs, failures = [], []
    print(f"Output: {root}\nConfigurations: {len(configs)}, runs: {total}", flush=True)
    for repeat in range(args.repeats):
        ordered = list(configs)
        random.Random(args.random_seed+repeat).shuffle(ordered)
        for config in ordered:
            print(f"[{len(runs)+len(failures)+1}/{total}] repeat={repeat+1} {config['id']}",
                  flush=True)
            try:
                rows = run_trial(args, root, config, repeat)
                runs.append((config["id"], repeat, rows))
            except RuntimeError as error:
                failures.append({"config_id": config["id"], "repeat": repeat+1,
                                 "error": str(error)})
                print(f"FAILED: {error}", flush=True)
            raw, summary, targets = aggregate(runs, args.targets, args.repeats)
            write_csv(root / "all.csv", raw)
            write_csv(root / "summary.csv", summary)
            write_csv(root / "target-recall.csv", targets)
            write_json(root / "failures.json", failures)
    write_json(root / "status.json", {"successful_runs": len(runs), "failed_runs": len(failures),
                                      "planned_runs": total})
    print(f"Finished: {len(runs)} successful, {len(failures)} failed. Results: {root}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
