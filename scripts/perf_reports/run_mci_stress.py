#!/usr/bin/env python3
"""Run random MCI churn and refresh file-based statistical curves after each checkpoint."""

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

import sweep_mci_thresholds as sweep


def read_completed(root):
    events_path = root / "curve.csv.events.csv"
    curve_path = root / "curve.csv"
    if not events_path.exists() or not curve_path.exists():
        return [], []
    with events_path.open(newline="") as source:
        events = [r for r in csv.DictReader(source) if all(v is not None for v in r.values())]
    stages = {r["stage"] for r in events}
    with curve_path.open(newline="") as source:
        rows = [r for r in csv.DictReader(source) if r["stage"] in stages and
                all(v is not None for v in r.values())]
    return events, rows


def validate(events, rows, args):
    if not events or events[0]["operation"] != "build":
        raise ValueError("missing initial checkpoint")
    if int(events[-1]["round"]) != args.rounds:
        raise ValueError("incomplete stress rounds")
    if int(events[0]["active_vectors"]) != args.initial_count:
        raise ValueError("wrong initial count")
    expected = {(r["stage"], ef) for r in events for ef in args.efs}
    actual = {(r["stage"], int(r["ef_search"])) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("missing or duplicated search measurements")
    live, round_counts = args.initial_count, {}
    for event in events[1:]:
        count = int(event["count"])
        operation = event["operation"]
        round_id = int(event["round"])
        if operation not in ("add", "delete") or count <= 0:
            raise ValueError("invalid mutation event")
        if args.mode == "alternate" and operation != ("add" if round_id % 2 else "delete"):
            raise ValueError("incorrect alternate operation")
        live += count if operation == "add" else -count
        round_counts[round_id] = round_counts.get(round_id, 0) + count
        if live != int(event["active_vectors"]):
            raise ValueError("event live-count mismatch")
    if round_counts != {r: args.step_count for r in range(1, args.rounds+1)}:
        raise ValueError("round sample-count mismatch")
    for row in rows:
        live = int(row["active_vectors"])
        if any(int(row[k]) != live for k in
               ("index_elements", "mci_total_nodes", "mci_covered_nodes")):
            raise ValueError("index live/physical/coverage mismatch")
        if int(row["inactive_nodes"]) != 0:
            raise ValueError("FORCE_REMOVE retained inactive nodes")
        if any(float(row[k]) != 1 for k in ("mci_route_ratio", "mci_raw_float_ratio")):
            raise ValueError("non-MCI or non-raw search route")
        if (not 0 <= float(row["recall_at_k"]) <= 1 or
                not math.isfinite(float(row["qps"])) or float(row["qps"]) <= 0):
            raise ValueError("invalid measurement")


def plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events, rows = read_completed(root)
    if not events or not rows:
        return 0
    stages = [r["stage"] for r in events]
    position = {stage: i for i, stage in enumerate(stages)}
    first_ef = min(int(r["ef_search"]) for r in rows)
    structural = [r for r in rows if int(r["ef_search"]) == first_ef]
    x = [position[r["stage"]] for r in structural]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(x, [int(r["active_vectors"]) for r in structural], "o-")
    axes[0].set_ylabel("Live vectors (count)")
    for ef in sorted({int(r["ef_search"]) for r in rows}):
        subset = [r for r in rows if int(r["ef_search"]) == ef]
        px = [position[r["stage"]] for r in subset]
        axes[1].plot(px, [float(r["recall_at_k"])*100 for r in subset], "o-", label=f"ef={ef}")
        axes[2].plot(px, [float(r["qps"]) for r in subset], "o-", label=f"ef={ef}")
    axes[1].set_ylabel("Recall@10 (%)")
    axes[2].set_ylabel("QPS (queries/s)")
    for key, label in (("index_memory_bytes", "Index total"),
                       ("vector_memory_bytes", "Vectors"),
                       ("graph_memory_bytes", "Graph"), ("mci_memory_bytes", "MCI")):
        axes[3].plot(x, [float(r[key])/2**20 for r in structural], "o-", label=label)
    axes[3].set_ylabel("Index accounting (MiB; excludes dataset/GT)")
    for key, label in (("mci_total_cliques", "Total"), ("mci_delta_cliques", "Delta")):
        axes[4].plot(x, [int(r[key]) for r in structural], "o-", label=label)
    axes[4].set_ylabel("Cliques (count)")
    for op in ("add", "delete"):
        subset = [r for r in events if r["operation"] == op]
        axes[5].plot([position[r["stage"]] for r in subset],
                     [float(r["mutation_seconds"]) for r in subset], "o-", label=op)
    axes[5].set_ylabel("Mutation time per checkpoint (s)")
    labels = ["Init" if r["operation"] == "build" else
              f"{r['round']}{'A' if r['operation']=='add' else 'D'}" for r in events]
    for i, axis in enumerate(axes):
        axis.set_xlabel("Checkpoint (D=delete, A=add)")
        axis.set_xticks(range(len(stages)), labels, rotation=60)
        axis.grid(alpha=0.25)
        if i:
            axis.legend()
    fig.suptitle("MCI random churn: completed checkpoints only")
    fig.savefig(root / "statistics.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    selected = sorted({round(i*(len(stages)-1)/min(4, len(stages)-1))
                       for i in range(min(4, len(stages)-1)+1)}) if len(stages)>1 else [0]
    for i in selected:
        subset = sorted((r for r in rows if r["stage"] == stages[i]),
                        key=lambda r: int(r["ef_search"]))
        axis.plot([100*float(r["recall_at_k"]) for r in subset],
                  [float(r["qps"]) for r in subset], "o-", label=stages[i])
    axis.set(xlabel="Recall@10 (%)", ylabel="QPS (queries/s)",
             title="QPS-recall: sampled completed checkpoints")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(root / "qps-recall.png", dpi=150)
    plt.close(fig)
    return len(events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("/root/data/codefilter-3m-384-angular-f32.hdf5"))
    parser.add_argument("--binary", type=Path,
                        default=sweep.REPO / "build-release/tools/eval/mci_mutation_benchmark")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-count", type=int, default=800000)
    parser.add_argument("--rounds", type=int, default=14)
    parser.add_argument("--step-count", type=int, default=0)
    parser.add_argument("--mode", choices=("toggle", "alternate"), default="toggle")
    parser.add_argument("--random-seed", type=int, default=20260908)
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--search-count", type=int, default=10000)
    parser.add_argument("--ef-search-values", default="40,80,160,320")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--delete-size", type=int, default=3)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if args.plot_only:
        print(f"Plotted {plot(root)} completed checkpoints in {root}")
        return 0
    import h5py
    with h5py.File(args.dataset, "r") as data:
        pool_count = len(data["train"])
    args.step_count = args.step_count or int(math.floor(pool_count/14 + 0.5))
    args.efs = sweep.numbers(args.ef_search_values, int, 1)
    if any(getattr(args, k) <= 0 for k in
           ("initial_count", "rounds", "step_count", "query_count", "search_count",
            "threads", "delete_size")):
        parser.error("counts, threads, and thresholds must be positive")
    if args.initial_count > pool_count or args.step_count > pool_count:
        parser.error("initial/step count exceeds the dataset")
    if not 0 <= args.random_seed < 2**64:
        parser.error("random seed must fit uint64")
    if args.mode == "alternate" and args.step_count > pool_count-args.initial_count:
        parser.error("not enough absent IDs for the first ADD")
    if root.exists() and any(root.iterdir()):
        parser.error("output directory is nonempty; refusing to overwrite")
    root.mkdir(parents=True, exist_ok=True)
    cmd = [str(args.binary.resolve()), "--dataset", str(args.dataset.resolve()),
           "--output", str(root / "curve.csv"), "--force-remove",
           "--stress-rounds", str(args.rounds), "--stress-initial-count", str(args.initial_count),
           "--stress-step-count", str(args.step_count), "--stress-mode", args.mode,
           "--random-seed", str(args.random_seed), "--query-count", str(args.query_count),
           "--search-count", str(args.search_count), "--ef-search-values", args.ef_search_values,
           "--build-threads", str(args.threads), "--search-threads", str(args.threads),
           "--mutation-batch-size", str(args.step_count),
           "--mci-delete-clique-size-threshold", str(args.delete_size)]
    sweep.write_json(root / "manifest.json", {
        "command": cmd, "parameters": {k: str(v) if isinstance(v, Path) else v
                                       for k, v in vars(args).items()},
        "pool_count": pool_count, "binary": sweep.fingerprint(args.binary, True),
        "dataset": sweep.fingerprint(args.dataset),
    })
    start = time.monotonic()
    print(f"Output: {root}; initial={args.initial_count}; step={args.step_count}; "
          f"rounds={args.rounds}; mode={args.mode}", flush=True)
    process = None
    plotted = 0
    try:
        with (root / "benchmark.log").open("w") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                events, _ = read_completed(root)
                if len(events) > plotted:
                    plotted = plot(root)
                    print(f"Completed checkpoints: {plotted}; curves updated", flush=True)
                time.sleep(5)
        if process.returncode:
            raise RuntimeError(f"benchmark failed with exit code {process.returncode}")
        events, rows = read_completed(root)
        validate(events, rows, args)
        plot(root)
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        sweep.write_json(root / "status.json", {"ok": False, "error": str(error),
                         "elapsed_seconds": time.monotonic()-start})
        raise
    sweep.write_json(root / "status.json", {"ok": True, "checkpoints": len(events),
                     "curve_points": len(rows), "elapsed_seconds": time.monotonic()-start})
    print(f"Finished: {len(events)} checkpoints, {len(rows)} curve points. {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
