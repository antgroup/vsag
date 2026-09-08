#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


STAGE_LABELS = {
    "initial_100pct": "Initial 100%",
    "delete_10pct": "Delete 10%",
    "delete_20pct": "Delete 20%",
    "add_back_10pct": "Add back 10%",
    "add_back_20pct": "Add back 20%",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot MCI mutation QPS-recall curves")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows_by_stage = defaultdict(list)
    with args.csv_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            rows_by_stage[row["stage"]].append(row)

    if not rows_by_stage:
        raise RuntimeError(f"no benchmark rows found in {args.csv_path}")

    _, axis = plt.subplots(figsize=(9, 6))
    for stage, rows in rows_by_stage.items():
        rows.sort(key=lambda row: int(row["ef_search"]))
        recalls = [float(row["recall_at_k"]) for row in rows]
        qps_values = [float(row["qps"]) for row in rows]
        axis.plot(
            recalls,
            qps_values,
            marker="o",
            linewidth=1.8,
            label=STAGE_LABELS.get(stage, stage),
        )
        for recall, qps, row in zip(recalls, qps_values, rows):
            axis.annotate(
                f"ef={row['ef_search']}",
                (recall, qps),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    axis.set_xlabel("Recall@k")
    axis.set_ylabel("QPS")
    axis.set_yscale("log")
    axis.set_title("HGraph/NSW + MCI mutation benchmark")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    print(f"plot written to {args.output}")


if __name__ == "__main__":
    main()
