#!/usr/bin/env python3

"""Merge quantizer and end-to-end raw JSON into auditable CSV files."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FIELDS = [
    "dataset",
    "method",
    "ef_search",
    "target_recall",
    "target_met",
    "code_size_bytes_per_vector",
    "train_ms",
    "encode_ms",
    "encode_vectors_per_second",
    "distance_mse",
    "distance_relative_mse",
    "distance_mean_relative_error",
    "distance_pair_protocol",
    "distance_pair_count",
    "query_setup_and_single_distance_ms",
    "query_setup_count",
    "query_setup_and_distance_total_ms",
    "distance_scan_count",
    "distance_scan_ms",
    "distance_scans_per_second",
    "distance_batch4_scan_count",
    "distance_batch4_scan_ms",
    "distance_batch4_scans_per_second",
    "code_pair_supported",
    "code_pair_count",
    "code_pair_ms",
    "code_pairs_per_second",
    "build_duration_s",
    "index_serialized_size_bytes",
    "index_memory_bytes",
    "build_peak_memory",
    "dataset_query_count",
    "timed_query_count",
    "successful_query_count",
    "timed_repetitions_per_query",
    "qps",
    "latency_p50_ms",
    "latency_p99_ms",
    "recall_at_10",
]

ABLATION_FIELDS = [
    "dataset",
    "method",
    "parameters",
    "trained_segments",
    "code_size_bytes_per_vector",
    "train_ms",
    "encode_ms",
    "encode_vectors_per_second",
    "distance_mse",
    "distance_relative_mse",
    "distance_mean_relative_error",
    "distance_pair_protocol",
    "distance_pair_count",
    "query_setup_and_single_distance_ms",
    "query_setup_count",
    "query_setup_and_distance_total_ms",
    "distance_scan_count",
    "distance_scan_ms",
    "distance_scans_per_second",
    "distance_batch4_scan_count",
    "distance_batch4_scan_ms",
    "distance_batch4_scans_per_second",
    "code_pair_supported",
    "code_pair_count",
    "code_pair_ms",
    "code_pairs_per_second",
]


def value_at(document: dict[str, Any], *path: str) -> Any:
    value: Any = document
    for component in path:
        if not isinstance(value, dict):
            return ""
        value = value.get(component, "")
    return value


def parse_common_recall_target(value: str) -> tuple[str, float]:
    dataset, separator, recall_text = value.partition("=")
    if not separator or not dataset:
        raise argparse.ArgumentTypeError(
            "common recall targets must use DATASET=RECALL syntax"
        )
    try:
        recall = float(recall_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid recall target: {recall_text}"
        ) from error
    if not 0.0 <= recall <= 1.0:
        raise argparse.ArgumentTypeError("common recall targets must be in [0, 1]")
    return dataset, recall


def load_rows(dataset_dir: Path, target_recall: float) -> list[dict[str, Any]]:
    quant_path = dataset_dir / "quantization.json"
    eval_path = dataset_dir / "eval.json"
    if not quant_path.is_file() or not eval_path.is_file():
        return []

    quant = json.loads(quant_path.read_text(encoding="utf-8"))
    has_precise_query_timing = quant.get("benchmark_schema_version", 1) >= 2
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    quant_methods = {entry["name"]: entry for entry in quant["methods"]}
    rows: list[dict[str, Any]] = []
    for case_name, search in evaluation.items():
        match = re.fullmatch(r"(saq|rabitq)_ef(\d+)", case_name)
        if match is None:
            continue
        method, ef_search = match.groups()
        build = evaluation.get(f"{method}_build", {})
        quant_result = quant_methods.get(method, {})
        recall = value_at(search, "recall_avg")
        dataset_query_count = value_at(search, "dataset_info", "query_count")
        timed_query_count = search.get("measurement_sample_count", "")
        successful_query_count = search.get("measurement_successful_query_count", "")
        repetitions_per_query: Any = ""
        if (
            isinstance(dataset_query_count, int)
            and dataset_query_count > 0
            and isinstance(timed_query_count, int)
        ):
            repetitions_per_query = timed_query_count / dataset_query_count
            if repetitions_per_query.is_integer():
                repetitions_per_query = int(repetitions_per_query)
        row = {
            "dataset": dataset_dir.name,
            "method": method,
            "ef_search": int(ef_search),
            "target_recall": target_recall,
            "target_met": isinstance(recall, (float, int)) and recall >= target_recall,
            "code_size_bytes_per_vector": quant_result.get(
                "code_size_bytes_per_vector", ""
            ),
            "train_ms": quant_result.get("train_ms", ""),
            "encode_ms": quant_result.get("encode_ms", ""),
            "encode_vectors_per_second": quant_result.get(
                "encode_vectors_per_second", ""
            ),
            "distance_mse": quant_result.get("distance_mse", ""),
            "distance_relative_mse": quant_result.get("distance_relative_mse", ""),
            "distance_mean_relative_error": quant_result.get(
                "distance_mean_relative_error", ""
            ),
            "distance_pair_protocol": quant_result.get(
                "distance_pair_protocol", ""
            ),
            "distance_pair_count": quant_result.get("distance_pair_count", ""),
            "query_setup_and_single_distance_ms": quant_result.get(
                "query_setup_and_single_distance_ms", ""
            )
            if has_precise_query_timing
            else "",
            "query_setup_count": quant_result.get("query_setup_count", "")
            if has_precise_query_timing
            else "",
            "query_setup_and_distance_total_ms": quant_result.get(
                "query_setup_and_distance_total_ms", ""
            )
            if has_precise_query_timing
            else "",
            "distance_scan_count": quant_result.get("distance_scan_count", ""),
            "distance_scan_ms": quant_result.get("distance_scan_ms", ""),
            "distance_scans_per_second": quant_result.get(
                "distance_scans_per_second", ""
            ),
            "distance_batch4_scan_count": quant_result.get(
                "distance_batch4_scan_count", ""
            ),
            "distance_batch4_scan_ms": quant_result.get(
                "distance_batch4_scan_ms", ""
            ),
            "distance_batch4_scans_per_second": quant_result.get(
                "distance_batch4_scans_per_second", ""
            ),
            "code_pair_supported": quant_result.get("code_pair_supported", ""),
            "code_pair_count": quant_result.get("code_pair_count", ""),
            "code_pair_ms": quant_result.get("code_pair_ms", ""),
            "code_pairs_per_second": quant_result.get("code_pairs_per_second", ""),
            "build_duration_s": build.get("duration(s)", ""),
            "index_serialized_size_bytes": build.get("index_serialized_size(B)", ""),
            "index_memory_bytes": build.get("index_memory(B)", ""),
            "build_peak_memory": build.get("memory_peak(build)", ""),
            "dataset_query_count": dataset_query_count,
            "timed_query_count": timed_query_count,
            "successful_query_count": successful_query_count,
            "timed_repetitions_per_query": repetitions_per_query,
            "qps": search.get("qps", ""),
            "latency_p50_ms": value_at(search, "latency_detail(ms)", "p50"),
            "latency_p99_ms": value_at(search, "latency_detail(ms)", "p99"),
            "recall_at_10": recall,
        }
        rows.append(row)
    return rows


def load_ablation_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    quant_path = dataset_dir / "quantization.json"
    if not quant_path.is_file():
        return []
    quant = json.loads(quant_path.read_text(encoding="utf-8"))
    has_precise_query_timing = quant.get("benchmark_schema_version", 1) >= 2
    rows: list[dict[str, Any]] = []
    for result in quant.get("methods", []):
        if not result.get("name", "").startswith("saq"):
            continue
        row = {field: result.get(field, "") for field in ABLATION_FIELDS}
        row["dataset"] = dataset_dir.name
        row["method"] = result.get("name", "")
        row["parameters"] = json.dumps(result.get("parameters", {}), sort_keys=True)
        row["trained_segments"] = json.dumps(
            result.get("trained_segments", []), sort_keys=True
        )
        if not has_precise_query_timing:
            row["query_setup_and_single_distance_ms"] = ""
            row["query_setup_and_distance_total_ms"] = ""
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] = FIELDS) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument(
        "--common-recall-target",
        action="append",
        default=[],
        type=parse_common_recall_target,
        metavar="DATASET=RECALL",
        help="select the fastest measured row reaching a declared dataset target",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    for dataset_dir in sorted(path for path in args.result_root.iterdir() if path.is_dir()):
        rows.extend(load_rows(dataset_dir, args.target_recall))
        ablation_rows.extend(load_ablation_rows(dataset_dir))
    rows.sort(key=lambda row: (row["dataset"], row["method"], row["ef_search"]))
    write_csv(args.result_root / "all_results.csv", rows)
    ablation_rows.sort(key=lambda row: (row["dataset"], row["method"]))
    write_csv(
        args.result_root / "ablation_results.csv", ablation_rows, ABLATION_FIELDS
    )

    selected: list[dict[str, Any]] = []
    groups = {(row["dataset"], row["method"]) for row in rows}
    for group in sorted(groups):
        candidates = [row for row in rows if (row["dataset"], row["method"]) == group]
        meeting_target = [row for row in candidates if row["target_met"]]
        if meeting_target:
            best = min(meeting_target, key=lambda row: float(row["latency_p50_ms"]))
        else:
            best = max(candidates, key=lambda row: float(row["recall_at_10"]))
        selected.append(best)
    write_csv(args.result_root / "target_recall_results.csv", selected)

    common_selected: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        method_rows = {
            method: [
                row
                for row in rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            for method in ("saq", "rabitq")
        }
        if any(not candidates for candidates in method_rows.values()):
            continue
        common_recall = min(
            max(float(row["recall_at_10"]) for row in candidates)
            for candidates in method_rows.values()
        )
        for method in ("saq", "rabitq"):
            candidates = [
                row
                for row in method_rows[method]
                if float(row["recall_at_10"]) >= common_recall
            ]
            best = min(candidates, key=lambda row: float(row["latency_p50_ms"]))
            comparable = dict(best)
            comparable["target_recall"] = common_recall
            comparable["target_met"] = True
            common_selected.append(comparable)
    write_csv(args.result_root / "common_recall_results.csv", common_selected)

    reported_common_selected: list[dict[str, Any]] = []
    for dataset, common_recall in sorted(dict(args.common_recall_target).items()):
        for method in ("saq", "rabitq"):
            candidates = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["method"] == method
                and float(row["recall_at_10"]) >= common_recall
            ]
            if not candidates:
                raise ValueError(
                    f"{method} has no measured {dataset} row reaching {common_recall}"
                )
            best = min(candidates, key=lambda row: float(row["latency_p50_ms"]))
            comparable = dict(best)
            comparable["target_recall"] = common_recall
            comparable["target_met"] = True
            reported_common_selected.append(comparable)
    if reported_common_selected:
        write_csv(
            args.result_root / "reported_common_recall_results.csv",
            reported_common_selected,
        )


if __name__ == "__main__":
    main()
