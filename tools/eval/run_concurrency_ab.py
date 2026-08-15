#!/usr/bin/env python3
"""Run an alternating baseline/candidate concurrency evaluation.

The harness deliberately delegates search measurement to eval_performance. It
only creates one-case YAML files, runs the evaluator serially, and preserves
each evaluator JSON result in a machine-readable report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_CONCURRENCY = [1, 2, 4, 8, 16, 32]
VARIANTS = ("baseline", "candidate")
CASE_NAME = "concurrency_run"
DEFAULT_ACCEPTANCE = {
    "qps_min_pct": 15.0,
    "p99_max_regression_pct": 10.0,
    "recall_max_abs_change": 0.01,
    "target_concurrency": [8, 16, 32],
    "min_valid_pairs": 3,
}


class HarnessError(ValueError):
    """Raised when the harness input cannot produce a valid evaluation."""


def _json_object(value: Any, field: str) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise HarnessError(f"{field} must contain valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise HarnessError(f"{field} must be a JSON object")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise HarnessError(f"{field} must be a positive integer")
    if isinstance(value, float) and not value.is_integer():
        raise HarnessError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise HarnessError(f"{field} must be a positive integer") from error
    if result <= 0:
        raise HarnessError(f"{field} must be a positive integer")
    return result


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise HarnessError(f"{field} must be a non-negative integer")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise HarnessError(f"{field} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise HarnessError(f"{field} must be a non-negative integer") from error
    if result < 0:
        raise HarnessError(f"{field} must be a non-negative integer")
    return result


def _nonnegative_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise HarnessError(f"{field} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise HarnessError(f"{field} must be a finite non-negative number") from error
    if not math.isfinite(result) or result < 0:
        raise HarnessError(f"{field} must be a finite non-negative number")
    return result


def _normalize_acceptance(raw: Dict[str, Any]) -> Dict[str, Any]:
    value = raw.get("acceptance", {})
    if not isinstance(value, dict):
        raise HarnessError("acceptance must be a JSON object")

    target_concurrency = value.get(
        "target_concurrency", DEFAULT_ACCEPTANCE["target_concurrency"]
    )
    if not isinstance(target_concurrency, list) or not target_concurrency:
        raise HarnessError("acceptance.target_concurrency must be a non-empty JSON array")
    normalized_target_concurrency = [
        _positive_int(item, "acceptance.target_concurrency") for item in target_concurrency
    ]
    if len(set(normalized_target_concurrency)) != len(normalized_target_concurrency):
        raise HarnessError("acceptance.target_concurrency values must be unique")

    return {
        "qps_min_pct": _nonnegative_float(
            value.get("qps_min_pct", DEFAULT_ACCEPTANCE["qps_min_pct"]),
            "acceptance.qps_min_pct",
        ),
        "p99_max_regression_pct": _nonnegative_float(
            value.get(
                "p99_max_regression_pct", DEFAULT_ACCEPTANCE["p99_max_regression_pct"]
            ),
            "acceptance.p99_max_regression_pct",
        ),
        "recall_max_abs_change": _nonnegative_float(
            value.get("recall_max_abs_change", DEFAULT_ACCEPTANCE["recall_max_abs_change"]),
            "acceptance.recall_max_abs_change",
        ),
        "target_concurrency": normalized_target_concurrency,
        "min_valid_pairs": _positive_int(
            value.get("min_valid_pairs", DEFAULT_ACCEPTANCE["min_valid_pairs"]),
            "acceptance.min_valid_pairs",
        ),
    }


def _yaml_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _file_export_target(path: Path) -> str:
    # FileExporter strips the six-character file:// prefix before opening the path.
    absolute_path = path.expanduser().resolve()
    return "file://" + absolute_path.as_posix()


def _json_text(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_spec(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the JSON spec used by the harness."""
    required = ("datapath", "index_path", "index_name", "create_params", "search_params")
    missing = [key for key in required if key not in raw]
    if missing:
        raise HarnessError("missing required spec fields: " + ", ".join(missing))

    search_mode = raw.get("search_mode", "knn")
    if search_mode != "knn":
        raise HarnessError("search_mode must be 'knn'; the in-memory evaluator supports KNN here")

    concurrency = raw.get("concurrency", DEFAULT_CONCURRENCY)
    if not isinstance(concurrency, list) or not concurrency:
        raise HarnessError("concurrency must be a non-empty JSON array")
    normalized_concurrency = [_positive_int(value, "concurrency") for value in concurrency]
    if len(set(normalized_concurrency)) != len(normalized_concurrency):
        raise HarnessError("concurrency values must be unique")

    shared_index_path = raw.get("index_path")
    if not shared_index_path:
        raise HarnessError("index_path is required and must be shared by both variants")
    shared_index_path = str(Path(str(shared_index_path)).expanduser())

    variants: Dict[str, Dict[str, str]] = {}
    for variant in VARIANTS:
        value = raw.get(variant)
        if not isinstance(value, dict):
            raise HarnessError(f"{variant} must be an object with eval_binary and shared_library")
        expected_keys = {"eval_binary", "shared_library"}
        missing = expected_keys - set(value)
        extra = set(value) - expected_keys
        if missing or extra:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if extra:
                details.append("unexpected " + ", ".join(sorted(extra)))
            raise HarnessError(f"{variant} schema error: " + "; ".join(details))
        if not value["eval_binary"] or not value["shared_library"]:
            raise HarnessError(f"{variant}.eval_binary and {variant}.shared_library are required")
        variants[variant] = {
            "eval_binary": str(value["eval_binary"]),
            "shared_library": str(Path(str(value["shared_library"])).expanduser()),
        }

    datapath = str(Path(str(raw["datapath"])).expanduser())
    create_params = _json_object(raw["create_params"], "create_params")
    search_params = _json_object(raw["search_params"], "search_params")
    timeout_seconds = raw.get("timeout_seconds")
    if timeout_seconds is not None:
        try:
            timeout_seconds = float(timeout_seconds)
        except (TypeError, ValueError, OverflowError) as error:
            raise HarnessError("timeout_seconds must be positive") from error
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise HarnessError("timeout_seconds must be positive")

    hash_files = raw.get("hash_files", False)
    if not isinstance(hash_files, bool):
        raise HarnessError("hash_files must be boolean")

    set_immutable = raw.get("set_immutable", False)
    if not isinstance(set_immutable, bool):
        raise HarnessError("set_immutable must be boolean")

    warmup_query_count = _nonnegative_int(
        raw.get("warmup_query_count", 10000), "warmup_query_count"
    )

    acceptance = _normalize_acceptance(raw)

    return {
        "datapath": datapath,
        "index_path": shared_index_path,
        "index_name": str(raw["index_name"]),
        "create_params": create_params,
        "create_params_text": _json_text(create_params),
        "search_params": search_params,
        "search_params_text": _json_text(search_params),
        "search_mode": search_mode,
        "set_immutable": set_immutable,
        "warmup_query_count": warmup_query_count,
        "topk": _positive_int(raw.get("topk", 10), "topk"),
        "search_query_count": _positive_int(
            raw.get("search_query_count", 100000), "search_query_count"
        ),
        "concurrency": normalized_concurrency,
        "rounds": _positive_int(raw.get("rounds", 1), "rounds"),
        "timeout_seconds": timeout_seconds,
        "hash_files": hash_files,
        "acceptance": acceptance,
        "baseline": variants["baseline"],
        "candidate": variants["candidate"],
    }


def build_case_yaml(
    spec: Dict[str, Any],
    variant: str,
    concurrency: int,
    result_path: Path,
    case_name: str = CASE_NAME,
) -> str:
    """Build one eval_performance YAML with JSON exported to a temporary file."""
    if variant not in VARIANTS:
        raise HarnessError(f"unknown variant: {variant}")
    index_path = spec["index_path"]
    return "\n".join(
        [
            "global:",
            f"  num_threads_searching: {concurrency}",
            "  exporters:",
            "    raw:",
            '      format: "json"',
            f"      to: {_yaml_string(_file_export_target(result_path))}",
            f"{case_name}:",
            f"  datapath: {_yaml_string(spec['datapath'])}",
            '  type: "search"',
            f"  index_name: {_yaml_string(spec['index_name'])}",
            f"  create_params: {_yaml_string(spec['create_params_text'])}",
            f"  search_params: {_yaml_string(spec['search_params_text'])}",
            f"  index_path: {_yaml_string(index_path)}",
            f"  search_mode: {_yaml_string(spec['search_mode'])}",
            f"  set_immutable: {'true' if spec['set_immutable'] else 'false'}",
            f"  warmup_query_count: {spec['warmup_query_count']}",
            f"  topk: {spec['topk']}",
            f"  search_query_count: {spec['search_query_count']}",
            "  disable_memory: true",
            "",
        ]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_binary(binary: str) -> str:
    path = Path(binary).expanduser()
    if path.is_file():
        return str(path)
    resolved = shutil.which(binary)
    if resolved:
        return resolved
    raise HarnessError(f"eval binary was not found: {binary}")


def _realpath(path: Path, field: str) -> str:
    try:
        return str(path.expanduser().resolve(strict=True))
    except OSError as error:
        raise HarnessError(f"{field} does not resolve to a readable file: {path}") from error


def _variant_environment(shared_library_realpath: str) -> Dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"}
    }
    env["LD_LIBRARY_PATH"] = str(Path(shared_library_realpath).parent)
    return env


def _parse_ldd_libvsag(output: str, variant: str) -> str:
    resolved_paths = []
    for line in output.splitlines():
        if "libvsag.so" not in line:
            continue
        if "not found" in line:
            raise HarnessError(f"ldd could not resolve libvsag for {variant}: {line.strip()}")
        match = re.match(r"\s*libvsag\.so(?:\.[^\s]+)?\s*=>\s*(\S+)\s+\(", line)
        if match:
            resolved_paths.append(match.group(1))
    if len(resolved_paths) != 1:
        raise HarnessError(
            f"ldd did not resolve exactly one libvsag.so dependency for {variant}: "
            f"found {len(resolved_paths)}"
        )
    return resolved_paths[0]


def _preflight_variant(variant: str, binary: str, shared_library: str) -> Dict[str, Any]:
    binary_realpath = _realpath(Path(binary), f"{variant}.eval_binary")
    library_path = Path(shared_library).expanduser()
    if not library_path.name.startswith("libvsag.so"):
        raise HarnessError(
            f"{variant}.shared_library must be libvsag.so or a versioned libvsag.so file"
        )
    library_realpath = _realpath(library_path, f"{variant}.shared_library")
    library_env = _variant_environment(library_realpath)
    ldd_path = shutil.which("ldd")
    if not ldd_path:
        raise HarnessError("ldd is required for shared library preflight")
    try:
        completed = subprocess.run(
            [ldd_path, binary_realpath],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=library_env,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HarnessError(f"ldd preflight failed for {variant}: {error}") from error
    if completed.returncode != 0:
        raise HarnessError(
            f"ldd preflight failed for {variant} with status {completed.returncode}: "
            f"{_trim(completed.stderr)}"
        )
    resolved_library = _parse_ldd_libvsag(completed.stdout, variant)
    resolved_library_realpath = _realpath(Path(resolved_library), f"{variant} resolved libvsag")
    if resolved_library_realpath != library_realpath:
        raise HarnessError(
            f"{variant} ldd resolved libvsag {resolved_library_realpath}, "
            f"but shared_library is {library_realpath}"
        )
    return {
        "eval_binary": binary,
        "evaluator_realpath": binary_realpath,
        "evaluator_sha256": _sha256(Path(binary_realpath)),
        "shared_library": shared_library,
        "shared_library_realpath": library_realpath,
        "shared_library_sha256": _sha256(Path(library_realpath)),
        "resolved_libvsag_realpath": resolved_library_realpath,
        "resolved_libvsag_sha256": _sha256(Path(resolved_library_realpath)),
        "ld_library_path": str(Path(library_realpath).parent),
        "environment": library_env,
    }


def _preflight_variants(
    normalized: Dict[str, Any], binaries: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    identities = {
        variant: _preflight_variant(
            variant, binaries[variant], normalized[variant]["shared_library"]
        )
        for variant in VARIANTS
    }
    comparisons = (
        ("evaluator_realpath", "evaluator realpath"),
        ("evaluator_sha256", "evaluator SHA-256"),
        ("resolved_libvsag_realpath", "resolved libvsag realpath"),
        ("resolved_libvsag_sha256", "resolved libvsag SHA-256"),
    )
    for key, label in comparisons:
        if identities["baseline"][key] == identities["candidate"][key]:
            raise HarnessError(f"baseline and candidate must have different {label}")
    return identities


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _metrics(result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    latency = result.get("latency_detail(ms)")
    if not isinstance(latency, dict):
        latency = {}
    return {
        "qps": _number(result.get("qps")),
        "p50_ms": _number(latency.get("p50")),
        "p99_ms": _number(latency.get("p99")),
        "recall_avg": _number(result.get("recall_avg")),
    }


def _trim(text: Any, limit: int = 4000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _run_one(
    variant_info: Dict[str, Any],
    spec: Dict[str, Any],
    variant: str,
    concurrency: int,
    round_number: int,
    pair_number: int,
    order_in_pair: int,
    run_number: int,
    config_path: Path,
    result_path: Path,
) -> Dict[str, Any]:
    config_path.write_text(
        build_case_yaml(spec, variant, concurrency, result_path), encoding="utf-8"
    )
    try:
        result_path.unlink()
    except FileNotFoundError:
        pass
    binary = variant_info["evaluator_realpath"]
    command = [binary, str(config_path)]
    base: Dict[str, Any] = {
        "run_id": f"run-{run_number:04d}",
        "pair_id": f"pair-{pair_number:04d}",
        "pair_number": pair_number,
        "round": round_number,
        "order_in_pair": order_in_pair,
        "variant": variant,
        "outer_concurrency": concurrency,
        "evaluator_binary": binary,
        "evaluator_realpath": variant_info["evaluator_realpath"],
        "evaluator_sha256": variant_info["evaluator_sha256"],
        "resolved_libvsag_realpath": variant_info["resolved_libvsag_realpath"],
        "resolved_libvsag_sha256": variant_info["resolved_libvsag_sha256"],
        "index_path": spec["index_path"],
        "command": command,
        "errors": 0,
    }

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=spec["timeout_seconds"],
            check=False,
            env=variant_info["environment"],
        )
    except subprocess.TimeoutExpired as error:
        base.update(
            {
                "status": "timeout",
                "errors": 1,
                "error": f"evaluation timed out after {spec['timeout_seconds']} seconds",
                "stdout": _trim(error.stdout or ""),
                "stderr": _trim(error.stderr or ""),
            }
        )
        return base
    except OSError as error:
        base.update({"status": "launch_error", "errors": 1, "error": str(error)})
        return base

    if completed.returncode != 0:
        base.update(
            {
                "status": "process_error",
                "errors": 1,
                "returncode": completed.returncode,
                "error": "eval_performance exited with a non-zero status",
                "stdout": _trim(completed.stdout),
                "stderr": _trim(completed.stderr),
            }
        )
        return base

    if not result_path.is_file():
        base.update(
            {
                "status": "invalid_json",
                "errors": 1,
                "returncode": completed.returncode,
                "error": f"JSON exporter did not create result file: {result_path}",
                "stdout": _trim(completed.stdout),
                "stderr": _trim(completed.stderr),
            }
        )
        return base

    try:
        raw_result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        base.update(
            {
                "status": "invalid_json",
                "errors": 1,
                "returncode": completed.returncode,
                "error": f"JSON result file could not be parsed: {error}",
                "stdout": _trim(completed.stdout),
                "stderr": _trim(completed.stderr),
            }
        )
        return base

    if not isinstance(raw_result, dict) or not isinstance(raw_result.get(CASE_NAME), dict):
        base.update(
            {
                "status": "invalid_result",
                "errors": 1,
                "returncode": completed.returncode,
                "error": f"JSON result does not contain object case '{CASE_NAME}'",
                "raw_result": raw_result,
                "stdout": _trim(completed.stdout),
                "stderr": _trim(completed.stderr),
            }
        )
        return base

    case_result = raw_result[CASE_NAME]
    metrics = _metrics(case_result)
    missing_metrics = [name for name, value in metrics.items() if value is None]
    for name in (
        "measurement_sample_count",
        "measurement_successful_query_count",
        "error_count",
    ):
        value = case_result.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            missing_metrics.append(name)
    if missing_metrics:
        base.update(
            {
                "status": "invalid_result",
                "errors": 1,
                "returncode": completed.returncode,
                "error": "JSON result is missing or invalid metrics: " + ", ".join(missing_metrics),
                "raw_result": raw_result,
                "stdout": _trim(completed.stdout),
                "stderr": _trim(completed.stderr),
            }
        )
        return base
    base.update(
        {
            "status": "ok",
            "returncode": completed.returncode,
            "qps": metrics["qps"],
            "p50_ms": metrics["p50_ms"],
            "p99_ms": metrics["p99_ms"],
            "recall_avg": metrics["recall_avg"],
            "measurement_sample_count": case_result.get("measurement_sample_count"),
            "measurement_successful_query_count": case_result.get(
                "measurement_successful_query_count"
            ),
            "error_count": case_result.get("error_count"),
            "raw_result": raw_result,
        }
    )
    if completed.stdout.strip():
        base["stdout"] = _trim(completed.stdout)
    if completed.stderr.strip():
        base["stderr"] = _trim(completed.stderr)
    return base


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    usable = [value for value in values if value is not None]
    return float(statistics.median(usable)) if usable else None


def summarize(runs: Sequence[Dict[str, Any]], concurrency: Sequence[int]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        for level in concurrency:
            selected = [
                run
                for run in runs
                if run["variant"] == variant and run["outer_concurrency"] == level
            ]
            successful = [run for run in selected if run["status"] == "ok"]
            summary.append(
                {
                    "variant": variant,
                    "outer_concurrency": level,
                    "run_count": len(selected),
                    "successful_run_count": len(successful),
                    "errors": sum(int(run["errors"]) for run in selected),
                    "qps": _median([run.get("qps") for run in successful]),
                    "p50_ms": _median([run.get("p50_ms") for run in successful]),
                    "p99_ms": _median([run.get("p99_ms") for run in successful]),
                    "recall_avg": _median([run.get("recall_avg") for run in successful]),
                    "recall_avg_min": min(
                        [run["recall_avg"] for run in successful if run.get("recall_avg") is not None],
                        default=None,
                    ),
                }
            )
    return summary


def _change_pct(candidate: float, baseline: float) -> Optional[float]:
    if baseline == 0.0:
        return None
    return (candidate / baseline - 1.0) * 100.0


def _pair_changes(
    pair_number: int,
    round_number: int,
    concurrency: int,
    pair_runs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "pair_number": pair_number,
        "round": round_number,
        "outer_concurrency": concurrency,
        "baseline_run_id": None,
        "candidate_run_id": None,
        "qps_change_pct": None,
        "p99_change_pct": None,
        "recall_abs_change": None,
        "valid": False,
        "reasons": [],
    }
    grouped = {
        variant: [run for run in pair_runs if run.get("variant") == variant]
        for variant in VARIANTS
    }

    unexpected_variants = [
        run.get("variant")
        for run in pair_runs
        if run.get("variant") not in VARIANTS
    ]
    if unexpected_variants:
        record["reasons"].append(
            "unexpected variant(s) in pair: " + ", ".join(map(str, unexpected_variants))
        )

    selected: Dict[str, Dict[str, Any]] = {}
    for variant in VARIANTS:
        variant_runs = grouped[variant]
        if len(variant_runs) != 1:
            record["reasons"].append(
                f"expected one {variant} run for pair_number={pair_number}, "
                f"found {len(variant_runs)}"
            )
            continue
        run = variant_runs[0]
        record[f"{variant}_run_id"] = run.get("run_id")
        if run.get("status") != "ok":
            detail = run.get("error", "no error detail")
            record["reasons"].append(
                f"{variant} run {run.get('run_id', '<unknown>')} failed: "
                f"status={run.get('status', '<missing>')}; {detail}"
            )
            continue
        selected[variant] = run

    if len(selected) != len(VARIANTS):
        return record

    count_fields = (
        "measurement_sample_count",
        "measurement_successful_query_count",
        "error_count",
    )
    record["measurement"] = {
        variant: {field: selected[variant].get(field) for field in count_fields}
        for variant in VARIANTS
    }
    for field in count_fields:
        values_for_field = [selected[variant].get(field) for variant in VARIANTS]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values_for_field
        ):
            record["reasons"].append(f"paired metric '{field}' is missing or invalid")
            continue
        if values_for_field[0] != values_for_field[1]:
            record["reasons"].append(
                f"paired metric '{field}' differs: "
                f"baseline={values_for_field[0]}, candidate={values_for_field[1]}"
            )
    if any(selected[variant].get("error_count") != 0 for variant in VARIANTS):
        record["reasons"].append("paired error_count must be zero")
    if record["reasons"]:
        return record

    metrics = {
        "qps": "qps",
        "p99": "p99_ms",
        "recall": "recall_avg",
    }
    values: Dict[str, Dict[str, Optional[float]]] = {
        metric: {
            variant: _number(selected[variant].get(run_key)) for variant in VARIANTS
        }
        for metric, run_key in metrics.items()
    }
    for metric, metric_values in values.items():
        for variant in VARIANTS:
            if metric_values[variant] is None:
                record["reasons"].append(
                    f"{variant} run {selected[variant].get('run_id', '<unknown>')} "
                    f"is missing metric '{metrics[metric]}'"
                )

    if any(value is None for metric_values in values.values() for value in metric_values.values()):
        return record

    baseline_qps = values["qps"]["baseline"]
    candidate_qps = values["qps"]["candidate"]
    baseline_p99 = values["p99"]["baseline"]
    candidate_p99 = values["p99"]["candidate"]
    baseline_recall = values["recall"]["baseline"]
    candidate_recall = values["recall"]["candidate"]
    assert baseline_qps is not None
    assert candidate_qps is not None
    assert baseline_p99 is not None
    assert candidate_p99 is not None
    assert baseline_recall is not None
    assert candidate_recall is not None

    if baseline_qps == 0.0:
        record["reasons"].append(
            f"pair_number={pair_number} baseline qps is zero; qps_change_pct is undefined"
        )
    else:
        record["qps_change_pct"] = _change_pct(candidate_qps, baseline_qps)

    if baseline_p99 == 0.0:
        record["reasons"].append(
            f"pair_number={pair_number} baseline p99_ms is zero; p99_change_pct is undefined"
        )
    else:
        record["p99_change_pct"] = _change_pct(candidate_p99, baseline_p99)

    record["recall_abs_change"] = abs(candidate_recall - baseline_recall)
    record["valid"] = not record["reasons"]
    return record


def evaluate_acceptance(
    runs: Sequence[Dict[str, Any]],
    concurrency: Sequence[int],
    rounds: int,
    thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    """Pair runs and evaluate threshold criteria independently of harness status."""
    grouped: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = {}
    for run in runs:
        key = (run.get("round"), run.get("outer_concurrency"), run.get("pair_number"))
        grouped.setdefault(key, []).append(run)

    expected_keys = set()
    pair_records: List[Dict[str, Any]] = []
    pairs_by_concurrency: Dict[int, List[Dict[str, Any]]] = {}
    concurrency_levels = list(concurrency)
    for target in thresholds["target_concurrency"]:
        if target not in concurrency_levels:
            concurrency_levels.append(target)

    for round_number in range(1, rounds + 1):
        for level_number, level in enumerate(concurrency):
            pair_number = (round_number - 1) * len(concurrency) + level_number + 1
            key = (round_number, level, pair_number)
            expected_keys.add(key)
            record = _pair_changes(
                pair_number,
                round_number,
                level,
                grouped.get(key, []),
            )
            pair_records.append(record)
            pairs_by_concurrency.setdefault(level, []).append(record)

    orphaned_reasons = []
    for key, pair_runs in grouped.items():
        if key not in expected_keys:
            orphaned_reasons.append(
                "unexpected run group "
                f"round={key[0]}, outer_concurrency={key[1]}, pair_number={key[2]} "
                f"contains {len(pair_runs)} run(s)"
            )

    target_set = set(thresholds["target_concurrency"])
    by_concurrency: List[Dict[str, Any]] = []
    acceptance_reasons = list(orphaned_reasons)
    for level in concurrency_levels:
        target = level in target_set
        level_pairs = pairs_by_concurrency.get(level, [])
        valid_pairs = [pair for pair in level_pairs if pair["valid"]]
        invalid_pairs = [pair for pair in level_pairs if not pair["valid"]]
        qps_change_pct = _median([pair["qps_change_pct"] for pair in valid_pairs])
        p99_change_pct = _median([pair["p99_change_pct"] for pair in valid_pairs])
        recall_abs_change = _median([pair["recall_abs_change"] for pair in valid_pairs])
        level_reasons: List[str] = []

        if target:
            if level not in concurrency:
                level_reasons.append(
                    f"target concurrency {level} is not present in the concurrency matrix"
                )
            for pair in invalid_pairs:
                detail = "; ".join(pair["reasons"])
                level_reasons.append(f"pair_number={pair['pair_number']}: {detail}")
            if len(valid_pairs) < thresholds["min_valid_pairs"]:
                level_reasons.append(
                    f"valid pair count {len(valid_pairs)} is below minimum "
                    f"{thresholds['min_valid_pairs']}"
                )
            if not valid_pairs:
                level_reasons.append("no valid baseline/candidate paired changes are available")
            if qps_change_pct is None:
                level_reasons.append("qps_change_pct median is unavailable")
            elif qps_change_pct < thresholds["qps_min_pct"]:
                level_reasons.append(
                    f"qps_change_pct median {qps_change_pct:.6g} is below "
                    f"minimum {thresholds['qps_min_pct']:.6g}"
                )
            if p99_change_pct is None:
                level_reasons.append("p99_change_pct median is unavailable")
            elif p99_change_pct > thresholds["p99_max_regression_pct"]:
                level_reasons.append(
                    f"p99_change_pct median {p99_change_pct:.6g} exceeds "
                    f"maximum regression {thresholds['p99_max_regression_pct']:.6g}"
                )
            if recall_abs_change is None:
                level_reasons.append("recall_abs_change median is unavailable")
            elif recall_abs_change > thresholds["recall_max_abs_change"]:
                level_reasons.append(
                    f"recall_abs_change median {recall_abs_change:.6g} exceeds "
                    f"maximum {thresholds['recall_max_abs_change']:.6g}"
                )
            level_status = "pass" if not level_reasons else "fail"
            acceptance_reasons.extend(
                f"concurrency {level}: {reason}" for reason in level_reasons
            )
        else:
            for pair in invalid_pairs:
                detail = "; ".join(pair["reasons"])
                level_reasons.append(f"pair_number={pair['pair_number']}: {detail}")
            if invalid_pairs:
                level_status = "fail"
                acceptance_reasons.extend(
                    f"concurrency {level}: {reason}" for reason in level_reasons
                )
            else:
                level_status = "not_target"

        by_concurrency.append(
            {
                "outer_concurrency": level,
                "target": target,
                "status": level_status,
                "pair_count": len(level_pairs),
                "valid_pair_count": len(valid_pairs),
                "invalid_pair_count": len(invalid_pairs),
                "qps_change_pct": qps_change_pct,
                "p99_change_pct": p99_change_pct,
                "recall_abs_change": recall_abs_change,
                "reasons": level_reasons,
            }
        )

    return {
        "status": "pass" if not acceptance_reasons else "fail",
        "thresholds": thresholds,
        "target_concurrency": thresholds["target_concurrency"],
        "pairing": "round + outer_concurrency + pair_number; one baseline and one candidate run",
        "pairs": pair_records,
        "by_concurrency": by_concurrency,
        "reasons": acceptance_reasons,
    }


def _cpu_info() -> Dict[str, Any]:
    logical_cpu_count = os.cpu_count()
    if not hasattr(os, "sched_getaffinity"):
        return {
            "sched_affinity": None,
            "affinity_cpu_count": None,
            "logical_cpu_count": logical_cpu_count,
            "reason": "os.sched_getaffinity is unavailable",
        }
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except OSError as error:
        return {
            "sched_affinity": None,
            "affinity_cpu_count": None,
            "logical_cpu_count": logical_cpu_count,
            "reason": f"os.sched_getaffinity failed: {error}",
        }
    return {
        "sched_affinity": affinity,
        "affinity_cpu_count": len(affinity),
        "logical_cpu_count": logical_cpu_count,
        "reason": None,
    }


def run_suite(spec: Dict[str, Any], output_path: Optional[Path] = None) -> Dict[str, Any]:
    normalized = normalize_spec(spec)
    binaries = {
        variant: _resolve_binary(normalized[variant]["eval_binary"]) for variant in VARIANTS
    }
    input_paths = [Path(normalized["datapath"]), Path(normalized["index_path"])]
    for path in input_paths:
        if not path.is_file():
            raise HarnessError(f"evaluation input does not exist or is not a file: {path}")

    identities = _preflight_variants(normalized, binaries)
    cpu = _cpu_info()
    warnings: List[str] = []
    warning_records: List[Dict[str, Any]] = []
    if cpu["affinity_cpu_count"] is not None:
        for level in normalized["concurrency"]:
            if level > cpu["affinity_cpu_count"]:
                warning = {
                    "type": "oversubscription",
                    "outer_concurrency": level,
                    "affinity_cpu_count": cpu["affinity_cpu_count"],
                    "message": (
                        f"concurrency {level} exceeds sched affinity CPU count "
                        f"{cpu['affinity_cpu_count']}"
                    ),
                }
                warning_records.append(warning)
                warnings.append(warning["message"])

    fixed_input: Dict[str, Any] = {
        "datapath": normalized["datapath"],
        "index_path": normalized["index_path"],
        "index_name": normalized["index_name"],
        "create_params": normalized["create_params"],
        "search_params": normalized["search_params"],
        "search_mode": normalized["search_mode"],
        "set_immutable": normalized["set_immutable"],
        "warmup_query_count": normalized["warmup_query_count"],
        "topk": normalized["topk"],
        "search_query_count": normalized["search_query_count"],
        "acceptance": normalized["acceptance"],
    }
    if normalized["hash_files"]:
        fixed_input["sha256"] = {
            "datapath": _sha256(Path(normalized["datapath"])),
            "indexes": {"shared": _sha256(Path(normalized["index_path"]))},
            "eval_binaries": {
                variant: identities[variant]["evaluator_sha256"] for variant in VARIANTS
            },
            "shared_libraries": {
                variant: identities[variant]["resolved_libvsag_sha256"] for variant in VARIANTS
            },
        }

    report: Dict[str, Any] = {
        "schema_version": 1,
        "harness": "vsag-concurrency-ab",
        "fixed_input": fixed_input,
        "variants": {
            variant: {
                key: value for key, value in identities[variant].items() if key != "environment"
            }
            for variant in VARIANTS
        },
        "concurrency": normalized["concurrency"],
        "rounds": normalized["rounds"],
        "execution_order": "pairwise; the leading variant alternates for each pair",
        "error_scope": "evaluator_invocation",
        "cpu": cpu,
        "warnings": warnings,
        "warning_records": warning_records,
        "errors": 0,
        "runs": [],
    }

    with tempfile.TemporaryDirectory(prefix="vsag-concurrency-ab-") as temp_dir:
        config_dir = Path(temp_dir)
        run_number = 0
        pair_number = 0
        for round_number in range(1, normalized["rounds"] + 1):
            for level_number, level in enumerate(normalized["concurrency"]):
                pair_number += 1
                if (round_number + level_number) % 2 == 0:
                    order = ("candidate", "baseline")
                else:
                    order = VARIANTS
                for order_in_pair, variant in enumerate(order, start=1):
                    run_number += 1
                    run = _run_one(
                        identities[variant],
                        normalized,
                        variant,
                        level,
                        round_number,
                        pair_number,
                        order_in_pair,
                        run_number,
                        config_dir / f"{run_number:04d}.yaml",
                        config_dir / f"{run_number:04d}.json",
                    )
                    report["runs"].append(run)

        report["summary"] = summarize(report["runs"], normalized["concurrency"])
        report["acceptance"] = evaluate_acceptance(
            report["runs"],
            normalized["concurrency"],
            normalized["rounds"],
            normalized["acceptance"],
        )
    report["errors"] = sum(int(run["errors"]) for run in report["runs"])
    report["status"] = "ok" if report["errors"] == 0 else "failed"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report


def _load_spec(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"failed to read JSON spec {path}: {error}") from error
    if not isinstance(value, dict):
        raise HarnessError("the JSON spec root must be an object")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path, help="JSON concurrency evaluation spec")
    parser.add_argument("--output", required=True, type=Path, help="machine-readable JSON report path")
    parser.add_argument(
        "--hash-files",
        action="store_true",
        help=(
            "include SHA-256 fingerprints for the dataset/shared index; "
            "evaluator and library fingerprints are always reported"
        ),
    )
    args = parser.parse_args(argv)

    try:
        spec = _load_spec(args.spec)
        if args.hash_files:
            spec["hash_files"] = True
        report = run_suite(spec, args.output)
    except HarnessError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for warning in report["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    print(
        f"wrote {args.output} ({len(report['runs'])} runs, errors={report['errors']}, "
        f"acceptance={report['acceptance']['status']})"
    )
    if report["status"] != "ok":
        return 1
    return 0 if report["acceptance"]["status"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
