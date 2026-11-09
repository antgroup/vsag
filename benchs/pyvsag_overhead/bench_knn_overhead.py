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

"""Measure single-query pyvsag HGraph k-NN call latency.

The module deliberately imports numpy and pyvsag only inside the benchmark entry
point so that the statistics helpers can be tested without either dependency.
"""

import argparse
import gc
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SEED = 3295
MAX_QUERY_VECTORS = 256


class BenchmarkError(RuntimeError):
    """A benchmark setup or execution failure with a user-facing message."""


def _sorted_finite(values: Iterable[float]) -> List[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("at least one latency sample is required")
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("latency samples must be finite")
    return ordered


def _percentile_sorted(ordered: Sequence[float], percent: float) -> float:
    """Return a linearly interpolated percentile of an already sorted sequence."""
    if not math.isfinite(percent) or not 0.0 <= percent <= 100.0:
        raise ValueError("percent must be between 0 and 100")
    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def percentile(values: Iterable[float], percent: float) -> float:
    """Return a percentile using the same linear method as numpy.percentile.

    For example, the p50 of [1, 2, 3, 4] is 2.5 and its p99 is 3.97.
    """
    return _percentile_sorted(_sorted_finite(values), percent)


def summarize_latencies_ns(latencies_ns: Iterable[float]) -> Dict[str, float]:
    """Aggregate per-call nanosecond latencies into the reported statistics."""
    ordered = _sorted_finite(latencies_ns)
    return {
        "count": len(ordered),
        "p50_ns": _percentile_sorted(ordered, 50.0),
        "p90_ns": _percentile_sorted(ordered, 90.0),
        "p99_ns": _percentile_sorted(ordered, 99.0),
        "mean_ns": math.fsum(ordered) / len(ordered),
        "min_ns": float(ordered[0]),
    }


def create_index_params(dim: int) -> str:
    """Create the HGraph fp32/l2 parameters used by the Python HGraph tests."""
    max_degree = max(1, int(dim / 4))
    return json.dumps(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": dim,
            "index_param": {
                "max_degree": max_degree,
                "ef_construction": max(200, max_degree),
                "base_quantization_type": "fp32",
                "precise_quantization_type": "fp32",
                "use_reorder": False,
            },
        }
    )


def _cpu_name() -> str:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as cpuinfo:
                for line in cpuinfo:
                    if line.startswith("model name") and ":" in line:
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass

    for candidate in (platform.processor(), os.environ.get("PROCESSOR_IDENTIFIER")):
        if candidate:
            return candidate
    return "unknown"


def _environment_info(numpy_module: object) -> Dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": numpy_module.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
    }


def _markdown_table(label: str, stats: Dict[str, float]) -> str:
    headings = ("label", "calls", "p50 (us)", "p90 (us)", "p99 (us)", "mean (us)", "min (us)")
    values = (
        label,
        str(stats["count"]),
        *(f"{stats[name] / 1000.0:.3f}" for name in ("p50_ns", "p90_ns", "p99_ns", "mean_ns", "min_ns")),
    )
    return "\n".join(
        (
            "| " + " | ".join(headings) + " |",
            "| " + " | ".join("---" for _ in headings) + " |",
            "| " + " | ".join(values) + " |",
        )
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-call latency of single-vector pyvsag HGraph knn_search."
    )
    parser.add_argument("--num-vectors", type=_positive_int, default=20000)
    parser.add_argument("--dim", type=_positive_int, default=128)
    parser.add_argument("--k", type=_positive_int, default=10)
    parser.add_argument("--iterations", type=_positive_int, default=200)
    parser.add_argument("--warmup", type=_nonnegative_int, default=20)
    parser.add_argument("--ef-search", type=_positive_int, default=100)
    parser.add_argument("--label", help="Implementation label used in output (default: installed-pyvsag)")
    parser.add_argument("--json-out", type=Path, help="Write configuration, environment, samples, and statistics as JSON")
    parser.add_argument(
        "--library",
        type=Path,
        help="Path to libvsag_c.so; sets PYVSAG_LIBRARY before importing pyvsag",
    )
    args = parser.parse_args(argv)
    if args.k > args.num_vectors:
        parser.error("--k must not exceed --num-vectors")
    return args


def _import_modules(library: Optional[Path]) -> Tuple[object, object, Optional[Path]]:
    try:
        numpy_module = importlib.import_module("numpy")
    except ImportError as exc:
        raise BenchmarkError("numpy is required; install it in the benchmark environment") from exc

    resolved_library = None
    if library is not None:
        resolved_library = library.expanduser().resolve()
        if not resolved_library.is_file():
            raise BenchmarkError(f"shared library does not exist or is not a file: {resolved_library}")
        os.environ["PYVSAG_LIBRARY"] = str(resolved_library)

    try:
        pyvsag_module = importlib.import_module("pyvsag")
    except Exception as exc:
        library_hint = f" with PYVSAG_LIBRARY={resolved_library}" if resolved_library else ""
        raise BenchmarkError(f"failed to import pyvsag{library_hint}: {exc}") from exc
    return numpy_module, pyvsag_module, resolved_library


def _run_benchmark(args: argparse.Namespace, numpy_module: object, pyvsag_module: object) -> Tuple[List[int], str, str, int]:
    np = numpy_module
    rng = np.random.default_rng(DEFAULT_SEED)
    base_vectors = np.ascontiguousarray(
        rng.random((args.num_vectors, args.dim)), dtype=np.float32
    )
    ids = np.ascontiguousarray(np.arange(args.num_vectors, dtype=np.int64))
    query_count = min(args.iterations + args.warmup, MAX_QUERY_VECTORS)
    query_vectors = np.ascontiguousarray(
        rng.random((query_count, args.dim)), dtype=np.float32
    )

    index_params = create_index_params(args.dim)
    search_params = json.dumps({"hgraph": {"ef_search": args.ef_search}})
    try:
        index = pyvsag_module.Index("hgraph", index_params)
    except Exception as exc:
        raise BenchmarkError(f"failed to create HGraph index: {exc}") from exc

    try:
        build_result = index.build(base_vectors, ids, args.num_vectors, args.dim)
    except Exception as exc:
        raise BenchmarkError(f"failed to build HGraph index: {exc}") from exc
    if build_result is False:
        raise BenchmarkError("failed to build HGraph index: Index.build returned False")

    try:
        for iteration in range(args.warmup):
            query = query_vectors[iteration % query_count]
            index.knn_search(vector=query, k=args.k, parameters=search_params)
    except Exception as exc:
        raise BenchmarkError(f"knn_search failed during warmup iteration {iteration}: {exc}") from exc

    latencies_ns = []
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        for iteration in range(args.iterations):
            query = query_vectors[(args.warmup + iteration) % query_count]
            started_ns = time.perf_counter_ns()
            try:
                index.knn_search(vector=query, k=args.k, parameters=search_params)
            except Exception as exc:
                raise BenchmarkError(
                    f"knn_search failed during measured iteration {iteration}: {exc}"
                ) from exc
            latencies_ns.append(time.perf_counter_ns() - started_ns)
    finally:
        if gc_was_enabled:
            gc.enable()

    return latencies_ns, index_params, search_params, query_count


def run(args: argparse.Namespace) -> int:
    numpy_module, pyvsag_module, resolved_library = _import_modules(args.library)
    environment = _environment_info(numpy_module)
    latencies_ns, index_params, search_params, query_count = _run_benchmark(
        args, numpy_module, pyvsag_module
    )
    stats = summarize_latencies_ns(latencies_ns)
    label = args.label or ("library-pyvsag" if resolved_library else "installed-pyvsag")

    result = {
        "schema_version": 1,
        "label": label,
        "configuration": {
            "num_vectors": args.num_vectors,
            "dim": args.dim,
            "k": args.k,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "ef_search": args.ef_search,
            "seed": DEFAULT_SEED,
            "query_vectors": query_count,
        },
        "index_parameters": json.loads(index_params),
        "search_parameters": json.loads(search_params),
        "environment": environment,
        "pyvsag": {
            "version": getattr(pyvsag_module, "__version__", "unknown"),
            "module_file": getattr(pyvsag_module, "__file__", "unknown"),
            "library": str(resolved_library) if resolved_library else None,
        },
        "statistics": stats,
        "samples_ns": latencies_ns,
    }

    if args.json_out:
        try:
            args.json_out.expanduser().parent.mkdir(parents=True, exist_ok=True)
            with args.json_out.expanduser().open("w", encoding="utf-8") as output:
                json.dump(result, output, indent=2, allow_nan=False)
                output.write("\n")
        except OSError as exc:
            raise BenchmarkError(f"failed to write JSON output {args.json_out}: {exc}") from exc

    print("Environment:")
    print(json.dumps(environment, indent=2))
    print("\nPer-call knn_search latency:")
    print(_markdown_table(label, stats))
    if args.json_out:
        print(f"\nJSON output: {args.json_out.expanduser()}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: unexpected benchmark failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
