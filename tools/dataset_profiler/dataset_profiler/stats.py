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

"""Small numeric helpers shared by the profiler and report writers."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

PERCENTILES = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)
HISTOGRAM_BINS = 20


def as_finite_float(value: Any) -> Optional[float]:
    """Convert a NumPy or Python number to a JSON-safe float."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    # Preserve small finite magnitudes: fixed-decimal rounding would serialize
    # values such as 5e-11 as zero in JSON, HTML, and AI payloads.
    return number


def _as_float64_array(values: Iterable[float]) -> np.ndarray:
    """Convert values without boxing ndarray elements into a Python list."""
    if isinstance(values, np.ndarray):
        return np.asarray(values, dtype=np.float64)
    return np.fromiter(values, dtype=np.float64)


def summarize(values: Iterable[float]) -> Dict[str, Any]:
    """Return count/mean/std/min/max/percentiles for finite values."""
    array = _as_float64_array(values)
    array = array[np.isfinite(array)]
    out: Dict[str, Any] = {"count": int(array.size)}
    if array.size == 0:
        out.update(
            {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        )
        for percentile in PERCENTILES:
            out["p%g" % percentile] = None
        return out
    out.update(
        {
            "mean": as_finite_float(np.mean(array)),
            "std": as_finite_float(np.std(array)),
            "min": as_finite_float(np.min(array)),
            "max": as_finite_float(np.max(array)),
        }
    )
    for percentile in PERCENTILES:
        key = "p%g" % percentile
        out[key] = as_finite_float(np.percentile(array, percentile, method="linear"))
    return out


def histogram(values: Iterable[float], bins: int = HISTOGRAM_BINS) -> Dict[str, Any]:
    """Return histogram edges/counts safe for JSON and SVG rendering."""
    array = _as_float64_array(values)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"bins": [], "counts": [], "bin_count": 0}
    bin_count = int(bins)
    if np.all(array == array[0]):
        # NumPy's automatic constant-range padding is +/-0.5, which rounds
        # away at large magnitudes.  Supply a scale-aware representable range.
        value = float(array[0])
        padding = max(abs(value) * 1e-3, math.ulp(value) * 4.0, 0.5)
        lower = value - padding
        upper = value + padding
        if math.isfinite(lower) and math.isfinite(upper) and lower < upper:
            counts, edges = np.histogram(
                array, bins=bin_count, range=(lower, upper)
            )
        else:
            # A one-ulp range cannot be split into ``bin_count`` bins.  Build
            # explicit, one-sided edges instead, always stepping toward the
            # finite interior so even +/-DBL_MAX cannot overflow.
            direction = -np.inf if value > 0 else np.inf
            edges = np.empty(bin_count + 1, dtype=np.float64)
            edges[0] = value
            for index in range(1, bin_count + 1):
                edges[index] = np.nextafter(edges[index - 1], direction)
            if direction == -np.inf:
                edges = edges[::-1]
            counts, edges = np.histogram(array, bins=edges)
    else:
        counts, edges = np.histogram(array, bins=bin_count)
    return {
        "bins": [as_finite_float(edge) for edge in edges],
        "counts": [int(count) for count in counts],
        "bin_count": bin_count,
    }


def downsample(values: np.ndarray, max_points: int = 1024) -> Dict[str, Any]:
    """Downsample a 1-D array for report rendering.

    Returns the (possibly reduced) values plus metadata describing whether the
    array was reduced and how many source dimensions each output point covers.
    """
    array = np.asarray(values)
    length = int(array.size)
    if length <= max_points:
        return {
            "values": [as_finite_float(value) for value in array],
            "source_length": length,
            "downsampled": False,
            "bucket_size": 1,
        }
    boundaries = np.linspace(0, length, max_points + 1, dtype=np.int64)
    reduced: List[Optional[float]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            reduced.append(None)
            continue
        chunk = np.asarray(array[start:end], dtype=np.float64)
        finite = chunk[np.isfinite(chunk)]
        reduced.append(as_finite_float(np.mean(finite)) if finite.size else None)
    return {
        "values": reduced,
        "source_length": length,
        "downsampled": True,
        "bucket_size": int(math.ceil(length / float(max_points))),
    }


def top_indices(values: np.ndarray, k: int, largest: bool = True) -> List[int]:
    """Return indices of the top-``k`` entries, ignoring non-finite values."""
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not np.any(finite):
        return []
    candidates = np.where(finite)[0]
    order = np.argsort(array[candidates], kind="stable")
    if largest:
        order = order[::-1]
    return [int(candidates[i]) for i in order[: int(k)]]


def jsonable(value: Any) -> Any:
    """Recursively convert NumPy values and non-finite floats for JSON."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return as_finite_float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def dump_json(report: Dict[str, Any]) -> str:
    """Serialize a report deterministically (sorted keys, fixed indentation)."""
    return json.dumps(
        jsonable(report), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"


def metric_keys(report: Dict[str, Any]) -> List[str]:
    """Enumerate dotted evidence keys for every metric leaf in a report."""
    keys: List[str] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key in sorted(node):
                walk(node[key], "%s.%s" % (prefix, key) if prefix else str(key))
            return
        if isinstance(node, (list, tuple)):
            if node and all(not isinstance(item, (dict, list, tuple)) for item in node):
                keys.append(prefix)
            return
        keys.append(prefix)

    walk(report.get("metrics", {}), "metrics")
    return keys
