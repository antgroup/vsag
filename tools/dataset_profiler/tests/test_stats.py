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

"""Tests for the shared numeric helpers."""

from __future__ import annotations

import json
import math

import numpy as np

from dataset_profiler.stats import (
    as_finite_float,
    downsample,
    dump_json,
    histogram,
    jsonable,
    metric_keys,
    summarize,
    top_indices,
)


def test_as_finite_float_maps_non_finite_to_none():
    assert as_finite_float(1.23456789) == 1.23456789
    assert as_finite_float(math.nan) is None
    assert as_finite_float(math.inf) is None
    assert as_finite_float(None) is None
    assert as_finite_float("x") is None
    assert as_finite_float(np.float64(2.5)) == 2.5


def test_small_finite_values_survive_serialization():
    value = 5e-12
    assert as_finite_float(value) == value
    assert json.loads(dump_json({"value": value}))["value"] == value


def test_summarize_reports_percentiles_and_skips_non_finite():
    stats = summarize([1.0, 2.0, 3.0, math.nan, math.inf, 4.0])
    assert stats["count"] == 4
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0
    assert stats["p50"] == 2.5
    assert "p99" in stats


def test_summarize_empty_input_is_all_none():
    stats = summarize([math.nan, math.inf])
    assert stats["count"] == 0
    assert stats["mean"] is None
    assert stats["p50"] is None


def test_histogram_is_json_safe():
    hist = histogram([0.0, 0.5, 1.0, 2.0, 2.5], bins=5)
    assert hist["bin_count"] == 5
    assert len(hist["bins"]) == 6
    assert sum(hist["counts"]) == 5
    json.dumps(hist)  # must not raise
    empty = histogram([math.nan])
    assert empty == {"bins": [], "counts": [], "bin_count": 0}


def test_downsample_keeps_small_arrays_and_buckets_large_ones():
    small = downsample(np.arange(10, dtype=float), max_points=16)
    assert small["downsampled"] is False
    assert small["values"] == [float(v) for v in range(10)]

    big = downsample(np.arange(100, dtype=float), max_points=10)
    assert big["downsampled"] is True
    assert big["source_length"] == 100
    assert big["bucket_size"] == 10
    assert len(big["values"]) == 10


def test_top_indices_ignores_non_finite():
    values = np.array([1.0, math.nan, 5.0, 3.0, math.inf])
    assert top_indices(values, 2) == [2, 3]
    assert top_indices(values, 2, largest=False) == [0, 3]
    assert top_indices(np.array([math.nan]), 3) == []


def test_jsonable_converts_numpy_types():
    payload = {
        "i": np.int64(3),
        "f": np.float32(1.5),
        "b": np.True_,
        "arr": np.arange(3),
        "nested": [{"n": np.int32(1)}],
    }
    converted = jsonable(payload)
    text = json.dumps(converted, allow_nan=False)
    assert json.loads(text) == {"i": 3, "f": 1.5, "b": True, "arr": [0, 1, 2], "nested": [{"n": 1}]}


def test_dump_json_is_deterministic_and_sorted():
    first = dump_json({"b": 1, "a": [2, 1]})
    second = dump_json({"a": [2, 1], "b": 1})
    assert first == second
    assert first.endswith("\n")
    assert json.loads(first)["a"] == [2, 1]


def test_metric_keys_enumerate_leaf_paths():
    report = {
        "metrics": {
            "norms": {"stats": {"mean": 1.0, "max": None}, "histogram": {"counts": [1, 2]}},
            "duplicates": {"exact": {"groups": [{"size": 2}]}},
        }
    }
    keys = metric_keys(report)
    assert "metrics.norms.stats.mean" in keys
    assert "metrics.norms.histogram.counts" in keys
    assert not any(key.startswith("metrics.duplicates.exact.groups") for key in keys)
