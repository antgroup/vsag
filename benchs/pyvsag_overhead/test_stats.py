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

import math

from bench_knn_overhead import percentile, summarize_latencies_ns
import pytest


def test_percentile_uses_linear_interpolation():
    values = [1, 2, 3, 4]

    assert percentile(values, 50) == pytest.approx(2.5)
    assert percentile(values, 90) == pytest.approx(3.7)
    assert percentile(values, 99) == pytest.approx(3.97)


def test_percentile_sorts_input_and_handles_one_sample():
    assert percentile([10, 0, 20], 50) == pytest.approx(10)
    assert percentile([42], 50) == pytest.approx(42)
    assert percentile([42], 99) == pytest.approx(42)


def test_latency_summary_for_known_sequence():
    stats = summarize_latencies_ns([400, 100, 300, 200])

    assert stats["count"] == 4
    assert stats["p50_ns"] == pytest.approx(250)
    assert stats["p90_ns"] == pytest.approx(370)
    assert stats["p99_ns"] == pytest.approx(397)
    assert stats["mean_ns"] == pytest.approx(250)
    assert stats["min_ns"] == pytest.approx(100)


@pytest.mark.parametrize("percent", [-1, 100.1, math.nan, math.inf])
def test_percentile_rejects_invalid_percent(percent):
    with pytest.raises(ValueError, match="percent"):
        percentile([1, 2, 3], percent)


@pytest.mark.parametrize("values", [[], [1, math.nan], [1, math.inf]])
def test_statistics_reject_empty_or_nonfinite_samples(values):
    with pytest.raises(ValueError):
        summarize_latencies_ns(values)
