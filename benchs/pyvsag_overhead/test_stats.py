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

import json
import math
from types import SimpleNamespace

import bench_knn_overhead as bench
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


def test_dim_eight_uses_hgraph_minimum_max_degree():
    params = json.loads(bench.create_index_params(8))

    assert params["index_param"]["max_degree"] == 4


class _FakeIndex:
    def __init__(self, fail_on):
        self.fail_on = fail_on
        self.calls = 0

    def knn_search(self, *, vector, k, parameters):
        self.calls += 1
        result_ids = [-1] * k if self.calls == self.fail_on else list(range(k))
        return result_ids, [0.0] * k


class _Pybind11Index(_FakeIndex):
    pass


class _CffiIndex(_FakeIndex):
    pass


@pytest.mark.parametrize("index_class", [_Pybind11Index, _CffiIndex], ids=["pybind11", "cffi"])
@pytest.mark.parametrize(
    ("warmup", "fail_on", "phase"),
    [(1, 1, "warmup"), (0, 1, "measured"), (1, 2, "measured")],
)
def test_failed_search_sentinel_is_rejected(index_class, warmup, fail_on, phase):
    index = index_class(fail_on)

    with pytest.raises(bench.BenchmarkError, match=f"-1 sentinel results during {phase}"):
        bench._measure_queries(index, [object()], 2, "{}", 1, warmup)

    assert index.calls == fail_on


def test_result_validation_is_outside_timed_interval(monkeypatch):
    events = []

    class _RecordingIds:
        def __len__(self):
            events.append("validate")
            return 1

        def __iter__(self):
            return iter([0])

    class _RecordingIndex:
        def knn_search(self, *, vector, k, parameters):
            events.append("search")
            return _RecordingIds(), [0.0]

    timestamps = iter([100, 125])

    def _clock():
        events.append("timer")
        return next(timestamps)

    monkeypatch.setattr(bench.time, "perf_counter_ns", _clock)

    assert bench._measure_queries(_RecordingIndex(), [object()], 1, "{}", 1, 0) == [25]
    assert events == ["timer", "search", "timer", "validate"]


def _patch_imports(monkeypatch, pyvsag_module, loader_module=None):
    numpy_module = SimpleNamespace(__version__="test")

    def _fake_import(name):
        if name == "numpy":
            return numpy_module
        if name == "pyvsag":
            return pyvsag_module
        if name == "pyvsag._lib" and loader_module is not None:
            return loader_module
        if name == "pyvsag._lib":
            raise ModuleNotFoundError("No module named 'pyvsag._lib'")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.delenv("PYVSAG_LIBRARY", raising=False)
    monkeypatch.setattr(bench.importlib, "import_module", _fake_import)
    return numpy_module


def test_library_override_rejects_pybind11_backend(monkeypatch, tmp_path):
    library = tmp_path / "libvsag_c.so"
    library.touch()
    pyvsag_module = SimpleNamespace(__name__="pyvsag")
    _patch_imports(monkeypatch, pyvsag_module)

    with pytest.raises(bench.BenchmarkError, match="requires the cffi pyvsag backend"):
        bench._import_modules(library)


def test_library_override_rejects_mismatched_cffi_library(monkeypatch, tmp_path):
    requested_library = tmp_path / "requested.so"
    other_library = tmp_path / "other.so"
    requested_library.touch()
    other_library.touch()
    pyvsag_module = SimpleNamespace(__name__="pyvsag")
    loader_module = SimpleNamespace(
        ffi=object(), lib=object(), _find_library=lambda: str(other_library)
    )
    _patch_imports(monkeypatch, pyvsag_module, loader_module)

    with pytest.raises(bench.BenchmarkError, match="cffi loader reported library"):
        bench._import_modules(requested_library)


def test_verified_module_and_library_are_printed(monkeypatch, tmp_path, capsys):
    library = tmp_path / "libvsag_c.so"
    library.touch()
    module_file = str(tmp_path / "pyvsag" / "__init__.py")
    pyvsag_module = SimpleNamespace(
        __name__="pyvsag", __file__=module_file, __version__="test"
    )
    loader_module = SimpleNamespace(
        ffi=object(), lib=object(), _find_library=lambda: str(library)
    )
    _patch_imports(monkeypatch, pyvsag_module, loader_module)
    monkeypatch.setattr(
        bench,
        "_run_benchmark",
        lambda *args: (
            [25],
            bench.create_index_params(8),
            '{"hgraph": {"ef_search": 100}}',
            1,
        ),
    )
    args = bench.parse_args(
        ["--library", str(library), "--iterations", "1", "--warmup", "0"]
    )

    assert bench.run(args) == 0
    stdout = capsys.readouterr().out
    assert f"pyvsag module: {module_file}" in stdout
    assert f"pyvsag library: {library.resolve()}" in stdout
