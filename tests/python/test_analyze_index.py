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

"""Tests for the pyvsag index analysis API and the pyvsag-analyze-index command."""

import json
import shutil
import struct
import subprocess
import sys

import numpy as np
import pyvsag
import pytest

DIM = 8
NUM_VECTORS = 64
NUM_QUERIES = 1
TOPK = 5


def _index_params():
    return json.dumps(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": DIM,
            "index_param": {
                "base_quantization_type": "fp32",
                "precise_quantization_type": "fp32",
            },
        }
    )


def _run(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


@pytest.fixture
def saved_index(tmp_path):
    """Build, save and return a small brute-force index plus its query file."""
    rng = np.random.RandomState(42)
    vectors = rng.randn(NUM_VECTORS, DIM).astype(np.float32)
    ids = np.arange(NUM_VECTORS, dtype=np.int64)
    index = pyvsag.Index("brute_force", _index_params())
    index.build(vectors=vectors, ids=ids, num_elements=NUM_VECTORS, dim=DIM)

    index_path = str(tmp_path / "index.vsag")
    index.save(index_path)

    queries = np.random.RandomState(7).randn(NUM_QUERIES, DIM).astype(np.float32)
    query_path = str(tmp_path / "queries.bin")
    with open(query_path, "wb") as stream:
        stream.write(struct.pack("<II", NUM_QUERIES, DIM))
        stream.write(queries.tobytes())

    return index_path, query_path


def test_analyze_index_static(saved_index):
    index_path, _ = saved_index

    report = pyvsag.analyze_index(index_path=index_path)

    assert report["schema_version"] == pyvsag.ANALYSIS_SCHEMA_VERSION
    assert report["index"]["name"] == "brute_force"
    assert report["index"]["dim"] == DIM
    assert report["index"]["data_type"] == "float32"
    assert report["index"]["metric"] == "l2"
    assert isinstance(report["static_analysis"]["statistics"], dict)
    assert report["static_analysis"]["statistics"]
    assert report["dynamic_analysis"] is None
    assert isinstance(report["warnings"], list)
    assert isinstance(report["compatibility"], dict)


def test_analyze_index_result_is_json_serializable(saved_index):
    index_path, _ = saved_index

    report = pyvsag.analyze_index(index_path=index_path)

    # json.dumps must not require custom encoders.
    serialized = json.dumps(report)
    assert json.loads(serialized)["schema_version"] == pyvsag.ANALYSIS_SCHEMA_VERSION


def test_analyze_index_dynamic(saved_index):
    index_path, query_path = saved_index

    report = pyvsag.analyze_index(
        index_path=index_path,
        query_path=query_path,
        query_data_type="dense",
        topk=TOPK,
    )

    dynamic = report["dynamic_analysis"]
    assert dynamic is not None
    assert dynamic["topk"] == TOPK
    assert dynamic["query_data_type"] == "dense"
    assert dynamic["query_path"] == query_path
    assert isinstance(dynamic["results"], dict)


def test_analyze_index_rejects_invalid_arguments(saved_index):
    index_path, query_path = saved_index

    with pytest.raises(ValueError):
        pyvsag.analyze_index(index_path="")

    with pytest.raises(ValueError):
        pyvsag.analyze_index(index_path=index_path, topk=0)

    with pytest.raises(TypeError):
        pyvsag.analyze_index(index_path=index_path, topk="5")

    with pytest.raises(TypeError):
        pyvsag.analyze_index(index_path=index_path, query_path=123)

    with pytest.raises(ValueError):
        pyvsag.analyze_index(index_path=index_path, query_data_type="fp64")

    with pytest.raises(ValueError):
        pyvsag.analyze_index(index_path=index_path, groundtruth_path=query_path)

    with pytest.raises(ValueError):
        pyvsag.analyze_index(index_path=index_path, build_parameter="not-json")

    with pytest.raises(ValueError):
        pyvsag.analyze_index(index_path=index_path, search_parameter="5")


def test_analyze_index_missing_file(tmp_path):
    missing = str(tmp_path / "missing.vsag")

    with pytest.raises(RuntimeError):
        pyvsag.analyze_index(index_path=missing)


def test_analyze_index_damaged_file(tmp_path):
    damaged = tmp_path / "damaged.vsag"
    damaged.write_bytes(b"not a vsag index")

    with pytest.raises(RuntimeError):
        pyvsag.analyze_index(index_path=str(damaged))


def test_cli_writes_json_to_stdout(saved_index):
    index_path, _ = saved_index

    returncode, stdout, stderr = _run(
        [sys.executable, "-m", "pyvsag.analyze_index_cli", "--index_path", index_path]
    )

    assert returncode == 0, stderr
    report = json.loads(stdout)
    assert report["schema_version"] == pyvsag.ANALYSIS_SCHEMA_VERSION
    assert report["index"]["name"] == "brute_force"


def test_console_script_is_installed(saved_index):
    executable = shutil.which("pyvsag-analyze-index")
    if executable is None:
        pytest.fail("pyvsag-analyze-index console script is not installed")

    index_path, _ = saved_index
    returncode, stdout, stderr = _run([executable, "--index_path", index_path])

    assert returncode == 0, stderr
    assert json.loads(stdout)["schema_version"] == pyvsag.ANALYSIS_SCHEMA_VERSION


def test_cli_failure_uses_non_zero_exit_code(tmp_path):
    missing = str(tmp_path / "missing.vsag")

    returncode, stdout, stderr = _run(
        [sys.executable, "-m", "pyvsag.analyze_index_cli", "--index_path", missing]
    )

    assert returncode != 0
    assert stdout.strip() == ""
    assert "error" in stderr.lower()
