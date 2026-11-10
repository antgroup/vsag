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

"""Tests of pyvsag.analyze_index() and the pyvsag-analyze-index command."""

import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

import pyvsag

DIM = 16
COUNT = 300
QUERY_COUNT = 8
TOPK = 10

HGRAPH_PARAMETER = {
    "dtype": "float32",
    "metric_type": "l2",
    "dim": DIM,
    "index_param": {"base_quantization_type": "fp32", "max_degree": 16, "ef_construction": 100},
}
BRUTE_FORCE_PARAMETER = {
    "dtype": "float32",
    "metric_type": "ip",
    "dim": DIM,
    "index_param": {"base_quantization_type": "fp32"},
}
SINDI_PARAMETER = {
    "dtype": "sparse",
    "metric_type": "ip",
    "index_param": {"term_id_limit": 2000, "doc_prune_ratio": 0.0, "window_size": 10000},
}
REPORT_KEYS = [
    "schema_version",
    "metadata",
    "static_analysis",
    "dynamic_analysis",
    "warnings",
    "compatibility",
]


def build_dense_index(directory, name, parameter):
    rng = np.random.RandomState(7)
    vectors = rng.standard_normal((COUNT, DIM)).astype(np.float32)
    index = pyvsag.Index(name, json.dumps(parameter))
    index.build(vectors, np.arange(COUNT, dtype=np.int64), COUNT, DIM)
    path = str(directory / (name + ".vsag"))
    index.save(path)
    return path


def write_dense_queries(path, rows=QUERY_COUNT, dim=DIM):
    data = np.random.RandomState(11).standard_normal((rows, dim)).astype("<f4")
    with open(str(path), "wb") as stream:
        stream.write(struct.pack("<II", rows, dim))
        stream.write(data.tobytes())
    return str(path)


def make_csr(rows, seed):
    rng = np.random.RandomState(seed)
    indptr, indices, values = [0], [], []
    for _ in range(rows):
        terms = np.sort(rng.choice(1000, size=int(rng.randint(3, 12)), replace=False))
        indices.extend(int(term) for term in terms)
        values.extend(float(value) for value in rng.uniform(0.1, 1.0, size=len(terms)))
        indptr.append(len(indices))
    return indptr, indices, values


def write_csr(path, csr, cols=1000):
    indptr, indices, values = csr
    with open(str(path), "wb") as stream:
        stream.write(struct.pack("<qqq", len(indptr) - 1, cols, len(indices)))
        stream.write(np.asarray(indptr, dtype="<i8").tobytes())
        stream.write(np.asarray(indices, dtype="<i4").tobytes())
        stream.write(np.asarray(values, dtype="<f4").tobytes())
    return str(path)


def build_sindi_index(directory):
    indptr, indices, values = make_csr(200, 3)
    index = pyvsag.Index("sindi", json.dumps(SINDI_PARAMETER))
    index.build(
        np.asarray(indptr, dtype=np.uint32),
        np.asarray(indices, dtype=np.uint32),
        np.asarray(values, dtype=np.float32),
        np.arange(len(indptr) - 1, dtype=np.int64),
    )
    path = str(directory / "sindi.vsag")
    index.save(path)
    return path


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("analyze_index")
    return {
        "directory": directory,
        "hgraph": build_dense_index(directory, "hgraph", HGRAPH_PARAMETER),
        "brute_force": build_dense_index(directory, "brute_force", BRUTE_FORCE_PARAMETER),
        "sindi": build_sindi_index(directory),
        "dense_queries": write_dense_queries(directory / "queries.bin"),
        "sparse_queries": write_csr(directory / "queries.csr", make_csr(4, 5)),
    }


def test_static_analysis_report_structure(files):
    report = pyvsag.analyze_index(files["hgraph"])
    assert list(report) == REPORT_KEYS
    assert report["schema_version"] == pyvsag.ANALYZE_INDEX_SCHEMA_VERSION == 1
    metadata = report["metadata"]
    assert metadata["index_path"] == files["hgraph"]
    assert metadata["index_type"] == "hgraph"
    assert metadata["dim"] == DIM
    assert metadata["data_type"] == "float32"
    assert metadata["metric"] == "l2"
    assert metadata["num_elements"] == COUNT
    assert metadata["parameter_source"] == "file"
    assert report["static_analysis"]["_analysis"]["analysis_type"] == "stats"
    assert report["dynamic_analysis"] is None
    assert report["warnings"] == []
    assert report["compatibility"]["serialization_format"] == "footer"
    assert json.loads(json.dumps(report)) == report


def test_dynamic_analysis_with_dense_queries(files):
    report = pyvsag.analyze_index(
        pathlib.Path(files["hgraph"]), query_path=pathlib.Path(files["dense_queries"]), topk=TOPK
    )
    dynamic = report["dynamic_analysis"]
    assert dynamic["query_count"] == QUERY_COUNT
    assert dynamic["query_dim"] == DIM
    assert dynamic["search_parameter"] == {"hgraph": {"ef_search": 100}}
    assert dynamic["search_parameter_source"] == "default"
    assert dynamic["result"]["_analysis"]["sample_count"] == QUERY_COUNT
    assert 0.0 <= dynamic["result"]["recall_query"] <= 1.0

    tuned = pyvsag.analyze_index(
        files["hgraph"],
        query_path=files["dense_queries"],
        search_parameter={"hgraph": {"ef_search": 20}},
        topk=np.int64(TOPK),
    )
    assert tuned["dynamic_analysis"]["search_parameter"] == {"hgraph": {"ef_search": 20}}
    assert tuned["dynamic_analysis"]["search_parameter_source"] == "user"


def test_metadata_defaults_and_build_parameter(files):
    report = pyvsag.analyze_index(files["brute_force"], query_path=files["dense_queries"], topk=5)
    assert [warning["code"] for warning in report["warnings"]] == ["metadata_defaults"]
    assert report["compatibility"]["defaulted_fields"] == {"dtype": "float32", "metric_type": "l2"}
    assert report["dynamic_analysis"]["result"]["result_count"] == QUERY_COUNT * 5

    explicit = pyvsag.analyze_index(files["brute_force"], build_parameter=BRUTE_FORCE_PARAMETER)
    assert explicit["metadata"]["metric"] == "ip"
    assert explicit["metadata"]["parameter_source"] == "user"
    assert explicit["metadata"]["build_parameter"] == BRUTE_FORCE_PARAMETER
    assert [warning["code"] for warning in explicit["warnings"]] == ["build_parameter_override"]


def test_sparse_analysis(files):
    report = pyvsag.analyze_index(files["sindi"], query_path=files["sparse_queries"], topk=TOPK)
    assert report["metadata"]["index_type"] == "sindi"
    assert report["metadata"]["dim"] is None
    assert report["dynamic_analysis"]["query_data_type"] == "sparse"
    assert report["dynamic_analysis"]["query_count"] == 4


@pytest.mark.parametrize(
    "arguments, code",
    [
        ({"index_path": "/nonexistent/index.vsag"}, "missing_file"),
        ({"query_data_type": "csr", "query_path": "queries"}, "invalid_argument"),
        ({"topk": 0, "query_path": "queries"}, "invalid_argument"),
        ({"build_parameter": "{not json"}, "invalid_argument"),
        ({"groundtruth_path": "queries"}, "invalid_argument"),
        ({"base_path": "queries", "query_path": "queries"}, "invalid_binary"),
        ({"topk": 2 ** 70, "query_path": "queries"}, "invalid_argument"),
    ],
)
def test_analysis_errors(files, arguments, code):
    arguments = dict(arguments)
    index_path = arguments.pop("index_path", files["hgraph"])
    for name in ("query_path", "groundtruth_path", "base_path"):
        if arguments.get(name) == "queries":
            arguments[name] = files["dense_queries"]
    with pytest.raises(pyvsag.IndexAnalysisError) as raised:
        pyvsag.analyze_index(index_path, **arguments)
    assert raised.value.code == code
    assert raised.value.message
    assert isinstance(raised.value, RuntimeError)
    assert str(raised.value).startswith("[{}] ".format(code))


@pytest.mark.parametrize(
    "arguments",
    [
        {"topk": "10"},
        {"topk": True},
        {"topk": 1.5},
        {"build_parameter": 5},
        {"search_parameter": [1]},
        {"query_path": 3},
        {"query_data_type": None},
    ],
)
def test_argument_types(files, arguments):
    with pytest.raises(TypeError):
        pyvsag.analyze_index(files["hgraph"], **arguments)


def test_index_path_is_required_and_options_are_keyword_only(files):
    with pytest.raises(TypeError):
        pyvsag.analyze_index(None)
    with pytest.raises(TypeError):
        pyvsag.analyze_index(files["hgraph"], "default")


def command_variants():
    variants = [[sys.executable, "-m", "pyvsag_analyze_index"]]
    # Scripts are installed next to the interpreter of a virtual environment, which need not
    # be on PATH.
    script = os.path.join(os.path.dirname(sys.executable), "pyvsag-analyze-index")
    if not os.path.isfile(script):
        script = shutil.which("pyvsag-analyze-index")
    if script is not None:
        variants.append([script])
    return variants


@pytest.fixture(params=command_variants(), ids=lambda command: os.path.basename(command[-1]))
def command(request):
    return request.param


def run(command, *arguments, **environment):
    env = dict(os.environ, **environment)
    return subprocess.run(
        command + list(arguments),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )


def test_command_prints_only_json_to_stdout(files, command):
    result = run(
        command,
        "--index_path",
        files["hgraph"],
        "-qp",
        files["dense_queries"],
        "-k",
        str(TOPK),
        VSAG_LOG_LEVEL="debug",
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert list(report) == REPORT_KEYS
    assert report["dynamic_analysis"]["query_count"] == QUERY_COUNT
    # The command reports what the Python API reports (timings aside).
    assert report["metadata"] == pyvsag.analyze_index(files["hgraph"])["metadata"]


def test_command_options(files, command):
    compact = run(command, "--index-path", files["sindi"], "--compact")
    assert compact.returncode == 0, compact.stderr
    assert compact.stdout.count("\n") == 1
    assert json.loads(compact.stdout)["metadata"]["index_type"] == "sindi"

    usage = run(command, "--help")
    assert usage.returncode == 0
    assert "--index_path" in usage.stdout and "--search-parameter" in usage.stdout

    version = run(command, "--version")
    assert version.returncode == 0
    assert version.stdout.strip() == "pyvsag {}".format(pyvsag.__version__)


@pytest.mark.parametrize(
    "arguments, status, message",
    [
        (["--index_path", "/nonexistent/index.vsag"], 1, "error [missing_file]"),
        (["--index_path", "HGRAPH", "--topk", "0", "-qp", "QUERIES"], 1, "[invalid_argument]"),
        ([], 2, "required"),
        (["--index_path", "HGRAPH", "--topk", "many"], 2, "invalid int value"),
        (["--index_path", "HGRAPH", "--query_data_type", "csr"], 2, "invalid choice"),
    ],
)
def test_command_failures(files, command, arguments, status, message):
    arguments = [
        files["hgraph"] if a == "HGRAPH" else files["dense_queries"] if a == "QUERIES" else a
        for a in arguments
    ]
    result = run(command, *arguments)
    assert result.returncode == status
    assert result.stdout == ""
    assert message in result.stderr
