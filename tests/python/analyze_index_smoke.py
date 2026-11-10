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

"""Smoke test of pyvsag.analyze_index() and pyvsag-analyze-index in an installed wheel.

    analyze_index_smoke.py fixtures DIR          write index and query files (needs numpy)
    analyze_index_smoke.py check DIR [--bare]    test the API and the command on them
    analyze_index_smoke.py all DIR               both steps in one environment

"check" needs only the installed wheel. Run it with ``python -I`` from outside the source tree,
ideally in a fresh virtual environment or a container without a compiler; ``--bare`` also
asserts that numpy, cmake and C++ compilers are absent from that environment.
"""

import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys

DIM = 16
COUNT = 300
QUERIES = 8
TOPK = 10
REPORT_KEYS = [
    "schema_version",
    "metadata",
    "static_analysis",
    "dynamic_analysis",
    "warnings",
    "compatibility",
]


def write_fixtures(directory):
    import numpy as np
    import pyvsag

    os.makedirs(directory, exist_ok=True)
    rng = np.random.RandomState(7)
    vectors = rng.standard_normal((COUNT, DIM)).astype(np.float32)
    hgraph = pyvsag.Index(
        "hgraph",
        json.dumps(
            {
                "dtype": "float32",
                "metric_type": "l2",
                "dim": DIM,
                "index_param": {"base_quantization_type": "sq8", "max_degree": 16},
            }
        ),
    )
    hgraph.build(vectors, np.arange(COUNT, dtype=np.int64), COUNT, DIM)
    hgraph.save(os.path.join(directory, "hgraph.vsag"))
    with open(os.path.join(directory, "queries.bin"), "wb") as stream:
        stream.write(struct.pack("<II", QUERIES, DIM))
        stream.write(rng.standard_normal((QUERIES, DIM)).astype("<f4").tobytes())

    indptr, indices, values = [0], [], []
    for _ in range(COUNT):
        terms = np.sort(rng.choice(1000, size=8, replace=False))
        indices.extend(int(term) for term in terms)
        values.extend(float(value) for value in rng.uniform(0.1, 1.0, size=8))
        indptr.append(len(indices))
    sindi = pyvsag.Index(
        "sindi",
        json.dumps(
            {
                "dtype": "sparse",
                "metric_type": "ip",
                "index_param": {"term_id_limit": 2000, "doc_prune_ratio": 0.0},
            }
        ),
    )
    sindi.build(
        np.asarray(indptr, dtype=np.uint32),
        np.asarray(indices, dtype=np.uint32),
        np.asarray(values, dtype=np.float32),
        np.arange(COUNT, dtype=np.int64),
    )
    sindi.save(os.path.join(directory, "sindi.vsag"))
    rows = QUERIES
    with open(os.path.join(directory, "queries.csr"), "wb") as stream:
        stream.write(struct.pack("<qqq", rows, 1000, indptr[rows]))
        stream.write(np.asarray(indptr[: rows + 1], dtype="<i8").tobytes())
        stream.write(np.asarray(indices[: indptr[rows]], dtype="<i4").tobytes())
        stream.write(np.asarray(values[: indptr[rows]], dtype="<f4").tobytes())
    print("analyze_index smoke fixtures written to {}".format(directory))


class Checker(object):
    def __init__(self):
        self.count = 0

    def __call__(self, condition, what):
        if not condition:
            raise SystemExit("analyze_index smoke test failed: {}".format(what))
        self.count += 1


def find_command():
    names = ["pyvsag-analyze-index", "pyvsag-analyze-index.exe"]
    for name in names:
        candidate = os.path.join(os.path.dirname(sys.executable), name)
        if os.path.isfile(candidate):
            return candidate
    return shutil.which(names[0])


def run(command, *arguments, **environment):
    return subprocess.run(
        [command] + list(arguments),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=dict(os.environ, **environment),
    )


def run_checks(directory, bare):
    check = Checker()
    if bare:
        check(importlib.util.find_spec("numpy") is None, "numpy must not be installed")
        for tool in ("cmake", "c++", "g++", "clang++"):
            check(shutil.which(tool) is None, "{} must not be installed".format(tool))

    import pyvsag

    package = os.path.dirname(os.path.abspath(pyvsag.__file__))
    check(
        not os.path.exists(os.path.join(package, os.pardir, "setup.py")),
        "pyvsag must be imported from an installed wheel, not from {}".format(package),
    )
    hgraph = os.path.join(directory, "hgraph.vsag")
    sindi = os.path.join(directory, "sindi.vsag")

    report = pyvsag.analyze_index(hgraph)
    check(list(report) == REPORT_KEYS, "report members: {}".format(list(report)))
    check(report["schema_version"] == pyvsag.ANALYZE_INDEX_SCHEMA_VERSION, "schema version")
    check(report["metadata"]["index_type"] == "hgraph", "hgraph metadata")
    check(report["metadata"]["num_elements"] == COUNT, "hgraph element count")
    check(report["dynamic_analysis"] is None, "static analysis only")
    check(json.loads(json.dumps(report)) == report, "the report is JSON-serializable")

    dynamic = pyvsag.analyze_index(
        hgraph, query_path=os.path.join(directory, "queries.bin"), topk=TOPK
    )["dynamic_analysis"]
    check(dynamic["query_count"] == QUERIES, "dense dynamic analysis")
    check("recall_query" in dynamic["result"], "dense recall")
    sparse = pyvsag.analyze_index(sindi, query_path=os.path.join(directory, "queries.csr"))
    check(sparse["dynamic_analysis"]["query_data_type"] == "sparse", "sparse dynamic analysis")
    try:
        pyvsag.analyze_index(os.path.join(directory, "missing.vsag"))
        check(False, "a missing file must raise IndexAnalysisError")
    except pyvsag.IndexAnalysisError as error:
        check(error.code == "missing_file", "missing_file error code")

    command = find_command()
    check(command is not None, "the wheel must install the pyvsag-analyze-index command")
    result = run(
        command,
        "--index_path",
        hgraph,
        "--query_path",
        os.path.join(directory, "queries.bin"),
        "--topk",
        str(TOPK),
        VSAG_LOG_LEVEL="debug",
    )
    check(result.returncode == 0, "command exit status 0: {}".format(result.stderr))
    check(list(json.loads(result.stdout)) == REPORT_KEYS, "stdout holds only the JSON report")
    result = run(command, "--index_path", sindi, "--compact")
    check(result.returncode == 0 and result.stdout.count("\n") == 1, "compact report")
    result = run(command, "--index_path", os.path.join(directory, "missing.vsag"))
    check(result.returncode == 1, "a failed analysis exits with status 1")
    check(result.stdout == "", "a failed analysis writes nothing to stdout")
    check("error [missing_file]" in result.stderr, "the error is reported on stderr")
    result = run(command)
    check(result.returncode == 2 and result.stdout == "", "an invalid command line exits with 2")
    result = run(command, "--version")
    check(result.stdout.strip() == "pyvsag {}".format(pyvsag.__version__), "--version")
    print(
        "analyze_index smoke test passed: {} checks, pyvsag {} from {}".format(
            check.count, pyvsag.__version__, package
        )
    )


def main(argv):
    if len(argv) < 2 or argv[0] not in ("fixtures", "check", "all"):
        raise SystemExit(__doc__)
    step, directory = argv[0], os.path.abspath(argv[1])
    if step in ("fixtures", "all"):
        write_fixtures(directory)
    if step in ("check", "all"):
        run_checks(directory, bare="--bare" in argv[2:])


if __name__ == "__main__":
    main(sys.argv[1:])
