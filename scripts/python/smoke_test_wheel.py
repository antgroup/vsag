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

"""Clean-environment smoke test for the ``pyvsag`` wheel.

Run after installing a built wheel (no VSAG source tree, CMake or C++ compiler required). It
exercises the public ``pyvsag.analyze_index`` API, the installed ``pyvsag-analyze-index``
console script and the failure exit code contract.
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

import pyvsag

DIM = 8
NUM_VECTORS = 32
NUM_QUERIES = 1


def build_and_save_index(index_path):
    vectors = np.random.RandomState(42).randn(NUM_VECTORS, DIM).astype(np.float32)
    ids = np.arange(NUM_VECTORS, dtype=np.int64)
    params = json.dumps(
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
    index = pyvsag.Index("brute_force", params)
    index.build(vectors=vectors, ids=ids, num_elements=NUM_VECTORS, dim=DIM)
    index.save(index_path)
    queries = np.random.RandomState(7).randn(NUM_QUERIES, DIM).astype(np.float32)
    return queries


def write_dense_query_file(path, queries):
    rows, cols = queries.shape
    with open(path, "wb") as stream:
        stream.write(struct.pack("<II", rows, cols))
        stream.write(queries.tobytes())


def run_command(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def fail(message):
    print("pyvsag wheel smoke test failed: {}".format(message), file=sys.stderr)
    return 1


def main():
    workdir = tempfile.mkdtemp(prefix="pyvsag-analyze-smoke-")
    try:
        index_path = os.path.join(workdir, "index.vsag")
        query_path = os.path.join(workdir, "queries.bin")
        queries = build_and_save_index(index_path)
        write_dense_query_file(query_path, queries)

        # --- Python API: static analysis ---------------------------------
        report = pyvsag.analyze_index(index_path=index_path)
        json.dumps(report)  # must be JSON serializable
        if report.get("schema_version") != pyvsag.ANALYSIS_SCHEMA_VERSION:
            return fail("unexpected schema_version: {!r}".format(report.get("schema_version")))
        if report["index"]["name"] != "brute_force":
            return fail("unexpected index name: {!r}".format(report["index"]["name"]))
        if report["index"]["dim"] != DIM:
            return fail("unexpected index dimension: {!r}".format(report["index"]["dim"]))
        statistics = report["static_analysis"].get("statistics")
        if not isinstance(statistics, dict) or not statistics:
            return fail("static analysis did not return statistics")
        if report["dynamic_analysis"] is not None:
            return fail("static analysis unexpectedly produced dynamic results")

        # --- Python API: dynamic analysis --------------------------------
        dynamic_report = pyvsag.analyze_index(
            index_path=index_path,
            query_path=query_path,
            query_data_type="dense",
            topk=5,
        )
        dynamic = dynamic_report.get("dynamic_analysis")
        if not isinstance(dynamic, dict):
            return fail("dynamic analysis was not reported")
        if dynamic.get("topk") != 5:
            return fail("dynamic analysis did not echo topk")
        if not isinstance(dynamic.get("results"), dict):
            return fail("dynamic analysis did not return results")

        # --- CLI: success path -------------------------------------------
        module_command = [
            sys.executable,
            "-m",
            "pyvsag.analyze_index_cli",
            "--index_path",
            index_path,
        ]
        returncode, stdout, stderr = run_command(module_command)
        if returncode != 0:
            return fail("CLI module exited with {}: {}".format(returncode, stderr))
        try:
            cli_report = json.loads(stdout)
        except ValueError as error:
            return fail("CLI stdout is not valid JSON ({}): {!r}".format(error, stdout))
        if cli_report.get("schema_version") != pyvsag.ANALYSIS_SCHEMA_VERSION:
            return fail("CLI report has an unexpected schema_version")

        executable = shutil.which("pyvsag-analyze-index")
        if executable is None:
            return fail("pyvsag-analyze-index console script is not installed")
        returncode, stdout, stderr = run_command(
            [executable, "--index_path", index_path, "--topk", "5"]
        )
        if returncode != 0:
            return fail("console script exited with {}: {}".format(returncode, stderr))
        try:
            executable_report = json.loads(stdout)
        except ValueError as error:
            return fail(
                "console script stdout is not valid JSON ({}): {!r}".format(error, stdout)
            )
        if executable_report.get("schema_version") != pyvsag.ANALYSIS_SCHEMA_VERSION:
            return fail("console script report has an unexpected schema_version")

        # --- CLI: failure exit code and stderr routing --------------------
        missing_path = os.path.join(workdir, "does-not-exist.vsag")
        returncode, stdout, stderr = run_command(
            [executable, "--index_path", missing_path]
        )
        if returncode == 0:
            return fail("CLI returned zero for a missing index file")
        if stdout.strip():
            return fail("CLI wrote output to stdout on failure: {!r}".format(stdout))
        if not stderr.strip():
            return fail("CLI did not report an error on stderr")

        print("pyvsag wheel smoke test passed")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
