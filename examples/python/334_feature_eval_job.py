#  Copyright 2024-present the vsag project
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Run an ``eval_performance`` YAML job from Python, or with the ``pyvsag-eval`` command.

Needs the optional dependencies: ``pip install "pyvsag[eval]"`` (h5py and pyyaml).

The example writes a small ann-benchmarks style HDF5 file and a job file into a temporary
directory, then runs the job in three ways: ``EvalDataset.from_hdf5``, ``run_job`` with the YAML
file and the ``pyvsag-eval`` command line entry point.
"""

import json
import os
import tempfile

import h5py
import numpy as np
import yaml

from pyvsag.eval import EvalDataset, compute_ground_truth, run_job
from pyvsag.eval.__main__ import main as pyvsag_eval

DIM = 32
TOP_K = 10


def write_dataset(path):
    rng = np.random.default_rng(7)
    base = rng.random((3000, DIM), dtype=np.float32)
    queries = rng.random((100, DIM), dtype=np.float32)
    neighbors, distances = compute_ground_truth(base, queries, TOP_K, "l2")
    with h5py.File(path, "w") as file:
        file.create_dataset("train", data=base)
        file.create_dataset("test", data=queries)
        file.create_dataset("neighbors", data=neighbors)
        file.create_dataset("distances", data=distances)
        file.attrs["distance"] = "euclidean"


def write_job(path, datapath, result_path, index_path):
    create_params = json.dumps(
        {
            "dim": DIM,
            "dtype": "float32",
            "metric_type": "l2",
            "index_param": {"base_quantization_type": "fp32", "max_degree": 32},
        }
    )
    search_params = json.dumps({"hgraph": {"ef_search": 100}})
    job = {
        "global": {
            "num_threads_building": 4,
            "exporters": {
                "table": {"format": "table", "to": "stdout"},
                "json": {"format": "json", "to": "file://" + result_path},
            },
        },
        # Build and search in memory.
        "hgraph_build_search": {
            "datapath": datapath,
            "type": "build,search",
            "index_name": "hgraph",
            "create_params": create_params,
            "search_params": search_params,
            "topk": TOP_K,
        },
        # Build and save the index; a second case can then search it from index_path.
        "hgraph_build": {
            "datapath": datapath,
            "type": "build",
            "index_name": "hgraph",
            "create_params": create_params,
            "index_path": index_path,
        },
        "hgraph_search": {
            "datapath": datapath,
            "type": "search",
            "index_name": "hgraph",
            "create_params": create_params,
            "search_params": search_params,
            "index_path": index_path,
            "topk": TOP_K,
        },
    }
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(job, file)


def main():
    with tempfile.TemporaryDirectory() as directory:
        datapath = os.path.join(directory, "data.hdf5")
        job_path = os.path.join(directory, "job.yaml")
        result_path = os.path.join(directory, "result.json")
        write_dataset(datapath)
        write_job(job_path, datapath, result_path, os.path.join(directory, "index"))

        # 1. Load the dataset by itself to use it with evaluate_build / evaluate_search.
        dataset = EvalDataset.from_hdf5(datapath)
        print(dataset)

        # 2. Run the job; export=True sends the results to the exporters of the "global"
        #    section (or prints a table when there are none).
        results = run_job(job_path, export=True)
        for name, result in results.items():
            print(name, result["action"])

        # 3. The same thing as a command line: `pyvsag-eval job.yaml` (or
        #    `python -m pyvsag.eval job.yaml`).
        pyvsag_eval([job_path])


if __name__ == "__main__":
    main()
