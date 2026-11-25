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

"""Measure VSAG indexes from Python with the native ``eval_performance`` measurement core.

Quick start::

    import numpy as np
    import pyvsag
    from pyvsag.eval import EvalDataset, evaluate_build, evaluate_search

    dataset = EvalDataset.from_arrays(base, queries, metric="l2").with_ground_truth(10)
    index = pyvsag.Index("hgraph", params)
    build = evaluate_build(index, dataset)
    search = evaluate_search(index, dataset, top_k=10, search_param=search_params)

YAML jobs compatible with ``eval_performance`` are run by :func:`run_job` or ``pyvsag-eval``.
"""

from .api import evaluate_build, evaluate_search, num_threads_building
from .dataset import EvalDataset, compute_ground_truth
from .export import export_results, format_results
from .job import run_job

__all__ = [
    "EvalDataset",
    "compute_ground_truth",
    "evaluate_build",
    "evaluate_search",
    "export_results",
    "format_results",
    "num_threads_building",
    "run_job",
]
