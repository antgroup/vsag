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

"""Programmatic entry points of :mod:`pyvsag.eval`."""

from . import _native
from .dataset import EvalDataset


def _check_arguments(index, dataset):
    if not hasattr(index, "_handle"):
        raise TypeError("index must be a pyvsag.Index")
    if not isinstance(dataset, EvalDataset):
        raise TypeError("dataset must be a pyvsag.eval.EvalDataset")


def _describe_index(index, options):
    options.setdefault("index_name", getattr(index, "_name", ""))
    options.setdefault("build_param", getattr(index, "_parameters", ""))


def evaluate_build(index, dataset, **options):
    """Build ``index`` from ``dataset.base`` and measure it.

    ``index`` must be a freshly created, still empty :class:`pyvsag.Index`. Keyword options are the
    ``eval_performance`` option names, e.g. ``enable_memory=False`` or ``num_threads_building=4``
    (applied for the duration of the build; to also size the index's own thread pool, create the
    index inside :func:`num_threads_building`). Returns a dict with the same keys as the C++ tool,
    such as ``duration(s)``, ``tps`` and ``index_memory(B)``.
    """
    _check_arguments(index, dataset)
    _describe_index(index, options)
    return _native.build(index._handle, dataset._base_dataset(), dataset.metric, options)


def evaluate_search(index, dataset, *, top_k=10, search_param=None, **options):
    """Run every query of ``dataset`` against a built ``index`` and measure it.

    Args:
        top_k: number of neighbours requested per query.
        search_param: the search parameters of the index, as a dict or a JSON string.
        **options: ``eval_performance`` option names such as ``num_threads_searching``,
            ``search_query_count``, ``enable_recall``, ``enable_percent_latency`` or
            ``use_id_based_recall``.

    Recall needs ``dataset.ground_truth`` with at least ``top_k`` neighbours per query (see
    :meth:`EvalDataset.with_ground_truth`); disable it with ``enable_recall=False`` and
    ``enable_percent_recall=False`` otherwise. Returns a dict with the same keys as the C++ tool,
    such as ``qps``, ``recall_avg``, ``latency_avg(ms)`` and ``latency_detail(ms)``.
    """
    _check_arguments(index, dataset)
    _describe_index(index, options)
    options["top_k"] = top_k
    if search_param is not None:
        options["search_param"] = search_param
    ground_truth = dataset.ground_truth
    return _native.search(
        index._handle,
        dataset._base_dataset(),
        dataset._queries_dataset(),
        ground_truth,
        0 if ground_truth is None else ground_truth.shape[1],
        dataset.metric,
        options,
    )


class num_threads_building:
    """Context manager that sets VSAG's process-wide build thread count and restores it."""

    def __init__(self, num_threads):
        self._num_threads = num_threads
        self._previous = None

    def __enter__(self):
        self._previous = _native.set_num_threads_building(self._num_threads)
        return self

    def __exit__(self, *exc_info):
        _native.set_num_threads_building(self._previous)
        return False
