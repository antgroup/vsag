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

"""Run ``eval_performance`` style jobs (a YAML file or the equivalent dict) from Python."""

import json
import os
import warnings

from .._index import Index
from .api import evaluate_build, evaluate_search, num_threads_building
from .dataset import EvalDataset
from .export import export_results, format_results

_REQUIRED_KEYS = ("datapath", "type", "index_name", "create_params")

# Case keys that are passed to the native measurement core, renamed where it uses other names.
_RENAMED_KEYS = {
    "topk": "top_k",
    "range": "radius",
    "search_params": "search_param",
}
_PASSTHROUGH_KEYS = (
    "search_mode",
    "search_query_count",
    "delete_index_after_search",
    "use_id_based_recall",
)
_DISABLE_KEYS = {
    "disable_recall": "enable_recall",
    "disable_percent_recall": "enable_percent_recall",
    "disable_qps": "enable_qps",
    "disable_tps": "enable_tps",
    "disable_memory": "enable_memory",
    "disable_latency": "enable_latency",
    "disable_percent_latency": "enable_percent_latency",
}
_OTHER_KEYS = (
    "datapath",
    "type",
    "index_name",
    "create_params",
    "index_path",
    "num_threads_building",
    "num_threads_searching",
)
_KNOWN_KEYS = (
    set(_RENAMED_KEYS) | set(_PASSTHROUGH_KEYS) | set(_DISABLE_KEYS) | set(_OTHER_KEYS)
)
_ACTIONS = ("build", "search", "build,search")
_DEFAULT_NUM_THREADS = 1


def load_job(job):
    """Return the job as a dict. ``job`` is a path to a YAML file or an already parsed dict."""
    if isinstance(job, dict):
        return job
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "reading YAML jobs requires pyyaml; install it with 'pip install pyvsag[eval]'"
        ) from None
    with open(os.fspath(job), "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError("the root node of the job file is not a map")
    return loaded


def _text(value):
    return value if isinstance(value, str) else json.dumps(value)


def _validate_case(name, case):
    if not isinstance(case, dict):
        raise ValueError("case %r must be a map" % name)
    for key in _REQUIRED_KEYS:
        if key not in case:
            raise ValueError("case %r is missing required key %r" % (name, key))
    action = case["type"]
    if action not in _ACTIONS:
        raise ValueError("case %r: type must be one of %s, got %r" % (name, list(_ACTIONS), action))
    if action != "build" and "search_params" not in case:
        raise ValueError("case %r: 'search_params' is required when type is %r" % (name, action))
    if action == "search" and not case.get("index_path"):
        raise ValueError("case %r: 'index_path' is required when type is 'search'" % name)
    unknown = sorted(set(case) - _KNOWN_KEYS)
    if unknown:
        warnings.warn("case %r: ignoring unknown keys %s" % (name, unknown), stacklevel=3)


def _native_options(case, global_options):
    options = {}
    for key in ("num_threads_building", "num_threads_searching"):
        if key in global_options:
            options[key] = global_options[key]
        if key in case:
            options[key] = case[key]
    for key, native_key in _RENAMED_KEYS.items():
        if key in case:
            options[native_key] = case[key]
    for key in _PASSTHROUGH_KEYS:
        if key in case:
            options[key] = case[key]
    for key, native_key in _DISABLE_KEYS.items():
        if case.get(key):
            options[native_key] = False
    return options


def _run_case(name, case, global_options, datasets):
    _validate_case(name, case)
    action = case["type"]
    options = _native_options(case, global_options)
    path = os.fspath(case["datapath"])
    if path not in datasets:
        datasets[path] = EvalDataset.from_hdf5(path)
    dataset = datasets[path]

    build_options = {
        key: options[key]
        for key in ("num_threads_building", "enable_memory", "enable_tps")
        if key in options
    }
    search_options = {key: value for key, value in options.items() if key != "num_threads_building"}
    threads = options.get("num_threads_building", _DEFAULT_NUM_THREADS)
    index_path = case.get("index_path")
    delete_after = bool(search_options.pop("delete_index_after_search", False))
    search_param = search_options.pop("search_param", None)
    top_k = search_options.pop("top_k", 10)

    with num_threads_building(threads):
        index = Index(case["index_name"], _text(case["create_params"]))
        result = {}
        if action in ("build", "build,search"):
            result = evaluate_build(index, dataset, **build_options)
            if action == "build" and index_path:
                directory = os.path.dirname(os.path.abspath(index_path))
                os.makedirs(directory, exist_ok=True)
                index.save(index_path)
        else:
            index.load(index_path)

        if action in ("search", "build,search"):
            search_result = evaluate_search(
                index, dataset, top_k=top_k, search_param=search_param, **search_options
            )
            if action == "search" and delete_after:
                os.remove(index_path)
            result.update(search_result)
            result["action"] = action
    return result


def run_job(job, export=False):
    """Run every case of an ``eval_performance`` job and return ``{case_name: result}``.

    ``job`` is the path of a YAML file (needs ``pyyaml``) or a dict with the same structure: an
    optional ``global`` section (``exporters``, ``num_threads_building``,
    ``num_threads_searching``; ``http_server`` is not supported and ignored) and one entry per
    case using the ``eval_performance`` YAML keys (``datapath``, ``type``, ``index_name``,
    ``create_params``, ``search_params``, ``topk``, ...). ``datapath`` is an ann-benchmarks HDF5
    file with dense vectors.

    Differences from the C++ tool: ``build,search`` keeps the index in memory instead of writing
    it to ``index_path``, and cases are run sequentially in this process.

    With ``export=True`` results are also sent to the global exporters, or printed as a table
    when there are none.
    """
    config = load_job(job)
    global_section = config.get("global") or {}
    global_options = {
        key: global_section[key]
        for key in ("num_threads_building", "num_threads_searching")
        if key in global_section
    }
    if "http_server" in global_section:
        warnings.warn("global.http_server is not supported by pyvsag.eval and is ignored")

    results = {}
    datasets = {}
    for name, case in config.items():
        if name == "global":
            continue
        results[str(name)] = _run_case(name, case, global_options, datasets)

    if export:
        exporters = global_section.get("exporters") or {}
        if not exporters:
            export_results(results, "table", "stdout")
        for exporter in exporters.values():
            export_results(
                results,
                exporter.get("format", "json"),
                exporter.get("to", "stdout"),
                exporter.get("vars"),
            )
    return results


__all__ = ["load_job", "run_job", "format_results"]
