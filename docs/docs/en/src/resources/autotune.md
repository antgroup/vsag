# AutoTune Tool (`autotune`)

`autotune` automates the VSAG build-and-search parameter sweep. AutoTune V1 evaluates real VSAG
indexes through the shared eval core, filters trials by constraints, and selects the best feasible
trial for one KNN workload.

The CLI JSON request and shared report schema are defined in the
[AutoTune V1 CLI JSON contract](autotune_api_v1.md).

V1 builds and tunes HGraph and IVF on dense float32 data. It also tunes search parameters for an
existing HGraph, IVF, or Pyramid index. The CLI reads HDF5; the C++ API accepts in-memory datasets
and indexes. The C++ API is experimental and build-tree-only; it is not installed with the VSAG
library.

## Building

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TOOLS=ON
cmake --build build-release --target autotune -j
```

The executable is `build-release/tools/autotune/autotune`.

A runnable request template is available at [the SIFT request template][sift-example].
It expects the dataset at `/tmp/sift-128-euclidean.hdf5`; edit `data_path` when necessary.

[sift-example]: https://github.com/antgroup/vsag/blob/main/tools/autotune/examples/sift_hgraph_autotune_request.json

## C++ API

The CLI and C++ entry points share candidate generation, trial planning, evaluation, constraint
filtering, and result selection. Only their input adapters differ:

- the CLI loads JSON and HDF5;
- `TuneIndex` accepts in-memory datasets and jointly tunes build and search parameters;
- `TuneSearch` accepts an already built or loaded `IndexPtr` and tunes search parameters only.

Consumers that link the build-tree `vsag::autotune` target can run the complete tune, select, and
load loop directly:

```cpp
vsag::autotune::IndexRequest request;
request.base = base;
request.metric_type = vsag::METRIC_L2;
request.workload = {queries, ground_truth, 10, 48};
request.index_spaces = {
    {"hgraph", create_candidate_space, search_candidate_space},
};
request.constraints = {
    {vsag::autotune::Metric::RECALL_AT_K, 0.95},
    {vsag::autotune::Metric::INDEX_MEMORY_MB, 8192.0},
};
request.objective = vsag::autotune::Metric::LATENCY_AVG_MS;

auto result = vsag::autotune::TuneIndex(request).value();
auto neighbors =
    result.index->KnnSearch(query, 10, result.search_parameters).value();
```

`base` and `queries` are dense float32 datasets. `base` IDs must be present and unique. The
ground-truth dataset stores `query_count` rows, uses `Dim()` as ground-truth K, and contains
flattened IDs from the base ID space. It is required when `RECALL_AT_K` is a constraint or the
objective. Dataset buffers must remain valid until the synchronous call returns.

`IndexSpace::create_parameter_space` and `search_parameter_space` remain JSON strings because each
index owns its native parameter schema. An empty `index_spaces` vector selects the built-in HGraph
and IVF spaces; otherwise arrays and `$range` expressions use the same semantics as the CLI
contract.

The result contains the loaded `index`, `index_name`, complete concrete `create_parameters`,
validated `search_parameters`, metrics, artifact path, and the complete report. `metrics` and
`report` are `JsonType` objects. The typed API returns the report in memory and never writes a
report file. The recommended artifact is retained, so another process can recreate an empty index
from `index_name + create_parameters` and deserialize that file. Unselected intermediate
artifacts are removed by default.

If no candidate satisfies every constraint, the completed call has
`status=TuneStatus::NO_FEASIBLE_CANDIDATE` and a structured `best_effort`; recommendation fields
such as `index` are not valid. Invalid requests and execution failures still return an error.

See
[`examples/cpp/326_feature_create_index_with_constraints.cpp`][factory-example] for a complete
example. Enable both tools and examples to build it:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TOOLS=ON \
  -DENABLE_EXAMPLES=ON
cmake --build build-release --target 326_feature_create_index_with_constraints -j
```

[factory-example]: https://github.com/antgroup/vsag/blob/main/examples/cpp/326_feature_create_index_with_constraints.cpp

## Request

Save a request such as the following as `request.json`. `dim`, `dtype`, and `metric_type` are read
from the dataset and need not be repeated. If provided, they must match the dataset.

```json
{
  "version": 1,
  "data_path": "/tmp/sift-128-euclidean.hdf5",
  "indexes": [
    {
      "name": "hgraph",
      "create_params": {
        "index_param": {
          "base_quantization_type": ["fp32", "sq8_uniform"],
          "max_degree": [16, 32],
          "ef_construction": 200
        }
      },
      "search_params": {
        "hgraph": {
          "ef_search": {
            "$range": {"start": 40, "stop": 1000}
          }
        }
      }
    }
  ],
  "workload": {
    "top_k": 10,
    "concurrency": 1
  },
  "constraints": {
    "recall_at_k": 0.95,
    "index_memory_mb": 8192
  },
  "objective": {
    "metric": "latency_avg_ms"
  },
  "tuning_config": {
    "workspace_path": "/tmp/vsag_autotune",
    "keep_intermediate": false,
    "max_trials": 1000
  }
}
```

Run it with:

```bash
./build-release/tools/autotune/autotune request.json
```

Every parameter leaf may be a scalar, an array, or a `$range` with `start`, `stop`, and `step`.
Explicit values are preserved. HGraph `ef_search` additionally accepts an integer `$range` without
`step`; AutoTune then doubles from `start` until it brackets a passing value and binary-searches
that interval for the smallest passing value. This assumes that recall does not decrease as
`ef_search` increases for the fixed configuration. Every evaluated point uses all queries. For
omitted tuning fields, V1 supplies a small built-in candidate set for HGraph or IVF. If `indexes`
is omitted, both indexes are evaluated.

Candidates with identical concrete create parameters share one build. Each concrete search
parameter set remains a separate trial. An adaptive `ef_search` range records only the concrete
points that were actually evaluated.

## Existing Index

Set `index_path` and provide exactly one index specification to tune search parameters without
rebuilding. Its `create_params` must be the concrete parameters used to create the serialized
index. AutoTune does not expand `create_params` in this mode, so native index fields that happen to
be arrays remain unchanged.

Programmatic callers use a separate `SearchRequest` and call `TuneSearch`:

```cpp
vsag::autotune::SearchRequest request;
request.index = existing_index;
request.workload = {queries, ground_truth, 10, 48};
request.parameter_space = R"({"hgraph":{"ef_search":[40,80,120]}})";
request.constraints = {{vsag::autotune::Metric::RECALL_AT_K, 0.95}};
request.objective = vsag::autotune::Metric::LATENCY_AVG_MS;

auto result = vsag::autotune::TuneSearch(request).value();
auto neighbors = existing_index->KnnSearch(query, 10, result.parameters).value();
```

This avoids serialization and file I/O and reuses the caller's index for all search trials.
AutoTune derives the index type and element count from `IndexPtr`. Search tuning therefore needs
queries but not the base dataset or metric type. When recall is requested, it also needs ground
truth and defines per-query `recall_at_k` as the intersection of the returned and ground-truth IDs
in their first `top_k` entries, divided by `top_k`; the reported value is the average over all
queries. Ground truth is optional when recall is neither a constraint nor the objective.

The CLI's `index_path` field remains an offline adapter: it uses the concrete create parameters to
create and deserialize an index, then enters the same search-only flow.

Pyramid is supported in this search-only mode. The HDF5 CLI adapter does not provide query paths,
so it can evaluate only Pyramid's native default/root search. Typed requests take paths directly
from `workload.queries->GetPaths()`: when paths are present, each query is evaluated against its
path; when absent, Pyramid applies its native root-search behavior. V1 supports only the default
unnamed hierarchy. To select different `ef_search` values for different paths, run one typed
AutoTune request per representative path workload, with the matching ground truth. V1 does not
aggregate path-specific recommendations into one result.

A complete example is available at
[`examples/cpp/327_feature_autotune_existing_index.cpp`][existing-index-example].
For Pyramid path tuning, see
[`examples/cpp/328_feature_autotune_existing_pyramid.cpp`][pyramid-example].
It tunes the same Pyramid index separately for 512-vector and 4096-vector leaf subgraphs under the
same recall target, illustrating why different paths may need different `ef_search` values.

[existing-index-example]: https://github.com/antgroup/vsag/blob/main/examples/cpp/327_feature_autotune_existing_index.cpp
[pyramid-example]: https://github.com/antgroup/vsag/blob/main/examples/cpp/328_feature_autotune_existing_pyramid.cpp

## Metrics

The following metrics can be used as constraints or as the objective:

| Metric | Constraint comparison |
| --- | --- |
| `recall_at_k`, `qps` | At least the requested value |
| `latency_avg_ms`, `latency_p99_ms` | At most the requested value |
| `index_memory_mb`, `index_size_mb` | At most the requested value |
| `build_seconds`, `search_seconds`, `build_and_search_seconds` | At most the requested value |

`build_seconds`, `index_size_mb`, and `build_and_search_seconds` are unavailable in
existing-index mode. `index_memory_mb` may be a constraint, but not the objective, because it is
constant across search candidates for the same existing index and cannot rank them. The objective
direction is inferred from the metric, so no `direction` field is needed.

AutoTune uses a dedicated latency/QPS pass and a separate recall/statistics pass, so a logical
query runs more than once per trial. `search_seconds` covers both in-memory passes and metric
collection, but excludes index deserialization. Candidates run in a fixed order without warm-up
or randomization. Measure latency and QPS on a representative, controlled machine and repeat runs
when these metrics determine the recommendation.

## Reading the Output

Standard output is a short summary ordered for copying:

- `recommendation`: the selected concrete parameters, metrics, and trial evidence. Build-and-search
  results also include create parameters and artifact/build evidence; search-only results do not.
- `best_effort`: present only when no trial satisfies every constraint.
- `failure`: present when the request or all evaluations fail.
- `report_path`: the full reproducible report.

The full report also contains every build and trial, constraint violations, the normalized request,
and structured per-stage failures. In search-only mode `builds` is empty and trials contain no
artifact or build fields. For both the CLI and typed `TuneIndex`, `keep_intermediate=false` retains
only the recommended generated index and removes unselected artifacts. Setting it to `true`
retains every generated index.

After initial request validation succeeds, the CLI and `RunAutoTune` persist the full report:
`output.result_path` overrides the path, otherwise a generated path under the workspace is used.
Early validation and request-file failures do not write a report. Typed `TuneIndex` and
`TuneSearch` never write a report file; callers receive the complete `JsonType` report in the
result.

## V1 Boundaries

V1 evaluates one KNN workload and performs a full sweep except for the supported HGraph
`ef_search` adaptive search. It does not provide filtered or range-search workloads, adaptive query
sampling, cross-request build cache, or model-based candidate generation.
