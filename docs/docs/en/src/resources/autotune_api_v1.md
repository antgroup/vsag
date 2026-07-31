# AutoTune V1 CLI JSON Input and Output Contract

This page defines the offline JSON/HDF5 adapter used by the `autotune` command-line tool. The
experimental, build-tree-only C++ API instead accepts typed `IndexRequest` and `SearchRequest`
objects. It is not installed with the VSAG library. All entry points use the same normalized
request and tuning engine after input conversion. AutoTune V1 supports one full-dataset KNN
workload on dense float32 data, with HGraph and IVF as the supported build-and-search index types.

Unknown fields are rejected in the request, index specification, workload, objective,
`tuning_config`, and `output` objects. Native fields inside `create_params` and `search_params` are
passed to the concrete VSAG index and are validated when that index is created or searched.

## Request Object

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `version` | integer | Yes | Must be `1`. |
| `data_path` | string | Yes | Readable HDF5 evaluation dataset. |
| `index_path` | string | No | Existing serialized index for search-only tuning. |
| `indexes` | array | No | Index specifications. Defaults to HGraph and IVF in normal mode. |
| `workload` | object | Yes | The KNN workload being optimized. |
| `constraints` | object | Yes | Non-empty metric-to-threshold map. |
| `objective` | object | Yes | Metric used to rank feasible candidates. |
| `tuning_config` | object | No | Workspace, artifact retention, and trial limit. |
| `output` | object | No | Report path and raw-eval control. |

### Dataset Rules

`data_path` must identify a readable regular file in the HDF5 format accepted by the VSAG
evaluation tool. V1 requires:

- dense vectors;
- float32 base and query vectors;
- at least one base vector, query vector, and ground-truth neighbor;
- `workload.top_k` no greater than the ground-truth width when recall is requested;
- a distance attribute that maps to `l2`, `ip`, or `cosine`.

AutoTune reads `dim`, `dtype`, and `metric_type` from the dataset and injects them into every
concrete `create_params`. A request may repeat one of these fields only when it exactly matches the
dataset.

## Index Specifications

Each `indexes[]` item has the following shape:

```json
{
  "name": "hgraph",
  "create_params": {
    "index_param": {
      "max_degree": [16, 32]
    }
  },
  "search_params": {
    "hgraph": {
      "ef_search": {"$range": {"start": 40, "stop": 1000}}
    }
  }
}
```

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `name` | string | Yes | `hgraph`, `ivf`, or search-only `pyramid`, case-insensitive. |
| `create_params` | object | No | VSAG create parameters or candidate expressions. |
| `search_params` | object | No | Search parameters, keyed by the same index name. |

When `indexes` is absent in normal build-and-search mode, AutoTune evaluates both HGraph and IVF.
When `index_path` is present, `indexes` is required and must contain exactly one item.
Pyramid is accepted only with an existing in-memory or serialized index.

### Candidate Expressions

Every leaf in `create_params` and `search_params` can be written as:

- a scalar, which fixes that value;
- a non-empty array, which provides discrete candidate values;
- an inclusive `$range` containing exactly `start`, `stop`, and non-zero `step`.

Examples:

```json
{
  "fixed": 32,
  "choices": [16, 32, 48],
  "integer_range": {"$range": {"start": 40, "stop": 120, "step": 40}},
  "float_range": {"$range": {"start": 0.1, "stop": 0.3, "step": 0.1}}
}
```

HGraph `ef_search` has one additional form:

```json
{"$range": {"start": 40, "stop": 1000}}
```

The bounds must be positive integers with `start <= stop`. This form requires a `recall_at_k`
constraint and an objective of `latency_avg_ms`, `latency_p99_ms`, `qps`, `search_seconds`, or
`build_and_search_seconds`. It cannot be combined with `timeout_ms` or `hops_limit`. For every
fixed combination of the other parameters, AutoTune evaluates concrete `ef_search` values and
doubles from `start`, capped at `stop`, until recall satisfies the constraint. It then
binary-searches the last failing/passing interval for the smallest passing value. Branching uses
recall only; latency and QPS remain measured metrics used by final constraint evaluation and
selection. This strategy assumes that recall does not decrease as `ef_search` increases while all
other parameters and the workload remain fixed.

Every probe evaluates the full query workload. If a probe fails to execute or does not produce
recall, AutoTune stops that adaptive range because it cannot choose the next interval safely.

V1 treats every JSON array and stepped range in these parameter objects as candidate choices. It
computes their full Cartesian product and removes identical complete candidates. A no-step range
is not accepted for other parameters.

The request is rejected when the planned worst-case evaluation count exceeds
`tuning_config.max_trials`. A normal concrete candidate costs one trial. An adaptive `ef_search`
range costs one when `start == stop`. Otherwise, the planner reserves the largest probe count
through any possible passing bound plus the binary-search cost of that bound's preceding interval.
For example, `40..1000` reserves 15 trials in the worst case.

### Built-in Missing-Field Candidates

Explicit user fields always win. The following values are supplied only when the corresponding
field is absent:

| Index | Missing field | V1 candidates |
| --- | --- | --- |
| HGraph | `base_quantization_type` | `fp32`, `sq8_uniform` |
| HGraph | `max_degree` | `16`, `32` |
| HGraph | `ef_construction` | `100`, `200` |
| HGraph | `ef_search` | Three values derived from `top_k`, with floors `40`, `80`, `120` |
| Pyramid, search-only | `ef_search` | Three values derived from `top_k`, with floors `40`, `80`, `120` |
| IVF | `base_quantization_type` | `fp32`, `sq8_uniform` |
| IVF | `buckets_count` | Up to `1024` and `2048`, capped by base-vector count |
| IVF, build-and-search | `scan_buckets_count` | Up to four values derived from concrete `buckets_count` |
| IVF, search-only | `scan_buckets_count` | `1`, `4`, `16`, `64` |

The exact HGraph and Pyramid search values are `max(40, top_k)`, `max(80, 2 * top_k)`, and
`max(120, 4 * top_k)`, with duplicates removed. Build-and-search IVF candidates never exceed the
concrete bucket count. AutoTune cannot infer an existing IVF index's bucket count, so search-only
mode uses the conservative generic values above. Values rejected by the loaded index become
failed trials; callers can avoid them by providing an explicit `scan_buckets_count` space.

Candidates with the same normalized index name and identical concrete `create_params` belong to
one build group. AutoTune builds that group once, serializes it once as evidence, and evaluates
every associated search candidate against the same in-memory index instance.

### Existing-Index Mode

With `index_path`:

- the path must identify a readable regular file;
- exactly one index specification is required;
- `create_params.index_param` is required;
- `create_params` is treated as one concrete native index configuration and is not expanded;
- AutoTune does not generate missing create candidates or invoke build;
- missing search fields may still receive built-in search candidates;
- `build_seconds`, `index_size_mb`, and `build_and_search_seconds` cannot be constraints or
  objectives;
- `index_memory_mb` can be a constraint but not the objective;
- the caller-owned index is never deleted.

The concrete create parameters are needed to instantiate the correct VSAG index before
deserialization. AutoTune does not infer them from the serialized file in V1.

For a typed `SearchRequest`, the index is already instantiated and loaded. AutoTune derives its
index type and element count from `IndexPtr`; the request supplies only the index, workload, search
`parameter_space`, constraints, objective, and config. It needs neither create parameters nor the
base dataset or metric type.

The HDF5 adapter does not provide Pyramid query paths, so the CLI can evaluate only Pyramid's
native default/root search. Path-specific Pyramid tuning requires a typed `SearchRequest` whose
query dataset supplies `Dataset::Paths()`. V1 forwards only the default unnamed hierarchy; named
Pyramid hierarchies are not supported by AutoTune.

## Workload

```json
{
  "workload": {
    "top_k": 10,
    "concurrency": 48
  }
}
```

| Field | Type | Required | Default | Validation |
| --- | --- | --- | --- | --- |
| `top_k` | positive integer | Yes | — | No greater than base count, and no greater than ground-truth width when recall is requested; representable by native `int`. |
| `concurrency` | positive integer | No | `1` | Maximum `200`. |

V1 always evaluates the complete query set in the HDF5 file and supports only KNN search. The
benchmark machine, system load, and runtime environment are part of the measured workload.
Latency and QPS are not portable across different machine specifications or load conditions.
For a typed `SearchRequest`, `top_k` is checked against `IndexPtr::GetNumElements()` instead of a
base dataset. Ground truth may be null unless recall is a constraint or the objective.

## Constraints and Objective

`constraints` maps metric names directly to thresholds. `objective.metric` names one metric. The
comparison and objective direction are inferred from the metric and cannot be overridden.

```json
{
  "constraints": {
    "recall_at_k": 0.95,
    "latency_p99_ms": 2.0,
    "index_memory_mb": 8192
  },
  "objective": {
    "metric": "latency_avg_ms"
  }
}
```

| Metric | Constraint | Objective direction | Existing-index use |
| --- | --- | --- | --- |
| `recall_at_k` | actual ≥ threshold | Maximize | Yes |
| `qps` | actual ≥ threshold | Maximize | Yes |
| `latency_avg_ms` | actual ≤ threshold | Minimize | Yes |
| `latency_p99_ms` | actual ≤ threshold | Minimize | Yes |
| `index_memory_mb` | actual ≤ threshold | Minimize | Constraint only |
| `index_size_mb` | actual ≤ threshold | Minimize | No |
| `build_seconds` | actual ≤ threshold | Minimize | No |
| `search_seconds` | actual ≤ threshold | Minimize | Yes |
| `build_and_search_seconds` | actual ≤ threshold | Minimize | No |

Thresholds must be finite and non-negative. `recall_at_k` must also be at most `1.0`.

Metric meanings in V1:

- For each query, `recall_at_k` is the size of the intersection between the first `top_k`
  returned IDs and ground-truth IDs, divided by `top_k`; the reported metric is the average over
  all queries. It needs ground truth but does not need base vectors or a metric type.
- `build_seconds` measures the index `Build` operation reported by the evaluation tool.
- It excludes index serialization, dataset loading, and candidate orchestration.
- `search_seconds` measures the wall time of a complete in-memory search evaluation trial. It
  excludes index deserialization, but includes all search passes and metric collection. Use
  latency or QPS for serving-performance goals.
- `build_and_search_seconds` is the sum of those two metrics for a build-and-search trial.
- `index_memory_mb` comes from the loaded index's positive `GetMemoryUsage()` result. When an index
  reports zero, AutoTune treats the metric as unavailable instead of as zero usage; a corresponding
  constraint is recorded as a missing-metric violation.
- `index_size_mb` is the generated serialized artifact's file size and is unavailable in
  search-only mode.
- AutoTune uses a dedicated latency/QPS pass and a separate recall/statistics pass. Latency wraps
  each `KnnSearch` call with `steady_clock`; QPS is successful queries divided by the wall time of
  the concurrent performance pass. The same logical query is therefore executed more than once
  per trial.

`search_seconds` remains evaluation cost because it includes both passes and monitor work; it is
not the time for one logical production query batch. Candidates run in a fixed order against a
reused index, without a warm-up or randomized ordering. Cache state and unrelated system load can
therefore affect performance metrics; benchmark in a representative, controlled environment and
repeat runs when latency or QPS drives selection.

## Tuning and Output Configuration

```json
{
  "tuning_config": {
    "workspace_path": "/tmp/vsag_autotune",
    "keep_intermediate": false,
    "max_trials": 1000
  },
  "output": {
    "result_path": "/tmp/vsag_autotune/report.json",
    "include_raw_eval": false
  }
}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `tuning_config.workspace_path` | `/tmp/vsag_autotune` | Run artifacts and CLI default reports. |
| `tuning_config.keep_intermediate` | `false` | Keep all generated indexes. |
| `tuning_config.max_trials` | `1000` | Maximum planned worst-case trials; hard maximum `100000`. |
| `output.result_path` | `<workspace>/run-<id>.json` | CLI full report path. |
| `output.include_raw_eval` | `false` | Include native eval JSON in build and trial records. |

`output.result_path` cannot alias `data_path` or `index_path`. After initial validation succeeds,
the offline JSON/CLI entry writes the full report. An explicit `output.result_path` overrides its
location; otherwise AutoTune generates a report path under `workspace_path`. Early validation and
request-file failures do not write one.

Typed `TuneIndex` and `TuneSearch` do not accept a report path and never write report files. Their
result objects return the complete report as `JsonType`. For both the CLI and typed `TuneIndex`,
`keep_intermediate=false` retains only the recommended generated index and removes unselected
artifacts; `true` retains every generated index. When there is no recommendation, the default is
to remove all generated indexes. Because `TuneSearch` creates no artifacts,
`workspace_path` and `keep_intermediate` have no effect on it.

## Full Report

A completed evaluation returns this top-level shape. The CLI also writes it to disk:

```json
{
  "version": 1,
  "status": "success",
  "recommendation": {},
  "best_effort": null,
  "builds": [],
  "trials": [],
  "request": {},
  "elapsed_seconds": 84.2,
  "report_path": "/tmp/vsag_autotune/run-....json"
}
```

| Field | Meaning |
| --- | --- |
| `version` | Report contract version, always `1`. |
| `status` | `success`, `no_feasible_candidate`, or `failed`. |
| `recommendation` | Best feasible trial, otherwise `null`. |
| `best_effort` | Closest successful trial when constraints are infeasible, otherwise `null`. |
| `builds` | One record per concrete generated build group; empty in search-only mode. |
| `trials` | One record per executed concrete search candidate. |
| `request` | Effective normalized request used by the tuning engine. |
| `elapsed_seconds` | AutoTune wall time through selection, excluding report writing and cleanup. |
| `report_path` | Persisted full report path; CLI/JSON adapter only. |
| `failure` | Present when the overall run failed. |

Early validation failures do not have `builds`, `trials`, or `report_path`, and no report file is
written. A CLI run in which all candidate evaluations fail does contain the attempted build and
trial records and is written when the report path itself is usable. Typed API reports never
contain `report_path`.

The normalized `request` contains derived dataset metadata, workload defaults, constraints,
objective, config, and output fields. Build-and-search mode adds `index_spaces`; search-only mode
adds `index_name` and `parameter_space`. CLI reports also retain the offline `data_path` and, when
present, `index_path` as top-level fields inside this normalized object.

### Status Semantics

- `success`: at least one successful trial satisfies every constraint; `recommendation` is set.
- `no_feasible_candidate`: successful trials exist, but none satisfies every constraint;
  `best_effort` is set for explanation and is not a valid recommendation. Typed calls return a
  value with `TuneStatus::NO_FEASIBLE_CANDIDATE`, not an error.
- `failed`: the request is invalid, execution/reporting fails, or no trial can be evaluated
  successfully with the objective metric.

### Recommendation and Best Effort

In build-and-search mode, `recommendation` contains:

```json
{
  "index_name": "hgraph",
  "create_params": {},
  "search_params": {},
  "workload": {"top_k": 10, "concurrency": 1},
  "metrics": {},
  "artifacts": {},
  "evidence": {"build_id": "build-0", "trial_id": "trial-0"}
}
```

In search-only mode, `create_params`, `artifacts`, and `evidence.build_id` are omitted; the
recommendation contains the concrete search parameters, workload, metrics, and
`evidence.trial_id`. `best_effort` has the same mode-specific fields plus
`constraint_evaluation`. It is selected first by the number of violated or missing constraints,
then by normalized violation magnitude. It is only explanatory.

### Build Records

`builds` is empty in search-only mode. In build-and-search mode, each `builds[]` item contains:

| Field | Meaning |
| --- | --- |
| `build_id` | Stable ID referenced by trials. |
| `index_name` | Concrete index type. |
| `create_params` | Concrete create parameters. |
| `status` | `success` or `failed`. |
| `metrics` | Available build-shared metrics. |
| `artifacts` | `source`, `index_path`, `use_existing_index`, and `retained`. |
| `failure` | Structured failure or `null`. |
| `elapsed_seconds` | Build-group preparation time. |
| `raw_eval_result` | Present only when requested and a native build eval ran. |

### Trial Records

Each `trials[]` item contains:

| Field | Meaning |
| --- | --- |
| `trial_id` | Stable trial ID. |
| `build_id` | Associated build group; build-and-search mode only. |
| `index_name` | Concrete index type. |
| `create_params` | Concrete create parameters; build-and-search mode only. |
| `search_params` | Concrete search parameters. |
| `status` | `success` or `failed`. |
| `metrics` | Available metrics used by constraint evaluation and selection. |
| `constraint_evaluation` | `satisfied` plus a `violations` array. |
| `artifacts` | Artifact evidence copied from the build group; build-and-search mode only. |
| `failure` | Structured failure or `null`. |
| `elapsed_seconds` | Search trial wall time. |
| `raw_eval_result` | Present only when requested and native search eval succeeded. |

For an adaptive HGraph `ef_search` range, the report contains one trial for each point actually
evaluated. Every recorded `search_params` value is concrete; the `$range` expression remains only
in the normalized request. Every recorded trial evaluates the full query workload.

A constraint violation contains `metric`, `comparison`, `expected`, and `actual`. `actual` is
`null` when the required metric is missing or non-finite.

Search trials in one build group reuse the same loaded index instance. A generated index is built
once and serialized once. In search-only mode the caller's index, or the index deserialized by the
CLI adapter, is reused directly for every trial.

### Artifact Semantics

Artifact fields appear only in build-and-search records in V1. `artifacts.source` is `generated`,
and `artifacts.index_path` is evidence of where the evaluated index was stored, not a promise that
the path still exists. Check `artifacts.retained`:

- `true`: this is the selected artifact, or retention of all generated indexes was requested;
- `false`: AutoTune planned to remove or already removed the generated artifact.

### Structured Failures

Failures use one common object:

```json
{
  "stage": "validation",
  "code": "invalid_request",
  "message": "request.workload.top_k is required"
}
```

Typical stages are `cli`, `validation`, `candidate_generation`, `build`, `search`, `evaluation`,
`selection`, and `report`. Per-build and per-trial failures remain in their corresponding records
so other candidates can continue.

| Stage | Code | Meaning |
| --- | --- | --- |
| `cli` | `request_file_error` | The CLI could not read or parse its request file. |
| `validation` | `invalid_request` | Request validation failed. |
| `candidate_generation` | `invalid_request` | A candidate expression or count was invalid. |
| `build` | `build_evaluation_failed` | One build group failed. |
| `search` | `build_failed` | Search was skipped because its build failed. |
| `search` | `search_evaluation_failed` | One search trial failed. |
| `evaluation` | `all_trials_failed` | Every candidate trial failed. |
| `selection` | `objective_metric_unavailable` | Trials succeeded, but none produced the objective metric. |
| `evaluation`, `report` | `execution_failed` | Top-level execution or report writing failed. |

## CLI Summary

The CLI prints a compact subset of the full report in this order:

1. `recommendation`, when present;
2. `best_effort`, when present;
3. `failure`, when present;
4. `status`;
5. `elapsed_seconds`;
6. `report_path`, when present;
7. `version`.

Null result branches and the detailed build/trial arrays are omitted from standard output. Read
`report_path` for complete evidence.

The CLI exits with code `1` for `status=failed` and command-line/request-file errors. It exits with
code `0` for both `success` and `no_feasible_candidate`; callers must inspect `status` rather than
using only the exit code to decide whether a recommendation exists.

## Build-Tree C++ Entry Point

The experimental optional `vsag::autotune` CMake target exposes the declarations in
`tools/autotune/autotune.h`. The target and header are build-tree-only and are not installed:

```cpp
tl::expected<IndexResult, Error> TuneIndex(const IndexRequest& request);
tl::expected<SearchResult, Error> TuneSearch(const SearchRequest& request);
JsonType RunAutoTune(const JsonType& request);  // CLI adapter
```

The JSON entry point is an offline adapter: it loads `data_path`, creates or deserializes an index
when `index_path` is present, and constructs the same internal request used by the typed entry
points. The target and header are intentionally not installed in V1. Calls inside one process are
serialized because the underlying evaluation path configures process-global OpenMP state.

`TuneIndex` jointly tunes build and search parameters:

```cpp
IndexRequest request;
request.base = base;
request.metric_type = METRIC_L2;
request.workload = {queries, ground_truth, 10, 48};
request.index_spaces = {{"hgraph", create_candidate_space, search_candidate_space}};
request.constraints = {{Metric::RECALL_AT_K, 0.95}};
request.objective = Metric::LATENCY_AVG_MS;
auto result = TuneIndex(request);
```

It returns a loaded, query-ready selected index together with concrete create and search
parameters. `metrics` and the complete `report` are returned as `JsonType`; typed calls never
persist the report. `TuneSearch` tunes an already built or loaded index:

```cpp
SearchRequest request;
request.index = existing_index;
request.workload = {queries, ground_truth, 10, 48};
request.parameter_space = search_candidate_space;
request.constraints = {{Metric::RECALL_AT_K, 0.95}};
request.objective = Metric::LATENCY_AVG_MS;
auto result = TuneSearch(request);
```

AutoTune derives the type and element count of `existing_index`. `TuneSearch` therefore does not
need base vectors or a metric type. Ground truth is required only when recall is a constraint or
the objective. Build-time metrics and `index_size_mb` are unavailable to `TuneSearch`.
`index_memory_mb` may be a constraint but cannot be the objective because it does not vary across
search candidates.

Both typed calls return `tl::unexpected<Error>` only for invalid requests and execution failures.
If evaluation completes but no candidate satisfies every constraint, they return a result with
`status=TuneStatus::NO_FEASIBLE_CANDIDATE` and structured `best_effort`; recommendation fields are
invalid in that state.
