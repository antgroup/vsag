# Search Reasoning (Recall Diagnostics)

Search Reasoning is a per-query diagnostic mechanism that explains why each
expected label was or was not recalled. Enable it by populating
`SearchRequest::expected_labels_` and calling `SearchWithRequest`; the result
dataset carries a JSON report accessible through `Dataset::GetReasoning()`.

## Quick Start

For a built SINDI_V2 index and a single-query sparse `query_dataset`:

```cpp
// Populate expected_labels_ with the labels of interest
vsag::SearchRequest request;
request.expected_labels_ = {100, 200, 300};
request.query_ = query_dataset;
request.topk_ = 10;
request.params_str_ = R"({"sindi_v2": {"n_candidate": 100}})";

auto results = index->SearchWithRequest(request);
if (!results.has_value()) {
    std::cerr << "SearchWithRequest failed: " << results.error().message << std::endl;
} else {
    std::cout << results.value()->GetReasoning() << std::endl;
}
```

## Report Schema (v1)

A full report is a JSON object with `expected_analysis` and `meta` sections.
Status-only reports produced by `MakeStatusReport` contain only `meta`, with exactly
`schema_version`, `status`, and `index_type`; they do not include `expected_analysis`
or the remaining metadata fields. Consumers must check `meta.status` and treat
those additional fields as optional rather than requiring a full report.

For example, a status-only report for an empty IVF index is:

```json
{"meta":{"schema_version":1,"status":"empty_index","index_type":"IVF"}}
```

### `expected_analysis`

| Field | Type | Description |
|-------|------|-------------|
| `summary` | string | "M/N expected labels found, P missed" |
| `missed_targets` | array | One entry per missed label |

Each `missed_targets` entry:

| Field | Type | Description |
|-------|------|-------------|
| `label` | int | Original label |
| `inner_id` | int | Internal ID |
| `diagnosis` | string | One of the diagnoses below |
| `true_distance` | float | Ground-truth distance |
| `quantized_distance` | float | Approximate distance seen during search |
| `was_visited` | bool | Whether the id was visited |
| `visited_at_hop` | int | Hop when first visited (-1 if never) |
| `was_evicted` | bool | Whether it was evicted from the candidate set |
| `filter_rejected` | bool | Whether the filter rejected it |
| `reorder_evicted` | bool | Whether reorder evicted it |

### `meta`

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Always 1 |
| `status` | string | Currently emitted: `ok`, `empty_index`. Reserved, not currently emitted: `skipped_range_search`, `unsupported_by_index`. |
| `index_type` | string | HGraph, IVF, SINDI, etc. |

The following metadata fields belong to full reports and are absent from
`MakeStatusReport` output; consumers must tolerate their absence:

| Field | Type | Description |
|-------|------|-------------|
| `search_mode` | string | knn or range |
| `topk` | int | Requested top k |
| `use_reorder` | bool | Whether reorder was active |
| `filter_active` | bool | Whether a filter was active |
| `termination_reason` | string | none, lower_bound_reached, hops_limit_reached, timeout |
| `total_hops` | int | Total search hops |
| `total_distance_computations` | int | Total distance evaluations |
| `available_diagnoses` | [string] | All diagnoses the engine can emit |
| `available_events` | [string] | Events this index supported for this search |
| `supports_range` | bool | Whether this index supports range-reasoning |

## Diagnoses

| Value | Meaning |
|-------|---------|
| `success` | Label was in the result set |
| `not_reachable` | Expected target was never visited |
| `filter_rejected` | Filter blocked the entry |
| `quantization_error` | Approximate distance was too far from true distance |
| `ef_too_small` | Visited but evicted (try larger ef) |
| `reorder_evicted` | Reorder evicted the entry |
| `unknown` | No clear reason determined |

For missed targets, rules use first-match priority: not visited, filter rejected,
quantization error, candidate eviction, reorder eviction, then unknown. In particular,
`quantization_error` takes priority over `ef_too_small` when both conditions hold.
The quantization check is the legacy heuristic `approximate > 1.5 * true`, applied
only when the true distance is positive. It is not a metric-independent error bound
or proof that quantization caused the miss; signed IP distances need particular care.

When present in a full report, `supports_range` is a boolean describing registered
range-reasoning support, not availability of the index's RangeSearch API. Unregistered
index types report false; absence from a status-only report does not mean false.

For range requests, HGraph ignores `expected_labels_` for reasoning and leaves
`GetReasoning()` as `{}`. Pyramid instead rejects range requests with non-empty
`expected_labels_` with an invalid-argument error. Neither behavior produces a
`skipped_range_search` or `unsupported_by_index` status report; those statuses
are reserved and not currently emitted.

## Per-Index Event Support

| Index | KNN | Range | Visit | Eviction | Filter Reject | Reorder | Reorder Eviction | Bucket Selection |
|-------|-----|-------|-------|----------|---------------|---------|------------------|------------------|
| HGraph | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| IVF | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SINDI | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| SINDI_V2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BruteForce | ✓ | ✓ | ✓ | — | ✓ | — | — | — |
| WARP | ✓ | ✓ | ✓ | — | ✓ | — | — | — |
| Pyramid | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | — |

## Advanced

To extend the framework:

1. Add a new `ReasoningDiagnosis` value in `reasoning_types.h`, map it in
   `ToString`, add a diagnosis rule in `reasoning_context.cpp`, add a test,
   and document it.
2. Add a new `ReasoningEvent` value in `reasoning_types.h`, implement a
   `RecordXxx` method on `ReasoningContext`, tag the instrumentation site
   with `// [reasoning]`, update the capability matrix, and document it.
3. For a new index: add a capability entry in `reasoning_capability.cpp`,
   implement `SearchWithRequest`, and insert recording hooks.
