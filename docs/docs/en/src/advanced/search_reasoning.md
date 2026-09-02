# Search Reasoning

VSAG's search reasoning mechanism explains **why specific expected vectors did (or did not) show up
in a search result**. When you search with a set of expected labels, the index records what happened
to each of them during the search — whether it was visited, rejected by a filter, evicted from the
candidate heap, or simply never reached — and attaches a structured JSON report to the result.

## When to use it

Use search reasoning to debug recall problems, to build recall regression tooling, or to answer
questions like "this vector should be a top-1 hit, why is it missing?".

## How to trigger

1. Build a `SearchRequest` and put the labels you expect to find into `expected_labels_`.
2. Call `SearchWithRequest(request)` on the index.
3. Read the report from the returned dataset via `GetReasoning()`.

```cpp
vsag::SearchRequest request;
request.query_ = query_dataset;              // single query vector
request.topk_ = 10;
request.params_str_ = R"({"hgraph": {"ef_search": 100}})";
request.expected_labels_ = {42, 1024};       // labels you expect to see

auto result = index->SearchWithRequest(request);
if (result.has_value()) {
    const std::string& reasoning = result.value()->GetReasoning();  // JSON report
}
```

`expected_labels_` left empty keeps the search behavior and results identical — the report is only
produced when at least one expected label is given.

## Report format

The JSON report has two sections:

```json
{
  "expected_analysis": {
    "summary": "1/2 expected labels found, 1 missed",
    "missed_targets": [
      {
        "label": 1024,
        "inner_id": 1007,
        "diagnosis": "quantization_error",
        "true_distance": 0.11,
        "quantized_distance": 0.24,
        "was_visited": true,
        "visited_at_hop": 2,
        "was_evicted": false,
        "filter_rejected": false,
        "reorder_evicted": false
      }
    ]
  },
  "meta": {
    "schema_version": 1,
    "status": "ok",
    "index_type": "HGraph",
    "search_mode": "knn",
    "topk": 10,
    "use_reorder": true,
    "filter_active": false,
    "termination_reason": "lower_bound_reached",
    "total_hops": 35,
    "total_distance_computations": 1840,
    "available_diagnoses": ["success", "not_reachable", "..."],
    "available_events": ["visit", "eviction", "..."],
    "supports_range": false
  }
}
```

- `expected_analysis.summary` — overall hit count over the expected labels.
- `expected_analysis.missed_targets` — one entry per **missed** expected label, with the diagnosis
  and the evidence trail (visit / eviction / filter / reorder flags and distances).
- `meta` — context of the search (index, mode, parameters) plus termination statistics.

## Diagnoses

Reported per missed expected label (first match wins):

| Diagnosis | Meaning |
|---|---|
| `not_reachable` | The vector's window/bucket was never scanned during the search. |
| `filter_rejected` | An active filter excluded the vector. |
| `quantization_error` | Approximate distance overestimated the true distance (roughly true × 1.5), so the vector lost to better-scored candidates. |
| `ef_too_small` | The vector was visited but lost in the candidate heap competition; consider increasing `ef_search` / `n_candidate`. |
| `reorder_evicted` | The vector survived approximate ranking but was dropped during the precise rerank stage. |
| `success` | Not applicable — used for expected labels that were actually found. |
| `unknown` | None of the above could be determined. |

## Index capability matrix

Not every index records the same evidence. The registry in
`src/impl/reasoning/reasoning_context.cpp` (`kReasoningCapabilities`) is the single source of truth:

| Index | KNN | Range | Events |
|---|---|---|---|
| HGraph | ✓ | – (skipped) | visit, eviction, filter_reject, reorder, reorder_eviction |
| IVF | ✓ | ✓ | visit, eviction, filter_reject, reorder, reorder_eviction, bucket_selection |
| SINDI | ✓ | ✓ | visit, filter_reject, reorder, reorder_eviction, bucket_selection |
| SINDI_V2 | ✓ | ✓ | visit, eviction, filter_reject, reorder, reorder_eviction, bucket_selection |
| BruteForce | ✓ | ✓ | visit, filter_reject |
| WARP | ✓ | ✓ | visit, filter_reject |
| Pyramid | ✓ | – (skipped) | visit, eviction, filter_reject, reorder, reorder_eviction |

Notes and limitations:

- **HGraph range search** with expected labels returns a minimal status report
  (`meta.status = "skipped_range_search"`) instead of an analysis — reasoning is KNN-only there.
- **SINDI_V2** reconstructs visits by re-scoring the expected targets with the exact pruned query
  terms used by the search; a zero score means no shared term (`not_reachable`).
- **SINDI_V2 KNN search** records an approximate eviction when a visited expected target is absent from the
  final candidate pool, and records precise rerank admission/heap eviction separately.
- An expected-label request on an empty SINDI_V2 index returns `meta.status = "empty_index"`.
- `termination_reason` can be `none`, `lower_bound_reached`, `hops_limit_reached` or `timeout`,
  depending on index support.

## Extending reasoning (for maintainers)

All enum definitions, the capability registry, and step-by-step recipes for adding a new diagnosis
or event live in `src/impl/reasoning/reasoning_context.h` / `.cpp`. Every instrumentation hook in the
codebase is marked with a `// [reasoning]` comment, so `grep -rn "\[reasoning\]" src/` enumerates the
complete set of collection points.
