# Search Statistics

VSAG can return query-time statistics together with a search result. Statistics are useful for
observability, debugging, and comparing search configurations.

## Reading statistics

`Dataset` provides two views of the same result statistics:

- `GetSearchMetrics()` returns a typed snapshot. New C++ code should prefer this API.
- `GetStatistics()` returns the existing JSON representation for backward compatibility.
- `GetStatistics(keys)` returns selected JSON values. A missing key is returned as an empty string.

```cpp
vsag::SearchRequest request;
request.query_ = query;
request.topk_ = 10;
request.params_str_ = R"({})";  // Statistics are enabled by default.

auto result = index->SearchWithRequest(request);
if (!result.has_value()) {
    // Handle result.error().
}

auto metrics = result.value()->GetSearchMetrics();
if (metrics.has_value()) {
    std::cout << metrics->distance_evaluations << std::endl;
    std::cout << metrics->DistanceEvaluations(
                     vsag::DistanceEvaluationPhase::APPROXIMATE)
              << std::endl;
}

// The compatible JSON view is still available.
std::cout << result.value()->GetStatistics() << std::endl;
```

The typed snapshot is the authoritative view when it is available. JSON is derived lazily from the
snapshot and cached by the result dataset.

## Enabling and disabling result statistics

Statistics are enabled by default. To disable them for one `SearchWithRequest` call, add the reserved
framework key to the top level of `params_str_`:

```cpp
request.params_str_ = R"({"want_statistics":false})";
```

When `want_statistics` is `false`:

- `GetSearchMetrics()` returns `std::nullopt`;
- `GetStatistics()` returns `"{}"`;
- search results and distances are unchanged.

Disabling statistics is not a guarantee of lower latency: parsing/removing the explicit option and
normalizing the request have a fixed cost, which can outweigh saved collection work for very small
searches. Prefer typed reads over JSON when possible, and benchmark your workload.

BruteForce suppresses statistics collection when this parameter is false. Other indexes may still
perform internal accounting, but they do not expose statistics in the result.

The legacy `KnnSearch` and `RangeSearch` overloads do not provide the per-request
`want_statistics` suppression flag and preserve their existing JSON-statistics behavior. Indexes that
support typed metrics may also attach a typed snapshot on these legacy paths; use
`SearchWithRequest` when per-request suppression is required.

## Typed-statistics support

Typed statistics are introduced incrementally. The current support matrix is:

| Index | `SearchWithRequest` typed snapshot | `"want_statistics": false` result behavior | Collection suppression |
| --- | --- | --- | --- |
| BruteForce | Yes | `nullopt` and `"{}"` | Yes |
| Other indexes | Not yet | `nullopt` and `"{}"` | Not guaranteed |

Indexes without typed support continue to expose their existing JSON statistics when statistics are
enabled.

## Baseline fields

The baseline schema is owned by the search-statistics framework. Index-specific extensions must not
overwrite these keys.

| Field | C++ type | Meaning |
| --- | --- | --- |
| `is_timeout` | `bool` | Whether the search timed out |
| `dist_cmp` | `uint32_t` | Legacy distance-comparison count |
| `hops` | `uint32_t` | Legacy graph-hop count |
| `io_cnt` | `uint32_t` | Number of search IO operations |
| `io_time_ms` | `uint32_t` | Search IO time in milliseconds |
| `reorder_distance_count` | `uint32_t` | Exact distances evaluated during reorder |
| `reorder_candidate_count` | `uint32_t` | Candidates considered during reorder |
| `reorder_lower_bound_probe_count` | `uint32_t` | Lower-bound probes used by reorder |
| `rabitq_filter_count` | `uint32_t` | RaBitQ filter-distance count |
| `rabitq_full_count` | `uint32_t` | RaBitQ full-distance count |
| `rabitq_filter_fallback_full_count` | `uint32_t` | Filter fallbacks to full distance |
| `rabitq_reorder_hint_full_count` | `uint32_t` | Full distances requested by reorder hints |
| `rabitq_reorder_fallback_full_count` | `uint32_t` | Reorder fallbacks to full distance |
| `query_computer_count` | `uint32_t` | Query-computer instances created |
| `parallel_search_fallback_count` | `uint32_t` | Parallel searches that fell back to serial execution |
| `distance_evaluations` | `uint64_t` | Total recorded distance evaluations |
| `distance_evaluations_by_phase` | `array<uint64_t, 3>` | Counts for routing, approximate search, and rerank |
| `distance_evaluations_by_backend` | `array<uint64_t, 16>` | Counts grouped by distance backend |
| `complete` | `bool` | Whether the framework considers accounting complete |

A value of zero is different from missing statistics. A collected empty search has a typed snapshot
whose counters are zero. A search with statistics disabled has no typed snapshot.

## Interpreting counters

- `APPROXIMATE` labels the main candidate-search phase, not distance accuracy. BruteForce's
  exact FP32 scan is recorded in this phase; routing and rerank remain zero for this example.
- `distance_evaluations` counts recorded distance evaluations, not distinct candidates or returned
  neighbors. Evaluating a candidate again can add another evaluation. Phase and backend arrays
  partition the recorded evaluations along different axes; do not add the two arrays together.
- `dist_cmp` preserves the legacy index-specific comparison counter. It can differ from the new
  total and must not be treated as its alias across indexes or search paths.
- `complete` describes diagnostic distance accounting, not recall, search success, or timeout.
  The collector marks it false on an unknown backend or saturated distance accounting. A true value
  is not proof that every possible execution path is instrumented; consult the support matrix and
  inspect `is_timeout` independently.
- The collector's 64-bit distance counters saturate at `UINT64_MAX` rather than wrapping, and mark
  accounting incomplete when an increment cannot be fully represented. Legacy 32-bit counters keep
  their existing update semantics; they do not gain a general saturation guarantee. Using an unsigned
  64-bit JSON setter preserves their stored values but does not widen collection or recover overflow.

For a runnable example with runtime contract checks, see
[`330_feature_search_statistics.cpp`](../../../../../examples/cpp/330_feature_search_statistics.cpp).

## Index-specific extensions

An index may add typed extension values using one of these types:

- `uint32_t`
- `uint64_t`
- `bool`
- `double`
- `std::string`

Extension availability depends on the index and search path. Applications read the const map returned
by `SearchResultMetrics::Extensions()` and must check whether a key exists. Baseline names are
reserved; only the internal collector can add extensions, and it rejects a colliding name.

## Compatibility

- Existing `GetStatistics()` and `GetStatistics(keys)` APIs remain available.
- The default is to return statistics, preserving existing search behavior.
- A dataset without collected statistics continues to return `"{}"` from `GetStatistics()`.
- Calling the legacy `Statistics(string)` setter clears any typed snapshot and makes the supplied
  JSON string authoritative.
- `GetSearchMetrics()` is a nonvirtual bridge and does not add a slot to the existing `Dataset`
  vtable.
- Per-request control uses the existing `params_str_` member, so `SearchRequest` keeps its previous
  object layout for binary compatibility.

## Maintenance contract

Changes to a search implementation must preserve these rules:

1. A typed snapshot is detached result data; live atomic counters must not escape the query. Snapshot
   creation happens after search workers complete. Counters are diagnostic values, not a transactional
   cross-counter view for concurrent writers.
2. JSON must be derived from typed metrics when a typed snapshot is attached.
3. Index extensions must not overwrite baseline fields.
4. Empty-result and early-return paths must honor the reserved `want_statistics` parameter.
5. A disabled result must be represented by `std::nullopt` and `"{}"`, not by a collected all-zero
   snapshot.
6. Adding a baseline field, phase, backend, or extension type requires serializer, documentation,
   unit-test, and functional-test updates in the same change.
7. Adding typed support to an index requires updating the support matrix and running the public
   behavior contract in `tests/test_search_statistics.cpp`.

Run the focused tests with:

```bash
./build-release/tests/unittests "[search_metrics]"
./build-release/tests/functests "[search_statistics]"
```
