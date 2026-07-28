# New Index Integration Checklist

Use this checklist when adding a new index implementation to VSAG. Keep the first pass small:
make the index creatable through the public factory, support the lifecycle methods it advertises,
and add feature flags only after the behavior is implemented and tested.

## Required

- [ ] Choose the public index name and type.
  - Add the user-facing index name constant in `include/vsag/constants.h` or
    `src/inner_string_params.h` if the new index needs one.
  - Add an `IndexType` value in `include/vsag/index.h` when callers must distinguish this index
    through `Index::GetIndexType()`.
  - Keep the public name stable. `src/factory/index_registry.cpp` normalizes factory names to lower
    case before lookup.

- [ ] Implement the index behind the public `Index` API.
  - Prefer the current `IndexImpl<T>` pattern in `src/index/index_impl.h` for new in-memory
    indexes: implement `T` as an `InnerIndexInterface` subclass under `src/algorithm/<name>/`.
  - Implement `static CheckAndMappingExternalParam(const JsonType&, const IndexCommonParam&)` so
    `IndexImpl<T>` can validate external JSON and construct the internal parameter object.
  - Implement `GetName()`, `GetIndexType()`, `GetNumElements()`, `Add()`, `KnnSearch()`,
    `Serialize(StreamWriter&)`, and `Deserialize(StreamReader&)` as required by the inner-index
    contract; implement `Build()` when the index supports it. `InnerIndexInterface::Add()` is pure
    virtual, so every subclass must override it even when the index only supports `Build()` and
    should throw `UNSUPPORTED_INDEX_OPERATION` without enabling the corresponding feature flag.
  - Leave unsupported operations on the base class defaults instead of advertising them.

- [ ] Wire creation through the factory and engine path.
  - Add a creator in `src/factory/index_creators.cpp`.
  - Register it in `register_all_index_creators()`.
  - Use `IndexCommonParam::CheckAndCreate()` from `src/index_common_param.cpp` for shared fields:
    `dtype`, `metric_type`, `dim`, optional `repr`, optional `extra_info_size`, allocator, thread
    pool, and old serialization format compatibility.
  - Add factory tests for the accepted name, invalid parameters, and unsupported parameter shapes in
    `src/factory/factory_test.cpp` or a focused test near the implementation.

- [ ] Add build-system wiring.
  - Add `src/algorithm/<name>/CMakeLists.txt` and include it from `src/algorithm/CMakeLists.txt`.
  - Add new sources to the closest existing target rather than creating a parallel build path.
  - Keep file suffixes as `.cpp`; do not edit `extern/` unless the dependency itself is part of the
    change.

- [ ] Define and validate index parameters.
  - Put implementation parameters in a `<name>_parameter.{h,cpp}` pair when the index has its own
    schema.
  - Implement JSON parsing, `ToJson()`, and `CheckCompatibility()` for serialized/recreated
    parameter checks.
  - Reject invalid dimensions, metric/data-type combinations, missing required blocks, and unknown
    modes with `ErrorType::INVALID_ARGUMENT` through the existing `CHECK_ARGUMENT` /
    `VsagException` flow.
  - Update `docs/docs/{en,zh}/src/resources/index_parameters.md` and the per-index docs if the
    parameter becomes user-facing.

- [ ] Implement lifecycle behavior deliberately.
  - Decide whether the index supports `Train()`, `Build()`, `ContinueBuild()`, `Add()` after build,
    and `Add()` from empty.
  - Decide whether `Remove()`, `UpdateId()`, `UpdateVector()`, `UpdateAttribute()`, and
    `UpdateExtraInfo()` are supported.
  - For every supported mutation, test empty datasets, duplicate IDs if applicable, missing IDs,
    immutable index behavior, and search correctness after mutation.
  - Keep `InitFeatures()` in sync with the implemented operations.

- [ ] Implement search behavior and result packing.
  - Support the public `KnnSearch()` overloads required by the index, including `BitsetPtr`,
    `std::function<bool(int64_t)>`, and `FilterPtr` filtering when advertised.
  - Implement `SearchWithRequest()` if the index supports the newer request path.
  - Return `Dataset` fields consistently: IDs, distances, `num_elements`, result dimension, and
    optional result statistics.
  - Parse search parameters with the same nested index-name convention used by existing indexes
    such as HGraph.

- [ ] Preserve serialization compatibility.
  - Implement both `Serialize(StreamWriter&)` and `Deserialize(StreamReader&)`; the base
    `InnerIndexInterface` adapts these to `BinarySet`, `ReaderSet`, and streams.
  - Store enough metadata to reject incompatible binaries, including parameter compatibility and
    `extra_info_size` when extra info is present.
  - Add round-trip tests through `BinarySet` and `ReaderSet` when the index supports both.
  - If the binary format changes for an existing index, update compatibility tests and document the
    migration path.

- [ ] Add tests before advertising features.
  - Unit tests should cover parameter parsing, build/add/search, serialization, feature flags,
    memory estimation if implemented, and error paths.
  - Functional tests under `tests/` should cover public API behavior that users can reach through
    `Factory::CreateIndex()`.
  - Keep C++ unit-test coverage for `src/` and `include/` at or above the project threshold.

## Optional Adaptation Points

Add these only when the new index actually implements the behavior. When implemented, enable the
matching `IndexFeature` values in `InitFeatures()` and add focused tests.

- [ ] Extra info (`extra_info` / `extrainfo`).
  - Parse `extra_info_size` through `IndexCommonParam`.
  - Store fixed-size per-vector payloads from `Dataset::GetExtraInfos()` and validate
    `Dataset::GetExtraInfoSize()` during `Build()`, `Add()`, and `UpdateExtraInfo()`.
  - Implement `GetExtraInfoByIds()` and populate search-result extra info when the feature is
    supported.
  - If the index supports extra-info filtering, document and test the search parameter that switches
    `Filter::CheckValid(const char*)` on.
  - See `docs/docs/en/src/advanced/extra_info.md` and `examples/cpp/320_feature_extra_info.cpp`.

- [ ] Statistics and analysis.
  - Implement `GetStats()` for static structure data that helps operators understand an index.
  - Implement `AnalyzeIndexBySearch(const SearchRequest&)` only for query-driven analysis.
  - Include result statistics with `Dataset::Statistics()` when search-time metrics are useful.
  - Keep tool output compatible with `tools/analyze_index` and
    `docs/docs/en/src/resources/analyze_index.md`.

- [ ] Range search.
  - Override the pure-virtual primary `RangeSearch(..., const FilterPtr&, ...)` required by
    `InnerIndexInterface`, even when the algorithm does not support range search; in that case,
    throw `UNSUPPORTED_INDEX_OPERATION` without enabling the corresponding feature flag.
  - Implement the other `RangeSearch()` overloads only when the algorithm can honor radius
    semantics and `limited_size`.
  - Test no-limit, limited, filtered, and empty-result cases.
  - See `docs/docs/en/src/advanced/range_search.md`.

- [ ] Filters and attributes.
  - Support `BitsetPtr`, `std::function<bool(int64_t)>`, or `FilterPtr` only when each path is
    wired through search.
  - If attribute filtering is supported, implement attribute storage/update paths and document
    accepted attribute schemas.
  - Test the difference between bitset invalidation and `Filter::CheckValid()` keep semantics.

- [ ] Allocator, resource, and threading integration.
  - Allocate long-lived structures with `IndexCommonParam::allocator_` or a derived allocator-aware
    component.
  - Use the `Resource` thread pool when build/search work is parallelized.
  - Verify custom allocator and custom thread-pool examples still describe the behavior accurately.
  - Mark concurrency features only after add/search/delete/update interactions are tested.

- [ ] Memory and introspection APIs.
  - Implement `EstimateMemory()`, `EstimateBuildMemory()`, `GetMemoryUsage()`, and
    `GetMemoryUsageDetail()` when the index can report meaningful numbers.
  - Implement `GetMinAndMaxId()`, `CheckIdExist()`, `ExportIDs()`, `GetVectorByIds()`,
    `GetDataByIds()`, `GetIndexDetailInfos()`, or `GetDetailDataByName()` only when the backing
    storage supports them.

- [ ] Model export, clone, merge, tune, feedback, and cache import/export.
  - Implement `Clone()` and `ExportModel()` when the index can be copied without sharing mutable
    storage incorrectly.
  - Implement `Merge()` only when parameter compatibility, ID remapping, and deletion semantics are
    clear.
  - Implement `Tune()`, `Feedback()`, `ExportCache()`, and `ImportCache()` only with explicit
    parameter parsing and tests.

- [ ] Bindings, examples, benchmarks, and docs.
  - Python bindings usually need updates only when the public API surface changes; current
    `pyvsag` users create indexes through names and JSON parameters.
  - Add C++ examples under `examples/cpp/` when the index introduces a new user workflow.
  - Add Python examples/tests under `tests/python/` if the behavior is reachable from `pyvsag`.
  - Add benchmark YAML under `benchs/` when reviewers need repeatable performance data.
  - Add English and Chinese website docs under `docs/docs/{en,zh}/src/` for user-facing indexes or
    parameters.

## Review Checklist

- [ ] `Factory::CreateIndex()` and `Engine::CreateIndex()` create the index by the documented name.
- [ ] `CheckFeature()` returns true only for implemented and tested behavior.
- [ ] Unsupported operations return `UNSUPPORTED_INDEX_OPERATION` through existing wrappers.
- [ ] Serialization round trips preserve IDs, vectors or compressed codes, parameters, deletions,
  attributes, and extra info that the index claims to support.
- [ ] Search results remain valid after every supported lifecycle transition.
- [ ] Documentation lists user-facing parameters, supported metrics/data types, and unsupported
  operations.
- [ ] Practical validation has run: unit/functional tests for changed code, plus formatting or
  `git diff --check` for documentation-only changes.
