// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <mutex>
#include <string>

#include "impl/allocator/allocator_wrapper.h"
#include "json_wrapper.h"
#include "reasoning_types.h"
#include "typing.h"

namespace vsag {

// ============================================================================
// Search Reasoning: per-query recall diagnostics.
//
// A search enables reasoning by filling SearchRequest::expected_labels_ and
// calling SearchWithRequest. The index creates a per-query ReasoningContext,
// records events during traversal, and attaches a JSON report to the result
// dataset (Dataset::Reasoning). The report explains why each expected target
// was or was not recalled.
//
// Quick answers:
//   * What can a diagnosis be?   See ReasoningDiagnosis in reasoning_types.h.
//   * Why did the search stop?   See ReasoningTermination in reasoning_types.h.
//   * What does an index record? GetReasoningCapability(index_type) - the
//     single support matrix, see reasoning_capability.cpp.
//   * Where are the hooks?       Every instrumentation site is tagged with a
//     "// [reasoning]" comment; `grep -rn "\[reasoning\]" src/` lists them all.
//
// How to add a new diagnosis:
//   1. add a value to ReasoningDiagnosis in reasoning_types.h
//   2. map it in ToString(ReasoningDiagnosis) (reasoning_types.cpp)
//   3. add a rule in DiagnoseTarget (rule order = causal chain priority,
//      first match wins)
//   4. add a SECTION in reasoning_context_test.cpp
//   5. add a row in docs/docs/{en,zh}/src/advanced/search-reasoning.md
//
// How to add a new event type:
//   1. add a value to ReasoningEvent in reasoning_types.h
//   2. add a RecordXxx method on ReasoningContext
//   3. update reasoning_capability.cpp for every index that emits it
//   4. tag the instrumentation site with "// [reasoning]"
//   5. add a row in the docs event table
// ============================================================================

class ReasoningContext {
public:
    ReasoningContext(Allocator* allocator);

    ~ReasoningContext();

    void
    InitializeExpectedTargets(const Vector<int64_t>& labels,
                              const UnorderedMap<int64_t, InnerIdType>& label_to_inner_id);

    void
    SetTrueDistance(InnerIdType id, float dist);

    void
    RecordVisit(InnerIdType id, float dist, uint32_t hop);

    void
    RecordEviction(InnerIdType id, uint32_t hop);

    void
    RecordFilterReject(InnerIdType id);

    void
    RecordReorder(InnerIdType id, float dist_before, float dist_after);

    void
    RecordReorderEviction(InnerIdType id, uint32_t hop);

    void
    SetTermination(ReasoningTermination termination);

    void
    MarkResult(const Vector<InnerIdType>& result_ids);

    void
    DiagnoseExpectedTargets();

    void
    RecordBucketSelection(const Vector<BucketIdType>& buckets);

    std::string
    GenerateReport() const;

    /// Builds a minimal report for searches where reasoning did not run
    /// (e.g. expected_labels set on an unsupported index or a skipped mode).
    /// The returned JSON carries only the meta section.
    static std::string
    MakeStatusReport(ReasoningReportStatus status, const std::string& index_type);

    void
    SetSearchParams(int64_t topk,
                    const std::string& index_type,
                    bool use_reorder,
                    bool filter_active,
                    bool is_range = false);

    void
    AddSearchHop();

    void
    AddDistanceComputation(uint32_t count = 1);

public:
    int64_t topk_{0};
    std::string index_type_{};
    bool use_reorder_{false};
    bool filter_active_{false};
    bool is_range_{false};

    uint32_t total_hops_{0};
    uint32_t total_dist_computations_{0};
    ReasoningTermination termination_{ReasoningTermination::kNone};

    UnorderedSet<InnerIdType> expected_inner_ids_;
    UnorderedMap<InnerIdType, ExpectedTargetTrace> expected_traces_;
    Vector<ReorderRecord> reorder_changes_;
    Vector<BucketIdType> selected_buckets_;

private:
    Allocator* allocator_{nullptr};
    mutable std::mutex mutex_;

    static ReasoningDiagnosis
    DiagnoseTarget(const ExpectedTargetTrace& trace);
};

}  // namespace vsag
