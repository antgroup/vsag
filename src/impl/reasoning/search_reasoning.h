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
#include <string>

#include "impl/allocator/allocator_wrapper.h"
#include "json_wrapper.h"
#include "typing.h"

namespace vsag {

struct ExpectedTargetTrace {
    int64_t label{0};
    InnerIdType inner_id{0};
    float true_distance{0.0F};
    float quantized_distance{0.0F};
    bool was_visited{false};
    int32_t visited_at_hop{-1};
    bool was_in_result_set{false};
    bool was_evicted{false};
    bool filter_rejected{false};
    bool reorder_evicted{false};
    std::string diagnosis{};

    ExpectedTargetTrace() = default;
};

struct ReorderRecord {
    InnerIdType id{0};
    float dist_before{0.0F};
    float dist_after{0.0F};
};

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
    SetTermination(const std::string& reason);

    void
    MarkResult(const Vector<InnerIdType>& result_ids);

    void
    DiagnoseExpectedTargets();

    std::string
    GenerateReport() const;

    void
    SetSearchParams(int64_t topk,
                    const std::string& index_type,
                    bool use_reorder,
                    bool filter_active);

    void
    AddSearchHop();

    void
    AddDistanceComputation(uint32_t count = 1);

    static constexpr const char* kTerminationLowerBoundReached = "lower_bound_reached";
    static constexpr const char* kTerminationHopsLimitReached = "hops_limit_reached";
    static constexpr const char* kTerminationTimeout = "timeout";

public:
    int64_t topk_{0};
    std::string index_type_{};
    bool use_reorder_{false};
    bool filter_active_{false};

    uint32_t total_hops_{0};
    uint32_t total_dist_computations_{0};
    std::string termination_reason_{};

    UnorderedSet<InnerIdType> expected_inner_ids_;
    UnorderedMap<InnerIdType, ExpectedTargetTrace> expected_traces_;
    Vector<ReorderRecord> reorder_changes_;

private:
    Allocator* allocator_{nullptr};

    static std::string
    DiagnoseTarget(const ExpectedTargetTrace& trace);
};

}  // namespace vsag
