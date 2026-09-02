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

#include "typing.h"

namespace vsag {

/// Why an expected target ended up in (or out of) the result set.
enum class ReasoningDiagnosis : uint8_t {
    kSuccess = 0,
    kNotReachable,
    kFilterRejected,
    kQuantizationError,
    kEfTooSmall,
    kReorderEvicted,
    kUnknown,
};

/// Why the graph traversal stopped.
enum class ReasoningTermination : uint8_t {
    kNone = 0,
    kLowerBoundReached,
    kHopsLimitReached,
    kTimeout,
};

/// Event types a search path can record.
enum class ReasoningEvent : uint8_t {
    kVisit = 0,
    kEviction,
    kFilterReject,
    kReorder,
    kReorderEviction,
    kBucketSelection,
};

/// Status of a generated Reasoning report.
enum class ReasoningReportStatus : uint8_t {
    kOk = 0,
    kSkippedRangeSearch,
    kUnsupportedByIndex,
    kEmptyIndex,
};

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
    ReasoningDiagnosis diagnosis{ReasoningDiagnosis::kUnknown};

    ExpectedTargetTrace() = default;
};

struct ReorderRecord {
    InnerIdType id{0};
    float dist_before{0.0F};
    float dist_after{0.0F};
};

const char*
ToString(ReasoningDiagnosis diagnosis);

const char*
ToString(ReasoningTermination termination);

const char*
ToString(ReasoningEvent event);

const char*
ToString(ReasoningReportStatus status);

inline constexpr uint64_t
ReasoningEventBit(ReasoningEvent event) {
    return static_cast<uint64_t>(1) << static_cast<uint8_t>(event);
}

inline constexpr bool
HasReasoningEvent(uint64_t event_mask, ReasoningEvent event) {
    return (event_mask & ReasoningEventBit(event)) != 0;
}

}  // namespace vsag
