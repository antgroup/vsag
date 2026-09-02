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

#include "reasoning_types.h"

namespace vsag {

const char*
ToString(ReasoningDiagnosis diagnosis) {
    switch (diagnosis) {
        case ReasoningDiagnosis::kSuccess:
            return "success";
        case ReasoningDiagnosis::kNotReachable:
            return "not_reachable";
        case ReasoningDiagnosis::kFilterRejected:
            return "filter_rejected";
        case ReasoningDiagnosis::kQuantizationError:
            return "quantization_error";
        case ReasoningDiagnosis::kEfTooSmall:
            return "ef_too_small";
        case ReasoningDiagnosis::kReorderEvicted:
            return "reorder_evicted";
        case ReasoningDiagnosis::kUnknown:
        default:
            return "unknown";
    }
}

const char*
ToString(ReasoningTermination termination) {
    switch (termination) {
        case ReasoningTermination::kNone:
            return "none";
        case ReasoningTermination::kLowerBoundReached:
            return "lower_bound_reached";
        case ReasoningTermination::kHopsLimitReached:
            return "hops_limit_reached";
        case ReasoningTermination::kTimeout:
            return "timeout";
        default:
            return "none";
    }
}

const char*
ToString(ReasoningEvent event) {
    switch (event) {
        case ReasoningEvent::kVisit:
            return "visit";
        case ReasoningEvent::kEviction:
            return "eviction";
        case ReasoningEvent::kFilterReject:
            return "filter_reject";
        case ReasoningEvent::kReorder:
            return "reorder";
        case ReasoningEvent::kReorderEviction:
            return "reorder_eviction";
        case ReasoningEvent::kBucketSelection:
            return "bucket_selection";
        default:
            return "unknown_event";
    }
}

const char*
ToString(ReasoningReportStatus status) {
    switch (status) {
        case ReasoningReportStatus::kOk:
            return "ok";
        case ReasoningReportStatus::kSkippedRangeSearch:
            return "skipped_range_search";
        case ReasoningReportStatus::kUnsupportedByIndex:
            return "unsupported_by_index";
        case ReasoningReportStatus::kEmptyIndex:
            return "empty_index";
        default:
            return "unknown_status";
    }
}

}  // namespace vsag
