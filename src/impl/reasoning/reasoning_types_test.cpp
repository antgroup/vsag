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

#include <catch2/catch_test_macros.hpp>

namespace vsag {

TEST_CASE("ReasoningDiagnosis ToString", "[reasoning]") {
    REQUIRE(std::string(ToString(ReasoningDiagnosis::kSuccess)) == "success");
    REQUIRE(std::string(ToString(ReasoningDiagnosis::kNotReachable)) == "not_reachable");
    REQUIRE(std::string(ToString(ReasoningDiagnosis::kFilterRejected)) == "filter_rejected");
    REQUIRE(std::string(ToString(ReasoningDiagnosis::kQuantizationError)) == "quantization_error");
    REQUIRE(std::string(ToString(ReasoningDiagnosis::kEfTooSmall)) == "ef_too_small");
    REQUIRE(std::string(ToString(ReasoningDiagnosis::kReorderEvicted)) == "reorder_evicted");
    REQUIRE(std::string(ToString(ReasoningDiagnosis::kUnknown)) == "unknown");
}

TEST_CASE("ReasoningTermination ToString", "[reasoning]") {
    REQUIRE(std::string(ToString(ReasoningTermination::kNone)) == "none");
    REQUIRE(std::string(ToString(ReasoningTermination::kLowerBoundReached)) ==
            "lower_bound_reached");
    REQUIRE(std::string(ToString(ReasoningTermination::kHopsLimitReached)) == "hops_limit_reached");
    REQUIRE(std::string(ToString(ReasoningTermination::kTimeout)) == "timeout");
}

TEST_CASE("ReasoningEvent ToString", "[reasoning]") {
    REQUIRE(std::string(ToString(ReasoningEvent::kVisit)) == "visit");
    REQUIRE(std::string(ToString(ReasoningEvent::kEviction)) == "eviction");
    REQUIRE(std::string(ToString(ReasoningEvent::kFilterReject)) == "filter_reject");
    REQUIRE(std::string(ToString(ReasoningEvent::kReorder)) == "reorder");
    REQUIRE(std::string(ToString(ReasoningEvent::kReorderEviction)) == "reorder_eviction");
    REQUIRE(std::string(ToString(ReasoningEvent::kBucketSelection)) == "bucket_selection");
}

TEST_CASE("ReasoningReportStatus ToString", "[reasoning]") {
    REQUIRE(std::string(ToString(ReasoningReportStatus::kOk)) == "ok");
    REQUIRE(std::string(ToString(ReasoningReportStatus::kSkippedRangeSearch)) ==
            "skipped_range_search");
    REQUIRE(std::string(ToString(ReasoningReportStatus::kUnsupportedByIndex)) ==
            "unsupported_by_index");
    REQUIRE(std::string(ToString(ReasoningReportStatus::kEmptyIndex)) == "empty_index");
}

TEST_CASE("ReasoningEvent bit helpers", "[reasoning]") {
    uint64_t mask =
        ReasoningEventBit(ReasoningEvent::kVisit) | ReasoningEventBit(ReasoningEvent::kReorder);
    REQUIRE(HasReasoningEvent(mask, ReasoningEvent::kVisit));
    REQUIRE(HasReasoningEvent(mask, ReasoningEvent::kReorder));
    REQUIRE_FALSE(HasReasoningEvent(mask, ReasoningEvent::kEviction));
}

}  // namespace vsag