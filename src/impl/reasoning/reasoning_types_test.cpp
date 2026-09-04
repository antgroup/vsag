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
#include <string_view>

namespace vsag {

TEST_CASE("Reasoning diagnosis and event name mapping", "[reasoning]") {
    REQUIRE(std::string_view(ToString(ReasoningDiagnosis::kSuccess)) == "success");
    REQUIRE(std::string_view(ToString(ReasoningDiagnosis::kNotReachable)) == "not_reachable");
    REQUIRE(std::string_view(ToString(ReasoningDiagnosis::kFilterRejected)) == "filter_rejected");
    REQUIRE(std::string_view(ToString(ReasoningDiagnosis::kQuantizationError)) ==
            "quantization_error");
    REQUIRE(std::string_view(ToString(ReasoningDiagnosis::kEfTooSmall)) == "ef_too_small");
    REQUIRE(std::string_view(ToString(ReasoningDiagnosis::kReorderEvicted)) == "reorder_evicted");
    REQUIRE(std::string_view(ToString(ReasoningDiagnosis::kUnknown)) == "unknown");

    REQUIRE(std::string_view(ToString(ReasoningEvent::kVisit)) == "visit");
    REQUIRE(std::string_view(ToString(ReasoningEvent::kEviction)) == "eviction");
    REQUIRE(std::string_view(ToString(ReasoningEvent::kFilterReject)) == "filter_reject");
    REQUIRE(std::string_view(ToString(ReasoningEvent::kReorder)) == "reorder");
    REQUIRE(std::string_view(ToString(ReasoningEvent::kReorderEviction)) == "reorder_eviction");
    REQUIRE(std::string_view(ToString(ReasoningEvent::kBucketSelection)) == "bucket_selection");
}

TEST_CASE("ReasoningContext termination reasons are centralized", "[reasoning]") {
    REQUIRE(std::string_view(ToString(ReasoningTermination::kLowerBoundReached)) ==
            "lower_bound_reached");
    REQUIRE(std::string_view(ToString(ReasoningTermination::kHopsLimitReached)) ==
            "hops_limit_reached");
    REQUIRE(std::string_view(ToString(ReasoningTermination::kTimeout)) == "timeout");
    REQUIRE(std::string_view(ToString(ReasoningTermination::kNone)) == "none");
}

}  // namespace vsag
