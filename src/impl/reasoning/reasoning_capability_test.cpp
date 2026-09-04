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

#include "reasoning_capability.h"

#include <catch2/catch_test_macros.hpp>

namespace vsag {

TEST_CASE("Reasoning capability registry is the support matrix", "[reasoning]") {
    SECTION("Known indexes resolve") {
        const auto* hgraph = GetReasoningCapability("HGraph");
        REQUIRE(hgraph != nullptr);
        REQUIRE(hgraph->supports_knn);
        REQUIRE_FALSE(hgraph->supports_range);
        REQUIRE(HasReasoningEvent(hgraph->event_mask, ReasoningEvent::kVisit));
        REQUIRE(HasReasoningEvent(hgraph->event_mask, ReasoningEvent::kEviction));
        REQUIRE(HasReasoningEvent(hgraph->event_mask, ReasoningEvent::kReorder));

        const auto* ivf = GetReasoningCapability("IVF");
        REQUIRE(ivf != nullptr);
        REQUIRE(ivf->supports_knn);
        REQUIRE(ivf->supports_range);
        REQUIRE(HasReasoningEvent(ivf->event_mask, ReasoningEvent::kBucketSelection));
        REQUIRE(HasReasoningEvent(ivf->event_mask, ReasoningEvent::kReorder));

        const auto* sindi = GetReasoningCapability("SINDI");
        REQUIRE(sindi != nullptr);
        REQUIRE(sindi->supports_range);
        REQUIRE(HasReasoningEvent(sindi->event_mask, ReasoningEvent::kBucketSelection));

        const auto* sindi_v2 = GetReasoningCapability("SINDI_V2");
        REQUIRE(sindi_v2 != nullptr);
        REQUIRE(sindi_v2->supports_knn);
        REQUIRE(sindi_v2->supports_range);
        REQUIRE(HasReasoningEvent(sindi_v2->event_mask, ReasoningEvent::kBucketSelection));
        REQUIRE(HasReasoningEvent(sindi_v2->event_mask, ReasoningEvent::kEviction));

        const auto* bruteforce = GetReasoningCapability("BruteForce");
        REQUIRE(bruteforce != nullptr);
        REQUIRE(bruteforce->supports_range);
        REQUIRE_FALSE(HasReasoningEvent(bruteforce->event_mask, ReasoningEvent::kReorder));

        const auto* warp = GetReasoningCapability("WARP");
        REQUIRE(warp != nullptr);
        REQUIRE(warp->supports_range);

        const auto* pyramid = GetReasoningCapability("Pyramid");
        REQUIRE(pyramid != nullptr);
        REQUIRE(HasReasoningEvent(pyramid->event_mask, ReasoningEvent::kEviction));
        REQUIRE(HasReasoningEvent(pyramid->event_mask, ReasoningEvent::kReorder));
    }

    SECTION("Unknown index resolves to nullptr") {
        REQUIRE(GetReasoningCapability("NonexistentIndex") == nullptr);
        REQUIRE(GetReasoningCapability("") == nullptr);
    }
}

}  // namespace vsag
