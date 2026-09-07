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

namespace vsag {

namespace {

constexpr ReasoningCapability K_HGRAPH_CAP{"HGraph",
                                           true,
                                           false,
                                           ReasoningEventBit(ReasoningEvent::kVisit) |
                                               ReasoningEventBit(ReasoningEvent::kEviction) |
                                               ReasoningEventBit(ReasoningEvent::kFilterReject) |
                                               ReasoningEventBit(ReasoningEvent::kReorder) |
                                               ReasoningEventBit(ReasoningEvent::kReorderEviction)};

constexpr ReasoningCapability K_IVF_CAP{"IVF",
                                        true,
                                        true,
                                        ReasoningEventBit(ReasoningEvent::kVisit) |
                                            ReasoningEventBit(ReasoningEvent::kEviction) |
                                            ReasoningEventBit(ReasoningEvent::kFilterReject) |
                                            ReasoningEventBit(ReasoningEvent::kReorder) |
                                            ReasoningEventBit(ReasoningEvent::kReorderEviction) |
                                            ReasoningEventBit(ReasoningEvent::kBucketSelection)};

constexpr ReasoningCapability K_SINDI_CAP{"SINDI",
                                          true,
                                          true,
                                          ReasoningEventBit(ReasoningEvent::kVisit) |
                                              ReasoningEventBit(ReasoningEvent::kFilterReject) |
                                              ReasoningEventBit(ReasoningEvent::kReorder) |
                                              ReasoningEventBit(ReasoningEvent::kReorderEviction) |
                                              ReasoningEventBit(ReasoningEvent::kBucketSelection)};

constexpr ReasoningCapability K_SINDI_V2_CAP{
    "SINDI_V2",
    true,
    true,
    ReasoningEventBit(ReasoningEvent::kVisit) | ReasoningEventBit(ReasoningEvent::kEviction) |
        ReasoningEventBit(ReasoningEvent::kFilterReject) |
        ReasoningEventBit(ReasoningEvent::kReorder) |
        ReasoningEventBit(ReasoningEvent::kReorderEviction) |
        ReasoningEventBit(ReasoningEvent::kBucketSelection)};

constexpr ReasoningCapability K_BRUTE_FORCE_CAP{
    "BruteForce",
    true,
    true,
    ReasoningEventBit(ReasoningEvent::kVisit) | ReasoningEventBit(ReasoningEvent::kFilterReject)};

constexpr ReasoningCapability K_WARP_CAP{
    "WARP",
    true,
    true,
    ReasoningEventBit(ReasoningEvent::kVisit) | ReasoningEventBit(ReasoningEvent::kFilterReject)};

constexpr ReasoningCapability K_PYRAMID_CAP{
    "Pyramid",
    true,
    false,
    ReasoningEventBit(ReasoningEvent::kVisit) | ReasoningEventBit(ReasoningEvent::kEviction) |
        ReasoningEventBit(ReasoningEvent::kFilterReject) |
        ReasoningEventBit(ReasoningEvent::kReorder) |
        ReasoningEventBit(ReasoningEvent::kReorderEviction)};

constexpr ReasoningCapability K_REASONING_CAPABILITIES[] = {K_HGRAPH_CAP,
                                                            K_IVF_CAP,
                                                            K_SINDI_CAP,
                                                            K_SINDI_V2_CAP,
                                                            K_BRUTE_FORCE_CAP,
                                                            K_WARP_CAP,
                                                            K_PYRAMID_CAP};

}  // namespace

const ReasoningCapability*
GetReasoningCapability(std::string_view index_type) {
    for (const auto& capability : K_REASONING_CAPABILITIES) {
        if (index_type == capability.index_type) {
            return &capability;
        }
    }
    return nullptr;
}

}  // namespace vsag