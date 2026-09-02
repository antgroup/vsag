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

// Single support matrix for reasoning capabilities. The index_type strings
// must match the indexes' GetName(). Keep this table in sync with the docs
// support matrix: docs/docs/{en,zh}/src/advanced/search-reasoning.md.
constexpr ReasoningCapability K_REASONING_CAPABILITIES[] = {
    {"HGraph",
     true,
     false,
     ReasoningEventBit(ReasoningEvent::kVisit) | ReasoningEventBit(ReasoningEvent::kEviction) |
         ReasoningEventBit(ReasoningEvent::kFilterReject) |
         ReasoningEventBit(ReasoningEvent::kReorder) |
         ReasoningEventBit(ReasoningEvent::kReorderEviction)},
    {"IVF",
     true,
     true,
     ReasoningEventBit(ReasoningEvent::kVisit) | ReasoningEventBit(ReasoningEvent::kEviction) |
         ReasoningEventBit(ReasoningEvent::kFilterReject) |
         ReasoningEventBit(ReasoningEvent::kReorder) |
         ReasoningEventBit(ReasoningEvent::kReorderEviction) |
         ReasoningEventBit(ReasoningEvent::kBucketSelection)},
    {"SINDI",
     true,
     true,
     ReasoningEventBit(ReasoningEvent::kVisit) | ReasoningEventBit(ReasoningEvent::kFilterReject) |
         ReasoningEventBit(ReasoningEvent::kReorder) |
         ReasoningEventBit(ReasoningEvent::kReorderEviction) |
         ReasoningEventBit(ReasoningEvent::kBucketSelection)},
    {"SINDI_V2",
     true,
     true,
     ReasoningEventBit(ReasoningEvent::kVisit) | ReasoningEventBit(ReasoningEvent::kEviction) |
         ReasoningEventBit(ReasoningEvent::kFilterReject) |
         ReasoningEventBit(ReasoningEvent::kReorder) |
         ReasoningEventBit(ReasoningEvent::kReorderEviction) |
         ReasoningEventBit(ReasoningEvent::kBucketSelection)},
    {"BruteForce",
     true,
     true,
     ReasoningEventBit(ReasoningEvent::kVisit) | ReasoningEventBit(ReasoningEvent::kFilterReject)},
    {"WARP",
     true,
     true,
     ReasoningEventBit(ReasoningEvent::kVisit) | ReasoningEventBit(ReasoningEvent::kFilterReject)},
    {"Pyramid",
     true,
     false,
     ReasoningEventBit(ReasoningEvent::kVisit) | ReasoningEventBit(ReasoningEvent::kEviction) |
         ReasoningEventBit(ReasoningEvent::kFilterReject) |
         ReasoningEventBit(ReasoningEvent::kReorder) |
         ReasoningEventBit(ReasoningEvent::kReorderEviction)},
};

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
