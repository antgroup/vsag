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
#include <string_view>

#include "reasoning_types.h"

namespace vsag {

/// Reasoning modes and events supported by one index implementation.
struct ReasoningCapability {
    std::string_view index_type;
    bool supports_knn{false};
    bool supports_range{false};
    uint64_t event_mask{0};
};

/// Returns nullptr when the index type has no Reasoning support.
const ReasoningCapability*
GetReasoningCapability(std::string_view index_type);

}  // namespace vsag
