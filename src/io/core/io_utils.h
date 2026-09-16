// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <limits>

#include "vsag_exception.h"

namespace vsag {

[[nodiscard]] inline bool
IsValidRange(uint64_t offset, uint64_t size, uint64_t extent) {
    return offset <= extent and size <= extent - offset;
}

[[nodiscard]] inline uint64_t
CheckedEnd(uint64_t offset, uint64_t size) {
    if (size > std::numeric_limits<uint64_t>::max() - offset) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "IO offset and size overflow");
    }
    return offset + size;
}

}  // namespace vsag
