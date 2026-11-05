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

#include <utility>

#include "io/backend/noncontinuous_backend.h"
#include "io/cache/no_cache.h"
#include "io/core/byte_io.h"

namespace vsag {

/**
 * A logical IO over shared monotonic extents. Shrink changes the logical size but cannot reclaim
 * physical extents because NonContinuousAllocator intentionally has no release operation.
 */
template <typename PhysicalIO>
class NonContinuousIO : public ByteIO<NonContinuousBackend<PhysicalIO>, NoCache> {
public:
    using Base = ByteIO<NonContinuousBackend<PhysicalIO>, NoCache>;

    template <typename... Args>
    NonContinuousIO(NonContinuousAllocator* extent_allocator, Allocator* allocator, Args&&... args)
        : Base(extent_allocator, allocator, std::forward<Args>(args)...) {
    }
};

}  // namespace vsag
