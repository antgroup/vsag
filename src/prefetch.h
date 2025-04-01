
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
#include <algorithm>

#include "simd/simd.h"

namespace vsag {
template <int N>
__inline void __attribute__((__always_inline__)) PrefetchImpl(const void* data) {
    if constexpr (N > 32) {
        return PrefetchImpl<32>(data);
    }
    Prefetch(data);
    PrefetchImpl<N - 1>(static_cast<const char*>(data) + 64);
}

void
PrefetchLines(const void* data, uint64_t size);

}  // namespace vsag
