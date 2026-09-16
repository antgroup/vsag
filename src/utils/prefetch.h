
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
    if constexpr (N > 24) {
        return PrefetchImpl<24>(data);
    }
    for (int i = 0; i < N; ++i) {
        __builtin_prefetch(static_cast<const char*>(data) + i * 64, 0, 3);
    }
}

void
PrefetchLines(const void* data, uint64_t size);

}  // namespace vsag
