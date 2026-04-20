
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

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "simd/simd.h"

namespace vsag {
template <int N>
#ifdef _MSC_VER
__forceinline void
PrefetchImpl(const void* data) {
#else
__inline void __attribute__((__always_inline__)) PrefetchImpl(const void* data) {
#endif
    if constexpr (N > 24) {
        return PrefetchImpl<24>(data);
    }
    for (int i = 0; i < N; ++i) {
#ifdef _MSC_VER
        _mm_prefetch(static_cast<const char*>(data) + i * 64, _MM_HINT_T0);
#else
        __builtin_prefetch(static_cast<const char*>(data) + i * 64, 0, 3);
#endif
    }
}

void
PrefetchLines(const void* data, uint64_t size);

}  // namespace vsag
