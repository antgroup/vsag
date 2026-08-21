
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

template <>
inline void
PrefetchImpl<0>(const void* data) {
}

// Prefetching sits on the per-neighbor search hot path, so this must stay
// inlinable: an out-of-line call plus dispatch would dominate the profile.
#define VSAG_PREFETCH_LINE(X)    \
    case X:                      \
        PrefetchImpl<X>(data);   \
        break;

inline void __attribute__((always_inline))
PrefetchLines(const void* data, uint64_t size) {
    uint64_t n = std::min<uint64_t>(size / 64, 63ULL);
    switch (n) {
        VSAG_PREFETCH_LINE(0);
        VSAG_PREFETCH_LINE(1);
        VSAG_PREFETCH_LINE(2);
        VSAG_PREFETCH_LINE(3);
        VSAG_PREFETCH_LINE(4);
        VSAG_PREFETCH_LINE(5);
        VSAG_PREFETCH_LINE(6);
        VSAG_PREFETCH_LINE(7);
        VSAG_PREFETCH_LINE(8);
        VSAG_PREFETCH_LINE(9);
        VSAG_PREFETCH_LINE(10);
        VSAG_PREFETCH_LINE(11);
        VSAG_PREFETCH_LINE(12);
        VSAG_PREFETCH_LINE(13);
        VSAG_PREFETCH_LINE(14);
        VSAG_PREFETCH_LINE(15);
        VSAG_PREFETCH_LINE(16);
        VSAG_PREFETCH_LINE(17);
        VSAG_PREFETCH_LINE(18);
        VSAG_PREFETCH_LINE(19);
        VSAG_PREFETCH_LINE(20);
        VSAG_PREFETCH_LINE(21);
        VSAG_PREFETCH_LINE(22);
        VSAG_PREFETCH_LINE(23);
        VSAG_PREFETCH_LINE(24);
        VSAG_PREFETCH_LINE(25);
        VSAG_PREFETCH_LINE(26);
        VSAG_PREFETCH_LINE(27);
        VSAG_PREFETCH_LINE(28);
        VSAG_PREFETCH_LINE(29);
        VSAG_PREFETCH_LINE(30);
        VSAG_PREFETCH_LINE(31);
        VSAG_PREFETCH_LINE(32);
        VSAG_PREFETCH_LINE(33);
        VSAG_PREFETCH_LINE(34);
        VSAG_PREFETCH_LINE(35);
        VSAG_PREFETCH_LINE(36);
        VSAG_PREFETCH_LINE(37);
        VSAG_PREFETCH_LINE(38);
        VSAG_PREFETCH_LINE(39);
        VSAG_PREFETCH_LINE(40);
        VSAG_PREFETCH_LINE(41);
        VSAG_PREFETCH_LINE(42);
        VSAG_PREFETCH_LINE(43);
        VSAG_PREFETCH_LINE(44);
        VSAG_PREFETCH_LINE(45);
        VSAG_PREFETCH_LINE(46);
        VSAG_PREFETCH_LINE(47);
        VSAG_PREFETCH_LINE(48);
        VSAG_PREFETCH_LINE(49);
        VSAG_PREFETCH_LINE(50);
        VSAG_PREFETCH_LINE(51);
        VSAG_PREFETCH_LINE(52);
        VSAG_PREFETCH_LINE(53);
        VSAG_PREFETCH_LINE(54);
        VSAG_PREFETCH_LINE(55);
        VSAG_PREFETCH_LINE(56);
        VSAG_PREFETCH_LINE(57);
        VSAG_PREFETCH_LINE(58);
        VSAG_PREFETCH_LINE(59);
        VSAG_PREFETCH_LINE(60);
        VSAG_PREFETCH_LINE(61);
        VSAG_PREFETCH_LINE(62);
        default:
            PrefetchImpl<63>(data);
            break;
    }
}

#undef VSAG_PREFETCH_LINE

}  // namespace vsag
