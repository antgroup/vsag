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

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace vsag::simd {

inline bool
mci_hash_contains_two_int32x4(const int32_t* first, const int32_t* second, int32_t value) {
#if defined(__SSE2__)
    __m128i cmp = _mm_set1_epi32(value);
    __m128i first_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(first));
    __m128i second_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(second));
    __m128i flag =
        _mm_or_si128(_mm_cmpeq_epi32(cmp, first_values), _mm_cmpeq_epi32(cmp, second_values));

    return _mm_movemask_epi8(flag) != 0;
#else
    for (uint32_t i = 0; i < 4; ++i) {
        if (first[i] == value || second[i] == value) {
            return true;
        }
    }
    return false;
#endif
}

}  // namespace vsag::simd
