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

#include <immintrin.h>

#include "pipnn_half_scalar.h"

namespace vsag::pipnn_half::detail {

void
DecodeRowBF16(const uint16_t* source, float* destination, uint64_t dim) {
    uint64_t k = 0;
    for (; dim - k >= 8; k += 8) {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + k));
        const __m256i bits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16);
        _mm256_storeu_ps(destination + k, _mm256_castsi256_ps(bits));
    }
    DecodeRowScalar<Format::BF16>(source + k, destination + k, dim - k);
}

}  // namespace vsag::pipnn_half::detail
