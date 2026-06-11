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

// Butterfly-pattern kernels used in Fast Hadamard Transform (FHT):
//   RotateOp:  data[i+j] = data[i+j] + data[i+j+step]
//              data[i+j+step] = data[i+j] - data[i+j+step]
//   KacsWalk:  in-place butterfly on two halves of the array.
//
// These use only load/store/add/sub from the traits interface.

#include <cmath>
#include <cstdint>

namespace vsag::simd {

using RotateOpFallback = void (*)(float*, int, int, int);
using KacsWalkFallback = void (*)(float*, uint64_t);

// RotateOp: butterfly on stride `step` within [idx, dim_).
// Requires step >= T::Width for the vectorized path.
template <typename T>
inline void
RotateOpImpl(float* data, int idx, int dim_, int step) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    for (int i = idx; i < dim_; i += step * 2) {
        int j = 0;
        for (; j + W <= step; j += W) {
            V g1 = T::load(&data[i + j]);
            V g2 = T::load(&data[i + j + step]);
            T::store(&data[i + j], T::add(g1, g2));
            T::store(&data[i + j + step], T::sub(g1, g2));
        }
        for (; j < step; ++j) {
            float g1 = data[i + j];
            float g2 = data[i + j + step];
            data[i + j] = g1 + g2;
            data[i + j + step] = g1 - g2;
        }
    }
}

// KacsWalk: in-place butterfly on data[0..len/2-1] vs data[offset..offset+len/2-1].
// For odd-length arrays, the middle element is scaled by sqrt(2).
template <typename T>
inline void
KacsWalkImpl(float* data, uint64_t len, KacsWalkFallback fallback = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (len / 2 < static_cast<uint64_t>(W)) {
            fallback(data, len);
            return;
        }
    }

    uint64_t base = len % 2;
    uint64_t offset = base + (len / 2);
    uint64_t i = 0;

    for (; i + W <= len / 2; i += W) {
        V x = T::load(&data[i]);
        V y = T::load(&data[i + offset]);
        T::store(&data[i], T::add(x, y));
        T::store(&data[i + offset], T::sub(x, y));
    }

    // Scalar tail
    for (; i < len / 2; i++) {
        float x = data[i];
        float y = data[i + offset];
        data[i] = x + y;
        data[i + offset] = x - y;
    }

    if (base != 0) {
        data[len / 2] *= std::sqrt(2.0f);
    }
}

}  // namespace vsag::simd
