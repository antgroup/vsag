
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

// Reduce-add kernel: sum(x[0..dim-1]).
// Used by FP32ReduceAdd across all ISAs.

#include <cstdint>

namespace vsag::simd {

using ReduceFallback = float (*)(const float*, uint64_t);

template <typename T>
inline float
ReduceAddImpl(const float* x, uint64_t dim, ReduceFallback fallback = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            return fallback(x, dim);
        }
    }

    V sum = T::zero();
    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        sum = T::add(sum, T::load(x + i));
    }
    float result = T::reduce_add(sum);
    if constexpr (W > 1) {
        if (dim > i) {
            result += fallback(x + i, dim - i);
        }
    }
    return result;
}

}  // namespace vsag::simd
