
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

// Pure-template squared-L2 kernel. See compute_ip.h for the rationale
// behind the Unroll parameter and the fallback function pointer.

namespace vsag::simd {

using FloatFallback = float (*)(const float*, const float*, uint64_t);

template <typename T, int Unroll = 1>
inline float
ComputeL2SqrImpl(const float* query,
                 const float* codes,
                 uint64_t dim,
                 FloatFallback fallback = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;
    constexpr int Stride = W * Unroll;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            return fallback(query, codes, dim);
        }
    }

    V acc[Unroll];
    for (int u = 0; u < Unroll; ++u) {
        acc[u] = T::zero();
    }

    uint64_t i = 0;
    for (; i + Stride <= dim; i += Stride) {
        for (int u = 0; u < Unroll; ++u) {
            V diff = T::sub(T::load(query + i + u * W), T::load(codes + i + u * W));
            acc[u] = T::fmadd(diff, diff, acc[u]);
        }
    }

    V sum = acc[0];
    for (int u = 1; u < Unroll; ++u) {
        sum = T::add(sum, acc[u]);
    }

    if constexpr (Unroll > 1) {
        for (; i + W <= dim; i += W) {
            V diff = T::sub(T::load(query + i), T::load(codes + i));
            sum = T::fmadd(diff, diff, sum);
        }
    }

    float result = T::reduce_add(sum);

    if constexpr (W > 1) {
        if (dim > i) {
            result += fallback(query + i, codes + i, dim - i);
        }
    }
    return result;
}

}  // namespace vsag::simd
