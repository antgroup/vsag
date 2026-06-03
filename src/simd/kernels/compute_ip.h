
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

// Pure-template inner-product kernel.
//
// Architecture-agnostic: each ISA translation unit instantiates
// ComputeIPImpl<Traits, Unroll>(...) with its own SimdTraits<>
// specialization, plus a function-pointer fallback that handles tail
// elements smaller than one vector. The fallback preserves the
// project's existing ISA-cascade behaviour (avx512 -> avx2 -> sse ->
// generic, neon -> generic) so we never lose performance on tail
// processing.
//
// Unroll > 1 gives the kernel multiple independent accumulators in the
// main loop, breaking the FMA dependency chain. Each ISA picks the
// unroll factor that matches the manually-unrolled code it replaces.

namespace vsag::simd {

using FloatFallback = float (*)(const float*, const float*, uint64_t);

template <typename T, int Unroll = 1>
inline float
ComputeIPImpl(const float* query,
              const float* codes,
              uint64_t dim,
              FloatFallback fallback = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;
    constexpr int Stride = W * Unroll;

    // For W == 1 (generic backend) the dim < W branch is dead code.
    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            return fallback(query, codes, dim);
        }
    }

    // Main loop with Unroll independent accumulators.
    V acc[Unroll];
    for (int u = 0; u < Unroll; ++u) {
        acc[u] = T::zero();
    }

    uint64_t i = 0;
    for (; i + Stride <= dim; i += Stride) {
        for (int u = 0; u < Unroll; ++u) {
            acc[u] = T::fmadd(T::load(query + i + u * W), T::load(codes + i + u * W), acc[u]);
        }
    }

    V sum = acc[0];
    for (int u = 1; u < Unroll; ++u) {
        sum = T::add(sum, acc[u]);
    }

    // Second-tier loop: remaining whole vectors, single accumulator.
    if constexpr (Unroll > 1) {
        for (; i + W <= dim; i += W) {
            sum = T::fmadd(T::load(query + i), T::load(codes + i), sum);
        }
    }

    float result = T::reduce_add(sum);

    // Tail: less than one vector left. Generic backend never reaches
    // here because W == 1 makes (dim > i) impossible (i increments by W).
    if constexpr (W > 1) {
        if (dim > i) {
            result += fallback(query + i, codes + i, dim - i);
        }
    }
    return result;
}

}  // namespace vsag::simd
