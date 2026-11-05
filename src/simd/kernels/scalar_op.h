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

// Scalar-broadcast kernels: to[i] = from[i] / scalar (DivScalar)
//                           data[i] *= val (VecRescale)
//
// These require traits to expose set1(float) in addition to the base
// interface. The fallback handles tail elements via the next-narrower ISA.

#include <cstdint>

namespace vsag::simd {

using ScalarFallback = void (*)(const float*, float*, uint64_t, float);
using RescaleFallback = void (*)(float*, uint64_t, float);

template <typename T>
inline void
DivScalarImpl(
    const float* from, float* to, uint64_t dim, float scalar, ScalarFallback fallback = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if (dim == 0) {
        return;
    }
    if (scalar == 0) {
        scalar = 1.0f;
    }

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            fallback(from, to, dim, scalar);
            return;
        }
    }

    V scalarVec = T::set1(scalar);
    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V vec = T::load(from + i);
        T::store(to + i, T::div(vec, scalarVec));
    }

    if constexpr (W > 1) {
        if (dim > i) {
            fallback(from + i, to + i, dim - i, scalar);
        }
    }
}

template <typename T>
inline void
VecRescaleImpl(float* data, uint64_t dim, float val, RescaleFallback fallback = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            fallback(data, dim, val);
            return;
        }
    }

    V scalarVec = T::set1(val);
    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V vec = T::load(data + i);
        T::store(data + i, T::mul(vec, scalarVec));
    }

    if constexpr (W > 1) {
        if (dim > i) {
            fallback(data + i, dim - i, val);
        }
    }
}

}  // namespace vsag::simd
