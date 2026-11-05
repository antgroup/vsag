
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

// Element-wise binary op kernel: z[i] = op(x[i], y[i]).
// Used by FP32Add / FP32Sub / FP32Mul / FP32Div across all ISAs.
//
// Op selects the per-element operation via a tag dispatched at compile time
// to the corresponding traits method (add/sub/mul/div). The Generic backend
// (Width == 1) compiles out the fallback branch via `if constexpr`.

#include <cstdint>

namespace vsag::simd {

using BinaryFallback = void (*)(const float*, const float*, float*, uint64_t);

enum class BinaryOp { Add, Sub, Mul, Div };

template <typename T, BinaryOp Op>
inline __attribute__((always_inline)) typename T::FloatVec
binary_apply(typename T::FloatVec a, typename T::FloatVec b) {
    if constexpr (Op == BinaryOp::Add) {
        return T::add(a, b);
    } else if constexpr (Op == BinaryOp::Sub) {
        return T::sub(a, b);
    } else if constexpr (Op == BinaryOp::Mul) {
        return T::mul(a, b);
    } else {
        return T::div(a, b);
    }
}

template <typename T, BinaryOp Op>
inline void
BinaryOpImpl(
    const float* x, const float* y, float* z, uint64_t dim, BinaryFallback fallback = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            fallback(x, y, z, dim);
            return;
        }
    }

    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V a = T::load(x + i);
        V b = T::load(y + i);
        T::store(z + i, binary_apply<T, Op>(a, b));
    }

    if constexpr (W > 1) {
        if (dim > i) {
            fallback(x + i, y + i, z + i, dim - i);
        }
    }
}

}  // namespace vsag::simd
