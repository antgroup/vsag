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

// SQ8 quantized distance kernels.
//
// SQ8ComputeIP:  sum += query[i] * (codes[i]/255 * diff[i] + lower_bound[i])
// SQ8ComputeL2Sqr: sum += (query[i] - (codes[i]/255 * diff[i] + lower_bound[i]))^2
// SQ8ComputeCodesIP: sum += adj1[i] * adj2[i]   where adj = codes/255*diff+lb
// SQ8ComputeCodesL2Sqr: sum += (adj1[i] - adj2[i])^2
//
// Parameterized on SQ8Traits<ISA>, which must expose:
//   FloatVec        - SIMD float vector type
//   Width           - number of float elements per vector
//   load_u8_as_float(const uint8_t* p) -> FloatVec
//                    Load Width uint8 values, zero-extend, convert to float
//   zero()          -> FloatVec
//   set1(float)     -> FloatVec
//   load(const float* p) -> FloatVec
//   mul(FloatVec, FloatVec) -> FloatVec
//   add(FloatVec, FloatVec) -> FloatVec
//   sub(FloatVec, FloatVec) -> FloatVec
//   fmadd(FloatVec a, FloatVec b, FloatVec c) -> FloatVec  (a*b+c)
//   reduce_add(FloatVec) -> float

namespace vsag::simd {

// --- SQ8ComputeIP ---

template <typename T>
inline float
SQ8ComputeIPImpl(const float* query,
                 const uint8_t* codes,
                 const float* lower_bound,
                 const float* diff,
                 uint64_t dim,
                 float (*fallback)(const float*,
                                   const uint8_t*,
                                   const float*,
                                   const float*,
                                   uint64_t) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            return fallback(query, codes, lower_bound, diff, dim);
        }
    }

    V sum = T::zero();
    V inv255 = T::set1(1.0f / 255.0f);

    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V code_floats = T::load_u8_as_float(codes + i);
        V query_vals = T::load(query + i);
        V diff_vals = T::load(diff + i);
        V lb_vals = T::load(lower_bound + i);

        V normalized = T::mul(code_floats, inv255);
        V adjusted = T::fmadd(normalized, diff_vals, lb_vals);
        sum = T::fmadd(query_vals, adjusted, sum);
    }

    float result = T::reduce_add(sum);

    if constexpr (W > 1) {
        if (i < dim) {
            result += fallback(query + i, codes + i, lower_bound + i, diff + i, dim - i);
        }
    }
    return result;
}

// --- SQ8ComputeL2Sqr ---

template <typename T>
inline float
SQ8ComputeL2SqrImpl(const float* query,
                    const uint8_t* codes,
                    const float* lower_bound,
                    const float* diff,
                    uint64_t dim,
                    float (*fallback)(const float*,
                                      const uint8_t*,
                                      const float*,
                                      const float*,
                                      uint64_t) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            return fallback(query, codes, lower_bound, diff, dim);
        }
    }

    V sum = T::zero();
    V inv255 = T::set1(1.0f / 255.0f);

    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V code_floats = T::load_u8_as_float(codes + i);
        V query_vals = T::load(query + i);
        V diff_vals = T::load(diff + i);
        V lb_vals = T::load(lower_bound + i);

        V normalized = T::mul(code_floats, inv255);
        V adjusted = T::fmadd(normalized, diff_vals, lb_vals);
        V d = T::sub(query_vals, adjusted);
        sum = T::fmadd(d, d, sum);
    }

    float result = T::reduce_add(sum);

    if constexpr (W > 1) {
        if (i < dim) {
            result += fallback(query + i, codes + i, lower_bound + i, diff + i, dim - i);
        }
    }
    return result;
}

template <typename T>
inline void
SQ8ComputeL2SqrBatch4Impl(
    const float* query,
    const uint8_t* codes1,
    const uint8_t* codes2,
    const uint8_t* codes3,
    const uint8_t* codes4,
    const float* lower_bound,
    const float* diff,
    uint64_t dim,
    float& dist1,
    float& dist2,
    float& dist3,
    float& dist4,
    float (*fallback)(const float*, const uint8_t*, const float*, const float*, uint64_t)) {
    using V = typename T::FloatVec;
    constexpr uint64_t W = T::Width;
    V sums[4] = {T::zero(), T::zero(), T::zero(), T::zero()};
    const uint8_t* codes[4] = {codes1, codes2, codes3, codes4};
    const V inv255 = T::set1(1.0f / 255.0f);

    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        const V q = T::load(query + i);
        const V lb = T::load(lower_bound + i);
        const V scale = T::load(diff + i);
        for (uint64_t j = 0; j < 4; ++j) {
            const V code = T::load_u8_as_float(codes[j] + i);
            const V adjusted = T::fmadd(T::mul(code, inv255), scale, lb);
            const V delta = T::sub(q, adjusted);
            sums[j] = T::fmadd(delta, delta, sums[j]);
        }
    }

    float* results[4] = {&dist1, &dist2, &dist3, &dist4};
    for (uint64_t j = 0; j < 4; ++j) {
        *results[j] = T::reduce_add(sums[j]);
        if (i < dim) {
            *results[j] += fallback(query + i, codes[j] + i, lower_bound + i, diff + i, dim - i);
        }
    }
}

// --- SQ8ComputeCodesIP ---

template <typename T>
inline float
SQ8ComputeCodesIPImpl(const uint8_t* codes1,
                      const uint8_t* codes2,
                      const float* lower_bound,
                      const float* diff,
                      uint64_t dim,
                      float (*fallback)(const uint8_t*,
                                        const uint8_t*,
                                        const float*,
                                        const float*,
                                        uint64_t) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            return fallback(codes1, codes2, lower_bound, diff, dim);
        }
    }

    V sum = T::zero();
    V inv255 = T::set1(1.0f / 255.0f);

    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V c1_floats = T::load_u8_as_float(codes1 + i);
        V c2_floats = T::load_u8_as_float(codes2 + i);
        V diff_vals = T::load(diff + i);
        V lb_vals = T::load(lower_bound + i);

        V adj1 = T::fmadd(T::mul(c1_floats, inv255), diff_vals, lb_vals);
        V adj2 = T::fmadd(T::mul(c2_floats, inv255), diff_vals, lb_vals);
        sum = T::fmadd(adj1, adj2, sum);
    }

    float result = T::reduce_add(sum);

    if constexpr (W > 1) {
        if (i < dim) {
            result += fallback(codes1 + i, codes2 + i, lower_bound + i, diff + i, dim - i);
        }
    }
    return result;
}

// --- SQ8ComputeCodesL2Sqr ---

template <typename T>
inline float
SQ8ComputeCodesL2SqrImpl(const uint8_t* codes1,
                         const uint8_t* codes2,
                         const float* lower_bound,
                         const float* diff,
                         uint64_t dim,
                         float (*fallback)(const uint8_t*,
                                           const uint8_t*,
                                           const float*,
                                           const float*,
                                           uint64_t) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            return fallback(codes1, codes2, lower_bound, diff, dim);
        }
    }

    V sum = T::zero();
    V inv255 = T::set1(1.0f / 255.0f);

    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V c1_floats = T::load_u8_as_float(codes1 + i);
        V c2_floats = T::load_u8_as_float(codes2 + i);
        V diff_vals = T::load(diff + i);
        V lb_vals = T::load(lower_bound + i);

        V adj1 = T::fmadd(T::mul(c1_floats, inv255), diff_vals, lb_vals);
        V adj2 = T::fmadd(T::mul(c2_floats, inv255), diff_vals, lb_vals);
        V d = T::sub(adj1, adj2);
        sum = T::fmadd(d, d, sum);
    }

    float result = T::reduce_add(sum);

    if constexpr (W > 1) {
        if (i < dim) {
            result += fallback(codes1 + i, codes2 + i, lower_bound + i, diff + i, dim - i);
        }
    }
    return result;
}

}  // namespace vsag::simd
