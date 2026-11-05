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

// Normalize-family kernels:
//   NormalizeWithCentroid:        to[i] = (from[i] - centroid[i]) / norm
//   InverseNormalizeWithCentroid: to[i] = from[i] * norm + centroid[i]
//
// Both require traits with set1() for scalar broadcast.
// NormalizeWithCentroid computes norm = sqrt(L2Sqr(from, centroid, dim))
// internally, using the template-based L2 kernel for the distance part.

#include <cmath>
#include <cstdint>

namespace vsag::simd {

using NormCentroidFallback = float (*)(const float*, const float*, float*, uint64_t);
using InvNormCentroidFallback = void (*)(const float*, const float*, float*, uint64_t, float);

// NormalizeWithCentroid: compute norm = sqrt(sum((from-centroid)^2)),
// then to = (from - centroid) / norm. Returns norm.
template <typename T>
inline float
NormalizeWithCentroidImpl(const float* from,
                          const float* centroid,
                          float* to,
                          uint64_t dim,
                          NormCentroidFallback fallback = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            return fallback(from, centroid, to, dim);
        }
    }

    // Phase 1: compute norm_sq = sum((from - centroid)^2)
    V sum = T::zero();
    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V f = T::load(from + i);
        V c = T::load(centroid + i);
        V diff = T::sub(f, c);
        sum = T::fmadd(diff, diff, sum);
    }
    float norm_sq = T::reduce_add(sum);
    // Scalar tail for norm computation
    for (uint64_t j = i; j < dim; ++j) {
        float d = from[j] - centroid[j];
        norm_sq += d * d;
    }

    float norm = (norm_sq < 1e-5f) ? 1.0f : std::sqrt(norm_sq);

    // Phase 2: to = (from - centroid) / norm
    V normVec = T::set1(norm);
    for (i = 0; i + W <= dim; i += W) {
        V f = T::load(from + i);
        V c = T::load(centroid + i);
        V diff = T::sub(f, c);
        T::store(to + i, T::div(diff, normVec));
    }
    // Scalar tail for division
    for (uint64_t j = i; j < dim; ++j) {
        to[j] = (from[j] - centroid[j]) / norm;
    }

    return norm;
}

// InverseNormalizeWithCentroid: to[i] = from[i] * norm + centroid[i]
template <typename T>
inline void
InverseNormalizeWithCentroidImpl(const float* from,
                                 const float* centroid,
                                 float* to,
                                 uint64_t dim,
                                 float norm,
                                 InvNormCentroidFallback fallback = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            fallback(from, centroid, to, dim, norm);
            return;
        }
    }

    V normVec = T::set1(norm);
    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V f = T::load(from + i);
        V c = T::load(centroid + i);
        T::store(to + i, T::fmadd(f, normVec, c));
    }

    if constexpr (W > 1) {
        if (dim > i) {
            for (uint64_t j = i; j < dim; ++j) {
                to[j] = from[j] * norm + centroid[j];
            }
        }
    }
}

}  // namespace vsag::simd
