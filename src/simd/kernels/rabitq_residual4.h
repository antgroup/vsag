
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

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "utils/float_utils.h"

namespace vsag::simd {

// Keep 32 as the on-wire plane stride even for a partial block.
template <typename T, typename LoadInt>
uint32_t
RaBitQResidual4(const int32_t* accumulators,
                uint32_t filter_bits,
                float delta,
                float sum_vl,
                float query_sum,
                float query_norm,
                float query_bucket_norm_sqr,
                float inv_sqrt_d,
                const float* f_add,
                const float* f_scale,
                const float* filter_norm_codes,
                uint32_t valid_size,
                float* dists,
                float* filter_inner_products,
                LoadInt load_int) {
    static_assert(T::Width == 4);
    constexpr float max = std::numeric_limits<float>::max();
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    const auto delta_v = T::set1(delta);
    const auto sum_vl_v = T::set1(sum_vl);
    const auto center = T::set1(0.5F * static_cast<float>((1U << filter_bits) - 1U) * query_sum);
    uint32_t mask = 0;
    for (uint32_t i = 0; i < valid_size; i += 4) {
        const auto remaining = valid_size - i;
        const auto lanes = remaining < 4 ? remaining : 4U;
        auto floats = [&](const float* p, float fill) {
            if (lanes == 4) {
                return T::load(p + i);
            }
            std::array<float, 4> tmp{fill, fill, fill, fill};
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                tmp[lane] = p[i + lane];
            }
            return T::load(tmp.data());
        };
        auto lookup = [&](uint32_t plane) {
            const auto* p = accumulators + plane * 32 + i;
            if (lanes == 4) {
                return T::add(T::mul(load_int(p), delta_v), sum_vl_v);
            }
            std::array<int32_t, 4> tmp{};
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                tmp[lane] = p[lane];
            }
            return T::add(T::mul(load_int(tmp.data()), delta_v), sum_vl_v);
        };
        auto ip = T::zero();
        if (filter_bits == 1) {
            ip = T::mul(T::sub(T::mul(T::set1(2.0F), lookup(0)), T::set1(query_sum)),
                        T::set1(inv_sqrt_d));
        } else {
            for (uint32_t plane = 0; plane < filter_bits; ++plane) {
                const float weight = static_cast<float>(1U << (filter_bits - plane - 1));
                ip = T::add(ip, T::mul(lookup(plane), T::set1(weight)));
            }
            // Invalid norm lanes are masked when publishing the result.
            ip = T::div(T::sub(ip, center), floats(filter_norm_codes, 1.0F));
        }
        const auto result = T::add(T::add(floats(f_add, 0.0F), T::set1(query_bucket_norm_sqr)),
                                   T::mul(T::mul(floats(f_scale, 0.0F), T::set1(query_norm)), ip));
        std::array<float, 4> lane_result, lane_ip;
        T::store(lane_result.data(), result);
        if (filter_inner_products != nullptr) {
            T::store(lane_ip.data(), ip);
        }
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const auto index = i + lane;
            const bool ok = IsFiniteFloatBits(f_add[index]) && IsFiniteFloatBits(f_scale[index]) &&
                            std::abs(f_scale[index]) > std::numeric_limits<float>::epsilon() &&
                            (filter_bits == 1 || filter_norm_codes[index] > 0.0F) &&
                            IsFiniteFloatBits(lane_result[lane]);
            dists[index] = ok ? lane_result[lane] : max;
            if (filter_inner_products != nullptr) {
                filter_inner_products[index] = ok ? lane_ip[lane] : nan;
            }
            if (ok) {
                mask |= 1U << index;
            }
        }
    }
    return mask;
}

}  // namespace vsag::simd
