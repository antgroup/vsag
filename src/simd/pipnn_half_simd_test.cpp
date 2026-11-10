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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "pipnn_half_scalar.h"
#include "simd_status.h"
#include "unittest.h"

namespace vsag::pipnn_half {
namespace {
uint32_t
Bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
}  // namespace

TEST_CASE("PIPNN half dispatch capability combinations", "[ut][simd]") {
    // Resolve through the production path. Never call synthetic selections.
    const auto scalar = ResolveDispatch({false, false, false});
    const auto optimized = ResolveDispatch({true, true, true});
    REQUIRE(scalar.decode_fp16 != nullptr);
    REQUIRE(scalar.decode_bf16 != nullptr);
    REQUIRE(scalar.preparation != nullptr);
#if defined(ENABLE_AVX2)
    REQUIRE(optimized.decode_bf16 == detail::DecodeRowBF16);
#else
    REQUIRE(optimized.decode_bf16 == scalar.decode_bf16);
#endif
#if defined(ENABLE_AVX2) && defined(ENABLE_PIPNN_F16C)
    REQUIRE(optimized.decode_fp16 == detail::DecodeRowFP16);
    REQUIRE(optimized.preparation == detail::PreparationAVX2);
#else
    REQUIRE(optimized.decode_fp16 == scalar.decode_fp16);
    REQUIRE(optimized.preparation == scalar.preparation);
#endif

    for (unsigned mask = 0; mask < 64; ++mask) {
        INFO("capability mask=" << mask);
        const Capabilities caps{bool(mask & 1),
                                bool(mask & 2),
                                bool(mask & 4),
                                bool(mask & 8),
                                bool(mask & 16),
                                bool(mask & 32)};
        const bool avx2 =
            caps.compiled_avx2 && caps.cpuinfo_initialized && caps.cpu_avx && caps.cpu_avx2;
        const bool f16c = avx2 && caps.compiled_f16c && caps.cpu_f16c;
        const auto selection = SelectImplementations(caps);
        REQUIRE(selection.decode_bf16_avx2 == avx2);
        REQUIRE(selection.decode_fp16_avx2_f16c == f16c);
        REQUIRE(selection.preparation_avx2_f16c == f16c);

        const auto dispatch = ResolveDispatch(selection);
        REQUIRE(dispatch.decode_bf16 == (avx2 ? optimized.decode_bf16 : scalar.decode_bf16));
        REQUIRE(dispatch.decode_fp16 == (f16c ? optimized.decode_fp16 : scalar.decode_fp16));
        REQUIRE(dispatch.preparation == (f16c ? optimized.preparation : scalar.preparation));
    }

    // Exercise the actual cached production dispatch, without executing any
    // function selected using fabricated CPU capabilities.
    const auto& actual = GetDispatch();
#if defined(ENABLE_AVX2)
    REQUIRE(actual.decode_bf16 ==
            (SimdStatus::SupportAVXAndAVX2() ? detail::DecodeRowBF16 : scalar.decode_bf16));
#else
    REQUIRE(actual.decode_bf16 == scalar.decode_bf16);
#endif
#if defined(ENABLE_AVX2) && defined(ENABLE_PIPNN_F16C)
    REQUIRE(actual.decode_fp16 ==
            (SimdStatus::SupportAVX2F16C() ? detail::DecodeRowFP16 : scalar.decode_fp16));
    REQUIRE(actual.preparation ==
            (SimdStatus::SupportAVX2F16C() ? detail::PreparationAVX2 : scalar.preparation));
#else
    REQUIRE(actual.decode_fp16 == scalar.decode_fp16);
    REQUIRE(actual.preparation == scalar.preparation);
#endif
}

TEST_CASE("PIPNN half row decoding dispatched and scalar agree", "[ut][simd]") {
    const auto& dispatch = GetDispatch();
    const auto scalar_dispatch = ResolveDispatch({false, false, false});
    REQUIRE(dispatch.decode_fp16 != nullptr);
    REQUIRE(dispatch.decode_bf16 != nullptr);
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVXAndAVX2()) {
        REQUIRE(dispatch.decode_bf16 == detail::DecodeRowBF16);
    } else {
        REQUIRE(dispatch.decode_bf16 == scalar_dispatch.decode_bf16);
    }
#else
    REQUIRE(dispatch.decode_bf16 == scalar_dispatch.decode_bf16);
#endif
    for (const auto format : {Format::FP16, Format::BF16}) {
        const auto decode = format == Format::FP16 ? dispatch.decode_fp16 : dispatch.decode_bf16;
        // Exhaust every bit pattern, including signaling NaNs and both zero signs.
        // Offset the buffers to exercise unaligned loads and stores.
        std::vector<uint16_t> input(65538);
        std::vector<float> output(65538, 123.0F);
        for (uint32_t i = 0; i < 65536; ++i) {
            input[i + 1] = static_cast<uint16_t>(i);
        }
        decode(input.data() + 1, output.data() + 1, 65536);
        // Exercise the scalar fallback even on hosts whose production dispatch
        // selects SIMD. Check every encoding and both guard elements.
        std::vector<float> scalar_output(65538, 123.0F);
        const auto scalar_decode =
            format == Format::FP16 ? scalar_dispatch.decode_fp16 : scalar_dispatch.decode_bf16;
        scalar_decode(input.data() + 1, scalar_output.data() + 1, 65536);
        REQUIRE(scalar_output.front() == 123.0F);
        REQUIRE(scalar_output.back() == 123.0F);
        REQUIRE(output.front() == 123.0F);
        REQUIRE(output.back() == 123.0F);
        for (uint32_t i = 0; i < 65536; ++i) {
            INFO("format=" << static_cast<int>(format) << " bits=" << i);
            REQUIRE(Bits(output[i + 1]) == Bits(Decode(input[i + 1], format)));
            REQUIRE(Bits(scalar_output[i + 1]) == Bits(Decode(input[i + 1], format)));
        }
        for (uint64_t dim = 0; dim <= 33; ++dim) {
            std::vector<float> actual(dim + 2, 123.0F);
            decode(input.data() + 1, actual.data() + 1, dim);
            REQUIRE(actual.front() == 123.0F);
            REQUIRE(actual.back() == 123.0F);
            for (uint64_t i = 0; i < dim; ++i) {
                REQUIRE(Bits(actual[i + 1]) == Bits(Decode(input[i + 1], format)));
            }
        }
    }
}

TEST_CASE("PIPNN half dispatched norm and projection", "[ut][simd]") {
    const auto& dispatch = GetDispatch();
    const auto scalar_dispatch = ResolveDispatch({false, false, false});
    REQUIRE(dispatch.preparation != nullptr);
    REQUIRE(scalar_dispatch.preparation != nullptr);
    // Finite values of both signs, zeros and subnormals in each format.
    const std::array<uint16_t, 12> fp16 = {0x0000,
                                           0x8000,
                                           0x0001,
                                           0x03ff,
                                           0x0400,
                                           0x3c00,
                                           0xbc00,
                                           0x3800,
                                           0xb800,
                                           0x4000,
                                           0x3555,
                                           0x7bff};
    const std::array<uint16_t, 12> bf16 = {0x0000,
                                           0x8000,
                                           0x0001,
                                           0x007f,
                                           0x0080,
                                           0x3f80,
                                           0xbf80,
                                           0x3f00,
                                           0xbf00,
                                           0x4000,
                                           0x3eab,
                                           0x4200};
    for (const auto format : {Format::FP16, Format::BF16}) {
        const auto& values = format == Format::FP16 ? fp16 : bf16;
        std::vector<uint16_t> input(67);
        std::vector<float> projection(67);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = values[i % values.size()];
            projection[i] = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.25F;
        }
        for (uint64_t dim = 0; dim <= 65; ++dim) {
            INFO("format=" << static_cast<int>(format) << " dim=" << dim);
            for (const float* b : {static_cast<const float*>(nullptr),
                                   static_cast<const float*>(projection.data() + 1)}) {
                const auto expected = scalar_dispatch.preparation(input.data() + 1, b, dim, format);
                const auto actual = dispatch.preparation(input.data() + 1, b, dim, format);
                double reference = 0.0;
                for (uint64_t i = 0; i < dim; ++i) {
                    const double value = Decode(input[i + 1], format);
                    reference += value * (b == nullptr ? value : b[i]);
                }
                const double tolerance = 1e-5 * std::max(1.0, std::abs(reference));
                REQUIRE(std::abs(expected - reference) <= tolerance);
                REQUIRE(std::abs(actual - reference) <= tolerance);
            }
        }
        // Non-finite values in both the vector body and the scalar tail.
        const uint16_t inf = format == Format::FP16 ? 0x7c00 : 0x7f80;
        const uint16_t nan = format == Format::FP16 ? 0x7e01 : 0x7fc1;
        for (const uint64_t pos : {uint64_t{0}, uint64_t{4}}) {
            std::array<uint16_t, 5> special{};
            std::array<float, 5> b = {1, 1, 1, 1, 1};
            special[pos] = inf;
            REQUIRE(std::isinf(scalar_dispatch.preparation(special.data(), nullptr, 5, format)));
            REQUIRE(std::isinf(scalar_dispatch.preparation(special.data(), b.data(), 5, format)));
            REQUIRE(std::isinf(dispatch.preparation(special.data(), nullptr, 5, format)));
            REQUIRE(std::isinf(dispatch.preparation(special.data(), b.data(), 5, format)));
            special[pos] = nan;
            REQUIRE(std::isnan(scalar_dispatch.preparation(special.data(), nullptr, 5, format)));
            REQUIRE(std::isnan(scalar_dispatch.preparation(special.data(), b.data(), 5, format)));
            REQUIRE(std::isnan(dispatch.preparation(special.data(), nullptr, 5, format)));
            REQUIRE(std::isnan(dispatch.preparation(special.data(), b.data(), 5, format)));
        }
    }
}
}  // namespace vsag::pipnn_half
