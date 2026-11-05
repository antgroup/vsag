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

#include <catch2/catch_approx.hpp>
#include <cstring>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/quantizer_adapter.h"
#include "quantization/scalar_quantization/half_precision_quantizer.h"
#include "quantization/scalar_quantization/scalar_quantizer.h"
#include "unittest.h"

using namespace vsag;

namespace {
template <typename Format>
constexpr DataTypes InputType =
    std::is_same_v<Format, FP16Format> ? DataTypes::DATA_TYPE_FP16 : DataTypes::DATA_TYPE_BF16;

template <typename Format, MetricType metric>
void
CheckPreservedBits() {
    using Inner = HalfPrecisionQuantizer<Format, metric>;
    using Adapter = QuantizerAdapter<Inner, uint16_t>;
    IndexCommonParam common;
    common.dim_ = 8;
    common.metric_ = metric;
    common.data_type_ = InputType<Format>;
    common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    Adapter adapter(nullptr, common);
    // Include signed zero, subnormals, infinities, and NaN payloads: a float
    // round trip is not an acceptable replacement for bit-preserving storage.
    const std::vector<uint16_t> bits = {0,
                                        0x8000,
                                        1,
                                        0x8001,
                                        0x7c01,
                                        0x7f81,
                                        0x7fff,
                                        0xffff,
                                        0x3c00,
                                        0xbc00,
                                        0x0080,
                                        0x8080,
                                        0x7c00,
                                        0xfc00,
                                        2,
                                        3};
    std::vector<uint8_t> codes(bits.size() * sizeof(uint16_t));
    REQUIRE(adapter.GetCodeSize() == common.dim_ * sizeof(uint16_t));
    REQUIRE(adapter.EncodeBatchImpl(reinterpret_cast<const float*>(bits.data()), codes.data(), 2));
    REQUIRE(std::memcmp(codes.data(), bits.data(), codes.size()) == 0);
    REQUIRE(adapter.EncodeOneImpl(reinterpret_cast<const float*>(bits.data() + common.dim_),
                                  codes.data()));
    REQUIRE(std::memcmp(codes.data(), bits.data() + common.dim_, adapter.GetCodeSize()) == 0);
    Computer<Adapter> computer(&adapter, common.allocator_.get());
    adapter.ProcessQueryImpl(reinterpret_cast<const float*>(bits.data()), computer);
    REQUIRE(std::memcmp(computer.buf_, bits.data(), adapter.GetCodeSize()) == 0);
    // Reusing the computer must overwrite the old query without reallocating semantics.
    adapter.ProcessQueryImpl(reinterpret_cast<const float*>(bits.data() + common.dim_), computer);
    REQUIRE(std::memcmp(computer.buf_, bits.data() + common.dim_, adapter.GetCodeSize()) == 0);
}

template <typename InputFormat, typename Inner, MetricType metric>
void
CheckConvertedReference() {
    using Adapter = QuantizerAdapter<Inner, uint16_t>;
    IndexCommonParam common;
    common.dim_ = 4;
    common.metric_ = metric;
    common.data_type_ = InputType<InputFormat>;
    common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    QuantizerParamPtr param;
    if constexpr (std::is_same_v<Inner, FP32Quantizer<metric>>) {
        param = std::make_shared<FP32QuantizerParameter>();
    }
    Adapter adapter(param, common);
    Inner reference(param, common);
    const std::vector<float> original = {3, 4, 0, 0, -2, 1, 2, -3, 0, 0, 0, 0};
    std::vector<uint16_t> bits(original.size());
    std::vector<float> expanded(original.size());
    for (uint64_t i = 0; i < bits.size(); ++i) {
        bits[i] = InputFormat::FloatToHalf(original[i]);
        expanded[i] = InputFormat::HalfToFloat(bits[i]);
    }
    REQUIRE(adapter.TrainImpl(reinterpret_cast<const float*>(bits.data()), 3));
    REQUIRE(reference.TrainImpl(expanded.data(), 3));
    const auto size = adapter.GetCodeSize();
    std::vector<uint8_t> actual(size * 3), expected(size * 3), one(size);
    REQUIRE(adapter.EncodeBatchImpl(reinterpret_cast<const float*>(bits.data()), actual.data(), 3));
    REQUIRE(reference.EncodeBatchImpl(expanded.data(), expected.data(), 3));
    REQUIRE(actual == expected);
    Computer<Adapter> computer(&adapter, common.allocator_.get());
    Computer<Inner> reference_computer(&reference, common.allocator_.get());
    for (uint64_t i = 0; i < 3; ++i) {
        REQUIRE(
            adapter.EncodeOneImpl(reinterpret_cast<const float*>(bits.data() + i * 4), one.data()));
        REQUIRE(std::memcmp(one.data(), expected.data() + i * size, size) == 0);
        adapter.ProcessQueryImpl(reinterpret_cast<const float*>(bits.data() + i * 4), computer);
        reference.ProcessQueryImpl(expanded.data() + i * 4, reference_computer);
        REQUIRE(std::memcmp(computer.buf_, reference_computer.buf_, adapter.GetQueryCodeSize()) ==
                0);
        float distance = 0, expected_distance = 0;
        adapter.ComputeDistImpl(computer, actual.data(), &distance);
        reference.ComputeDistImpl(reference_computer, expected.data(), &expected_distance);
        REQUIRE(distance == Catch::Approx(expected_distance));
    }
}

template <typename Format, MetricType metric>
void
CheckCommonQuantizers() {
    CheckConvertedReference<Format, FP32Quantizer<metric>, metric>();
    CheckConvertedReference<Format, SQ8Quantizer<metric>, metric>();
}
}  // namespace

TEST_CASE("Matched half L2 and IP preserve input bits", "[ut][HalfInputQuantizer]") {
    CheckPreservedBits<FP16Format, MetricType::METRIC_TYPE_L2SQR>();
    CheckPreservedBits<FP16Format, MetricType::METRIC_TYPE_IP>();
    CheckPreservedBits<BF16Format, MetricType::METRIC_TYPE_L2SQR>();
    CheckPreservedBits<BF16Format, MetricType::METRIC_TYPE_IP>();
}

TEST_CASE("Matched half cosine retains float normalization", "[ut][HalfInputQuantizer]") {
    CheckConvertedReference<FP16Format,
                            FP16Quantizer<MetricType::METRIC_TYPE_COSINE>,
                            MetricType::METRIC_TYPE_COSINE>();
    CheckConvertedReference<BF16Format,
                            BF16Quantizer<MetricType::METRIC_TYPE_COSINE>,
                            MetricType::METRIC_TYPE_COSINE>();
}

TEST_CASE("Cross format half inputs retain conversion", "[ut][HalfInputQuantizer]") {
    CheckConvertedReference<FP16Format,
                            BF16Quantizer<MetricType::METRIC_TYPE_L2SQR>,
                            MetricType::METRIC_TYPE_L2SQR>();
    CheckConvertedReference<BF16Format,
                            FP16Quantizer<MetricType::METRIC_TYPE_IP>,
                            MetricType::METRIC_TYPE_IP>();
    CheckConvertedReference<FP16Format,
                            BF16Quantizer<MetricType::METRIC_TYPE_COSINE>,
                            MetricType::METRIC_TYPE_COSINE>();
    CheckConvertedReference<BF16Format,
                            FP16Quantizer<MetricType::METRIC_TYPE_COSINE>,
                            MetricType::METRIC_TYPE_COSINE>();
}

TEST_CASE("Half inputs support FP32 and SQ8 conversion", "[ut][HalfInputQuantizer]") {
    CheckCommonQuantizers<FP16Format, MetricType::METRIC_TYPE_L2SQR>();
    CheckCommonQuantizers<FP16Format, MetricType::METRIC_TYPE_IP>();
    CheckCommonQuantizers<FP16Format, MetricType::METRIC_TYPE_COSINE>();
    CheckCommonQuantizers<BF16Format, MetricType::METRIC_TYPE_L2SQR>();
    CheckCommonQuantizers<BF16Format, MetricType::METRIC_TYPE_IP>();
    CheckCommonQuantizers<BF16Format, MetricType::METRIC_TYPE_COSINE>();
}
