
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

#include "sq4_uniform_quantizer.h"

#include <limits>

#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_test.h"
#include "scalar_quantizer_test_utils.h"
#include "unittest.h"
using namespace vsag;

const auto dims = fixtures::get_common_used_dims();
const auto counts = {10, 101};

TEST_CASE("SQ4 Uniform validates truncation rate", "[ut][SQ4UniformQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    SECTION("accepts finite boundary values") {
        for (const float rate : {0.0F, 0.05F, 0.5F}) {
            CAPTURE(rate);
            REQUIRE_NOTHROW(
                SQ4UniformQuantizer<MetricType::METRIC_TYPE_L2SQR>(4, allocator.get(), rate));
        }
    }

    SECTION("rejects non-finite and out-of-range values") {
        const float invalid_rates[] = {-0.01F,
                                       0.5001F,
                                       std::numeric_limits<float>::quiet_NaN(),
                                       std::numeric_limits<float>::infinity(),
                                       -std::numeric_limits<float>::infinity()};
        for (const float rate : invalid_rates) {
            CAPTURE(rate);
            try {
                (void)SQ4UniformQuantizer<MetricType::METRIC_TYPE_L2SQR>(4, allocator.get(), rate);
                FAIL("invalid SQ4 uniform truncation rate was accepted");
            } catch (const VsagException& error) {
                REQUIRE(error.error_.type == ErrorType::INVALID_ARGUMENT);
            }
        }
    }
}

template <MetricType metric>
void
TestQuantizerEncodeDecodeMetricSQ4Uniform(uint64_t dim,
                                          int count,
                                          float error = 1e-5,
                                          float error_same = 1e-2) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SQ4UniformQuantizer<metric> quantizer(dim, allocator.get());
    TestQuantizerEncodeDecode(quantizer, dim, count, error);
    TestQuantizerEncodeDecodeSame(quantizer, dim, count, 15, error_same);
}

TEST_CASE("SQ4 Uniform Encode and Decode", "[ut][SQ4UniformQuantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 2 * 1.0f / 15.0f;
    for (auto dim : dims) {
        for (auto count : counts) {
            auto error_same = (float)(dim * 255 * 0.01);
            TestQuantizerEncodeDecodeMetricSQ4Uniform<metrics[0]>(dim, count, error, error_same);
            TestQuantizerEncodeDecodeMetricSQ4Uniform<metrics[1]>(dim, count, error, error_same);
        }
    }
}

TEST_CASE("SQ4 Uniform encodes zero range to zero", "[ut][SQ4UniformQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 4;
    SQ4UniformQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(dim, allocator.get());
    TestUniformZeroRangeEncodesToZero(quantizer, dim, dim / 2);
}

template <MetricType metric>
void
TestComputeMetricSQ4Uniform(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SQ4UniformQuantizer<metric> quantizer(dim, allocator.get());
    TestComputeCodesSame<SQ4UniformQuantizer<metric>, metric>(quantizer, dim, count, error);
}

TEST_CASE("SQ4 Uniform Compute", "[ut][SQ4UniformQuantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 4 * 1.0f / 15.0f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestComputeMetricSQ4Uniform<metrics[0]>(dim, count, error);
            TestComputeMetricSQ4Uniform<metrics[1]>(dim, count, error);
        }
    }
}

template <MetricType metric>
void
TestSerializeAndDeserializeMetricSQ4Uniform(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SQ4UniformQuantizer<metric> quantizer1(dim, allocator.get());
    SQ4UniformQuantizer<metric> quantizer2(dim, allocator.get());
    TestSerializeAndDeserialize<SQ4UniformQuantizer<metric>, metric, true>(
        quantizer1, quantizer2, dim, count, error);
}

TEST_CASE("SQ4 Uniform Serialize and Deserialize", "[ut][SQ4UniformQuantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    for (auto dim : dims) {
        float error = 4 * 1.0f / 15.0f;
        for (auto count : counts) {
            TestSerializeAndDeserializeMetricSQ4Uniform<metrics[0]>(dim, count, error);
            //            TestSerializeAndDeserializeMetricSQ4Uniform<metrics[1]>(dim, count, error);
            TestSerializeAndDeserializeMetricSQ4Uniform<metrics[2]>(dim, count, error);
        }
    }
}
