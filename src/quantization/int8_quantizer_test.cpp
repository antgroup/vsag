
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

#include "quantization/int8_quantizer.h"

#include <cstdint>
#include <cstdlib>
#include <initializer_list>

#include "catch2/catch_test_macros.hpp"
#include "impl/allocator/safe_allocator.h"
#include "metric_type.h"
#include "quantization/computer.h"
#include "quantization/quantizer.h"
#include "quantizer_test.h"

namespace vsag {

constexpr auto dims = {64, 128};
constexpr auto counts = {10, 101};

template <MetricType metric>
void
TestQuantizerEncodeDecodeINT8(Quantizer<INT8Quantizer<metric>>& quant,
                              int64_t dim,
                              int count,
                              float error = 1e-5,
                              bool retrain = true) {
    auto vecs = fixtures::generate_int8_codes(count, dim);
    if (retrain) {
        quant.ReTrain(reinterpret_cast<DataTypePtr>(vecs.data()), count);
    }
    // Test EncodeOne & DecodeOne
    for (uint64_t i = 0; i < count; ++i) {
        std::vector<uint8_t> codes(quant.GetCodeSize());
        quant.EncodeOne(reinterpret_cast<DataTypePtr>(vecs.data() + i * dim), codes.data());
        std::vector<int8_t> out_vec(dim);
        quant.DecodeOne(codes.data(), reinterpret_cast<DataTypePtr>(out_vec.data()));
        float sum = 0.0F;
        for (int j = 0; j < dim; ++j) {
            sum += std::abs(static_cast<DataType>(vecs[i * dim + j]) -
                            static_cast<DataType>(out_vec[j]));
        }
        REQUIRE(sum < error * dim);
    }

    // Test EncodeBatch & DecodeBatch
    std::vector<uint8_t> codes(quant.GetCodeSize() * count);
    quant.EncodeBatch(reinterpret_cast<DataTypePtr>(vecs.data()), codes.data(), count);
    std::vector<int8_t> out_vec(dim * count);
    quant.DecodeBatch(codes.data(), reinterpret_cast<DataTypePtr>(out_vec.data()), count);
    for (int64_t i = 0; i < count; ++i) {
        float sum = 0.0F;
        for (int j = 0; j < dim; ++j) {
            sum += std::abs(static_cast<DataType>(vecs[i * dim + j]) -
                            static_cast<DataType>(out_vec[i * dim + j]));
        }
        REQUIRE(sum < error * dim);
    }
}

template <MetricType metric>
void
TestQuantizerEncodeDecodeMetricINT8(int64_t dim, int64_t count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    INT8Quantizer<metric> quantizer(dim, allocator.get());
    TestQuantizerEncodeDecodeINT8(quantizer, dim, count, error);
}

TEST_CASE("INT8 Quantizer Encode and Decode", "[ut][INT8Quantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 2e-5f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestQuantizerEncodeDecodeMetricINT8<metrics[0]>(dim, count, error);
            TestQuantizerEncodeDecodeMetricINT8<metrics[1]>(dim, count, error);
        }
    }
}

template <MetricType metric>
void
TestComputeCodesINT8(Quantizer<INT8Quantizer<metric>>& quantizer,
                     size_t dim,
                     uint32_t count,
                     float error = 1e-4f,
                     bool retrain = true) {
    auto vecs = fixtures::generate_int8_codes(count, dim);
    if (retrain) {
        quantizer.ReTrain(reinterpret_cast<DataTypePtr>(vecs.data()), count);
    }
    for (int i = 0; i < count; ++i) {
        auto idx1 = random() % count;
        auto idx2 = random() % count;
        std::vector<uint8_t> codes1(quantizer.GetCodeSize());
        std::vector<uint8_t> codes2(quantizer.GetCodeSize());
        quantizer.EncodeOne(reinterpret_cast<DataTypePtr>(vecs.data() + idx1 * dim), codes1.data());
        quantizer.EncodeOne(reinterpret_cast<DataTypePtr>(vecs.data() + idx2 * dim), codes2.data());
        float gt = 0.0;
        float value = quantizer.Compute(codes1.data(), codes2.data());
        if constexpr (metric == vsag::MetricType::METRIC_TYPE_IP ||
                      metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            gt = 1 - INT8InnerProduct(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
        } else if constexpr (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            gt = INT8L2Sqr(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
        }
        REQUIRE(std::abs(gt - value) < error);
    }
}

template <MetricType metric>
void
TestComputeMetricINT8(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    INT8Quantizer<metric> quantizer(dim, allocator.get());
    TestComputeCodesINT8<metric>(quantizer, dim, count, error);
    // TestComputeCodesSame<FP32Quantizer<metric>, metric>(quantizer, dim, count, 65536);
    // TestComputer<FP32Quantizer<metric>, metric>(quantizer, dim, count, error);
}

TEST_CASE("INT8 Compute", "[ut][INT8Quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 2e-5f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestComputeMetricINT8<metrics[0]>(dim, count, error);
            // TestComputeMetricINT8<metrics[1]>(dim, count, error);
            // TestComputeMetricINT8<metrics[2]>(dim, count, error);
        }
    }
}

}  // namespace vsag