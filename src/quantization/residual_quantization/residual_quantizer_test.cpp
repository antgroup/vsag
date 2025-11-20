
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

#include "residual_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_test.h"

using namespace vsag;

const auto dims = fixtures::get_common_used_dims(10, 114);
const auto counts = {101, 1001};

template <typename T, MetricType metric>
void
TestComputeMetricRQ(std::string base_quantizer_type, uint64_t dim, int count, float error = 2.0) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto param = std::make_shared<ResidualQuantizerParameter>();
    constexpr static const char* param_template = R"(
                {{
                    "rq_base_quantization_type": "{}",
                    "rq_centroids_count": {}
                }}
            )";
    auto param_str = fmt::format(param_template, base_quantizer_type, 1);
    auto param_json = vsag::JsonType::Parse(param_str);
    param->FromJson(param_json);

    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    ResidualQuantizer<T, metric> quantizer(param, common_param);

    REQUIRE(quantizer.NameImpl() == QUANTIZATION_TYPE_VALUE_RQ);
    TestComputer<ResidualQuantizer<T, metric>, metric>(quantizer, dim, count, error);
}

template <typename T, MetricType metric>
void
TestSerializeDeserializeRQ(std::string base_quantizer_type, uint64_t dim, int count) {
    float numeric_error = 2.0;
    float related_error = 0.1F;
    float unbounded_numeric_error_rate = 0.2F;
    float unbounded_related_error_rate = 0.2F;

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto param = std::make_shared<ResidualQuantizerParameter>();
    constexpr static const char* param_template = R"(
                {{
                    "rq_base_quantization_type": "{}",
                    "rq_centroids_count": {}
                }}
            )";
    auto param_str = fmt::format(param_template, base_quantizer_type, 1);
    auto param_json = vsag::JsonType::Parse(param_str);
    param->FromJson(param_json);

    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    ResidualQuantizer<T, metric> quantizer1(param, common_param);
    ResidualQuantizer<T, metric> quantizer2(param, common_param);

    TestSerializeAndDeserialize<ResidualQuantizer<T, metric>, metric, false, false>(
        quantizer1,
        quantizer2,
        dim,
        count,
        numeric_error,
        related_error,
        unbounded_numeric_error_rate,
        unbounded_related_error_rate,
        false);
}

TEST_CASE("RQ Compute", "[ut][ResidualQuantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_COSINE};

    for (auto dim : dims) {
        for (auto count : counts) {
            std::cout << dim << " " << count << std::endl;
            TestComputeMetricRQ<FP32Quantizer<metrics[0]>, metrics[0]>(
                "fp32", dim, count, 0.1);
//            TestComputeMetricRQ<FP32Quantizer<metrics[1]>, metrics[1]>(
//                "fp32", dim, count, 0.1);
//            TestComputeMetricRQ<FP32Quantizer<metrics[2]>, metrics[2]>(
//                "fp32", dim, count, 0.1);

            TestComputeMetricRQ<FP16Quantizer<metrics[0]>, metrics[0]>(
                "fp16", dim, count, 1);
//            TestComputeMetricRQ<FP16Quantizer<metrics[1]>, metrics[1]>(
//                "fp16", dim, count, 0.1);
//            TestComputeMetricRQ<FP16Quantizer<metrics[2]>, metrics[2]>(
//                "fp16", dim, count, 0.1);

            TestComputeMetricRQ<SQ8Quantizer<metrics[0]>, metrics[0]>(
                "sq8", dim, count, 10);
//            TestComputeMetricRQ<SQ8Quantizer<metrics[1]>, metrics[1]>(
//                "sq8", dim, count, 10);
//            TestComputeMetricRQ<SQ8Quantizer<metrics[2]>, metrics[2]>(
//                "sq8", dim, count, 10);

//            TestComputeMetricRQ<SQ4UniformQuantizer<metrics[0]>, metrics[0]>(
//                "sq4_uniform", dim, count, 20);
//            TestComputeMetricRQ<SQ4UniformQuantizer<metrics[1]>, metrics[1]>(
//                "sq4_uniform", dim, count, 20);
//            TestComputeMetricRQ<SQ4UniformQuantizer<metrics[2]>, metrics[2]>(
//                "sq4_uniform", dim, count, 20);
        }
    }
}

TEST_CASE("RQ Serialize and Deserialize", "[ut][ResidualQuantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_COSINE};

    for (auto dim : dims) {
        for (auto count : counts) {
            TestSerializeDeserializeRQ<FP32Quantizer<MetricType::METRIC_TYPE_IP>, metrics[0]>(
                "fp32", dim, count);
            TestSerializeDeserializeRQ<FP32Quantizer<MetricType::METRIC_TYPE_IP>, metrics[1]>(
                "fp32", dim, count);
            TestSerializeDeserializeRQ<FP32Quantizer<MetricType::METRIC_TYPE_IP>, metrics[2]>(
                "fp32", dim, count);

            TestSerializeDeserializeRQ<FP16Quantizer<MetricType::METRIC_TYPE_IP>, metrics[0]>(
                "fp16", dim, count);
            TestSerializeDeserializeRQ<FP16Quantizer<MetricType::METRIC_TYPE_IP>, metrics[1]>(
                "fp16", dim, count);
            TestSerializeDeserializeRQ<FP16Quantizer<MetricType::METRIC_TYPE_IP>, metrics[2]>(
                "fp16", dim, count);
        }
    }
}
