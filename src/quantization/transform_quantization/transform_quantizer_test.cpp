
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

#include "transform_quantizer.h"

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
TestComputeMetricTQ(std::string tq_chain, uint64_t dim, int count, float error = 2.0) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto param = std::make_shared<TransformQuantizerParameter>();
    constexpr static const char* param_template = R"(
        {{
            "tq_chain": "{}",
            "pca_dim": {}
        }}
    )";
    auto param_str = fmt::format(param_template, tq_chain, dim - 1);
    auto param_json = vsag::JsonType::Parse(param_str);
    param->FromJson(param_json);

    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    TransformQuantizer<T, metric> quantizer(param, common_param);

    REQUIRE(quantizer.NameImpl() == QUANTIZATION_TYPE_VALUE_TQ);
    TestComputeCodes<TransformQuantizer<T, metric>, metric>(quantizer, dim, count, error);
    TestComputer<TransformQuantizer<T, metric>, metric>(quantizer, dim, count, error);
}

template <typename T, MetricType metric>
void
TestSerializeDeserializeTQ(std::string tq_chain, uint64_t dim, int count) {
    float numeric_error = 2.0;
    float related_error = 0.1F;
    float unbounded_numeric_error_rate = 0.2F;
    float unbounded_related_error_rate = 0.2F;

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto param = std::make_shared<TransformQuantizerParameter>();
    constexpr static const char* param_template = R"(
                {{
                    "tq_chain": "{}",
                    "pca_dim": {}
                }}
            )";
    auto param_str = fmt::format(param_template, tq_chain, dim - 2);
    auto param_json = vsag::JsonType::Parse(param_str);
    param->FromJson(param_json);

    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    TransformQuantizer<T, metric> quantizer1(param, common_param);
    TransformQuantizer<T, metric> quantizer2(param, common_param);

    TestSerializeAndDeserialize<TransformQuantizer<T, metric>, metric>(quantizer1,
                                                                       quantizer2,
                                                                       dim,
                                                                       count,
                                                                       numeric_error,
                                                                       related_error,
                                                                       unbounded_numeric_error_rate,
                                                                       unbounded_related_error_rate,
                                                                       false);
}

TEST_CASE("TQ Compute", "[ut][TransformQuantizer]") {
    constexpr MetricType metrics[1] = {MetricType::METRIC_TYPE_L2SQR};
    std::string tq_chain = GENERATE("rom, pca, fp32", "rom, fp32", "fht, fp32");

    for (auto dim : dims) {
        if (dim < 100) {
            continue;
        }
        for (auto count : counts) {
            if (tq_chain.find("fp32") != std::string::npos) {
                TestComputeMetricTQ<FP32Quantizer<metrics[0]>, metrics[0]>(tq_chain, dim, count);
            }
            // note that when use pca or rom, the "absolute distance" and error is meaningless
            // so, in other pr, use Inverse Pair to test the distance
        }
    }
}

TEST_CASE("TQ Serialize and Deserialize", "[ut][TransformQuantizer]") {
    constexpr MetricType metrics[1] = {MetricType::METRIC_TYPE_L2SQR};
    std::string tq_chain = GENERATE("rom, pca, fp32", "rom, fp32", "fht, fp32");

    for (auto dim : dims) {
        if (dim < 100) {
            continue;
        }
        for (auto count : counts) {
            if (tq_chain.find("fp32") != std::string::npos) {
                TestSerializeDeserializeTQ<FP32Quantizer<metrics[0]>, metrics[0]>(
                    tq_chain, dim, count);
            }
        }
    }
}
