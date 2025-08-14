
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

template <MetricType metric>
void
TestComputeMetricTQ(std::string tq_chain, uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    TransformQuantizerParamPtr param = std::make_shared<TransformQuantizerParameter>();
    auto param_template = R"(
        {{
            "tq_chain": "{}",
            "pca_dim": {}
        }}
    )";
    auto param_str = fmt::format(param_template, tq_chain, dim - 1);
    auto param_json = vsag::JsonType::parse(param_str);
    param->FromJson(param_json);

    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    TransformQuantizer<metric> quantizer(param, common_param);

    REQUIRE(quantizer.NameImpl() == QUANTIZATION_TYPE_VALUE_TQ);

    TestComputeCodes<TransformQuantizer<metric>, metric>(quantizer, dim, count, error);
    TestComputer<TransformQuantizer<metric>, metric>(quantizer, dim, count, error);
}

TEST_CASE("TQ Compute", "[ut][TransformQuantizer]") {
    constexpr MetricType metrics[1] = {MetricType::METRIC_TYPE_L2SQR};

    auto tq_chain = GENERATE("rom, pca, fp32", "rom, fp32", "fht, fp32");
    float error = 0.1;
    for (auto dim : dims) {
        if (dim < 100) {
            continue;
        }
        for (auto count : counts) {
            TestComputeMetricTQ<metrics[0]>(tq_chain, dim, count, error);
        }
    }
}