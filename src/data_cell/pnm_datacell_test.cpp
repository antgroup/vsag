
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


#include "pnm_datacell.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "quantization/fp32_quantizer.h"

#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <utility>
#include "vsag/engine.h"


TEST_CASE("PnmDatacell Basic Test", "[ut][FlattenDataCell] ") {
    pnmesdk_conf conf;
    conf.timeout = 10000;
    pnmesdk_init(&conf);
    vsag::FP32QuantizerParamPtr param = std::make_shared<vsag::FP32QuantizerParameter>();
    param->hold_molds = false;
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 2;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = vsag::Engine::CreateDefaultAllocator();
    vsag::PnmDatacell<vsag::FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>> pnm_datacell(param, common_param);
    // float vectors[] = {
    //     0.0f, 0.0f,
    //     1.0f, 0.0f,
    //     0.0f, 1.0f
    // };
    //
    // vsag::InnerIdType ids[3];
    // pnm_datacell.BatchInsertVector(vectors, 3, ids);
    //
    // float query_vector[] = {0.0f, 0.0f};
    // auto computer = pnm_datacell.FactoryComputer(query_vector);
    //
    // float result_dists[3];
    // pnm_datacell.Query(result_dists, computer, ids, 3);
    //
    // REQUIRE(std::abs(result_dists[0] - 0.0f) < 1e-6);
    // REQUIRE(std::abs(result_dists[1] - 1.0f) < 1e-6);
    // REQUIRE(std::abs(result_dists[2] - 1.0f) < 1e-6);

    pnmesdk_uninit(&conf);
}