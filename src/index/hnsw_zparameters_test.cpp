
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

#include "hnsw_zparameters.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("create hnsw with correct parameter", "[ut][hnsw]") {
    vsag::IndexCommonParam commom_param;
    commom_param.dim_ = 128;
    commom_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    commom_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    auto build_parameter_json = R"(
        {
            "max_degree": 16,
            "ef_construction": 100
        }
        )";

    nlohmann::json parsed_params = nlohmann::json::parse(build_parameter_json);
    vsag::HnswParameters::FromJson(parsed_params, commom_param);
}
