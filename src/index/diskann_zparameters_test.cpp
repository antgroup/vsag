
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

#include "diskann_zparameters.h"

#include <catch2/catch_test_macros.hpp>

#include "index_common_param.h"

TEST_CASE("create diskann with correct parameter", "[ut][diskann]") {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    auto build_parameter_json = R"(
        {
            "max_degree": 16,
            "ef_construction": 200,
            "pq_dims": 32,
            "pq_sample_rate": 0.5
        }
        )";
    vsag::JsonType parsed_params = vsag::JsonType::Parse(build_parameter_json);
    vsag::DiskannParameters::FromJson(parsed_params, common_param);
}
