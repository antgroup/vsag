
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

#include "ivf_parameter.h"

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "parameter_test.h"

TEST_CASE("IVF Parameters Test", "[ut][IVFParameter]") {
    auto param_str = R"({
        "type": "ivf",
        "buckets_params": {
            "io_params": {
                "type": "block_memory_io"
            },
            "quantization_params": {
                "type": "fp32"
            },
            "buckets_count": 3
        }
    })";
    vsag::JsonType param_json = vsag::JsonType::parse(param_str);
    auto param = std::make_shared<vsag::IVFParameter>();
    param->FromJson(param_json);
    REQUIRE(param->bucket_param->buckets_count == 3);
}
