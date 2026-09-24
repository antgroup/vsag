
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

#include "sparse_vector_datacell_parameter.h"

#include "parameter_test.h"
#include "quantization/sparse_quantization/sparse_quantizer_parameter.h"
#include "unittest.h"

namespace vsag {

TEST_CASE("SparseVectorDataCellParameter ToJson Test", "[ut][SparseVectorDataCellParameter]") {
    std::string param_str = R"(
    {
        "io_params": {
            "type": "memory_io"
        },
        "quantization_params": {
            "type": "sparse"
        }
    })";
    auto param = std::make_shared<SparseVectorDataCellParameter>();
    auto json = JsonType::Parse(param_str);
    param->FromJson(json);
    ParameterTest::TestToJson(param);
}

TEST_CASE("SparseVectorDataCellParameter FP16 value type", "[ut][SparseVectorDataCellParameter]") {
    auto fp16 = std::make_shared<SparseVectorDataCellParameter>();
    fp16->FromJson(JsonType::Parse(R"({
        "io_params": {"type": "memory_io"},
        "quantization_params": {"type": "sparse", "value_type": "fp16"}
    })"));
    REQUIRE(fp16->quantizer_parameter->ToJson()[SPARSE_QUANTIZER_VALUE_TYPE_KEY].GetString() ==
            QUANTIZATION_TYPE_VALUE_FP16);

    auto restored = std::make_shared<SparseVectorDataCellParameter>();
    restored->FromJson(fp16->ToJson());
    REQUIRE(fp16->CheckCompatibility(restored));

    auto fp32 = std::make_shared<SparseVectorDataCellParameter>();
    fp32->FromJson(JsonType::Parse(R"({
        "io_params": {"type": "memory_io"},
        "quantization_params": {"type": "sparse"}
    })"));
    REQUIRE_FALSE(fp16->CheckCompatibility(fp32));

    auto invalid = fp16->ToJson();
    invalid[QUANTIZATION_PARAMS_KEY][SPARSE_QUANTIZER_VALUE_TYPE_KEY].SetString("bf16");
    REQUIRE_THROWS(restored->FromJson(invalid));
}

}  // namespace vsag
