
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

#include "brute_force_parameter.h"

#include <fmt/format-inl.h>

#include "../utils.h"
#include "inner_string_params.h"
#include "vsag/constants.h"
namespace vsag {

static const std::unordered_map<std::string, std::vector<std::string>> EXTERNAL_MAPPING = {
    {BRUTE_FORCE_QUANTIZATION_TYPE, {QUANTIZATION_PARAMS_KEY, QUANTIZATION_TYPE_KEY}},
    {BRUTE_FORCE_IO_TYPE, {IO_PARAMS_KEY, IO_TYPE_KEY}}};

static const std::string BRUTE_FORCE_PARAMS_TEMPLATE =
    R"(
    {
        "type": "{INDEX_BRUTE_FORCE}",
        "{IO_PARAMS_KEY}": {
            "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}"
        },
        "{QUANTIZATION_PARAMS_KEY}": {
            "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
            "subspace": 64,
            "nbits": 8
        }
    })";

static void
mapping_external_param_to_inner(const JsonType& external_json, JsonType& inner_json) {
    for (const auto& [key, value] : external_json.items()) {
        const auto& iter = EXTERNAL_MAPPING.find(key);

        if (iter != EXTERNAL_MAPPING.end()) {
            const auto& vec = iter->second;
            auto* json = &inner_json;
            for (const auto& str : vec) {
                json = &(json->operator[](str));
            }
            *json = value;
        } else {
            throw std::invalid_argument(fmt::format("BruteForce have no config param: {}", key));
        }
    }
}

BruteForceParameter::BruteForceParameter() : flatten_param_(nullptr) {
}

void
BruteForceParameter::FromJson(const JsonType& json) {
    std::string str = format_map(BRUTE_FORCE_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto inner_json = JsonType::parse(str);
    mapping_external_param_to_inner(json, inner_json);

    this->flatten_param_ = std::make_shared<FlattenDataCellParameter>();
    this->flatten_param_->FromJson(inner_json);
}

JsonType
BruteForceParameter::ToJson() {
    auto json = this->flatten_param_->ToJson();
    json["type"] = INDEX_BRUTE_FORCE;

    return json;
}
}  // namespace vsag
