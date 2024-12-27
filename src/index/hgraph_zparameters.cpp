
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

#include "hgraph_zparameters.h"

#include "../utils.h"
#include "common.h"
#include "fmt/format-inl.h"
#include "inner_string_params.h"
#include "vsag/constants.h"

namespace vsag {

static const std::unordered_map<std::string, std::vector<std::string>> EXTERNAL_MAPPING = {
    {HGRAPH_USE_REORDER, {HGRAPH_USE_REORDER_KEY}},
    {HGRAPH_BASE_QUANTIZATION_TYPE, {HGRAPH_BASE_CODES_KEY, QUANTIZATION_TYPE_KEY}},
    {HGRAPH_PRECISE_QUANTIZATION_TYPE, {HGRAPH_PRECISE_CODES_KEY, QUANTIZATION_TYPE_KEY}},
    {HGRAPH_GRAPH_MAX_DEGREE, {HGRAPH_GRAPH_KEY, GRAPH_PARAMS_KEY, GRAPH_PARAM_MAX_DEGREE}},
    {HGRAPH_BUILD_EF_CONSTRUCTION, {BUILD_PARAMS_KEY, BUILD_EF_CONSTRUCTION}},
    {HGRAPH_INIT_CAPACITY, {HGRAPH_GRAPH_KEY, GRAPH_PARAMS_KEY, GRAPH_PARAM_INIT_MAX_CAPACITY}},
    {HGRAPH_BUILD_THREAD_COUNT, {BUILD_PARAMS_KEY, BUILD_THREAD_COUNT}}};

static const std::string HGRAPH_PARAMS_TEMPLATE =
    R"(
    {
        "{HGRAPH_USE_REORDER_KEY}": false,
        "{HGRAPH_GRAPH_KEY}": {
            "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
            "{IO_PARAMS_KEY}": {
                "{BLOCK_IO_BLOCK_SIZE_KEY}": {DEFAULT_BLOCK_SIZE}
            },
            "type": "nsw",
            "{GRAPH_PARAMS_KEY}": {
                "{GRAPH_PARAM_MAX_DEGREE}": 64,
                "{GRAPH_PARAM_INIT_MAX_CAPACITY}": 100
            }
        },
        "{HGRAPH_BASE_CODES_KEY}": {
            "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
            "{IO_PARAMS_KEY}": {
                "{BLOCK_IO_BLOCK_SIZE_KEY}": {DEFAULT_BLOCK_SIZE}
            },
            "codes_type": "flatten_codes",
            "codes_param": {},
            "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_PQ}",
            "{QUANTIZATION_PARAMS_KEY}": {
                "subspace": 64,
                "nbits": 8
            }
        },
        "{HGRAPH_PRECISE_CODES_KEY}": {
            "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
            "{IO_PARAMS_KEY}": {},
            "codes_type": "flatten_codes",
            "codes_param": {},
            "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
            "{QUANTIZATION_PARAMS_KEY}": {}
        },
        "{BUILD_PARAMS_KEY}": {
            "{BUILD_EF_CONSTRUCTION}": 400,
            "{BUILD_THREAD_COUNT}": 100
        }
    })";

HGraphParameters::HGraphParameters(JsonType& hgraph_param, const IndexCommonParam& common_param)
    : common_param_(common_param),
      default_hgraph_params_(format_map(HGRAPH_PARAMS_TEMPLATE, DEFAULT_MAP)) {
    this->str_ = default_hgraph_params_;
    this->check_common_param();
    this->init_by_options();
    this->refresh_json_by_string();
    this->ParseStringParam(hgraph_param);
    this->refresh_string_by_json();
}

void
HGraphParameters::check_common_param() const {
    if (this->common_param_.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw std::invalid_argument(fmt::format("HGraph not support {} datatype", DATATYPE_INT8));
    }
}

void
HGraphParameters::ParseStringParam(JsonType& hgraph_param) {
    for (const auto& [key, value] : hgraph_param.items()) {
        this->CheckAndSetKeyValue(key, value);
    }
    this->refresh_string_by_json();
}

void
HGraphParameters::CheckAndSetKeyValue(const std::string& key, JsonType& value) {
    const auto& iter = EXTERNAL_MAPPING.find(key);

    if (key == HGRAPH_BASE_QUANTIZATION_TYPE) {
        std::string value_str = value;
        if (value_str != QUANTIZATION_TYPE_VALUE_SQ8 && value_str != QUANTIZATION_TYPE_VALUE_FP32 &&
            value_str != QUANTIZATION_TYPE_VALUE_SQ4 &&
            value_str != QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM &&
            value_str != QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM) {
            throw std::invalid_argument(
                fmt::format("parameters[{}] must in [{}, {}, {}, {}, {}], now is {}",
                            HGRAPH_BASE_QUANTIZATION_TYPE,
                            QUANTIZATION_TYPE_VALUE_SQ8,
                            QUANTIZATION_TYPE_VALUE_FP32,
                            QUANTIZATION_TYPE_VALUE_SQ4,
                            QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM,
                            QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM,
                            value_str));
        }
    }

    if (iter != EXTERNAL_MAPPING.end()) {
        const auto& vec = iter->second;
        auto* json = &json_;
        for (const auto& str : vec) {
            json = &(json->operator[](str));
        }
        *json = value;
    } else {
        throw std::invalid_argument(fmt::format("HGraph have no config param: {}", key));
    }
}

void
HGraphParameters::init_by_options() {
    const std::string DEFAULT_BLOCK_SIZE = std::to_string(Options::Instance().block_size_limit());
    std::unordered_map<std::string, std::string> option_map;
    option_map.insert({"DEFAULT_BLOCK_SIZE", DEFAULT_BLOCK_SIZE});
    this->str_ = format_map(this->str_, option_map);
}

HGraphSearchParameters
HGraphSearchParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::parse(json_string);

    HGraphSearchParameters obj;

    // set obj.ef_search
    CHECK_ARGUMENT(params.contains(INDEX_HGRAPH),
                   fmt::format("parameters must contains {}", INDEX_HGRAPH));

    CHECK_ARGUMENT(
        params[INDEX_HGRAPH].contains(HNSW_PARAMETER_EF_RUNTIME),
        fmt::format("parameters[{}] must contains {}", INDEX_HGRAPH, HNSW_PARAMETER_EF_RUNTIME));
    obj.ef_search = params[INDEX_HGRAPH][HNSW_PARAMETER_EF_RUNTIME];
    CHECK_ARGUMENT((1 <= obj.ef_search) and (obj.ef_search <= 1000),
                   fmt::format("ef_search({}) must in range[1, 1000]", obj.ef_search));

    return obj;
}
}  // namespace vsag
