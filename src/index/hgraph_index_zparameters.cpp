
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

#include "hgraph_index_zparameters.h"

#include <fmt/format-inl.h>

#include <utility>

#include "../utils.h"
#include "common.h"
#include "inner_string_params.h"
#include "vsag/constants.h"

// NOLINTBEGIN(readability-simplify-boolean-expr)

namespace vsag {

static const std::unordered_map<std::string, std::vector<std::string>> EXTERNAL_MAPPING = {
    {HGRAPH_USE_REORDER, {HGRAPH_USE_REORDER_KEY}},
    {HGRAPH_BASE_QUANTIZATION_TYPE,
     {HGRAPH_BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, QUANTIZATION_TYPE_KEY}},
    {HGRAPH_BASE_IO_TYPE, {HGRAPH_BASE_CODES_KEY, IO_PARAMS_KEY, IO_TYPE_KEY}},
    {HGRAPH_PRECISE_IO_TYPE, {HGRAPH_PRECISE_CODES_KEY, IO_PARAMS_KEY, IO_TYPE_KEY}},
    {HGRAPH_BASE_FILE_PATH, {HGRAPH_BASE_CODES_KEY, IO_PARAMS_KEY, IO_FILE_PATH}},
    {HGRAPH_PRECISE_FILE_PATH, {HGRAPH_PRECISE_CODES_KEY, IO_PARAMS_KEY, IO_FILE_PATH}},
    {HGRAPH_PRECISE_QUANTIZATION_TYPE,
     {HGRAPH_PRECISE_CODES_KEY, QUANTIZATION_PARAMS_KEY, QUANTIZATION_TYPE_KEY}},
    {HGRAPH_GRAPH_MAX_DEGREE, {HGRAPH_GRAPH_KEY, GRAPH_PARAM_MAX_DEGREE}},
    {HGRAPH_BUILD_EF_CONSTRUCTION, {BUILD_PARAMS_KEY, BUILD_EF_CONSTRUCTION}},
    {HGRAPH_INIT_CAPACITY, {HGRAPH_GRAPH_KEY, GRAPH_PARAM_INIT_MAX_CAPACITY}},
    {HGRAPH_BUILD_THREAD_COUNT, {BUILD_PARAMS_KEY, BUILD_THREAD_COUNT}}};

static const std::string HGRAPH_PARAMS_TEMPLATE =
    R"(
    {
        "type": "{INDEX_TYPE_HGRAPH}",
        "{HGRAPH_USE_REORDER_KEY}": false,
        "{HGRAPH_GRAPH_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{GRAPH_PARAM_MAX_DEGREE}": 64,
            "{GRAPH_PARAM_INIT_MAX_CAPACITY}": 100
        },
        "{HGRAPH_BASE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "codes_type": "flatten_codes",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_PQ}",
                "subspace": 64,
                "nbits": 8
            }
        },
        "{HGRAPH_PRECISE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "codes_type": "flatten_codes",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}"
            }
        },
        "{BUILD_PARAMS_KEY}": {
            "{BUILD_EF_CONSTRUCTION}": 400,
            "{BUILD_THREAD_COUNT}": 100
        }
    })";

static void
mapping_external_param_to_inner(const JsonType& external_json, JsonType& inner_json);

HGraphIndexParameter::HGraphIndexParameter(IndexCommonParam common_param)
    : common_param_(std::move(common_param)) {
}

void
HGraphIndexParameter::FromJson(const JsonType& json) {
    this->check_common_param();

    std::string str = format_map(HGRAPH_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto inner_json = JsonType::parse(str);
    mapping_external_param_to_inner(json, inner_json);

    this->hgraph_parameter_ = std::make_shared<HGraphParameter>();
    this->hgraph_parameter_->FromJson(inner_json);
}

void
HGraphIndexParameter::check_common_param() const {
    if (this->common_param_.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw std::invalid_argument(fmt::format("HGraph not support {} datatype", DATATYPE_INT8));
    }
}

JsonType
HGraphIndexParameter::ToJson() {
    JsonType json;
    if (this->hgraph_parameter_) {
        json = this->hgraph_parameter_->ToJson();
    }
    return json;
}

void
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
            throw std::invalid_argument(fmt::format("HGraph have no config param: {}", key));
        }
    }
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

// NOLINTEND(readability-simplify-boolean-expr)
