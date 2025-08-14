
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

#include "hgraph_parameter.h"

#include "data_cell/graph_datacell_parameter.h"
#include "data_cell/graph_interface_parameter.h"
#include "data_cell/sparse_vector_datacell_parameter.h"
#include "inner_string_params.h"
#include "vsag/constants.h"

namespace vsag {

HGraphParameter::HGraphParameter(const JsonType& json) : HGraphParameter() {
    this->FromJson(json);
}

HGraphParameter::HGraphParameter() : name(INDEX_TYPE_HGRAPH) {
}

void
HGraphParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(json.contains(HGRAPH_USE_REORDER_KEY),
                   fmt::format("hgraph parameters must contains {}", HGRAPH_USE_REORDER_KEY));
    this->use_reorder = json[HGRAPH_USE_REORDER_KEY];

    if (json.contains(HGRAPH_USE_ELP_OPTIMIZER_KEY)) {
        this->use_elp_optimizer = json[HGRAPH_USE_ELP_OPTIMIZER_KEY];
    }

    if (json.contains(HGRAPH_IGNORE_REORDER_KEY)) {
        this->ignore_reorder = json[HGRAPH_IGNORE_REORDER_KEY];
    }

    if (json.contains(HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY)) {
        this->build_by_base = json[HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY];
    }

    if (json.contains(USE_ATTRIBUTE_FILTER_KEY)) {
        this->use_attribute_filter = json[USE_ATTRIBUTE_FILTER_KEY];
    }

    CHECK_ARGUMENT(json.contains(HGRAPH_BASE_CODES_KEY),
                   fmt::format("hgraph parameters must contains {}", HGRAPH_BASE_CODES_KEY));
    const auto& base_codes_json = json[HGRAPH_BASE_CODES_KEY];
    if (data_type == DataTypes::DATA_TYPE_SPARSE) {
        this->base_codes_param = std::make_shared<SparseVectorDataCellParameter>();
    } else {
        this->base_codes_param = std::make_shared<FlattenDataCellParameter>();
    }
    this->base_codes_param->FromJson(base_codes_json);

    if (use_reorder) {
        CHECK_ARGUMENT(json.contains(HGRAPH_PRECISE_CODES_KEY),
                       fmt::format("hgraph parameters must contains {}", HGRAPH_PRECISE_CODES_KEY));
        const auto& precise_codes_json = json[HGRAPH_PRECISE_CODES_KEY];
        if (data_type == DataTypes::DATA_TYPE_SPARSE) {
            this->precise_codes_param = std::make_shared<SparseVectorDataCellParameter>();
        } else {
            this->precise_codes_param = std::make_shared<FlattenDataCellParameter>();
        }
        this->precise_codes_param->FromJson(precise_codes_json);
    }

    CHECK_ARGUMENT(json.contains(HGRAPH_GRAPH_KEY),
                   fmt::format("hgraph parameters must contains {}", HGRAPH_GRAPH_KEY));
    const auto& graph_json = json[HGRAPH_GRAPH_KEY];

    GraphStorageTypes graph_storage_type = GraphStorageTypes::GRAPH_STORAGE_TYPE_FLAT;
    if (graph_json.contains(GRAPH_STORAGE_TYPE_KEY)) {
        const auto& graph_storage_type_str = graph_json[GRAPH_STORAGE_TYPE_KEY];
        if (graph_storage_type_str == GRAPH_STORAGE_TYPE_COMPRESSED) {
            graph_storage_type = GraphStorageTypes::GRAPH_STORAGE_TYPE_COMPRESSED;
        }

        if (graph_storage_type_str != GRAPH_STORAGE_TYPE_COMPRESSED &&
            graph_storage_type_str != GRAPH_STORAGE_TYPE_FLAT) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                fmt::format("invalid graph_storage_type: {}", graph_storage_type_str.dump()));
        }
    }
    this->bottom_graph_param =
        GraphInterfaceParameter::GetGraphParameterByJson(graph_storage_type, graph_json);

    hierarchical_graph_param = std::make_shared<SparseGraphDatacellParameter>();
    hierarchical_graph_param->max_degree_ = this->bottom_graph_param->max_degree_ / 2;
    if (graph_storage_type == GraphStorageTypes::GRAPH_STORAGE_TYPE_FLAT) {
        auto graph_param =
            std::dynamic_pointer_cast<GraphDataCellParameter>(this->bottom_graph_param);
        if (graph_param != nullptr) {
            hierarchical_graph_param->remove_flag_bit_ = graph_param->remove_flag_bit_;
            hierarchical_graph_param->support_delete_ = graph_param->support_remove_;
        } else {
            hierarchical_graph_param->support_delete_ = false;
        }
    } else {
        hierarchical_graph_param->support_delete_ = false;
    }

    if (json.contains(BUILD_PARAMS_KEY)) {
        const auto& build_params = json[BUILD_PARAMS_KEY];
        if (build_params.contains(BUILD_EF_CONSTRUCTION)) {
            this->ef_construction = build_params[BUILD_EF_CONSTRUCTION];
        }
        if (build_params.contains(BUILD_THREAD_COUNT)) {
            this->build_thread_count = build_params[BUILD_THREAD_COUNT];
        }
    }

    if (graph_json.contains(GRAPH_TYPE_KEY)) {
        graph_type = graph_json[GRAPH_TYPE_KEY];
        if (graph_type == GRAPH_TYPE_ODESCENT) {
            odescent_param = std::make_shared<ODescentParameter>();
            odescent_param->FromJson(graph_json);
        }
    }

    CHECK_ARGUMENT(json.contains(HGRAPH_EXTRA_INFO_KEY),
                   fmt::format("hgraph parameters must contains {}", HGRAPH_EXTRA_INFO_KEY));
    const auto& extra_info_json = json[HGRAPH_EXTRA_INFO_KEY];
    this->extra_info_param = std::make_shared<ExtraInfoDataCellParameter>();
    this->extra_info_param->FromJson(extra_info_json);
    if (json.contains(SUPPORT_DUPLICATE)) {
        this->support_duplicate = json[SUPPORT_DUPLICATE];
    }
}

JsonType
HGraphParameter::ToJson() const {
    JsonType json;
    json["type"] = INDEX_TYPE_HGRAPH;

    json[HGRAPH_USE_REORDER_KEY] = this->use_reorder;
    json[HGRAPH_USE_ELP_OPTIMIZER_KEY] = this->use_elp_optimizer;
    json[HGRAPH_BASE_CODES_KEY] = this->base_codes_param->ToJson();
    if (use_reorder) {
        json[HGRAPH_PRECISE_CODES_KEY] = this->precise_codes_param->ToJson();
    }
    json[HGRAPH_GRAPH_KEY] = this->bottom_graph_param->ToJson();

    json[BUILD_PARAMS_KEY][BUILD_EF_CONSTRUCTION] = this->ef_construction;
    json[BUILD_PARAMS_KEY][BUILD_THREAD_COUNT] = this->build_thread_count;
    json[HGRAPH_EXTRA_INFO_KEY] = this->extra_info_param->ToJson();
    json[SUPPORT_DUPLICATE] = this->support_duplicate;
    json[HGRAPH_STORE_RAW_VECTOR] = this->store_raw_vector;
    json[USE_ATTRIBUTE_FILTER_KEY] = this->use_attribute_filter;
    return json;
}

bool
HGraphParameter::CheckCompatibility(const ParamPtr& other) const {
    auto hgraph_param = std::dynamic_pointer_cast<HGraphParameter>(other);
    if (hgraph_param == nullptr) {
        logger::error("HGraphParameter::CheckCompatibility: other is not HGraphParameter");
        return false;
    }
    auto have_reorder = this->use_reorder && not this->ignore_reorder;
    auto have_reorder_other = hgraph_param->use_reorder && not hgraph_param->ignore_reorder;
    if (have_reorder != have_reorder_other) {
        logger::error(
            "HGraphParameter::CheckCompatibility: use_reorder and ignore_reorder must be the same");
        return false;
    }
    if (not this->base_codes_param->CheckCompatibility(hgraph_param->base_codes_param)) {
        logger::error("HGraphParameter::CheckCompatibility: base_codes_param is not compatible");
        return false;
    }
    if (have_reorder) {
        if (not this->precise_codes_param ||
            not this->precise_codes_param->CheckCompatibility(hgraph_param->precise_codes_param)) {
            logger::error(
                "HGraphParameter::CheckCompatibility: precise_codes_param is not compatible");
            return false;
        }
    }
    if (not this->bottom_graph_param->CheckCompatibility(hgraph_param->bottom_graph_param)) {
        logger::error("HGraphParameter::CheckCompatibility: bottom_graph_param is not compatible");
        return false;
    }
    if (use_attribute_filter != hgraph_param->use_attribute_filter) {
        logger::error("HGraphParameter::CheckCompatibility: use_attribute_filter must be the same");
        return false;
    }
    if (support_duplicate != hgraph_param->support_duplicate) {
        logger::error("HGraphParameter::CheckCompatibility: support_duplicate must be the same");
        return false;
    }
    return true;
}

HGraphSearchParameters
HGraphSearchParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::parse(json_string);

    HGraphSearchParameters obj;

    // set obj.ef_search
    CHECK_ARGUMENT(params.contains(INDEX_TYPE_HGRAPH),
                   fmt::format("parameters must contains {}", INDEX_TYPE_HGRAPH));

    CHECK_ARGUMENT(
        params[INDEX_TYPE_HGRAPH].contains(HGRAPH_PARAMETER_EF_RUNTIME),
        fmt::format(
            "parameters[{}] must contains {}", INDEX_TYPE_HGRAPH, HGRAPH_PARAMETER_EF_RUNTIME));
    obj.ef_search = params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_EF_RUNTIME];
    if (params[INDEX_TYPE_HGRAPH].contains(HGRAPH_USE_EXTRA_INFO_FILTER)) {
        obj.use_extra_info_filter = params[INDEX_TYPE_HGRAPH][HGRAPH_USE_EXTRA_INFO_FILTER];
    }

    if (params[INDEX_TYPE_HGRAPH].contains(SEARCH_MAX_TIME_COST_MS)) {
        obj.timeout_ms = params[INDEX_TYPE_HGRAPH][SEARCH_MAX_TIME_COST_MS];
        obj.enable_time_record = true;
    }

    return obj;
}
}  // namespace vsag
