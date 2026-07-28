
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

#include <cmath>

#include "datacell/extra_info_datacell_parameter.h"
#include "datacell/flatten_datacell_parameter.h"
#include "datacell/graph_datacell_parameter.h"
#include "datacell/graph_interface_parameter.h"
#include "datacell/sparse_graph_datacell_parameter.h"
#include "datacell/sparse_vector_datacell_parameter.h"
#include "impl/odescent/odescent_graph_parameter.h"
#include "inner_string_params.h"
#include "utils/param_compat_macros.h"
#include "vsag/constants.h"

namespace vsag {

HGraphParameter::HGraphParameter(const JsonType& json) : HGraphParameter() {
    this->FromJson(json);
}

HGraphParameter::HGraphParameter() : name(INDEX_TYPE_HGRAPH) {
}

void
HGraphParameter::FromJson(const JsonType& json) {
    InnerIndexParameter::FromJson(json);

    if (json.Contains(HGRAPH_USE_ELP_OPTIMIZER_KEY)) {
        this->use_elp_optimizer = json[HGRAPH_USE_ELP_OPTIMIZER_KEY].GetBool();
    }

    if (json.Contains(HGRAPH_IGNORE_REORDER_KEY)) {
        this->ignore_reorder = json[HGRAPH_IGNORE_REORDER_KEY].GetBool();
    }

    if (json.Contains(HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY)) {
        this->build_by_base = json[HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY].GetBool();
    }

    CHECK_ARGUMENT(json.Contains(BASE_CODES_KEY),
                   fmt::format("hgraph parameters must contains {}", BASE_CODES_KEY));
    const auto& base_codes_json = json[BASE_CODES_KEY];
    this->base_codes_param = CreateFlattenParam(base_codes_json);

    if (use_reorder && this->reorder_source != HGRAPH_REORDER_SOURCE_BASE) {
        CHECK_ARGUMENT(json.Contains(PRECISE_CODES_KEY),
                       fmt::format("hgraph parameters must contains {}", PRECISE_CODES_KEY));
        const auto& precise_codes_json = json[PRECISE_CODES_KEY];
        this->precise_codes_param = CreateFlattenParam(precise_codes_json);
    }

    CHECK_ARGUMENT(json.Contains(GRAPH_KEY),
                   fmt::format("hgraph parameters must contains {}", GRAPH_KEY));
    const auto& graph_json = json[GRAPH_KEY];

    GraphStorageTypes graph_storage_type = GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT;
    if (graph_json.Contains(GRAPH_STORAGE_TYPE_KEY)) {
        const auto graph_storage_type_str = graph_json[GRAPH_STORAGE_TYPE_KEY].GetString();
        if (graph_storage_type_str == GRAPH_STORAGE_TYPE_VALUE_COMPRESSED) {
            graph_storage_type = GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_COMPRESSED;
        }

        if (graph_storage_type_str != GRAPH_STORAGE_TYPE_VALUE_COMPRESSED &&
            graph_storage_type_str != GRAPH_STORAGE_TYPE_VALUE_FLAT) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                fmt::format("invalid graph_storage_type: {}", graph_storage_type_str));
        }
    }
    this->bottom_graph_param =
        GraphInterfaceParameter::GetGraphParameterByJson(graph_storage_type, graph_json);

    hierarchical_graph_param = std::make_shared<SparseGraphDatacellParameter>();
    hierarchical_graph_param->max_degree_ = this->bottom_graph_param->max_degree_ / 2;
    if (graph_storage_type == GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT) {
        auto graph_param =
            std::dynamic_pointer_cast<GraphDataCellParameter>(this->bottom_graph_param);
        if (graph_param != nullptr) {
            hierarchical_graph_param->remove_flag_bit_ = graph_param->remove_flag_bit_;
            hierarchical_graph_param->support_delete_ = graph_param->support_remove_;
            hierarchical_graph_param->use_reverse_edges_ = graph_param->use_reverse_edges_;
        } else {
            hierarchical_graph_param->support_delete_ = false;
        }
    } else {
        hierarchical_graph_param->support_delete_ = false;
    }

    if (json.Contains(EF_CONSTRUCTION_KEY)) {
        this->ef_construction = json[EF_CONSTRUCTION_KEY].GetUint64();
        CHECK_ARGUMENT(this->ef_construction > 0, "ef_construction must be positive");
    }

    if (json.Contains(ALPHA_KEY)) {
        this->alpha = json[ALPHA_KEY].GetFloat();
    }

    if (json.Contains(BUILD_THREAD_COUNT_KEY)) {
        this->build_thread_count = json[BUILD_THREAD_COUNT_KEY].GetUint64();
    }

    if (graph_json.Contains(GRAPH_TYPE_KEY)) {
        graph_type = graph_json[GRAPH_TYPE_KEY].GetString();
        if (graph_type == GRAPH_TYPE_VALUE_ODESCENT) {
            odescent_param = std::make_shared<ODescentParameter>();
            odescent_param->FromJson(graph_json);
        }
    }

    if (json.Contains(SUPPORT_DUPLICATE)) {
        this->support_duplicate = json[SUPPORT_DUPLICATE].GetBool();
        if (this->bottom_graph_param != nullptr) {
            this->bottom_graph_param->support_duplicate_ = this->support_duplicate;
        }
    }
    if (json.Contains(DEDUPLICATE_STORAGE)) {
        this->deduplicate_storage = json[DEDUPLICATE_STORAGE].GetBool();
    }
    if (this->deduplicate_storage && not this->support_duplicate) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "deduplicate_storage requires support_duplicate to be true");
    }
    if (json.Contains(DUPLICATE_DISTANCE_THRESHOLD)) {
        this->duplicate_distance_threshold = json[DUPLICATE_DISTANCE_THRESHOLD].GetFloat();
    }
    if (json.Contains(SUPPORT_FORCE_REMOVE)) {
        this->support_force_remove = json[SUPPORT_FORCE_REMOVE].GetBool();
    }
    if (this->deduplicate_storage && this->support_force_remove) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            "deduplicate_storage does not support force remove because duplicate groups share "
            "physical vector slots");
    }
    if (json.Contains(HGRAPH_PERSIST_SOURCE_ID_KEY)) {
        this->persist_source_id = json[HGRAPH_PERSIST_SOURCE_ID_KEY].GetBool();
    }
}

JsonType
HGraphParameter::ToJson() const {
    JsonType json = InnerIndexParameter::ToJson();
    json[TYPE_KEY].SetString(INDEX_TYPE_HGRAPH);

    json[HGRAPH_USE_ELP_OPTIMIZER_KEY].SetBool(this->use_elp_optimizer);
    json[HGRAPH_IGNORE_REORDER_KEY].SetBool(this->ignore_reorder);
    json[REORDER_SOURCE_KEY].SetString(this->reorder_source);
    json[BASE_CODES_KEY].SetJson(this->base_codes_param->ToJson());
    json[GRAPH_KEY].SetJson(this->bottom_graph_param->ToJson());
    json[EF_CONSTRUCTION_KEY].SetUint64(this->ef_construction);
    json[ALPHA_KEY].SetFloat(this->alpha);
    json[SUPPORT_DUPLICATE].SetBool(this->support_duplicate);
    json[DEDUPLICATE_STORAGE].SetBool(this->deduplicate_storage);
    json[DUPLICATE_DISTANCE_THRESHOLD].SetFloat(this->duplicate_distance_threshold);
    json[SUPPORT_FORCE_REMOVE].SetBool(this->support_force_remove);
    json[HGRAPH_PERSIST_SOURCE_ID_KEY].SetBool(this->persist_source_id);
    json[TRAIN_SAMPLE_COUNT_KEY].SetInt(this->train_sample_count);
    return json;
}

bool
HGraphParameter::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(HGraphParameter, p, other);
    auto have_reorder = this->use_reorder && not this->ignore_reorder;
    auto have_reorder_other = p->use_reorder && not p->ignore_reorder;
    if (have_reorder != have_reorder_other) {
        logger::error(
            "HGraphParameter::CheckCompatibility: use_reorder and ignore_reorder must be the same");
        return false;
    }
    CHECK_SUB_PARAM(*this, *p, base_codes_param);
    if (have_reorder) {
        CHECK_FIELD_EQ(*this, *p, reorder_source);
        if (this->reorder_source != HGRAPH_REORDER_SOURCE_BASE) {
            if (not this->precise_codes_param ||
                not this->precise_codes_param->CheckCompatibility(p->precise_codes_param)) {
                logger::error(
                    "HGraphParameter::CheckCompatibility: precise_codes_param is not compatible");
                return false;
            }
        }
    }
    CHECK_SUB_PARAM(*this, *p, bottom_graph_param);
    CHECK_FIELD_EQ(*this, *p, use_attribute_filter);
    CHECK_FIELD_EQ(*this, *p, support_duplicate);
    CHECK_FIELD_EQ(*this, *p, deduplicate_storage);
    CHECK_FIELD_EQ(*this, *p, duplicate_distance_threshold);
    CHECK_FIELD_EQ(*this, *p, support_force_remove);
    return true;
}

HGraphSearchParameters
HGraphSearchParameters::FromJson(const std::string& json_string) {
    auto params = JsonType::Parse(json_string);

    HGraphSearchParameters obj;

    // set obj.ef_search
    CHECK_ARGUMENT(params.Contains(INDEX_TYPE_HGRAPH),
                   fmt::format("parameters must contains {}", INDEX_TYPE_HGRAPH));

    obj.IndexSearchParameter::FromJson(params[INDEX_TYPE_HGRAPH]);

    CHECK_ARGUMENT(
        params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_EF_RUNTIME),
        fmt::format(
            "parameters[{}] must contains {}", INDEX_TYPE_HGRAPH, HGRAPH_PARAMETER_EF_RUNTIME));
    const auto& ef_search_json = params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_EF_RUNTIME];
    CHECK_ARGUMENT(ef_search_json.IsNumberInteger(), "ef_search must be an integer");
    if (ef_search_json.IsNumberUnsigned()) {
        CHECK_ARGUMENT(ef_search_json.GetUint64() <=
                           static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                       "ef_search exceeds int64_t range");
    }
    obj.ef_search = ef_search_json.GetInt();
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_HOPS_LIMIT)) {
        obj.hops_limit = params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_HOPS_LIMIT].GetInt();
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_USE_EXTRA_INFO_FILTER)) {
        obj.use_extra_info_filter =
            params[INDEX_TYPE_HGRAPH][HGRAPH_USE_EXTRA_INFO_FILTER].GetBool();
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_RABITQ_ONE_BIT_SEARCH)) {
        obj.rabitq_one_bit_search =
            params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_RABITQ_ONE_BIT_SEARCH].GetBool();
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_BRUTE_FORCE_THRESHOLD)) {
        obj.brute_force_threshold =
            params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_BRUTE_FORCE_THRESHOLD].GetFloat();
        CHECK_ARGUMENT(  // NOLINT
            (0.0F <= obj.brute_force_threshold) and (obj.brute_force_threshold <= 1.0F),
            fmt::format("brute_force_threshold({}) must in range[0.0, 1.0]",
                        obj.brute_force_threshold));
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(RABITQ_QUANTIZATION_ERROR_RATE_KEY)) {
        obj.rabitq_error_rate =
            params[INDEX_TYPE_HGRAPH][RABITQ_QUANTIZATION_ERROR_RATE_KEY].GetFloat();
        CHECK_ARGUMENT(std::isfinite(obj.rabitq_error_rate),
                       fmt::format("rabitq_error_rate must be finite and positive, got {}",
                                   obj.rabitq_error_rate));
        CHECK_ARGUMENT(obj.rabitq_error_rate > 0.0F,
                       fmt::format("rabitq_error_rate must be finite and positive, got {}",
                                   obj.rabitq_error_rate));
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HNSW_PARAMETER_SKIP_RATIO)) {
        obj.skip_ratio = params[INDEX_TYPE_HGRAPH][HNSW_PARAMETER_SKIP_RATIO].GetFloat();
        CHECK_ARGUMENT((0.0F <= obj.skip_ratio) and (obj.skip_ratio <= 1.0F),  // NOLINT
                       fmt::format("skip_ratio({}) must be in range [0.0, 1.0]", obj.skip_ratio));
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HNSW_PARAMETER_SKIP_STRATEGY)) {
        CHECK_ARGUMENT(
            params[INDEX_TYPE_HGRAPH][HNSW_PARAMETER_SKIP_STRATEGY].IsString(),
            fmt::format("parameters[{}] must be string type", HNSW_PARAMETER_SKIP_STRATEGY));
        obj.skip_strategy_type = parse_filter_search_skip_strategy_type(
            params[INDEX_TYPE_HGRAPH][HNSW_PARAMETER_SKIP_STRATEGY].GetString());
    }

    return obj;
}
}  // namespace vsag
