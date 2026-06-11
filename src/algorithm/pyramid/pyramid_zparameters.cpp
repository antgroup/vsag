
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

#include "pyramid_zparameters.h"

#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>
#include <unordered_set>
#include <utility>

#include "common.h"
#include "impl/logger/logger.h"
#include "index/diskann_zparameters.h"
#include "io/memory_io_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "utils/param_compat_macros.h"
#include "vsag/constants.h"

// NOLINTBEGIN(readability-simplify-boolean-expr)

namespace vsag {
namespace {

PyramidSearchParameters::HierarchyOp
parse_hierarchy_op(const std::string& op) {
    if (op == "union") {
        return PyramidSearchParameters::HierarchyOp::UNION;
    }
    if (op == "intersection") {
        return PyramidSearchParameters::HierarchyOp::INTERSECTION;
    }
    CHECK_ARGUMENT(false, fmt::format("unsupported pyramid hierarchy_op {}", op));
    return PyramidSearchParameters::HierarchyOp::SINGLE;
}

void
append_hierarchy_selector(PyramidSearchParameters& params,
                          std::unordered_set<std::string>& seen_names,
                          const std::string& hierarchy_name) {
    CHECK_ARGUMENT(not hierarchy_name.empty(), "pyramid hierarchy name must not be empty");
    CHECK_ARGUMENT(seen_names.insert(hierarchy_name).second,
                   fmt::format("duplicate hierarchy name {}", hierarchy_name));
    params.hierarchies.emplace_back(hierarchy_name);
}

}  // namespace

void
PyramidHierarchyParameters::FromJson(const JsonType& json) {
    if (json.IsString()) {
        name = json.GetString();
        CHECK_ARGUMENT(not name.empty(), "hierarchy name must not be empty");
        return;
    }

    CHECK_ARGUMENT(json.IsObject(), "hierarchy definition must be a string or an object");
    CHECK_ARGUMENT(json.Contains("name"), "hierarchy must contain name");
    CHECK_ARGUMENT(json["name"].IsString(), "hierarchy name must be a string");
    name = json["name"].GetString();
    CHECK_ARGUMENT(not name.empty(), "hierarchy name must not be empty");

    if (json.Contains(NO_BUILD_LEVELS)) {
        const auto& no_build_levels_json = json[NO_BUILD_LEVELS];
        CHECK_ARGUMENT(no_build_levels_json.IsArray(),
                       fmt::format("hierarchy {} no_build_levels must be an array", name));
        no_build_levels = no_build_levels_json.GetVector();
        std::sort(no_build_levels.begin(), no_build_levels.end());
    }
    if (json.Contains(GRAPH_PARAM_MAX_DEGREE_KEY)) {
        max_degree = json[GRAPH_PARAM_MAX_DEGREE_KEY].GetInt();
        CHECK_ARGUMENT(max_degree > 0,
                       fmt::format("hierarchy {} max_degree must be positive", name));
    }
    if (json.Contains(EF_CONSTRUCTION_KEY)) {
        const auto ef_construction_value = json[EF_CONSTRUCTION_KEY].GetInt();
        CHECK_ARGUMENT(ef_construction_value > 0,
                       fmt::format("hierarchy {} ef_construction must be positive", name));
        ef_construction = static_cast<uint64_t>(ef_construction_value);
    }
    if (json.Contains(ALPHA_KEY)) {
        alpha = json[ALPHA_KEY].GetFloat();
        CHECK_ARGUMENT(alpha > 0.0F, fmt::format("hierarchy {} alpha must be positive", name));
    }
    if (json.Contains(INDEX_MIN_SIZE)) {
        const auto index_min_size_value = json[INDEX_MIN_SIZE].GetInt();
        CHECK_ARGUMENT(index_min_size_value >= 0,
                       fmt::format("hierarchy {} index_min_size must be non-negative", name));
        CHECK_ARGUMENT(index_min_size_value <= std::numeric_limits<uint32_t>::max(),
                       fmt::format("hierarchy {} index_min_size exceeds uint32_t", name));
        index_min_size = static_cast<uint32_t>(index_min_size_value);
    }
    for (auto level : no_build_levels) {
        CHECK_ARGUMENT(
            level >= 0,
            fmt::format("hierarchy {} no_build_levels values must be non-negative", name));
    }
}

JsonType
PyramidHierarchyParameters::ToJson() const {
    JsonType json;
    json["name"].SetString(name);
    json[NO_BUILD_LEVELS].SetVector(no_build_levels);
    json[GRAPH_PARAM_MAX_DEGREE_KEY].SetInt(max_degree);
    json[EF_CONSTRUCTION_KEY].SetInt(ef_construction);
    json[ALPHA_KEY].SetFloat(alpha);
    json[INDEX_MIN_SIZE].SetInt(index_min_size);
    return json;
}

bool
PyramidHierarchyParameters::CheckCompatibility(const PyramidHierarchyParameters& other) const {
    if (name != other.name) {
        logger::error("PyramidHierarchyParameters::CheckCompatibility: names are not compatible");
        return false;
    }
    if (no_build_levels.size() != other.no_build_levels.size() ||
        not std::is_permutation(
            no_build_levels.begin(), no_build_levels.end(), other.no_build_levels.begin())) {
        logger::error(
            "PyramidHierarchyParameters::CheckCompatibility: no_build_levels are not compatible");
        return false;
    }
    if (max_degree != other.max_degree || ef_construction != other.ef_construction ||
        alpha != other.alpha || index_min_size != other.index_min_size) {
        logger::error(
            "PyramidHierarchyParameters::CheckCompatibility: build params are not compatible");
        return false;
    }
    return true;
}

void
PyramidParameters::FromJson(const JsonType& json) {
    InnerIndexParameter::FromJson(json);
    // init graph param
    const auto& graph_json = json[GRAPH_KEY];

    graph_param = GraphInterfaceParameter::GetGraphParameterByJson(
        GraphStorageTypes::GRAPH_STORAGE_TYPE_SPARSE, graph_json);
    this->alpha = graph_json[ALPHA_KEY].GetFloat();
    this->max_degree = graph_json[GRAPH_PARAM_MAX_DEGREE_KEY].GetInt();

    this->graph_type = graph_json[GRAPH_TYPE_KEY].GetString();
    if (this->graph_type == GRAPH_TYPE_ODESCENT) {
        this->odescent_param = std::make_shared<ODescentParameter>();
        this->odescent_param->FromJson(graph_json);
    } else {
        if (json.Contains(EF_CONSTRUCTION_KEY)) {
            this->ef_construction = json[EF_CONSTRUCTION_KEY].GetInt();
        }
    }

    this->base_codes_param = CreateFlattenParam(json[BASE_CODES_KEY]);

    if (json.Contains(NO_BUILD_LEVELS)) {
        const auto& no_build_levels_json = json[NO_BUILD_LEVELS];
        CHECK_ARGUMENT(no_build_levels_json.IsArray(),
                       fmt::format("build_without_levels must be a list of integers"));
        this->no_build_levels = no_build_levels_json.GetVector();
        std::sort(this->no_build_levels.begin(), this->no_build_levels.end());
    }

    this->use_reorder = json[USE_REORDER_KEY].GetBool();
    if (this->use_reorder) {
        this->precise_codes_param = CreateFlattenParam(json[PRECISE_CODES_KEY]);
    }

    if (json.Contains(INDEX_MIN_SIZE)) {
        this->index_min_size = json[INDEX_MIN_SIZE].GetInt();
    }

    if (json.Contains(SUPPORT_DUPLICATE)) {
        this->support_duplicate = json[SUPPORT_DUPLICATE].GetBool();
    }

    this->has_hierarchies = false;
    this->hierarchies.clear();
    if (json.Contains(PYRAMID_HIERARCHIES)) {
        const auto& hierarchies_json = json[PYRAMID_HIERARCHIES];
        CHECK_ARGUMENT(hierarchies_json.IsArray(), "hierarchies must be an array");
        CHECK_ARGUMENT(not hierarchies_json.GetInnerJson()->empty(),
                       "hierarchies must not be empty");
        this->has_hierarchies = true;

        std::unordered_set<std::string> seen_names;
        for (const auto& hierarchy_raw_json : *hierarchies_json.GetInnerJson()) {
            JsonType hierarchy_json;
            *hierarchy_json.GetInnerJson() = hierarchy_raw_json;

            PyramidHierarchyParameters hierarchy;
            hierarchy.no_build_levels = this->no_build_levels;
            hierarchy.max_degree = this->max_degree;
            hierarchy.ef_construction = this->ef_construction;
            hierarchy.alpha = this->alpha;
            hierarchy.index_min_size = this->index_min_size;
            hierarchy.FromJson(hierarchy_json);
            CHECK_ARGUMENT(seen_names.insert(hierarchy.name).second,
                           fmt::format("duplicate hierarchy name {}", hierarchy.name));
            this->hierarchies.emplace_back(std::move(hierarchy));
        }
    }
}
JsonType
PyramidParameters::ToJson() const {
    JsonType json = InnerIndexParameter::ToJson();
    json[NO_BUILD_LEVELS].SetVector(no_build_levels);
    json[BASE_CODES_KEY].SetJson(base_codes_param->ToJson());

    auto graph_json = graph_param->ToJson();
    graph_json[ALPHA_KEY].SetFloat(this->alpha);
    graph_json[GRAPH_TYPE_KEY].SetString(this->graph_type);
    if (this->graph_type == GRAPH_TYPE_ODESCENT) {
        graph_json.UpdateJson(odescent_param->ToJson());
    } else {
        json[EF_CONSTRUCTION_KEY].SetInt(this->ef_construction);
    }
    json[GRAPH_KEY].SetJson(graph_json);
    json[USE_REORDER_KEY].SetBool(this->use_reorder);
    json[INDEX_MIN_SIZE].SetInt(index_min_size);
    json[SUPPORT_DUPLICATE].SetBool(support_duplicate);
    if (this->use_reorder) {
        json[PRECISE_CODES_KEY].SetJson(precise_codes_param->ToJson());
    }
    if (this->has_hierarchies) {
        auto* hierarchies_json = json[PYRAMID_HIERARCHIES].GetInnerJson();
        *hierarchies_json = nlohmann::json::array();
        for (const auto& hierarchy : this->hierarchies) {
            auto hierarchy_json = hierarchy.ToJson();
            hierarchies_json->push_back(*hierarchy_json.GetInnerJson());
        }
    }
    return json;
}

bool
PyramidParameters::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(PyramidParameters, p, other);
    CHECK_SUB_PARAM(*this, *p, graph_param);
    CHECK_SUB_PARAM(*this, *p, base_codes_param);
    if (this->has_hierarchies != p->has_hierarchies) {
        logger::error(
            "PyramidParameters::CheckCompatibility: hierarchies mode settings are not compatible");
        return false;
    }
    if (this->has_hierarchies) {
        if (this->hierarchies.size() != p->hierarchies.size()) {
            logger::error(
                "PyramidParameters::CheckCompatibility: hierarchy counts are not compatible");
            return false;
        }
        for (const auto& hierarchy : this->hierarchies) {
            bool found = false;
            for (const auto& other_hierarchy : p->hierarchies) {
                if (hierarchy.name != other_hierarchy.name) {
                    continue;
                }
                if (not hierarchy.CheckCompatibility(other_hierarchy)) {
                    logger::error(
                        "PyramidParameters::CheckCompatibility: hierarchies are not compatible");
                    return false;
                }
                found = true;
                break;
            }
            if (not found) {
                logger::error("PyramidParameters::CheckCompatibility: hierarchy {} is missing",
                              hierarchy.name);
                return false;
            }
        }
    }
    if (not this->has_hierarchies &&
        (no_build_levels.size() != p->no_build_levels.size() ||
         not std::is_permutation(
             no_build_levels.begin(), no_build_levels.end(), p->no_build_levels.begin()))) {
        logger::error("PyramidParameters::CheckCompatibility: no_build_levels are not compatible");
        return false;
    }
    CHECK_FIELD_EQ(*this, *p, use_reorder);
    if (this->use_reorder) {
        CHECK_SUB_PARAM(*this, *p, precise_codes_param);
    }
    CHECK_FIELD_EQ(*this, *p, index_min_size);
    CHECK_FIELD_EQ(*this, *p, support_duplicate);
    return true;
}

PyramidSearchParameters
PyramidSearchParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::Parse(json_string);

    PyramidSearchParameters obj;

    // set obj.ef_search
    CHECK_ARGUMENT(params.Contains(INDEX_PYRAMID),
                   fmt::format("parameters must contains {}", INDEX_PYRAMID));
    obj.IndexSearchParameter::FromJson(params[INDEX_PYRAMID]);

    CHECK_ARGUMENT(
        params[INDEX_PYRAMID].Contains(PYRAMID_PARAMETER_EF_SEARCH),
        fmt::format("parameters[{}] must contains {}", INDEX_PYRAMID, PYRAMID_PARAMETER_EF_SEARCH));
    obj.ef_search = params[INDEX_PYRAMID][PYRAMID_PARAMETER_EF_SEARCH].GetInt();
    if (params[INDEX_PYRAMID].Contains(PYRAMID_PARAMETER_SUBINDEX_EF_SEARCH)) {
        obj.subindex_ef_search =
            params[INDEX_PYRAMID][PYRAMID_PARAMETER_SUBINDEX_EF_SEARCH].GetInt();
    }
    std::unordered_set<std::string> seen_names;
    if (params[INDEX_PYRAMID].Contains(PYRAMID_PARAMETER_HIERARCHIES)) {
        const auto& hierarchies_json = params[INDEX_PYRAMID][PYRAMID_PARAMETER_HIERARCHIES];
        CHECK_ARGUMENT(hierarchies_json.IsArray(), "pyramid hierarchies must be an array");
        CHECK_ARGUMENT(not hierarchies_json.GetInnerJson()->empty(),
                       "pyramid hierarchies must not be empty");

        for (const auto& hierarchy_raw_json : *hierarchies_json.GetInnerJson()) {
            JsonType hierarchy_json;
            *hierarchy_json.GetInnerJson() = hierarchy_raw_json;
            CHECK_ARGUMENT(hierarchy_json.IsString(), "pyramid hierarchy name must be a string");
            append_hierarchy_selector(obj, seen_names, hierarchy_json.GetString());
        }
    }
    if (params[INDEX_PYRAMID].Contains(PYRAMID_PARAMETER_HIERARCHY_OP)) {
        CHECK_ARGUMENT(obj.hierarchies.size() >= 2,
                       "pyramid hierarchy_op requires at least two hierarchies");
        CHECK_ARGUMENT(params[INDEX_PYRAMID][PYRAMID_PARAMETER_HIERARCHY_OP].IsString(),
                       "pyramid hierarchy_op must be a string");
        obj.hierarchy_op =
            parse_hierarchy_op(params[INDEX_PYRAMID][PYRAMID_PARAMETER_HIERARCHY_OP].GetString());
    }
    CHECK_ARGUMENT(obj.hierarchies.size() <= 1 || obj.hierarchy_op != HierarchyOp::SINGLE,
                   "pyramid hierarchy_op is required for multiple hierarchies");
    return obj;
}

bool
PyramidSearchParameters::HasHierarchySelector() const {
    return not hierarchies.empty();
}
}  // namespace vsag

// NOLINTEND(readability-simplify-boolean-expr)
