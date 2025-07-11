
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

#include <fmt/format.h>

#include <nlohmann/json.hpp>

#include "vsag/constants.h"

namespace vsag {

HnswParameters
HnswParameters::FromJson(JsonType& hnsw_param_obj, const IndexCommonParam& index_common_param) {
    HnswParameters obj;

    if (index_common_param.data_type_ == DataTypes::DATA_TYPE_FLOAT) {
        obj.type = DataTypes::DATA_TYPE_FLOAT;
    } else if (index_common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        obj.type = DataTypes::DATA_TYPE_INT8;
        if (index_common_param.metric_ != MetricType::METRIC_TYPE_IP) {
            throw std::invalid_argument(fmt::format(
                "no support for INT8 when using {}, {} as metric", METRIC_L2, METRIC_COSINE));
        }
    }

    if (index_common_param.metric_ == MetricType::METRIC_TYPE_L2SQR) {
        obj.space = std::make_shared<hnswlib::L2Space>(index_common_param.dim_);
    } else if (index_common_param.metric_ == MetricType::METRIC_TYPE_IP) {
        obj.space = std::make_shared<hnswlib::InnerProductSpace>(index_common_param.dim_, obj.type);
    } else if (index_common_param.metric_ == MetricType::METRIC_TYPE_COSINE) {
        obj.normalize = true;
        obj.space = std::make_shared<hnswlib::InnerProductSpace>(index_common_param.dim_, obj.type);
    }

    // set obj.max_degree
    CHECK_ARGUMENT(hnsw_param_obj.contains(HNSW_PARAMETER_M),
                   fmt::format("parameters[{}] must contains {}", INDEX_HNSW, HNSW_PARAMETER_M));
    CHECK_ARGUMENT(hnsw_param_obj[HNSW_PARAMETER_M].is_number_integer(),
                   fmt::format("parameters[{}] must be integer type", HNSW_PARAMETER_M));
    obj.max_degree = hnsw_param_obj[HNSW_PARAMETER_M];
    auto max_degree_threshold = std::max(index_common_param.dim_, 128L);
    CHECK_ARGUMENT(  // NOLINT
        (4 <= obj.max_degree) and (obj.max_degree <= max_degree_threshold),
        fmt::format("max_degree({}) must in range[4, {}]", obj.max_degree, max_degree_threshold));

    // set obj.ef_construction
    CHECK_ARGUMENT(
        hnsw_param_obj.contains(HNSW_PARAMETER_CONSTRUCTION),
        fmt::format("parameters[{}] must contains {}", INDEX_HNSW, HNSW_PARAMETER_CONSTRUCTION));
    CHECK_ARGUMENT(hnsw_param_obj[HNSW_PARAMETER_CONSTRUCTION].is_number_integer(),
                   fmt::format("parameters[{}] must be integer type", HNSW_PARAMETER_CONSTRUCTION));
    obj.ef_construction = hnsw_param_obj[HNSW_PARAMETER_CONSTRUCTION];
    auto construction_threshold = std::max(1000L, AMPLIFICATION_FACTOR * obj.max_degree);
    CHECK_ARGUMENT((obj.max_degree <= obj.ef_construction) and  // NOLINT
                       (obj.ef_construction <= construction_threshold),
                   fmt::format("ef_construction({}) must in range[$max_degree({}), {}]",
                               obj.ef_construction,
                               obj.max_degree,
                               construction_threshold));

    // set obj.use_static
    obj.use_static = hnsw_param_obj.contains(HNSW_PARAMETER_USE_STATIC) &&
                     hnsw_param_obj[HNSW_PARAMETER_USE_STATIC];

    // set obj.use_conjugate_graph
    if (hnsw_param_obj.contains(PARAMETER_USE_CONJUGATE_GRAPH)) {
        obj.use_conjugate_graph = hnsw_param_obj[PARAMETER_USE_CONJUGATE_GRAPH];
    } else {
        obj.use_conjugate_graph = false;
    }
    return obj;
}

HnswSearchParameters
HnswSearchParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::parse(json_string);

    HnswSearchParameters obj;

    // set obj.ef_search
    std::string index_name;
    if (params.contains(INDEX_HNSW)) {
        index_name = INDEX_HNSW;
    } else if (params.contains(INDEX_FRESH_HNSW)) {
        index_name = INDEX_FRESH_HNSW;
    } else {
        throw std::invalid_argument(
            fmt::format("parameters must contains {}/{}", INDEX_HNSW, INDEX_FRESH_HNSW));
    }

    CHECK_ARGUMENT(
        params[index_name].contains(HNSW_PARAMETER_EF_RUNTIME),
        fmt::format("parameters[{}] must contains {}", index_name, HNSW_PARAMETER_EF_RUNTIME));
    obj.ef_search = params[index_name][HNSW_PARAMETER_EF_RUNTIME];

    // set obj.use_conjugate_graph search
    if (params[index_name].contains(PARAMETER_USE_CONJUGATE_GRAPH_SEARCH)) {
        obj.use_conjugate_graph_search = params[index_name][PARAMETER_USE_CONJUGATE_GRAPH_SEARCH];
    } else {
        obj.use_conjugate_graph_search = true;
    }

    if (params[index_name].contains(HNSW_PARAMETER_SKIP_RATIO)) {
        obj.skip_ratio = params[index_name][HNSW_PARAMETER_SKIP_RATIO];
    }

    return obj;
}

HnswParameters
FreshHnswParameters::FromJson(JsonType& hnsw_param_obj,
                              const IndexCommonParam& index_common_param) {
    HnswParameters obj = HnswParameters::FromJson(hnsw_param_obj, index_common_param);
    obj.use_static = false;
    // set obj.use_reversed_edges
    obj.use_reversed_edges = true;
    return obj;
}

}  // namespace vsag
