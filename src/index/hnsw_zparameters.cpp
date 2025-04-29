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

#include <fmt/format-inl.h>

#include <nlohmann/json.hpp>

#include "../algorithm/hnswlib/hnswlib.h"
#include "../common.h"
#include "vsag/constants.h"
namespace vsag {

CreateHnswParameters
CreateHnswParameters::FromJson(const std::string& json_string) {
    nlohmann::json params = nlohmann::json::parse(json_string);

    CHECK_ARGUMENT(params.contains(PARAMETER_DTYPE),
                   fmt::format("parameters must contains {}", PARAMETER_DTYPE));
    CHECK_ARGUMENT(
        params[PARAMETER_DTYPE] == DATATYPE_FLOAT32,
        fmt::format("parameters[{}] supports {} only now", PARAMETER_DTYPE, DATATYPE_FLOAT32));
    CHECK_ARGUMENT(params.contains(PARAMETER_METRIC_TYPE),
                   fmt::format("parameters must contains {}", PARAMETER_METRIC_TYPE));
    CHECK_ARGUMENT(params.contains(PARAMETER_DIM),
                   fmt::format("parameters must contains {}", PARAMETER_DIM));

    CreateHnswParameters obj;

    // set obj.space
    CHECK_ARGUMENT(params.contains(INDEX_HNSW),
                   fmt::format("parameters must contains {}", INDEX_HNSW));
    if (params[PARAMETER_METRIC_TYPE] == METRIC_L2) {
        obj.space = std::make_shared<hnswlib::L2Space>(params[PARAMETER_DIM]);
    } else if (params[PARAMETER_METRIC_TYPE] == METRIC_IP) {
        obj.space = std::make_shared<hnswlib::InnerProductSpace>(params[PARAMETER_DIM]);
    } else {
        std::string metric = params[PARAMETER_METRIC_TYPE];
        throw std::invalid_argument(fmt::format("parameters[{}] must in [{}, {}], now is {}",
                                                PARAMETER_METRIC_TYPE,
                                                METRIC_L2,
                                                METRIC_IP,
                                                metric));
    }

    // set obj.max_degree
    CHECK_ARGUMENT(params[INDEX_HNSW].contains(HNSW_PARAMETER_M),
                   fmt::format("parameters[{}] must contains {}", INDEX_HNSW, HNSW_PARAMETER_M));
    obj.max_degree = params[INDEX_HNSW][HNSW_PARAMETER_M];
    CHECK_ARGUMENT((5 <= obj.max_degree),
                   fmt::format("max_degree({}) must in range[5, )", obj.max_degree));

    // set obj.ef_construction
    CHECK_ARGUMENT(
        params[INDEX_HNSW].contains(HNSW_PARAMETER_CONSTRUCTION),
        fmt::format("parameters[{}] must contains {}", INDEX_HNSW, HNSW_PARAMETER_CONSTRUCTION));
    obj.ef_construction = params[INDEX_HNSW][HNSW_PARAMETER_CONSTRUCTION];
    CHECK_ARGUMENT((obj.max_degree <= obj.ef_construction) and (obj.ef_construction <= 1000),
                   fmt::format("ef_construction({}) must in range[$max_degree({}), 64]",
                               obj.ef_construction,
                               obj.max_degree));

    // set obj.use_static
    obj.use_static = params[INDEX_HNSW].contains(HNSW_PARAMETER_USE_STATIC) &&
                     params[INDEX_HNSW][HNSW_PARAMETER_USE_STATIC];

    if (params[INDEX_HNSW].contains(PARAMETER_USE_EXTRA_PQ_FILE)) {
        obj.extra_file = params[INDEX_HNSW][PARAMETER_USE_EXTRA_PQ_FILE];
    } else {
        obj.extra_file = "";
    }

    // set obj.use_conjugate_graph
    if (params[INDEX_HNSW].contains(PARAMETER_USE_CONJUGATE_GRAPH)) {
        obj.use_conjugate_graph = params[INDEX_HNSW][PARAMETER_USE_CONJUGATE_GRAPH];
    } else {
        obj.use_conjugate_graph = false;
    }

    if (params[INDEX_HNSW].contains("sq_num_bits")) {
        obj.sq_num_bits = params[INDEX_HNSW]["sq_num_bits"];
    } else {
        obj.sq_num_bits = -1;
    }

    // set obj.alpha
    if (params[INDEX_HNSW].contains(PARAMETER_ALPHA)) {
        obj.alpha = params[INDEX_HNSW][PARAMETER_ALPHA];
        CHECK_ARGUMENT((0.8 <= obj.alpha) and (obj.alpha <= 2.0),
                       fmt::format("alpha({}) must in range[0.8, 2.0]", obj.alpha));
    } else {
        obj.alpha = 1;
    }

    if (params[INDEX_HNSW].contains(PARAMETER_REDUNDANT_RATE)) {
        obj.redundant_rate = params[INDEX_HNSW][PARAMETER_REDUNDANT_RATE];
        CHECK_ARGUMENT((0 <= obj.redundant_rate) and (obj.redundant_rate <= 1),
                       fmt::format("alpha({}) must in range[0, 1.0]", obj.redundant_rate));
    } else {
        obj.redundant_rate = 1;
    }

    if (params[INDEX_HNSW].contains("graph_type")) {
        std::string graph_type = params[INDEX_HNSW]["graph_type"];
        if (graph_type == "odescent") {
            obj.odesent_parameters = std::make_shared<ODesentParameters>();
            obj.odesent_parameters->alpha = obj.alpha;
            if (params[INDEX_HNSW].contains("sample_rate")) {
                obj.odesent_parameters->sample_rate = params[INDEX_HNSW]["sample_rate"];
            }
            if (params[INDEX_HNSW].contains("graph_iter_turn")) {
                obj.odesent_parameters->graph_iter_turn = params[INDEX_HNSW]["graph_iter_turn"];
            }
            if (params[INDEX_HNSW].contains("use_thread")) {
                obj.odesent_parameters->use_thread = params[INDEX_HNSW]["use_thread"];
            }
        }
    }

    return obj;
}

HnswSearchParameters
HnswSearchParameters::FromJson(const std::string& json_string) {
    nlohmann::json params = nlohmann::json::parse(json_string);

    HnswSearchParameters obj;

    // set obj.ef_search
    CHECK_ARGUMENT(params.contains(INDEX_HNSW),
                   fmt::format("parameters must contains {}", INDEX_HNSW));

    CHECK_ARGUMENT(
        params[INDEX_HNSW].contains(HNSW_PARAMETER_EF_RUNTIME),
        fmt::format("parameters[{}] must contains {}", INDEX_HNSW, HNSW_PARAMETER_EF_RUNTIME));
    obj.ef_search = params[INDEX_HNSW][HNSW_PARAMETER_EF_RUNTIME];
    CHECK_ARGUMENT((1 <= obj.ef_search) and (obj.ef_search <= 1000),
                   fmt::format("ef_search({}) must in range[1, 1000]", obj.ef_search));

    // set obj.use_conjugate_graph search
    if (params[INDEX_HNSW].contains(PARAMETER_USE_CONJUGATE_GRAPH_SEARCH)) {
        obj.use_conjugate_graph_search = params[INDEX_HNSW][PARAMETER_USE_CONJUGATE_GRAPH_SEARCH];
    } else {
        obj.use_conjugate_graph_search = true;
    }

    return obj;
}

CreateFreshHnswParameters
CreateFreshHnswParameters::FromJson(const std::string& json_string) {
    auto parrent_obj = CreateHnswParameters::FromJson(json_string);
    CreateFreshHnswParameters obj;

    obj.max_degree = parrent_obj.max_degree;
    obj.ef_construction = parrent_obj.ef_construction;
    obj.space = parrent_obj.space;
    obj.use_static = false;

    // set obj.use_reversed_edges
    obj.use_reversed_edges = true;
    return obj;
}

}  // namespace vsag
