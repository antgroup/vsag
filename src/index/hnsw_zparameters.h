
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

#pragma once

#include <memory>
#include <string>

#include "../algorithm/hnswlib/hnswlib.h"
#include "../data_type.h"
#include "index_common_param.h"

namespace vsag {

struct HnswParameters {
public:
    static HnswParameters
    FromJson(JsonType& hnsw_param_obj, const IndexCommonParam& index_common_param);

public:
    // required vars
    std::shared_ptr<hnswlib::SpaceInterface> space;
    int64_t max_degree;
    int64_t ef_construction;
    bool use_conjugate_graph{false};
    bool use_static{false};
    bool normalize{false};
    bool use_reversed_edges{false};
    DataTypes type{DataTypes::DATA_TYPE_FLOAT};

protected:
    HnswParameters() = default;
};

struct FreshHnswParameters : public HnswParameters {
public:
    static HnswParameters
    FromJson(JsonType& hnsw_param_obj, const IndexCommonParam& index_common_param);

private:
    FreshHnswParameters() = default;
};

struct HnswSearchParameters {
public:
    static HnswSearchParameters
    FromJson(const std::string& json_string);

public:
    // required vars
    int64_t ef_search;
    float skip_ratio{0.9};
    bool use_conjugate_graph_search;

private:
    HnswSearchParameters() = default;
};

}  // namespace vsag
