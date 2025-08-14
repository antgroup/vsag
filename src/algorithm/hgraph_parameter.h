
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

#include "data_cell/extra_info_datacell_parameter.h"
#include "data_cell/flatten_datacell_parameter.h"
#include "data_cell/graph_interface_parameter.h"
#include "data_cell/sparse_graph_datacell_parameter.h"
#include "data_type.h"
#include "impl/odescent_graph_parameter.h"
#include "parameter.h"
#include "vsag/constants.h"

namespace vsag {

class HGraphParameter : public Parameter {
public:
    explicit HGraphParameter(const JsonType& json);

    HGraphParameter();

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const ParamPtr& other) const override;

public:
    FlattenInterfaceParamPtr base_codes_param{nullptr};
    FlattenInterfaceParamPtr precise_codes_param{nullptr};
    GraphInterfaceParamPtr bottom_graph_param{nullptr};
    SparseGraphDatacellParamPtr hierarchical_graph_param{nullptr};
    ExtraInfoDataCellParamPtr extra_info_param{nullptr};
    ODescentParameterPtr odescent_param{nullptr};

    std::string graph_type{GRAPH_TYPE_NSW};

    bool use_reorder{false};
    bool use_elp_optimizer{false};
    bool ignore_reorder{false};
    bool build_by_base{false};
    bool store_raw_vector{false};

    bool use_attribute_filter{false};
    uint64_t ef_construction{400};
    uint64_t build_thread_count{100};

    bool support_duplicate{false};

    DataTypes data_type{DataTypes::DATA_TYPE_FLOAT};

    std::string name;
};

using HGraphParameterPtr = std::shared_ptr<HGraphParameter>;

class HGraphSearchParameters {
public:
    static HGraphSearchParameters
    FromJson(const std::string& json_string);

public:
    int64_t ef_search{30};
    bool use_reorder{false};
    bool use_extra_info_filter{false};
    bool enable_time_record{false};
    double timeout_ms{std::numeric_limits<double>::max()};

private:
    HGraphSearchParameters() = default;
};

}  // namespace vsag
