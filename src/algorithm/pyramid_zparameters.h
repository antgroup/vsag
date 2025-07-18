
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

#include <functional>

#include "algorithm/hgraph_parameter.h"
#include "data_cell/flatten_interface.h"
#include "data_cell/graph_datacell_parameter.h"
#include "data_cell/graph_interface.h"
#include "impl/odescent_graph_parameter.h"
#include "index/index_common_param.h"
#include "typing.h"
#include "vsag/index.h"

namespace vsag {

struct PyramidParameters : public Parameter {
public:
    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

public:
    GraphInterfaceParamPtr graph_param{nullptr};
    FlattenDataCellParamPtr flatten_data_cell_param{nullptr};
    ODescentParameterPtr odescent_param{nullptr};

    std::vector<int32_t> no_build_levels;
    uint64_t ef_construction{100};
};

class PyramidSearchParameters {
public:
    static PyramidSearchParameters
    FromJson(const std::string& json_string);

public:
    int64_t ef_search{100};

private:
    PyramidSearchParameters() = default;
};

using PyramidParamPtr = std::shared_ptr<PyramidParameters>;

}  // namespace vsag
