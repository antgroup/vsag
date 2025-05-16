
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

#include "graph_interface_parameter.h"
#include "vsag/constants.h"

namespace vsag {
class SparseGraphDatacellParameter : public GraphInterfaceParameter {
public:
    SparseGraphDatacellParameter()
        : GraphInterfaceParameter(GraphStorageTypes::GRAPH_STORAGE_TYPE_SPARSE) {
    }

    void
    FromJson(const JsonType& json) override {
        if (json.contains(GRAPH_PARAM_MAX_DEGREE)) {
            this->max_degree_ = json[GRAPH_PARAM_MAX_DEGREE];
        }
        if (json.contains(GRAPH_SUPPORT_REMOVE)) {
            this->support_delete_ = json[GRAPH_SUPPORT_REMOVE];
        }
        if (json.contains(REMOVE_FLAG_BIT)) {
            this->remove_flag_bit_ = json[REMOVE_FLAG_BIT];
        }
    }

    JsonType
    ToJson() override {
        JsonType json;
        json[GRAPH_PARAM_MAX_DEGREE] = this->max_degree_;
        json[GRAPH_SUPPORT_REMOVE] = support_delete_;
        json[REMOVE_FLAG_BIT] = remove_flag_bit_;
        return json;
    }

public:
    uint64_t max_degree_{64};
    bool support_delete_{false};
    uint32_t remove_flag_bit_{8};
};

using SparseGraphDatacellParamPtr = std::shared_ptr<SparseGraphDatacellParameter>;
}  // namespace vsag
