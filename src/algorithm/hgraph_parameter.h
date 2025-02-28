
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

#include "data_cell/flatten_datacell_parameter.h"
#include "data_cell/graph_interface_parameter.h"
#include "parameter.h"

namespace vsag {

class HGraphParameter : public Parameter {
public:
    explicit HGraphParameter(const JsonType& json);

    HGraphParameter();

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() override;

public:
    FlattenDataCellParamPtr base_codes_param_{nullptr};
    FlattenDataCellParamPtr precise_codes_param_{nullptr};
    GraphInterfaceParamPtr bottom_graph_param_{nullptr};

    bool use_reorder_{false};
    uint64_t ef_construction_{400};
    uint64_t build_thread_count_{100};

    std::string name_;
};

using HGraphParameterPtr = std::shared_ptr<HGraphParameter>;

}  // namespace vsag
