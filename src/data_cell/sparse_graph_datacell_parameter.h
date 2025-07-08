
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
#include "inner_string_params.h"
#include "logger.h"
#include "vsag/constants.h"

namespace vsag {
class SparseGraphDatacellParameter : public GraphInterfaceParameter {
public:
    SparseGraphDatacellParameter();

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

public:
    bool support_delete_{false};
    uint32_t remove_flag_bit_{8};
};

using SparseGraphDatacellParamPtr = std::shared_ptr<SparseGraphDatacellParameter>;
}  // namespace vsag
