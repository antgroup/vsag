
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

#include <fmt/format.h>

#include "common.h"
#include "datacell/flatten_interface_parameter.h"
#include "inner_string_params.h"
#include "reorder_parameter.h"
#include "utils/pointer_define.h"
namespace vsag {
DEFINE_POINTER(PqrReorderParameter);

class PqrReorderParameter : public ReorderParameter {
public:
    explicit PqrReorderParameter() : ReorderParameter(PQR_REORDER) {
    }

    void
    FromJson(const vsag::JsonType& json) override {
        residual_param_ = CreateFlattenParam(json);
    }

    JsonType
    ToJson() const override {
        JsonType json = residual_param_->ToJson();
        json[REORDER_TYPE].SetString(this->name_);
        return json;
    }

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override {
        auto dynamic_other = std::dynamic_pointer_cast<PqrReorderParameter>(other);
        if (!dynamic_other) {
            return false;
        }
        return this->residual_param_->CheckCompatibility(dynamic_other->residual_param_);
    }

public:
    FlattenInterfaceParamPtr residual_param_;  // used to quantize residual
};

}  // namespace vsag
