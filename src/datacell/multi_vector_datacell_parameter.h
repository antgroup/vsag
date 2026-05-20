
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

#include "flatten_interface_parameter.h"
#include "impl/logger/logger.h"
#include "inner_string_params.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(MultiVectorDataCellParam, MultiVectorDataCellParameter);

class MultiVectorDataCellParameter : public FlattenInterfaceParameter {
public:
    explicit MultiVectorDataCellParameter() : FlattenInterfaceParameter(MULTI_VECTOR_DATA_CELL) {
    }

    void
    FromJson(const JsonType& json) override {
        CHECK_ARGUMENT(
            json.Contains(IO_PARAMS_KEY),
            fmt::format("multi-vector datacell parameters must contain {}", IO_PARAMS_KEY));
        this->io_parameter = IOParameter::GetIOParameterByJson(json[IO_PARAMS_KEY]);
        this->quantizer_parameter = std::make_shared<FP32QuantizerParameter>();
    }

    JsonType
    ToJson() const override {
        JsonType json;
        json[IO_PARAMS_KEY].SetJson(this->io_parameter->ToJson());
        json[CODES_TYPE_KEY].SetString(MULTI_VECTOR_CODES);
        return json;
    }

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override {
        auto mv_param = std::dynamic_pointer_cast<MultiVectorDataCellParameter>(other);
        if (not mv_param) {
            logger::error(
                "MultiVectorDataCellParameter::CheckCompatibility: "
                "other parameter is not MultiVectorDataCellParameter");
            return false;
        }
        return true;
    }
};
}  // namespace vsag
