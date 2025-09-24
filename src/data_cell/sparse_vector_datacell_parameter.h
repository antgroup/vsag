
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

#include "flatten_interface.h"
#include "inner_string_params.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(SparseVectorDataCellParam, SparseVectorDataCellParameter);
class SparseVectorDataCellParameter : public FlattenInterfaceParameter {
public:
    explicit SparseVectorDataCellParameter() : FlattenInterfaceParameter(SPARSE_VECTOR_DATA_CELL) {
    }

    void
    FromJson(const JsonType& json) override {
        CHECK_ARGUMENT(json.Contains(IO_PARAMS_KEY),
                       fmt::format("sparse datacell parameters must contains {}", IO_PARAMS_KEY));
        this->io_parameter = IOParameter::GetIOParameterByJson(json[IO_PARAMS_KEY]);
        CHECK_ARGUMENT(
            json.Contains(QUANTIZATION_PARAMS_KEY),
            fmt::format("sparse datacell parameters must contains {}", QUANTIZATION_PARAMS_KEY));
        this->quantizer_parameter =
            QuantizerParameter::GetQuantizerParameterByJson(json[QUANTIZATION_PARAMS_KEY]);
        CHECK_ARGUMENT(
            this->quantizer_parameter->GetTypeName() == QUANTIZATION_TYPE_VALUE_SPARSE,
            fmt::format("sparse datacell only support {}", QUANTIZATION_TYPE_VALUE_SPARSE));
    }

    JsonType
    ToJson() const override {
        JsonType json;
        json[IO_PARAMS_KEY].SetJson(this->io_parameter->ToJson());
        json[QUANTIZATION_PARAMS_KEY].SetJson(this->quantizer_parameter->ToJson());
        json[CODES_TYPE_KEY].SetString(SPARSE_CODES);

        return json;
    }

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override {
        auto sparse_param = std::dynamic_pointer_cast<SparseVectorDataCellParameter>(other);
        if (not sparse_param) {
            logger::error(
                "SparseVectorDataCellParameter::CheckCompatibility: "
                "other parameter is not SparseVectorDataCellParameter");
            return false;
        }
        return this->quantizer_parameter->CheckCompatibility(sparse_param->quantizer_parameter);
    }
};
}  // namespace vsag
