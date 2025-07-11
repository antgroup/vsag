
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

namespace vsag {

class SparseVectorDataCellParameter : public FlattenInterfaceParameter {
public:
    explicit SparseVectorDataCellParameter() : FlattenInterfaceParameter(SPARSE_VECTOR_DATA_CELL) {
    }

    void
    FromJson(const JsonType& json) override {
        CHECK_ARGUMENT(json.contains(IO_PARAMS_KEY),
                       fmt::format("sparse datacell parameters must contains {}", IO_PARAMS_KEY));
        this->io_parameter = IOParameter::GetIOParameterByJson(json[IO_PARAMS_KEY]);
        CHECK_ARGUMENT(
            json.contains(QUANTIZATION_PARAMS_KEY),
            fmt::format("sparse datacell parameters must contains {}", QUANTIZATION_PARAMS_KEY));
        this->quantizer_parameter =
            QuantizerParameter::GetQuantizerParameterByJson(json[QUANTIZATION_PARAMS_KEY]);
        CHECK_ARGUMENT(
            this->quantizer_parameter->GetTypeName() == QUANTIZATION_TYPE_VALUE_SPARSE,
            fmt::format("sparse datacell only support {}", QUANTIZATION_TYPE_VALUE_SPARSE));
    }

    JsonType
    ToJson() override {
        JsonType json;
        json[IO_PARAMS_KEY] = this->io_parameter->ToJson();
        json[QUANTIZATION_PARAMS_KEY] = this->quantizer_parameter->ToJson();
        return json;
    }
};

using SparseVectorDataCellParamPtr = std::shared_ptr<SparseVectorDataCellParameter>;

}  // namespace vsag
