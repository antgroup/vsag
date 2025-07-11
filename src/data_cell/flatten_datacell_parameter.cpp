
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

#include "flatten_datacell_parameter.h"

#include <fmt/format.h>

#include "inner_string_params.h"

namespace vsag {
FlattenDataCellParameter::FlattenDataCellParameter()
    : FlattenInterfaceParameter(FLATTEN_DATA_CELL) {
}

void
FlattenDataCellParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(json.contains(IO_PARAMS_KEY),
                   fmt::format("flatten interface parameters must contains {}", IO_PARAMS_KEY));
    this->io_parameter = IOParameter::GetIOParameterByJson(json[IO_PARAMS_KEY]);

    CHECK_ARGUMENT(
        json.contains(QUANTIZATION_PARAMS_KEY),
        fmt::format("flatten interface parameters must contains {}", QUANTIZATION_PARAMS_KEY));
    this->quantizer_parameter =
        QuantizerParameter::GetQuantizerParameterByJson(json[QUANTIZATION_PARAMS_KEY]);
    this->name = FLATTEN_DATA_CELL;
}

JsonType
FlattenDataCellParameter::ToJson() {
    JsonType json;
    json[IO_PARAMS_KEY] = this->io_parameter->ToJson();
    json[QUANTIZATION_PARAMS_KEY] = this->quantizer_parameter->ToJson();
    return json;
}
}  // namespace vsag
