
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

#include "impl/logger/logger.h"
#include "inner_string_params.h"
#include "quantization/rabitq_quantization/rabitq_quantizer_parameter.h"
#include "utils/param_compat_macros.h"

namespace vsag {
FlattenDataCellParameter::FlattenDataCellParameter()
    : FlattenInterfaceParameter(FLATTEN_DATA_CELL) {
}

void
FlattenDataCellParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(json.Contains(IO_PARAMS_KEY),
                   fmt::format("flatten interface parameters must contains {}", IO_PARAMS_KEY));
    this->io_parameter = IOParameter::GetIOParameterByJson(json[IO_PARAMS_KEY]);

    if (json.Contains(SUPPLEMENT_IO_PARAMS_KEY)) {
        // Auto-fill file_path for file-backed supplement IO if not provided
        // explicitly, by appending "_supplement" to the base io_params'
        // file_path. This keeps the user-facing config minimal: only
        // base_supplement_io_type is required to enable mixed IO.
        auto supplement_json = json[SUPPLEMENT_IO_PARAMS_KEY];
        if (not supplement_json.Contains(IO_FILE_PATH_KEY)) {
            const auto& base_io_json = json[IO_PARAMS_KEY];
            std::string base_path = base_io_json.Contains(IO_FILE_PATH_KEY)
                                        ? base_io_json[IO_FILE_PATH_KEY].GetString()
                                        : std::string(DEFAULT_FILE_PATH_VALUE);
            supplement_json[IO_FILE_PATH_KEY].SetString(base_path + "_supplement");
        }
        this->supplement_io_parameter = IOParameter::GetIOParameterByJson(supplement_json);
    }

    CHECK_ARGUMENT(
        json.Contains(QUANTIZATION_PARAMS_KEY),
        fmt::format("flatten interface parameters must contains {}", QUANTIZATION_PARAMS_KEY));

    // When the caller asks for the split codes layout, force the quantizer
    // to its "split" variant so that users do not have to write
    // "rabitq_version": "split" by hand.
    // codes_type=rabitq_split is the single source of truth: any
    // user-supplied rabitq_version is silently overridden here so that the
    // hgraph default template (which always seeds rabitq_version=standard)
    // does not collide with codes_type=rabitq_split.
    const bool is_split_codes =
        json.Contains(CODES_TYPE_KEY) && json[CODES_TYPE_KEY].GetString() == RABITQ_SPLIT_CODES;
    auto quant_json = json[QUANTIZATION_PARAMS_KEY];
    if (is_split_codes) {
        if (quant_json.Contains(RABITQ_QUANTIZATION_VERSION_KEY)) {
            const auto user_version = quant_json[RABITQ_QUANTIZATION_VERSION_KEY].GetString();
            if (not RaBitQuantizerParameter::IsSplitVersion(user_version)) {
                logger::warn(
                    "rabitq_version={} is overridden to {} because codes_type=rabitq_split",
                    user_version,
                    RaBitQuantizerParameter::RABITQ_VERSION_SPLIT);
            }
        }
        quant_json[RABITQ_QUANTIZATION_VERSION_KEY].SetString(
            RaBitQuantizerParameter::RABITQ_VERSION_SPLIT);
    }
    this->quantizer_parameter = QuantizerParameter::GetQuantizerParameterByJson(quant_json);
    this->name = FLATTEN_DATA_CELL;
    if (is_split_codes) {
        this->name = RABITQ_SPLIT_DATA_CELL;
    }
}

JsonType
FlattenDataCellParameter::ToJson() const {
    JsonType json;
    json[CODES_TYPE_KEY].SetString(this->name == RABITQ_SPLIT_DATA_CELL ? RABITQ_SPLIT_CODES
                                                                        : FLATTEN_CODES);
    json[IO_PARAMS_KEY].SetJson(this->io_parameter->ToJson());
    if (this->supplement_io_parameter != nullptr) {
        json[SUPPLEMENT_IO_PARAMS_KEY].SetJson(this->supplement_io_parameter->ToJson());
    }
    json[QUANTIZATION_PARAMS_KEY].SetJson(this->quantizer_parameter->ToJson());
    return json;
}
bool
FlattenDataCellParameter::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(FlattenDataCellParameter, p, other);
    CHECK_FIELD_EQ(*this, *p, name);
    CHECK_SUB_PARAM(*this, *p, quantizer_parameter);
    return true;
}
}  // namespace vsag
