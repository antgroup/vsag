
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

#include <cstdint>
#include <string>

#include "quantization/quantizer_parameter.h"
#include "typing.h"
#include "utils/param_compat_macros.h"
#include "utils/pointer_define.h"
namespace vsag {
DEFINE_POINTER2(SparseQuantizerParam, SparseQuantizerParameter);

enum class SparseQuantizerValueType : uint8_t {
    FP32 = 0,
    FP16 = 1,
};

static constexpr const char* SPARSE_QUANTIZER_VALUE_TYPE_KEY = "value_type";

class SparseQuantizerParameter : public QuantizerParameter {
public:
    SparseQuantizerParameter() : QuantizerParameter(QUANTIZATION_TYPE_VALUE_SPARSE) {
    }

    ~SparseQuantizerParameter() override = default;

    void
    FromJson(const JsonType& json) override {
        value_type = SparseQuantizerValueType::FP32;
        if (json.Contains(SPARSE_QUANTIZER_VALUE_TYPE_KEY)) {
            const auto value_type_json = json[SPARSE_QUANTIZER_VALUE_TYPE_KEY];
            CHECK_ARGUMENT(value_type_json.IsString(), "sparse value_type must be a string");
            const std::string value_type_name = value_type_json.GetString();
            CHECK_ARGUMENT(
                value_type_name == QUANTIZATION_TYPE_VALUE_FP32 ||
                    value_type_name == QUANTIZATION_TYPE_VALUE_FP16,
                fmt::format("sparse value_type must be fp32 or fp16, got {}", value_type_name));
            if (value_type_name == QUANTIZATION_TYPE_VALUE_FP16) {
                value_type = SparseQuantizerValueType::FP16;
            }
        }
    }

    JsonType
    ToJson() const override {
        JsonType json;
        json[TYPE_KEY].SetString(this->GetTypeName());
        if (value_type == SparseQuantizerValueType::FP16) {
            json[SPARSE_QUANTIZER_VALUE_TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_FP16);
        }
        return json;
    }

    bool
    CheckCompatibility(const ParamPtr& other) const override {
        PARAM_CAST_OR_RETURN(SparseQuantizerParameter, parameter, other);
        CHECK_FIELD_EQ(*this, *parameter, value_type);
        return true;
    }

public:
    SparseQuantizerValueType value_type{SparseQuantizerValueType::FP32};
};
}  // namespace vsag
