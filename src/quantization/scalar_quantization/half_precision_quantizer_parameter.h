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

#include <memory>

#include "half_precision_traits.h"
#include "inner_string_params.h"
#include "quantization/quantizer_parameter.h"

namespace vsag {

template <typename Format>
class HalfPrecisionQuantizerParameter : public QuantizerParameter {
public:
    HalfPrecisionQuantizerParameter() : QuantizerParameter(Format::TYPE_NAME) {
    }

    ~HalfPrecisionQuantizerParameter() override = default;

    void
    FromJson(const JsonType& json) override {
    }

    JsonType
    ToJson() const override {
        JsonType json;
        json[TYPE_KEY].SetString(Format::TYPE_NAME);
        return json;
    }
};

using FP16QuantizerParameter = HalfPrecisionQuantizerParameter<FP16Format>;
using BF16QuantizerParameter = HalfPrecisionQuantizerParameter<BF16Format>;

using FP16QuantizerParamPtr = std::shared_ptr<FP16QuantizerParameter>;
using FP16QuantizerParamUPtr = std::unique_ptr<FP16QuantizerParameter>;
using FP16QuantizerParamConstPtr = std::shared_ptr<const FP16QuantizerParameter>;
using FP16QuantizerParamConstUPtr = std::unique_ptr<const FP16QuantizerParameter>;

using BF16QuantizerParamPtr = std::shared_ptr<BF16QuantizerParameter>;
using BF16QuantizerParamUPtr = std::unique_ptr<BF16QuantizerParameter>;
using BF16QuantizerParamConstPtr = std::shared_ptr<const BF16QuantizerParameter>;
using BF16QuantizerParamConstUPtr = std::unique_ptr<const BF16QuantizerParameter>;

}  // namespace vsag
