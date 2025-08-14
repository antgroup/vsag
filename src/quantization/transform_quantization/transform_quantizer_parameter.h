
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

#include "quantization/quantizer_parameter.h"

namespace vsag {
class TransformQuantizerParameter : public QuantizerParameter {
public:
    TransformQuantizerParameter();

    ~TransformQuantizerParameter() override = default;

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

    static std::vector<std::string>
    SplitString(const std::string& input, char delimiter = ',');

    static std::string
    MergeStrings(const std::vector<std::string>& vec, char delimiter = ',');

public:
    std::vector<std::string> tq_chain_;
    JsonType base_quantizer_json_;  // store param of base quantizer
};

using TransformQuantizerParamPtr = std::shared_ptr<TransformQuantizerParameter>;

}  // namespace vsag
