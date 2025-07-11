
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
class RaBitQuantizerParameter : public QuantizerParameter {
public:
    RaBitQuantizerParameter();

    ~RaBitQuantizerParameter() override = default;

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

public:
    uint64_t pca_dim_{0};
    uint64_t num_bits_per_dim_query_{32};
    bool use_fht_{false};
};

using RaBitQuantizerParamPtr = std::shared_ptr<RaBitQuantizerParameter>;

}  // namespace vsag
