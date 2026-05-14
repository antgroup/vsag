
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

#include <string>

#include "quantization/quantizer_parameter.h"
#include "utils/pointer_define.h"
namespace vsag {
DEFINE_POINTER2(RaBitQuantizerParam, RaBitQuantizerParameter);
class RaBitQuantizerParameter : public QuantizerParameter {
public:
    static constexpr const char* DEFAULT_RABITQ_VERSION = "standard";
    static constexpr const char* RABITQ_VERSION_SPLIT_1BIT_7BIT = "split_1bit_7bit";
    static constexpr float DEFAULT_RABITQ_ERROR_RATE = 1.9F;

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
    uint64_t num_bits_per_dim_base_{1};
    std::string rabitq_version_{DEFAULT_RABITQ_VERSION};
    float rabitq_error_rate_{DEFAULT_RABITQ_ERROR_RATE};
    bool use_fht_{false};
};
}  // namespace vsag
