
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
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER2(TurboQuantizerParam, TurboQuantizerParameter);

class TurboQuantizerParameter : public QuantizerParameter {
public:
    TurboQuantizerParameter();

    ~TurboQuantizerParameter() override = default;

    void
    FromJson(const JsonType& json) override;

    [[nodiscard]] JsonType
    ToJson() const override;

    [[nodiscard]] bool
    CheckCompatibility(const ParamPtr& other) const override;

    uint64_t bits_per_dim_{4};
    bool use_fht_{true};
    bool enable_qjl_{true};
    uint64_t qjl_projection_dim_{0};
};

}  // namespace vsag
