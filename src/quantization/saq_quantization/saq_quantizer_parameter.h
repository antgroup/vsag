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

DEFINE_POINTER2(SAQQuantizerParam, SAQQuantizerParameter);

class SAQQuantizerParameter : public QuantizerParameter {
public:
    static constexpr float DEFAULT_AVG_BITS = 4.0F;
    static constexpr uint64_t DEFAULT_SEGMENT_COUNT = 0;
    static constexpr uint64_t DEFAULT_ADJUSTMENT_ROUNDS = 6;
    static constexpr bool DEFAULT_USE_PCA = true;
    static constexpr bool DEFAULT_RANDOM_ROTATION = true;
    static constexpr uint64_t DEFAULT_RANDOM_ROTATION_SEED = 20260825;
    static constexpr uint64_t MIN_ADJUSTMENT_ROUNDS = 0;
    static constexpr uint64_t MAX_ADJUSTMENT_ROUNDS = 32;

    SAQQuantizerParameter();

    ~SAQQuantizerParameter() override = default;

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const ParamPtr& other) const override;

public:
    float avg_bits_{DEFAULT_AVG_BITS};
    uint64_t segment_count_{DEFAULT_SEGMENT_COUNT};
    uint64_t adjustment_rounds_{DEFAULT_ADJUSTMENT_ROUNDS};
    bool use_pca_{DEFAULT_USE_PCA};
    bool random_rotation_{DEFAULT_RANDOM_ROTATION};
};

}  // namespace vsag
