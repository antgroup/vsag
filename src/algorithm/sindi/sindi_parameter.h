
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

#include "algorithm/inner_index_parameter.h"
#include "index_common_param.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER(SINDIParameter);

class SINDIParameter : public InnerIndexParameter {
public:
    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

    SINDIParameter() = default;

public:
    // index
    uint32_t term_id_limit{0};

    uint32_t window_size{0};

    float doc_prune_ratio{0};

    // temporal parameter
    bool deserialize_without_footer{false};
};

class SINDISearchParameter : public Parameter {
public:
    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    SINDISearchParameter() = default;

public:
    // search
    uint32_t n_candidate{0};

    // data cell
    float query_prune_ratio{0};
    float term_prune_ratio{0};
};

}  // namespace vsag
