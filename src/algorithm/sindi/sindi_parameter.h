
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

#include "index/index_common_param.h"
#include "parameter.h"

namespace vsag {

static constexpr uint32_t DEFAULT_WINDOW_SIZE = 100000;
static constexpr bool DEFAULT_USE_REORDER = false;
static constexpr float DEFAULT_QUERY_PRUNE_RATIO = 0.0F;
static constexpr float DEFAULT_DOC_PRUNE_RATIO = 0.0F;
static constexpr float DEFAULT_TERM_PRUNE_RATIO = 0.0F;
static constexpr uint32_t DEFAULT_N_CANDIDATE = 0;

struct SINDIParameter : public Parameter {
public:
    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    SINDIParameter() = default;

public:
    // index
    uint32_t window_size{0};
    bool use_reorder{false};

    float doc_prune_ratio{0};
};

using SINDIParameterPtr = std::shared_ptr<SINDIParameter>;

struct SINDISearchParameter : public Parameter {
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
