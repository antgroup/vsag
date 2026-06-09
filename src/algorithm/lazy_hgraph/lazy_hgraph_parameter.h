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

#include "algorithm/bruteforce/bruteforce_parameter.h"
#include "algorithm/hgraph/hgraph_parameter.h"

namespace vsag {

DEFINE_POINTER(LazyHGraphParameter);

class LazyHGraphParameter : public InnerIndexParameter {
public:
    LazyHGraphParameter() = default;

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const ParamPtr& other) const override;

public:
    uint64_t transition_threshold{1000};
    BruteForceParameterPtr flat_param{nullptr};
    HGraphParameterPtr graph_param{nullptr};
};

}  // namespace vsag
