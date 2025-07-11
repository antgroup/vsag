
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

#include "brute_force_parameter.h"

#include <fmt/format.h>

#include "vsag/constants.h"

namespace vsag {

BruteForceParameter::BruteForceParameter() : flatten_param(nullptr) {
}

void
BruteForceParameter::FromJson(const JsonType& json) {
    this->flatten_param = std::make_shared<FlattenDataCellParameter>();
    this->flatten_param->FromJson(json);
}

JsonType
BruteForceParameter::ToJson() {
    auto json = this->flatten_param->ToJson();
    json["type"] = INDEX_BRUTE_FORCE;
    return json;
}
}  // namespace vsag
