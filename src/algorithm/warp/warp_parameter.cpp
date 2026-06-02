
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

#include "warp_parameter.h"

#include <fmt/format.h>

namespace vsag {

void
WarpParameter::FromJson(const JsonType& json) {
    InnerIndexParameter::FromJson(json);
    CHECK_ARGUMENT(json.Contains(BASE_CODES_KEY),
                   fmt::format("warp parameters must contains {}", BASE_CODES_KEY));
    const auto& base_codes_json = json[BASE_CODES_KEY];
    this->base_codes_param = CreateFlattenParam(base_codes_json);
}

JsonType
WarpParameter::ToJson() const {
    JsonType json = InnerIndexParameter::ToJson();
    json[TYPE_KEY].SetString(INDEX_WARP);
    json[BASE_CODES_KEY].SetJson(this->base_codes_param->ToJson());
    return json;
}

bool
WarpParameter::CheckCompatibility(const vsag::ParamPtr& other) const {
    if (not InnerIndexParameter::CheckCompatibility(other)) {
        return false;
    }
    auto other_param = std::dynamic_pointer_cast<WarpParameter>(other);
    if (other_param == nullptr) {
        return false;
    }
    return this->base_codes_param->CheckCompatibility(other_param->base_codes_param);
}

}  // namespace vsag
