
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

#include "attribute_inverted_interface_parameter.h"

#include "inner_string_params.h"

namespace vsag {

void
AttributeInvertedInterfaceParameter::FromJson(const JsonType& json) {
    if (json.Contains(ATTR_HAS_BUCKETS_KEY)) {
        this->has_buckets_ = json[ATTR_HAS_BUCKETS_KEY].GetBool();
    }
}

JsonType
AttributeInvertedInterfaceParameter::ToJson() const {
    JsonType json;
    json[ATTR_HAS_BUCKETS_KEY].SetBool(this->has_buckets_);
    return json;
}
bool
AttributeInvertedInterfaceParameter::CheckCompatibility(const ParamPtr& other) const {
    auto other_param = std::dynamic_pointer_cast<AttributeInvertedInterfaceParameter>(other);
    if (other_param == nullptr) {
        return false;
    }
    return has_buckets_ == other_param->has_buckets_;
}

}  // namespace vsag
