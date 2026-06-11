
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

#include "gno_imi_parameter.h"

#include <fmt/format.h>

#include "inner_string_params.h"
#include "utils/param_compat_macros.h"

namespace vsag {

GNOIMIParameter::GNOIMIParameter() = default;

void
GNOIMIParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(
        json.Contains(GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY),
        fmt::format("ivf parameters must contains {}", GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY));
    this->first_order_buckets_count =
        static_cast<int32_t>(json[GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY].GetInt());

    if (json.Contains(GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY)) {
        this->second_order_buckets_count =
            static_cast<int32_t>(json[GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY].GetInt());
    } else {
        this->second_order_buckets_count = this->first_order_buckets_count;
    }
}

JsonType
GNOIMIParameter::ToJson() const {
    JsonType json;
    json[GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY].SetInt(this->first_order_buckets_count);
    json[GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY].SetInt(this->second_order_buckets_count);
    return json;
}
bool
GNOIMIParameter::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(GNOIMIParameter, p, other);
    CHECK_FIELD_EQ(*this, *p, first_order_buckets_count);
    CHECK_FIELD_EQ(*this, *p, second_order_buckets_count);
    return true;
}
}  // namespace vsag
