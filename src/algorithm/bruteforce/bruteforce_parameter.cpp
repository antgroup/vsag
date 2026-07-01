
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

#include "bruteforce_parameter.h"

#include <unordered_set>

#include "datacell/flatten_datacell_parameter.h"
#include "inner_string_params.h"
#include "utils/param_compat_macros.h"
#include "vsag/constants.h"

namespace vsag {

VSAG_PARAM_SCHEMA(BruteForceParameter)
VSAG_PARAM_TYPE_TAG(TYPE_KEY, INDEX_TYPE_BRUTE_FORCE)
VSAG_PARAM_SUBPARAM_REQUIRED(FlattenInterfaceParameter,
                             base_codes_param,
                             BASE_CODES_KEY,
                             CreateFlattenParam)
VSAG_PARAM_SCHEMA_END()

BruteForceParameter::BruteForceParameter() : base_codes_param(nullptr) {
}

void
BruteForceParameter::FromJson(const JsonType& json) {
    InnerIndexParameter::FromJson(json);
    schema().Parse(this, json);
}

JsonType
BruteForceParameter::ToJson() const {
    JsonType json = InnerIndexParameter::ToJson();
    schema().Serialize(this, json);
    return json;
}

bool
BruteForceParameter::CheckCompatibility(const ParamPtr& other) const {
    if (not InnerIndexParameter::CheckCompatibility(other)) {
        return false;
    }
    PARAM_CAST_OR_RETURN(BruteForceParameter, p, other);
    return schema().Equal(this, p.get());
}
}  // namespace vsag
