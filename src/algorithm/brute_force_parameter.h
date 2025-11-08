
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
#include "index_search_parameter.h"
#include "inner_index_parameter.h"
#include "typing.h"
#include "utils/pointer_define.h"
namespace vsag {
DEFINE_POINTER2(FlattenDataCellParam, FlattenDataCellParameter);
class BruteForceParameter : public InnerIndexParameter {
public:
    explicit BruteForceParameter();

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

public:
    FlattenInterfaceParamPtr base_codes_param{nullptr};
};

DEFINE_POINTER(BruteForceParameter);

class BruteForceSearchParameters : public IndexSearchParameter {
public:
    static BruteForceSearchParameters
    FromJson(const std::string& json_string) {
        if (json_string.empty()) {
            return BruteForceSearchParameters();
        }
        auto params = JsonType::Parse(json_string);
        BruteForceSearchParameters obj;
        obj.IndexSearchParameter::FromJson(params);
        return obj;
    }
};

}  // namespace vsag
