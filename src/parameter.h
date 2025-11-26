
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

#include "common.h"
#include "inner_string_params.h"
#include "typing.h"
#include "utils/pointer_define.h"
namespace vsag {

DEFINE_POINTER2(Param, Parameter);

class Parameter {
public:
    static std::string
    TryToParseType(const JsonType& json) {
        CHECK_ARGUMENT(json.Contains(TYPE_KEY),
                       "params must have type");  // TODO(LHT): "type" rename
        return json[TYPE_KEY].GetString();
    }

public:
    Parameter() = default;

    virtual ~Parameter() = default;

    virtual void
    FromJson(const JsonType& json) = 0;

    void
    FromString(const std::string& str) {
        auto json = JsonType::Parse(str);  // TODO(LHT129): try catch
        this->FromJson(json);
    }

    virtual JsonType
    ToJson() const = 0;

    std::string
    ToString() const {
        return this->ToJson().Dump(4);
    }

    virtual bool
    CheckCompatibility(const ParamPtr& other) const {
        return this->ToString() == other->ToString();
    }
};

}  // namespace vsag
