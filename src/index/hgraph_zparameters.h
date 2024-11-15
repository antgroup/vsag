
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

#include <nlohmann/json.hpp>
#include <string>

#include "index_common_param.h"
#include "typing.h"

namespace vsag {
class HGraphParameters {
public:
    explicit HGraphParameters(const IndexCommonParam& common_param, const std::string& str = "");

    void
    ParseStringParam(const std::string& str);

    void
    CheckAndSetKeyValue(const std::string& key, JsonType& value);

    std::string
    GetString() {
        this->refresh_string_by_json();
        return this->str_;
    }

    JsonType
    GetJson() {
        this->refresh_json_by_string();
        return this->json_;
    }

private:
    inline void
    refresh_json_by_string() {
        this->json_ = JsonType::parse(str_);
    }

    inline void
    refresh_string_by_json() {
        this->str_ = this->json_.dump();
    }

    void
    check_common_param() const;

private:
    JsonType json_;

    std::string str_{DEFAULT_HGRAPH_PARAMS};

    const IndexCommonParam common_param_;

    static const std::string DEFAULT_HGRAPH_PARAMS;

    static const std::unordered_map<std::string, std::vector<std::string>> EXTERNAL_MAPPING;
};

}  // namespace vsag
