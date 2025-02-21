
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

#include "util_functions.h"

namespace vsag {

std::string
format_map(const std::string& str, const std::unordered_map<std::string, std::string>& mappings) {
    std::string result = str;

    for (const auto& [key, value] : mappings) {
        size_t pos = result.find("{" + key + "}");
        while (pos != std::string::npos) {
            result.replace(pos, key.length() + 2, value);
            pos = result.find("{" + key + "}");
        }
    }
    return result;
}

void
mapping_external_param_to_inner(const JsonType& external_json,
                                ConstParamMap& param_map,
                                JsonType& inner_json) {
    for (const auto& [key, value] : external_json.items()) {
        const auto& iter = param_map.find(key);

        if (iter != param_map.end()) {
            const auto& vec = iter->second;
            auto* json = &inner_json;
            for (const auto& str : vec) {
                json = &(json->operator[](str));
            }
            *json = value;
        } else {
            throw std::invalid_argument(fmt::format("HGraph have no config param: {}", key));
        }
    }
}

}  // namespace vsag
