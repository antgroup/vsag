
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

}  // namespace vsag
