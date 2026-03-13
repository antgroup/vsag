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

#include "index_registry.h"

#include <algorithm>
#include <map>
#include <string>

#include "common.h"

namespace vsag {
namespace {

auto&
index_registry() {
    static std::map<std::string, IndexCreator> registry;
    return registry;
}

std::string
normalize_index_name(std::string index_name) {
    std::transform(index_name.begin(), index_name.end(), index_name.begin(), ::tolower);
    return index_name;
}

}  // namespace

bool
register_index_creator(const std::string& index_name, IndexCreator creator) {
    auto normalized_name = normalize_index_name(index_name);
    return index_registry().insert_or_assign(normalized_name, creator).second;
}

tl::expected<std::shared_ptr<Index>, Error>
create_registered_index(const std::string& index_name,
                        JsonType& parsed_params,
                        const IndexCommonParam& index_common_params) {
    auto normalized_name = normalize_index_name(index_name);
    auto& registry = index_registry();
    auto iterator = registry.find(normalized_name);
    if (iterator == registry.end()) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::UNSUPPORTED_INDEX, "failed to create index(unsupported): ", normalized_name);
    }
    return iterator->second(parsed_params, index_common_params);
}

}  // namespace vsag
