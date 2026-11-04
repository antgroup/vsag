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

#include "index/search_session_provider.h"
#include "vsag/index.h"

namespace vsag {

tl::expected<std::unique_ptr<SearchSession>, Error>
Index::OpenSearchSession(const DatasetPtr& query,
                         int64_t k_per_call,
                         const std::string& parameters,
                         const FilterPtr& filter,
                         Allocator* allocator) const {
    const auto* provider = dynamic_cast<const SearchSessionProvider*>(this);
    if (provider == nullptr) {
        return tl::unexpected(
            Error(ErrorType::UNSUPPORTED_INDEX_OPERATION, "Index does not support SearchSession"));
    }
    return provider->CreateSearchSession(query, k_per_call, parameters, filter, allocator);
}

}  // namespace vsag
