
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

#include "hgraph_index_zparameters.h"

#include <fmt/format-inl.h>

#include <utility>

#include "common.h"
#include "inner_string_params.h"
#include "utils/util_functions.h"
#include "vsag/constants.h"

// NOLINTBEGIN(readability-simplify-boolean-expr)

namespace vsag {
HGraphSearchParameters
HGraphSearchParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::parse(json_string);

    HGraphSearchParameters obj;

    // set obj.ef_search
    CHECK_ARGUMENT(params.contains(INDEX_HGRAPH),
                   fmt::format("parameters must contains {}", INDEX_HGRAPH));

    CHECK_ARGUMENT(
        params[INDEX_HGRAPH].contains(HNSW_PARAMETER_EF_RUNTIME),
        fmt::format("parameters[{}] must contains {}", INDEX_HGRAPH, HNSW_PARAMETER_EF_RUNTIME));
    obj.ef_search = params[INDEX_HGRAPH][HNSW_PARAMETER_EF_RUNTIME];
    CHECK_ARGUMENT((1 <= obj.ef_search) and (obj.ef_search <= 1000),
                   fmt::format("ef_search({}) must in range[1, 1000]", obj.ef_search));

    return obj;
}
}  // namespace vsag

// NOLINTEND(readability-simplify-boolean-expr)
