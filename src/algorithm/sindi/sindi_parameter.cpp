
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

#include "sindi_parameter.h"

#include "inner_string_params.h"

namespace vsag {

void
SINDIParameter::FromJson(const JsonType& json) {
    if (json.contains(SPARSE_DOC_PRUNE_RATIO)) {
        doc_prune_ratio = json[SPARSE_DOC_PRUNE_RATIO];
        CHECK_ARGUMENT((0.0F <= doc_prune_ratio and doc_prune_ratio <= 0.5F),
                       fmt::format("doc_prune_ratio must in [0, 0.5], got {}", doc_prune_ratio));
    } else {
        doc_prune_ratio = DEFAULT_DOC_PRUNE_RATIO;
    }

    if (json.contains(SPARSE_USE_REORDER)) {
        use_reorder = json[SPARSE_USE_REORDER];
    } else {
        use_reorder = DEFAULT_USE_REORDER;
    }

    if (json.contains(SPARSE_WINDOW_SIZE)) {
        window_size = json[SPARSE_WINDOW_SIZE];
        CHECK_ARGUMENT(
            (10'000 <= window_size and window_size <= 1'000'000),
            fmt::format("window_size must in [10000, 1000000], but now is {}", window_size));
    } else {
        window_size = DEFAULT_WINDOW_SIZE;
    }

    if (json.contains(SPARSE_DESERIALIZE_WITHOUT_FOOTER)) {
        deserialize_without_footer = json[SPARSE_DESERIALIZE_WITHOUT_FOOTER];
    }
}

JsonType
SINDIParameter::ToJson() const {
    JsonType json;
    json[SPARSE_DOC_PRUNE_RATIO] = doc_prune_ratio;
    json[SPARSE_USE_REORDER] = use_reorder;
    json[SPARSE_WINDOW_SIZE] = window_size;

    return json;
}

void
SINDISearchParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(json.contains(INDEX_SINDI),
                   fmt::format("parameters must contains {}", INDEX_SINDI));
    if (json[INDEX_SINDI].contains(SPARSE_TERM_PRUNE_RATIO)) {
        term_prune_ratio = json[INDEX_SINDI][SPARSE_TERM_PRUNE_RATIO];
        CHECK_ARGUMENT((0.0F <= term_prune_ratio and term_prune_ratio <= 0.5F),
                       fmt::format("term_prune_ratio must in [0, 0.5], got {}", term_prune_ratio));
    } else {
        term_prune_ratio = DEFAULT_TERM_PRUNE_RATIO;
    }

    if (json[INDEX_SINDI].contains(SPARSE_QUERY_PRUNE_RATIO)) {
        query_prune_ratio = json[INDEX_SINDI][SPARSE_QUERY_PRUNE_RATIO];
        CHECK_ARGUMENT(
            (0.0F <= query_prune_ratio and query_prune_ratio <= 0.5F),
            fmt::format("query_prune_ratio must in [0, 0.5], got {}", query_prune_ratio));
    } else {
        query_prune_ratio = DEFAULT_QUERY_PRUNE_RATIO;
    }
    if (json[INDEX_SINDI].contains(SPARSE_N_CANDIDATE)) {
        n_candidate = json[INDEX_SINDI][SPARSE_N_CANDIDATE];
    } else {
        n_candidate = DEFAULT_N_CANDIDATE;
    }
}
JsonType
SINDISearchParameter::ToJson() const {
    JsonType json;
    json[INDEX_SINDI] = JsonType();
    json[INDEX_SINDI][SPARSE_QUERY_PRUNE_RATIO] = query_prune_ratio;
    json[INDEX_SINDI][SPARSE_N_CANDIDATE] = n_candidate;
    json[INDEX_SINDI][SPARSE_TERM_PRUNE_RATIO] = term_prune_ratio;
    return json;
}

}  // namespace vsag
