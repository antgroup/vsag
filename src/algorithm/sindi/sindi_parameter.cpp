
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
SINDIParameters::FromJson(const JsonType& json) {
    if (json.contains(SPARSE_QUERY_PRUNE_RATIO)) {
        query_prune_ratio = json[SPARSE_QUERY_PRUNE_RATIO];
    } else {
        query_prune_ratio = DEFAULT_QUERY_PRUNE_RATIO;
    }

    if (json.contains(SPARSE_DOC_PRUNE_RATIO)) {
        doc_prune_ratio = json[SPARSE_DOC_PRUNE_RATIO];
    } else {
        doc_prune_ratio = DEFAULT_DOC_PRUNE_RATIO;
    }

    if (json.contains(SPARSE_TERM_PRUNE_RATIO)) {
        term_prune_ratio = json[SPARSE_TERM_PRUNE_RATIO];
    } else {
        term_prune_ratio = DEFAULT_TERM_PRUNE_RATIO;
    }

    if (json.contains(SPARSE_USE_REORDER)) {
        use_reorder = json[SPARSE_USE_REORDER];
    } else {
        use_reorder = DEFAULT_USE_REORDER;
    }

    if (json.contains(SPARSE_WINDOW_SIZE)) {
        window_size = json[SPARSE_WINDOW_SIZE];
    } else {
        window_size = DEFAULT_WINDOW_SIZE;
    }

    if (json.contains(SPARSE_N_CANDIDATE)) {
        n_candidate = json[SPARSE_N_CANDIDATE];
    } else {
        n_candidate = DEFAULT_N_CANDIDATE;
    }
}

JsonType
SINDIParameters::ToJson() const {
    JsonType json;
    json[SPARSE_QUERY_PRUNE_RATIO] = query_prune_ratio;
    json[SPARSE_DOC_PRUNE_RATIO] = doc_prune_ratio;
    json[SPARSE_TERM_PRUNE_RATIO] = term_prune_ratio;
    json[SPARSE_USE_REORDER] = use_reorder;
    json[SPARSE_WINDOW_SIZE] = window_size;
    json[SPARSE_N_CANDIDATE] = n_candidate;

    return json;
}

}  // namespace vsag
