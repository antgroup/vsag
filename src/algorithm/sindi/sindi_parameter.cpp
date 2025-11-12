
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
    if (json.Contains(SPARSE_TERM_ID_LIMIT)) {
        term_id_limit = json[SPARSE_TERM_ID_LIMIT].GetInt();

        CHECK_ARGUMENT(
            (0 < term_id_limit and term_id_limit <= 10'000'000),
            fmt::format("term_id_limit must in (0, 10'000'000], but now is {}", term_id_limit));
    } else {
        term_id_limit = DEFAULT_TERM_ID_LIMIT;
    }

    if (json.Contains(SPARSE_DOC_PRUNE_RATIO)) {
        doc_prune_ratio = json[SPARSE_DOC_PRUNE_RATIO].GetFloat();
        CHECK_ARGUMENT((0.0F <= doc_prune_ratio and doc_prune_ratio <= 0.9F),
                       fmt::format("doc_prune_ratio must in [0, 0.9], got {}", doc_prune_ratio));
    } else {
        doc_prune_ratio = DEFAULT_DOC_PRUNE_RATIO;
    }

    if (json.Contains(USE_REORDER_KEY)) {
        use_reorder = json[USE_REORDER_KEY].GetBool();
    } else {
        use_reorder = DEFAULT_USE_REORDER;
    }

    if (json.Contains(SPARSE_WINDOW_SIZE)) {
        window_size = json[SPARSE_WINDOW_SIZE].GetInt();
        CHECK_ARGUMENT(
            (10'000 <= window_size and window_size <= 1'000'000),
            fmt::format("window_size must in [10000, 1000000], but now is {}", window_size));
    } else {
        window_size = DEFAULT_WINDOW_SIZE;
    }

    if (json.Contains(SPARSE_DESERIALIZE_WITHOUT_FOOTER)) {
        deserialize_without_footer = json[SPARSE_DESERIALIZE_WITHOUT_FOOTER].GetBool();
    }
}

JsonType
SINDIParameter::ToJson() const {
    JsonType json;
    json[SPARSE_TERM_ID_LIMIT].SetInt(term_id_limit);
    json[SPARSE_DOC_PRUNE_RATIO].SetFloat(doc_prune_ratio);
    json[USE_REORDER_KEY].SetBool(use_reorder);
    json[SPARSE_WINDOW_SIZE].SetInt(window_size);
    return json;
}

bool
SINDIParameter::CheckCompatibility(const vsag::ParamPtr& other) const {
    auto sindi_param = std::dynamic_pointer_cast<SINDIParameter>(other);
    if (sindi_param == nullptr) {
        return false;
    }
    if (this->term_id_limit != sindi_param->term_id_limit) {
        return false;
    }
    if (this->window_size != sindi_param->window_size) {
        return false;
    }
    if (this->doc_prune_ratio != sindi_param->doc_prune_ratio) {
        return false;
    }
    if (this->use_reorder != sindi_param->use_reorder) {
        return false;
    }
    return true;
}

void
SINDISearchParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(json.Contains(INDEX_SINDI),
                   fmt::format("parameters must contains {}", INDEX_SINDI));
    if (json[INDEX_SINDI].Contains(SPARSE_TERM_PRUNE_RATIO)) {
        term_prune_ratio = json[INDEX_SINDI][SPARSE_TERM_PRUNE_RATIO].GetFloat();
        CHECK_ARGUMENT((0.0F <= term_prune_ratio and term_prune_ratio <= 0.9F),
                       fmt::format("term_prune_ratio must in [0, 0.9], got {}", term_prune_ratio));
    } else {
        term_prune_ratio = DEFAULT_TERM_PRUNE_RATIO;
    }

    if (json[INDEX_SINDI].Contains(SPARSE_QUERY_PRUNE_RATIO)) {
        query_prune_ratio = json[INDEX_SINDI][SPARSE_QUERY_PRUNE_RATIO].GetFloat();
        CHECK_ARGUMENT(
            (0.0F <= query_prune_ratio and query_prune_ratio <= 0.9F),
            fmt::format("query_prune_ratio must in [0, 0.9], got {}", query_prune_ratio));
    } else {
        query_prune_ratio = DEFAULT_QUERY_PRUNE_RATIO;
    }
    if (json[INDEX_SINDI].Contains(SPARSE_N_CANDIDATE)) {
        n_candidate = json[INDEX_SINDI][SPARSE_N_CANDIDATE].GetInt();
    } else {
        n_candidate = DEFAULT_N_CANDIDATE;
    }
}
JsonType
SINDISearchParameter::ToJson() const {
    JsonType json;
    json[INDEX_SINDI].SetJson(JsonType());
    json[INDEX_SINDI][SPARSE_QUERY_PRUNE_RATIO].SetFloat(query_prune_ratio);
    json[INDEX_SINDI][SPARSE_N_CANDIDATE].SetInt(n_candidate);
    json[INDEX_SINDI][SPARSE_TERM_PRUNE_RATIO].SetFloat(term_prune_ratio);
    return json;
}

}  // namespace vsag
