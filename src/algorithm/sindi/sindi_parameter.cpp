
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

#include "impl/logger/logger.h"
#include "inner_string_params.h"
#include "utils/param_compat_macros.h"

namespace vsag {
std::string
SparseValueQuantizationTypeToString(SparseValueQuantizationType type) {
    switch (type) {
        case SparseValueQuantizationType::FP32:
            return QUANTIZATION_TYPE_VALUE_FP32;
        case SparseValueQuantizationType::SQ8:
            return QUANTIZATION_TYPE_VALUE_SQ8;
        case SparseValueQuantizationType::FP16:
            return QUANTIZATION_TYPE_VALUE_FP16;
        default:
            CHECK_ARGUMENT(false, "unknown sparse value quantization type");
    }
    return QUANTIZATION_TYPE_VALUE_FP32;
}

static SparseValueQuantizationType
parse_sparse_value_quant_type(const std::string& type_name) {
    if (type_name == QUANTIZATION_TYPE_VALUE_FP32) {
        return SparseValueQuantizationType::FP32;
    }
    if (type_name == QUANTIZATION_TYPE_VALUE_SQ8) {
        return SparseValueQuantizationType::SQ8;
    }
    if (type_name == QUANTIZATION_TYPE_VALUE_FP16) {
        return SparseValueQuantizationType::FP16;
    }
    CHECK_ARGUMENT(
        false, fmt::format("use_quantization must be false, true, or fp16, but got {}", type_name));
    return SparseValueQuantizationType::FP32;
}

namespace {

constexpr auto LEGACY_USE_TERM_LISTS_HEAP_INSERT_KEY = "use_term_lists_heap_insert";

}  // namespace

void
SINDIParameter::FromJson(const JsonType& json) {
    if (json.Contains(SPARSE_TERM_ID_LIMIT)) {
        term_id_limit = json[SPARSE_TERM_ID_LIMIT].GetInt();

        CHECK_ARGUMENT(
            (0 < term_id_limit and term_id_limit <= 50'000'000),
            fmt::format("term_id_limit must be in (0, 50'000'000], but got {}", term_id_limit));
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

    sparse_value_quant_type = SparseValueQuantizationType::FP32;
    if (json.Contains(USE_QUANTIZATION)) {
        auto use_quantization = json[USE_QUANTIZATION];
        if (use_quantization.IsString()) {
            sparse_value_quant_type = parse_sparse_value_quant_type(use_quantization.GetString());
        } else {
            CHECK_ARGUMENT(use_quantization.IsBool(),
                           "use_quantization must be false, true, or fp16");
            if (use_quantization.GetBool()) {
                sparse_value_quant_type = SparseValueQuantizationType::SQ8;
            }
        }
    }

    if (json.Contains(SPARSE_WINDOW_SIZE)) {
        window_size = json[SPARSE_WINDOW_SIZE].GetInt();
        CHECK_ARGUMENT(
            (10'000 <= window_size and window_size <= 60'000),
            fmt::format("window_size must in [10000, 60000], but now is {}", window_size));
    } else {
        window_size = DEFAULT_WINDOW_SIZE;
    }

    if (json.Contains(SPARSE_AVG_DOC_TERM_LENGTH)) {
        avg_doc_term_length = json[SPARSE_AVG_DOC_TERM_LENGTH].GetInt();
        CHECK_ARGUMENT((0 < avg_doc_term_length),
                       fmt::format("avg_doc_term_length must be greater than 0, but now is {}",
                                   avg_doc_term_length));
    } else {
        avg_doc_term_length = DEFAULT_AVG_DOC_TERM_LENGTH;
    }

    if (json.Contains(SPARSE_DESERIALIZE_WITHOUT_FOOTER)) {
        deserialize_without_footer = json[SPARSE_DESERIALIZE_WITHOUT_FOOTER].GetBool();
    }

    if (json.Contains(SPARSE_DESERIALIZE_WITHOUT_BUFFER)) {
        deserialize_without_buffer = json[SPARSE_DESERIALIZE_WITHOUT_BUFFER].GetBool();
    }

    if (json.Contains(SPARSE_REMAP_TERM_IDS)) {
        remap_term_ids = json[SPARSE_REMAP_TERM_IDS].GetBool();
    }

    if (json.Contains(SPARSE_RERANK_TYPE)) {
        rerank_type = json[SPARSE_RERANK_TYPE].GetString();
    } else {
        rerank_type = SPARSE_RERANK_TYPE_FP32;
    }
    CHECK_ARGUMENT(rerank_type == SPARSE_RERANK_TYPE_FP32 || rerank_type == SPARSE_RERANK_TYPE_DMQ8,
                   fmt::format("rerank_type must be fp32 or dmq8, got {}", rerank_type));
    CHECK_ARGUMENT(use_reorder || rerank_type == SPARSE_RERANK_TYPE_FP32,
                   "rerank_type=dmq8 requires use_reorder=true");

    if (json.Contains(SPARSE_IMMUTABLE)) {
        immutable = json[SPARSE_IMMUTABLE].GetBool();
    }
}

JsonType
SINDIParameter::ToJson() const {
    JsonType json;
    json[SPARSE_TERM_ID_LIMIT].SetInt(term_id_limit);
    json[SPARSE_DOC_PRUNE_RATIO].SetFloat(doc_prune_ratio);
    json[USE_REORDER_KEY].SetBool(use_reorder);
    if (sparse_value_quant_type == SparseValueQuantizationType::FP16) {
        json[USE_QUANTIZATION].SetString(QUANTIZATION_TYPE_VALUE_FP16);
    } else {
        json[USE_QUANTIZATION].SetBool(sparse_value_quant_type == SparseValueQuantizationType::SQ8);
    }
    json[SPARSE_WINDOW_SIZE].SetInt(window_size);
    json[SPARSE_AVG_DOC_TERM_LENGTH].SetInt(avg_doc_term_length);
    json[SPARSE_REMAP_TERM_IDS].SetBool(remap_term_ids);
    if (immutable) {
        json[SPARSE_IMMUTABLE].SetBool(true);
    }
    json[SPARSE_RERANK_TYPE].SetString(rerank_type);
    return json;
}

bool
SINDIParameter::CheckCompatibility(const vsag::ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(SINDIParameter, p, other);
    CHECK_FIELD_EQ(*this, *p, term_id_limit);
    CHECK_FIELD_EQ(*this, *p, window_size);
    CHECK_FIELD_EQ(*this, *p, doc_prune_ratio);
    CHECK_FIELD_EQ(*this, *p, use_reorder);
    CHECK_FIELD_EQ(*this, *p, sparse_value_quant_type);
    CHECK_FIELD_EQ(*this, *p, avg_doc_term_length);
    CHECK_FIELD_EQ(*this, *p, remap_term_ids);
    CHECK_FIELD_EQ(*this, *p, immutable);
    CHECK_FIELD_EQ(*this, *p, rerank_type);
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

    if (json[INDEX_SINDI].Contains(LEGACY_USE_TERM_LISTS_HEAP_INSERT_KEY)) {
        logger::warn(
            "SINDI search parameter use_term_lists_heap_insert is ignored. "
            "Remove this key; heap insertion is derived from doc_prune_ratio "
            "and query_prune_ratio with the current SINDI prune-ratio threshold");
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
