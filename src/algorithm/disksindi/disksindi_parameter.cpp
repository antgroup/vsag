
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

#include "disksindi_parameter.h"

#include "inner_string_params.h"
#include "io/memory_block_io_parameter.h"
#include "io/reader_io_parameter.h"

namespace {

constexpr const char* DISKSINDI_TERM_IO_KEY = "term_io";
constexpr const char* DISKSINDI_RERANK_IO_KEY = "rerank_io";

}  // namespace

namespace vsag {

namespace {

bool
is_file_backed_io(const std::string& io_type_name) {
    return io_type_name == IO_TYPE_VALUE_MMAP_IO || io_type_name == IO_TYPE_VALUE_BUFFER_IO ||
           io_type_name == IO_TYPE_VALUE_ASYNC_IO;
}

JsonType
get_rerank_io_json(const JsonType& json) {
    auto rerank_io_json = json[DISKSINDI_RERANK_IO_KEY];
    auto rerank_io_type = Parameter::TryToParseType(rerank_io_json);
    if (is_file_backed_io(rerank_io_type) && not rerank_io_json.Contains(IO_FILE_PATH_KEY)) {
        CHECK_ARGUMENT(
            json.Contains(DISKSINDI_TERM_IO_KEY) &&
                json[DISKSINDI_TERM_IO_KEY].Contains(IO_FILE_PATH_KEY),
            fmt::format("rerank_io type {} requires file_path when term_io.file_path is absent",
                        rerank_io_type));
        rerank_io_json[IO_FILE_PATH_KEY].SetString(
            json[DISKSINDI_TERM_IO_KEY][IO_FILE_PATH_KEY].GetString() + ".rerank");
    }
    return rerank_io_json;
}

}  // namespace

void
DiskSINDIParameter::FromJson(const JsonType& json) {
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

    if (json.Contains(USE_QUANTIZATION)) {
        use_quantization = json[USE_QUANTIZATION].GetBool();
    } else {
        use_quantization = false;
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

    if (json.Contains(DISKSINDI_TERM_IO_KEY)) {
        term_io_parameter = IOParameter::GetIOParameterByJson(json[DISKSINDI_TERM_IO_KEY]);
        CHECK_ARGUMENT(term_io_parameter != nullptr, "invalid term_io parameter");
        CHECK_ARGUMENT(term_io_parameter->GetTypeName() != IO_TYPE_VALUE_MEMORY_IO,
                       "DiskSINDI term_io does not support memory_io");
    } else {
        term_io_parameter = std::make_shared<ReaderIOParameter>();
    }

    if (json.Contains(DISKSINDI_RERANK_IO_KEY)) {
        rerank_io_parameter = IOParameter::GetIOParameterByJson(get_rerank_io_json(json));
        CHECK_ARGUMENT(rerank_io_parameter != nullptr, "invalid rerank_io parameter");
        if (rerank_io_parameter->GetTypeName() == IO_TYPE_VALUE_MEMORY_IO) {
            rerank_io_parameter = std::make_shared<MemoryBlockIOParameter>();
        }
    } else {
        rerank_io_parameter = std::make_shared<MemoryBlockIOParameter>();
    }
}

JsonType
DiskSINDIParameter::ToJson() const {
    JsonType json;
    json[SPARSE_TERM_ID_LIMIT].SetInt(term_id_limit);
    json[SPARSE_DOC_PRUNE_RATIO].SetFloat(doc_prune_ratio);
    json[USE_REORDER_KEY].SetBool(use_reorder);
    json[USE_QUANTIZATION].SetBool(use_quantization);
    json[SPARSE_WINDOW_SIZE].SetInt(window_size);
    json[SPARSE_AVG_DOC_TERM_LENGTH].SetInt(avg_doc_term_length);
    json[SPARSE_REMAP_TERM_IDS].SetBool(remap_term_ids);
    if (term_io_parameter != nullptr) {
        json[DISKSINDI_TERM_IO_KEY].SetJson(term_io_parameter->ToJson());
    }
    if (rerank_io_parameter != nullptr) {
        json[DISKSINDI_RERANK_IO_KEY].SetJson(rerank_io_parameter->ToJson());
    }
    return json;
}

bool
DiskSINDIParameter::CheckCompatibility(const vsag::ParamPtr& other) const {
    auto disksindi_param = std::dynamic_pointer_cast<DiskSINDIParameter>(other);
    if (disksindi_param == nullptr) {
        return false;
    }
    if (this->term_id_limit != disksindi_param->term_id_limit) {
        return false;
    }
    if (this->window_size != disksindi_param->window_size) {
        return false;
    }
    if (this->doc_prune_ratio != disksindi_param->doc_prune_ratio) {
        return false;
    }
    if (this->use_reorder != disksindi_param->use_reorder) {
        return false;
    }
    if (this->use_quantization != disksindi_param->use_quantization) {
        return false;
    }
    if (this->avg_doc_term_length != disksindi_param->avg_doc_term_length) {
        return false;
    }
    if (this->remap_term_ids != disksindi_param->remap_term_ids) {
        return false;
    }
    return true;
}

void
DiskSINDISearchParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(json.Contains(INDEX_DISKSINDI),
                   fmt::format("parameters must contains {}", INDEX_DISKSINDI));
    if (json[INDEX_DISKSINDI].Contains(SPARSE_TERM_PRUNE_RATIO)) {
        term_prune_ratio = json[INDEX_DISKSINDI][SPARSE_TERM_PRUNE_RATIO].GetFloat();
        CHECK_ARGUMENT((0.0F <= term_prune_ratio and term_prune_ratio <= 0.9F),
                       fmt::format("term_prune_ratio must in [0, 0.9], got {}", term_prune_ratio));
    } else {
        term_prune_ratio = DEFAULT_TERM_PRUNE_RATIO;
    }

    if (json[INDEX_DISKSINDI].Contains(SPARSE_QUERY_PRUNE_RATIO)) {
        query_prune_ratio = json[INDEX_DISKSINDI][SPARSE_QUERY_PRUNE_RATIO].GetFloat();
        CHECK_ARGUMENT(
            (0.0F <= query_prune_ratio and query_prune_ratio <= 0.9F),
            fmt::format("query_prune_ratio must in [0, 0.9], got {}", query_prune_ratio));
    } else {
        query_prune_ratio = DEFAULT_QUERY_PRUNE_RATIO;
    }
    if (json[INDEX_DISKSINDI].Contains(SPARSE_N_CANDIDATE)) {
        n_candidate = json[INDEX_DISKSINDI][SPARSE_N_CANDIDATE].GetInt();
    } else {
        n_candidate = DEFAULT_N_CANDIDATE;
    }

    if (json[INDEX_DISKSINDI].Contains(SPARSE_USE_TERM_LISTS_HEAP_INSERT)) {
        use_term_lists_heap_insert =
            json[INDEX_DISKSINDI][SPARSE_USE_TERM_LISTS_HEAP_INSERT].GetBool();
    } else {
        use_term_lists_heap_insert = true;
    }
}
JsonType
DiskSINDISearchParameter::ToJson() const {
    JsonType json;
    json[INDEX_DISKSINDI].SetJson(JsonType());
    json[INDEX_DISKSINDI][SPARSE_QUERY_PRUNE_RATIO].SetFloat(query_prune_ratio);
    json[INDEX_DISKSINDI][SPARSE_N_CANDIDATE].SetInt(n_candidate);
    json[INDEX_DISKSINDI][SPARSE_TERM_PRUNE_RATIO].SetFloat(term_prune_ratio);
    json[INDEX_DISKSINDI][SPARSE_USE_TERM_LISTS_HEAP_INSERT].SetBool(use_term_lists_heap_insert);
    return json;
}

}  // namespace vsag
