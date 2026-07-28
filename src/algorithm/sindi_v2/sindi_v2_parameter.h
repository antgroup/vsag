
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

#pragma once

#include <string>

#include "algorithm/inner_index_parameter.h"
#include "algorithm/sindi/sindi_parameter.h"
#include "index_common_param.h"
#include "io/common/io_parameter.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER(SINDIV2Parameter);

class SINDIV2Parameter : public InnerIndexParameter {
public:
    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

    SINDIV2Parameter() = default;

public:
    // index
    uint32_t term_id_limit{0};

    uint32_t window_size{0};

    float doc_prune_ratio{0};

    bool use_reorder{false};

    bool use_quantization{false};

    SparseValueQuantizationType sparse_value_quant_type{SparseValueQuantizationType::FP32};

    bool remap_term_ids{false};

    std::string rerank_type{SPARSE_RERANK_TYPE_FP32};

    uint32_t dmq_shared_codebook_threshold{DEFAULT_SPARSE_DMQ_SHARED_CODEBOOK_THRESHOLD};

    bool immutable{false};

    uint32_t rerank_layout{0};

    uint32_t avg_doc_term_length{100};

    IOParamPtr term_io_parameter{nullptr};
    IOParamPtr rerank_io_parameter{nullptr};
};

class SINDIV2SearchParameter : public Parameter {
public:
    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    SINDIV2SearchParameter() = default;

public:
    // search
    uint32_t n_candidate{0};

    // choose heap insert path during search
    bool use_term_lists_heap_insert{true};

    // data cell
    float query_prune_ratio{0};
    float term_prune_ratio{0};
    uint64_t term_prune_threshold{0};
};

}  // namespace vsag
