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

#include "common.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer_parameter.h"

namespace vsag {

/**
 * Parameters of the rotated local scalar quantizer with split codes.
 *
 * The vector is first randomly rotated (the same FHT + random-sign rotation that
 * RaBitQ uses), then scalar-quantized with (x + y) bits per dimension where each
 * vector owns its FP32 minimum and FP32 step:
 *
 *   d_hat = v_min + delta * c,  c in [0, 2^(x+y))
 *
 * The code c is stored as two records: the high x bits ("filter") next to the
 * per-vector metadata, and the low y bits ("supplement") in a second storage.
 * The parameter names follow the RaBitQ x+y split configuration: `base` is x, the
 * filter bits used during traversal, and `precise` is y, the supplement bits read
 * for the full distance.
 *
 * Parameters (JSON index parameters):
 *   sq_local_bits_per_dim_base     x, filter bits, in [1, 7]
 *   sq_local_bits_per_dim_precise  y, supplement bits, in [1, 7]; x + y <= 8
 *   sq_local_bits_per_dim_query    32 => float query (default), 8 => SQ8 quantized query
 */
class SQLocalSplitQuantizerParameter : public QuantizerParameter {
public:
    explicit SQLocalSplitQuantizerParameter(
        std::string name = QUANTIZATION_TYPE_VALUE_SQ_LOCAL_SPLIT)
        : QuantizerParameter(std::move(name)) {
    }

    void
    FromJson(const JsonType& json) override {
        if (json.Contains(SQ_LOCAL_BITS_PER_DIM_BASE)) {
            CHECK_ARGUMENT(json[SQ_LOCAL_BITS_PER_DIM_BASE].IsNumberInteger(),
                           fmt::format("{} must be an integer", SQ_LOCAL_BITS_PER_DIM_BASE));
            filter_bits_ = json[SQ_LOCAL_BITS_PER_DIM_BASE].GetInt();
        }
        if (json.Contains(SQ_LOCAL_BITS_PER_DIM_PRECISE)) {
            CHECK_ARGUMENT(json[SQ_LOCAL_BITS_PER_DIM_PRECISE].IsNumberInteger(),
                           fmt::format("{} must be an integer", SQ_LOCAL_BITS_PER_DIM_PRECISE));
            supplement_bits_ = json[SQ_LOCAL_BITS_PER_DIM_PRECISE].GetInt();
        }
        if (json.Contains(SQ_LOCAL_BITS_PER_DIM_QUERY)) {
            CHECK_ARGUMENT(json[SQ_LOCAL_BITS_PER_DIM_QUERY].IsNumberInteger(),
                           fmt::format("{} must be an integer", SQ_LOCAL_BITS_PER_DIM_QUERY));
            query_bits_ = json[SQ_LOCAL_BITS_PER_DIM_QUERY].GetInt();
        }
        this->Validate();
    }

    [[nodiscard]] JsonType
    ToJson() const override {
        JsonType json;
        json[TYPE_KEY].SetString(this->GetTypeName());
        json[SQ_LOCAL_BITS_PER_DIM_BASE].SetInt(filter_bits_);
        json[SQ_LOCAL_BITS_PER_DIM_PRECISE].SetInt(supplement_bits_);
        json[SQ_LOCAL_BITS_PER_DIM_QUERY].SetInt(query_bits_);
        return json;
    }

    void
    Validate() const {
        CHECK_ARGUMENT(
            filter_bits_ >= 1 and filter_bits_ <= 7,
            fmt::format(
                "{} must be in [1, 7], but got {}", SQ_LOCAL_BITS_PER_DIM_BASE, filter_bits_));
        CHECK_ARGUMENT(supplement_bits_ >= 1 and supplement_bits_ <= 7,
                       fmt::format("{} must be in [1, 7], but got {}",
                                   SQ_LOCAL_BITS_PER_DIM_PRECISE,
                                   supplement_bits_));
        CHECK_ARGUMENT(filter_bits_ + supplement_bits_ <= 8,
                       fmt::format("{} ({}) + {} ({}) must not exceed 8",
                                   SQ_LOCAL_BITS_PER_DIM_BASE,
                                   filter_bits_,
                                   SQ_LOCAL_BITS_PER_DIM_PRECISE,
                                   supplement_bits_));
        CHECK_ARGUMENT(query_bits_ == 32 or query_bits_ == 8,
                       fmt::format("{} must be 32 (float query) or 8 (SQ8 query), but got {}",
                                   SQ_LOCAL_BITS_PER_DIM_QUERY,
                                   query_bits_));
    }

    /** x: filter bits, stored with the per-vector metadata. */
    [[nodiscard]] uint64_t
    FilterBits() const {
        return filter_bits_;
    }

    /** y: supplement bits, stored in the second record. */
    [[nodiscard]] uint64_t
    PreciseBits() const {
        return supplement_bits_;
    }

    [[nodiscard]] uint64_t
    SupplementBits() const {
        return supplement_bits_;
    }

    [[nodiscard]] uint64_t
    BaseBits() const {
        return filter_bits_ + supplement_bits_;
    }

    /** 32 = float query, 8 = SQ8 quantized query. */
    [[nodiscard]] uint64_t
    QueryBits() const {
        return query_bits_;
    }

public:
    uint64_t filter_bits_{3};
    uint64_t supplement_bits_{5};
    uint64_t query_bits_{32};
};

using SQLocalSplitQuantizerParamPtr = std::shared_ptr<SQLocalSplitQuantizerParameter>;

}  // namespace vsag
