
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

#include <cstring>
#include <vector>

#include "index_common_param.h"
#include "inner_string_params.h"
#include "algorithm/sparse_distance.h"
#include "quantization/quantizer.h"
#include "quantization/quantizer_parameter.h"
#include "sparse_quantizer_parameter.h"
#include "vsag/dataset.h"

namespace vsag {

struct BufferEntry {
    uint32_t id;
    float val;
};

/***
 * @brief Sparse Quantizer stores sparse vectors with only non-zero elements.
 *
 * code layout:
 * +----------+------------------+------------------+-----+------------------+
 * | len      | ids[0] | ids[1] | ... | ids[len-1] | vals[0] | ... | vals[len-1] |
 * | [4B]     | [4B]   | [4B]   |     | [4B]       | [4B]    |     | [4B]        |
 * +----------+------------------+------------------+-----+------------------+
 *
 * - len: number of non-zero elements (uint32)
 * - ids: dimension indexes sorted in ascending order
 * - vals: values matching ids by position
 * - Only supports METRIC_TYPE_IP (inner product)
 */
template <MetricType metric = MetricType::METRIC_TYPE_IP>
class SparseQuantizer : public Quantizer<SparseQuantizer<metric>> {
public:
    explicit SparseQuantizer(const SparseQuantizerParamPtr& param,
                             const IndexCommonParam& common_param);
    explicit SparseQuantizer(Allocator* allocator);

    explicit SparseQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    bool
    TrainImpl(const float* data, uint64_t count);

    bool
    EncodeOneImpl(const float* data, uint8_t* codes) const;

    bool
    EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, float* data);

    bool
    DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count);

    inline float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    inline void
    ProcessQueryImpl(const float* query, Computer<SparseQuantizer>& computer) const;

    inline void
    ComputeDistImpl(Computer<SparseQuantizer>& computer, const uint8_t* codes, float* dists) const;

    inline void
    ScanBatchDistImpl(Computer<SparseQuantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    inline void
    ReleaseComputerImpl(Computer<SparseQuantizer<metric>>& computer) const;

    inline void
    SerializeImpl(StreamWriter& writer);

    inline void
    DeserializeImpl(StreamReader& reader);

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_SPARSE;
    }
};
template <MetricType metric>
SparseQuantizer<metric>::SparseQuantizer(const QuantizerParamPtr& param,
                                         const IndexCommonParam& common_param)
    : SparseQuantizer<metric>(std::dynamic_pointer_cast<SparseQuantizerParameter>(param),
                              common_param) {
}
template <MetricType metric>
SparseQuantizer<metric>::SparseQuantizer(const SparseQuantizerParamPtr& param,
                                         const IndexCommonParam& common_param)
    : SparseQuantizer<metric>(common_param.allocator_.get()) {
    this->dim_ = common_param.dim_;
    this->code_size_ = common_param.dim_ * sizeof(float);
}

template <MetricType metric>
SparseQuantizer<metric>::SparseQuantizer(Allocator* allocator)
    : Quantizer<SparseQuantizer<metric>>(0, allocator) {
    this->metric_ = metric;
    this->is_trained_ = false;
}

template <MetricType metric>
void
SparseQuantizer<metric>::DeserializeImpl(StreamReader& reader) {
}

template <MetricType metric>
void
SparseQuantizer<metric>::SerializeImpl(StreamWriter& writer) {
}

template <MetricType metric>
void
SparseQuantizer<metric>::ReleaseComputerImpl(Computer<SparseQuantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

template <MetricType metric>
void
SparseQuantizer<metric>::ScanBatchDistImpl(Computer<SparseQuantizer<metric>>& computer,
                                           uint64_t count,
                                           const uint8_t* codes,
                                           float* dists) const {
    // it is impossible to get batch codes of sparse vector
    throw VsagException(ErrorType::INTERNAL_ERROR,
                        "no support for ScanBatchDistImpl in sparse quantizer");
}
template <MetricType metric>
void
SparseQuantizer<metric>::ComputeDistImpl(Computer<SparseQuantizer>& computer,
                                         const uint8_t* codes,
                                         float* dists) const {
    dists[0] = ComputeImpl(computer.buf_, codes);
}

template <MetricType metric>
void
SparseQuantizer<metric>::ProcessQueryImpl(const float* query,
                                          Computer<SparseQuantizer>& computer) const {
    const auto* sparse_query = reinterpret_cast<const SparseVector*>(query);
    try {
        computer.buf_ = reinterpret_cast<uint8_t*>(this->allocator_->Allocate(
            sizeof(uint32_t) + sparse_query->len_ * sizeof(BufferEntry)));
    } catch (const std::bad_alloc& e) {
        computer.buf_ = nullptr;
        logger::error("bad alloc when init computer buf");
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                            "bad alloc when init computer buf in sparse quantizer");
    }
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        EncodeOneImpl(query, computer.buf_);
    } else {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "no support for other metric type in sparse quantizer");
    }
}

template <MetricType metric>
float
SparseQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    if constexpr (metric != MetricType::METRIC_TYPE_IP) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "no support for other metric type in sparse quantizer");
    }
    uint32_t len1 = 0;
    std::memcpy(&len1, codes1, sizeof(len1));
    const auto* ids1 = codes1 + sizeof(uint32_t);
    const auto* vals1 = codes1 + sizeof(uint32_t) + len1 * sizeof(uint32_t);

    uint32_t len2 = 0;
    std::memcpy(&len2, codes2, sizeof(len2));
    const auto* ids2 = codes2 + sizeof(uint32_t);
    const auto* vals2 = codes2 + sizeof(uint32_t) + len2 * sizeof(uint32_t);
    float inner_product = 0.0f;
    uint32_t idx1 = 0, idx2 = 0;
    while (idx1 < len1 && idx2 < len2) {
        auto id1 = read_sparse_code_id(ids1, idx1);
        auto id2 = read_sparse_code_id(ids2, idx2);
        if (id1 < id2) {
            idx1++;
        } else if (id1 > id2) {
            idx2++;
        } else {
            inner_product += read_sparse_code_val(vals1, idx1) * read_sparse_code_val(vals2, idx2);
            idx1++;
            idx2++;
        }
    }
    return 1 - inner_product;
}

template <MetricType metric>
bool
SparseQuantizer<metric>::DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count) {
    throw VsagException(ErrorType::INTERNAL_ERROR, "no support for decode in sparse quantizer");
}

template <MetricType metric>
bool
SparseQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, float* data) {
    throw VsagException(ErrorType::INTERNAL_ERROR, "no support for decode in sparse quantizer");
}

template <MetricType metric>
bool
SparseQuantizer<metric>::EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count) {
    throw VsagException(ErrorType::INTERNAL_ERROR,
                        "no support for batch encode in sparse quantizer");
}

template <MetricType metric>
bool
SparseQuantizer<metric>::EncodeOneImpl(const float* data, uint8_t* codes) const {
    const SparseVector& sv = *reinterpret_cast<const SparseVector*>(data);
    std::memcpy(codes, &sv.len_, sizeof(sv.len_));
    std::vector<BufferEntry> entries(sv.len_);
    for (uint32_t i = 0; i < sv.len_; ++i) {
        entries[i].id = sv.ids_[i];
        entries[i].val = sv.vals_[i];
    }
    std::sort(entries.begin(), entries.end(), [](const BufferEntry& a, const BufferEntry& b) {
        return a.id < b.id;
    });
    auto* ids = codes + sizeof(uint32_t);
    auto* vals = codes + sizeof(uint32_t) + sv.len_ * sizeof(uint32_t);
    for (uint32_t i = 0; i < sv.len_; ++i) {
        std::memcpy(ids + i * sizeof(entries[i].id), &entries[i].id, sizeof(entries[i].id));
        std::memcpy(vals + i * sizeof(entries[i].val), &entries[i].val, sizeof(entries[i].val));
    }
    return true;
}

template <MetricType metric>
bool
SparseQuantizer<metric>::TrainImpl(const float* data, uint64_t count) {
    this->is_trained_ = true;
    return true;
}

}  // namespace vsag
