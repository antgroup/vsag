
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

#include <algorithm>

#include "container_types.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "quantization/quantizer_parameter.h"
#include "simd/fp16_simd.h"
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
 * FP32 code layout (also used for queries):
 * +----------+------------------+------------------+-----+------------------+
 * | len      | entry[0]         | entry[1]         | ... | entry[len-1]     |
 * | [4B]     | id[4B] + val[4B] | id[4B] + val[4B] |     | id[4B] + val[4B] |
 * +----------+------------------+------------------+-----+------------------+
 *
 * FP16 base code layout:
 * +----------+------------------+------------------+
 * | len      | ids              | values           |
 * | [4B]     | uint32[len]      | uint16[len]      |
 * +----------+------------------+------------------+
 *
 * - len: number of non-zero elements (uint32)
 * - entry[i]: sorted by id in ascending order
 *   - id: dimension index (uint32)
 *   - val: value at this dimension (float for FP32, binary16 for FP16 base codes)
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
        return IsFP16() ? QUANTIZATION_TYPE_VALUE_SPARSE_FP16 : QUANTIZATION_TYPE_VALUE_SPARSE;
    }

    [[nodiscard]] bool
    IsFP16() const {
        return value_type_ == SparseQuantizerValueType::FP16;
    }

    [[nodiscard]] SparseQuantizerValueType
    GetValueType() const {
        return value_type_;
    }

    [[nodiscard]] uint64_t
    GetCodeSizeByLength(uint32_t length) const;

    void
    DecodeSparseVector(const uint8_t* codes, uint32_t* ids, float* values) const;

private:
    static uint64_t
    GetFP32CodeSize(uint32_t length);

    static float
    ComputeFP32Codes(const uint8_t* codes1, const uint8_t* codes2);

    static float
    ComputeFP32QueryToFP16Codes(const uint8_t* query_codes, const uint8_t* base_codes);

    static float
    ComputeFP16Codes(const uint8_t* codes1, const uint8_t* codes2);

    void
    EncodeFP32(const SparseVector& sparse_vector, uint8_t* codes) const;

private:
    SparseQuantizerValueType value_type_{SparseQuantizerValueType::FP32};
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
    CHECK_ARGUMENT(param != nullptr, "invalid sparse quantizer parameter");
    value_type_ = param->value_type;
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
    dists[0] = IsFP16() ? ComputeFP32QueryToFP16Codes(computer.buf_, codes)
                        : ComputeFP32Codes(computer.buf_, codes);
}

template <MetricType metric>
void
SparseQuantizer<metric>::ProcessQueryImpl(const float* query,
                                          Computer<SparseQuantizer>& computer) const {
    const auto* sparse_query = reinterpret_cast<const SparseVector*>(query);
    try {
        computer.buf_ = reinterpret_cast<uint8_t*>(
            this->allocator_->Allocate(GetFP32CodeSize(sparse_query->len_)));
    } catch (const std::bad_alloc& e) {
        computer.buf_ = nullptr;
        logger::error("bad alloc when init computer buf");
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                            "bad alloc when init computer buf in sparse quantizer");
    }
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        EncodeFP32(*sparse_query, computer.buf_);
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
    return IsFP16() ? ComputeFP16Codes(codes1, codes2) : ComputeFP32Codes(codes1, codes2);
}

template <MetricType metric>
float
SparseQuantizer<metric>::ComputeFP32Codes(const uint8_t* codes1, const uint8_t* codes2) {
    const uint32_t len1 = *reinterpret_cast<const uint32_t*>(codes1);
    const auto* entries1 = reinterpret_cast<const BufferEntry*>(codes1 + sizeof(uint32_t));
    const uint32_t len2 = *reinterpret_cast<const uint32_t*>(codes2);
    const auto* entries2 = reinterpret_cast<const BufferEntry*>(codes2 + sizeof(uint32_t));
    float inner_product = 0.0f;
    uint32_t idx1 = 0, idx2 = 0;
    while (idx1 < len1 && idx2 < len2) {
        if (entries1[idx1].id < entries2[idx2].id) {
            idx1++;
        } else if (entries1[idx1].id > entries2[idx2].id) {
            idx2++;
        } else {
            inner_product += entries1[idx1].val * entries2[idx2].val;
            idx1++;
            idx2++;
        }
    }
    return 1.0F - inner_product;
}

template <MetricType metric>
float
SparseQuantizer<metric>::ComputeFP32QueryToFP16Codes(const uint8_t* query_codes,
                                                     const uint8_t* base_codes) {
    const uint32_t query_length = *reinterpret_cast<const uint32_t*>(query_codes);
    const auto* query_entries =
        reinterpret_cast<const BufferEntry*>(query_codes + sizeof(uint32_t));
    const uint32_t base_length = *reinterpret_cast<const uint32_t*>(base_codes);
    const auto* base_ids = reinterpret_cast<const uint32_t*>(base_codes + sizeof(uint32_t));
    const auto* base_values = reinterpret_cast<const uint16_t*>(
        base_codes + sizeof(uint32_t) + static_cast<uint64_t>(base_length) * sizeof(uint32_t));
    float inner_product = 0.0F;
    uint32_t query_index = 0;
    uint32_t base_index = 0;
    while (query_index < query_length && base_index < base_length) {
        if (query_entries[query_index].id < base_ids[base_index]) {
            ++query_index;
        } else if (query_entries[query_index].id > base_ids[base_index]) {
            ++base_index;
        } else {
            inner_product +=
                query_entries[query_index].val * generic::FP16ToFloat(base_values[base_index]);
            ++query_index;
            ++base_index;
        }
    }
    return 1.0F - inner_product;
}

template <MetricType metric>
float
SparseQuantizer<metric>::ComputeFP16Codes(const uint8_t* codes1, const uint8_t* codes2) {
    const uint32_t length1 = *reinterpret_cast<const uint32_t*>(codes1);
    const auto* ids1 = reinterpret_cast<const uint32_t*>(codes1 + sizeof(uint32_t));
    const auto* values1 = reinterpret_cast<const uint16_t*>(
        codes1 + sizeof(uint32_t) + static_cast<uint64_t>(length1) * sizeof(uint32_t));
    const uint32_t length2 = *reinterpret_cast<const uint32_t*>(codes2);
    const auto* ids2 = reinterpret_cast<const uint32_t*>(codes2 + sizeof(uint32_t));
    const auto* values2 = reinterpret_cast<const uint16_t*>(
        codes2 + sizeof(uint32_t) + static_cast<uint64_t>(length2) * sizeof(uint32_t));
    float inner_product = 0.0F;
    uint32_t index1 = 0;
    uint32_t index2 = 0;
    while (index1 < length1 && index2 < length2) {
        if (ids1[index1] < ids2[index2]) {
            ++index1;
        } else if (ids1[index1] > ids2[index2]) {
            ++index2;
        } else {
            inner_product +=
                generic::FP16ToFloat(values1[index1]) * generic::FP16ToFloat(values2[index2]);
            ++index1;
            ++index2;
        }
    }
    return 1.0F - inner_product;
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
    if (not IsFP16()) {
        EncodeFP32(sv, codes);
        return true;
    }

    *reinterpret_cast<uint32_t*>(codes) = sv.len_;
    auto* ids = reinterpret_cast<uint32_t*>(codes + sizeof(uint32_t));
    auto* values = reinterpret_cast<uint16_t*>(codes + sizeof(uint32_t) +
                                               static_cast<uint64_t>(sv.len_) * sizeof(uint32_t));
    const uint64_t encoded_size =
        sizeof(uint32_t) + static_cast<uint64_t>(sv.len_) * (sizeof(uint32_t) + sizeof(uint16_t));
    std::fill(codes + encoded_size, codes + GetCodeSizeByLength(sv.len_), 0);
    if (sv.len_ < 2 || std::is_sorted(sv.ids_, sv.ids_ + sv.len_)) {
        for (uint32_t i = 0; i < sv.len_; ++i) {
            ids[i] = sv.ids_[i];
            values[i] = generic::FloatToFP16(sv.vals_[i]);
        }
        return true;
    }

    Vector<BufferEntry> entries(sv.len_, this->allocator_);
    for (uint32_t i = 0; i < sv.len_; ++i) {
        entries[i].id = sv.ids_[i];
        entries[i].val = sv.vals_[i];
    }
    std::sort(entries.begin(), entries.end(), [](const BufferEntry& a, const BufferEntry& b) {
        return a.id < b.id;
    });
    for (uint32_t i = 0; i < sv.len_; ++i) {
        ids[i] = entries[i].id;
        values[i] = generic::FloatToFP16(entries[i].val);
    }
    return true;
}

template <MetricType metric>
void
SparseQuantizer<metric>::EncodeFP32(const SparseVector& sparse_vector, uint8_t* codes) const {
    *reinterpret_cast<uint32_t*>(codes) = sparse_vector.len_;
    auto* entries = reinterpret_cast<BufferEntry*>(codes + sizeof(uint32_t));
    for (uint32_t i = 0; i < sparse_vector.len_; ++i) {
        entries[i].id = sparse_vector.ids_[i];
        entries[i].val = sparse_vector.vals_[i];
    }
    std::sort(entries,
              entries + sparse_vector.len_,
              [](const BufferEntry& left, const BufferEntry& right) { return left.id < right.id; });
}

template <MetricType metric>
uint64_t
SparseQuantizer<metric>::GetFP32CodeSize(uint32_t length) {
    return sizeof(uint32_t) + static_cast<uint64_t>(length) * sizeof(BufferEntry);
}

template <MetricType metric>
uint64_t
SparseQuantizer<metric>::GetCodeSizeByLength(uint32_t length) const {
    if (not IsFP16()) {
        return GetFP32CodeSize(length);
    }
    constexpr uint64_t alignment = alignof(uint32_t);
    const uint64_t encoded_size =
        sizeof(uint32_t) + static_cast<uint64_t>(length) * (sizeof(uint32_t) + sizeof(uint16_t));
    return (encoded_size + alignment - 1) / alignment * alignment;
}

template <MetricType metric>
void
SparseQuantizer<metric>::DecodeSparseVector(const uint8_t* codes,
                                            uint32_t* ids,
                                            float* values) const {
    const uint32_t length = *reinterpret_cast<const uint32_t*>(codes);
    if (not IsFP16()) {
        const auto* entries = reinterpret_cast<const BufferEntry*>(codes + sizeof(uint32_t));
        for (uint32_t i = 0; i < length; ++i) {
            ids[i] = entries[i].id;
            values[i] = entries[i].val;
        }
        return;
    }
    const auto* stored_ids = reinterpret_cast<const uint32_t*>(codes + sizeof(uint32_t));
    const auto* stored_values = reinterpret_cast<const uint16_t*>(
        codes + sizeof(uint32_t) + static_cast<uint64_t>(length) * sizeof(uint32_t));
    for (uint32_t i = 0; i < length; ++i) {
        ids[i] = stored_ids[i];
        values[i] = generic::FP16ToFloat(stored_values[i]);
    }
}

template <MetricType metric>
bool
SparseQuantizer<metric>::TrainImpl(const float* data, uint64_t count) {
    this->is_trained_ = true;
    return true;
}

}  // namespace vsag
