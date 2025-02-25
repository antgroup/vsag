
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

#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "index/index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "rabitq_quantizer_parameter.h"
#include "simd/normalize.h"
#include "simd/rabitq_simd.h"
#include "typing.h"

namespace vsag {

/** Implement of RaBitQ Quantization
 *
 *  Supports bit-level quantization
 *
 *  Reference:
 *  Jianyang Gao and Cheng Long. 2024. RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search. Proc. ACM Manag. Data 2, 3, Article 167 (June 2024), 27 pages. https://doi.org/10.1145/3654970
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class RaBitQuantizer : public Quantizer<RaBitQuantizer<metric>> {
public:
    using norm_type = float;
    using error_type = float;

    explicit RaBitQuantizer(int dim, Allocator* allocator);

    explicit RaBitQuantizer(const RaBitQuantizerParamPtr& param,
                            const IndexCommonParam& common_param);

    explicit RaBitQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);

    inline float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    inline void
    ProcessQueryImpl(const DataType* query, Computer<RaBitQuantizer>& computer) const;

    inline void
    ComputeDistImpl(Computer<RaBitQuantizer>& computer, const uint8_t* codes, float* dists) const;

    inline void
    ComputeBatchDistImpl(Computer<RaBitQuantizer<metric>>& computer,
                         uint64_t count,
                         const uint8_t* codes,
                         float* dists) const;

    inline void
    ReleaseComputerImpl(Computer<RaBitQuantizer<metric>>& computer) const;

    inline void
    SerializeImpl(StreamWriter& writer);

    inline void
    DeserializeImpl(StreamReader& reader);

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_RABITQ;
    }

private:
    inline float
    L2_UBE(float norm_base_raw, float norm_query_raw, float est_ip_norm) const {
        float p1 = norm_base_raw * norm_base_raw;
        float p2 = norm_query_raw * norm_query_raw;
        float p3 = -2 * norm_base_raw * norm_query_raw * est_ip_norm;
        float ret = p1 + p2 + p3;
        return ret;
    }

private:
    std::vector<float> centroid_;  // TODO(ZXY): use centroids (e.g., IVF or Graph) outside

    uint64_t query_code_size_{0};  // TODO(ZXY): support various type of query (FP32, SQ4...)
    uint64_t query_offset_norm_{0};

    /***
     * code layout: sq-code(required) + norm(required) + error(required)
     */
    uint64_t offset_code_{0};
    uint64_t offset_norm_{0};
    uint64_t offset_error_{0};
};

template <MetricType metric>
RaBitQuantizer<metric>::RaBitQuantizer(int dim, Allocator* allocator)
    : Quantizer<RaBitQuantizer<metric>>(dim, allocator) {
    static_assert(metric == MetricType::METRIC_TYPE_L2SQR, "Unsupported metric type");

    centroid_.resize(dim, 0);

    // base code layout
    size_t align_size = std::max(sizeof(error_type), sizeof(norm_type));
    size_t code_original_size = (dim + 7) / 8;

    this->code_size_ = 0;

    offset_code_ = this->code_size_;
    this->code_size_ += ((code_original_size + align_size - 1) / align_size) * align_size;

    offset_norm_ = this->code_size_;
    this->code_size_ += ((sizeof(norm_type) + align_size - 1) / align_size) * align_size;

    offset_error_ = this->code_size_;
    this->code_size_ += ((sizeof(error_type) + align_size - 1) / align_size) * align_size;

    // query code layout
    this->query_code_size_ = ((sizeof(DataType) * this->dim_) / align_size) * align_size;
    query_offset_norm_ = this->query_code_size_;
}

template <MetricType metric>
RaBitQuantizer<metric>::RaBitQuantizer(const RaBitQuantizerParamPtr& param,
                                       const IndexCommonParam& common_param)
    : RaBitQuantizer<metric>(common_param.dim_, common_param.allocator_.get()){};

template <MetricType metric>
RaBitQuantizer<metric>::RaBitQuantizer(const QuantizerParamPtr& param,
                                       const IndexCommonParam& common_param)
    : RaBitQuantizer<metric>(std::dynamic_pointer_cast<RaBitQuantizerParameter>(param),
                             common_param){};

template <MetricType metric>
bool
RaBitQuantizer<metric>::TrainImpl(const DataType* data, uint64_t count) {
    if (count == 0 or data == nullptr) {
        return false;
    }

    if (this->is_trained_) {
        return true;
    }

    // get centroid
    for (int d = 0; d < this->dim_; d++) {
        centroid_[d] = 0;
    }
    for (uint64_t i = 0; i < count; ++i) {
        for (uint64_t d = 0; d < this->dim_; d++) {
            centroid_[d] += data[d + i * this->dim_];
        }
    }
    for (uint64_t d = 0; d < this->dim_; d++) {
        centroid_[d] = centroid_[d] / (float)count;
    }

    // TODO(ZXY): generate random orthogonal matrix

    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    // 0. init
    std::fill(codes, codes + this->code_size_, 0);

    // 1. random projection
    // TODO(ZXY) use random projection

    // 2. normalize
    Vector<DataType> norm_data(this->allocator_);
    norm_data.resize(this->dim_);
    norm_type norm = NormalizeWithCentroid(data, centroid_.data(), norm_data.data(), this->dim_);

    // 3. encode with BQ
    for (uint64_t d = 0; d < this->dim_; ++d) {
        if (norm_data[d] >= 0.0f) {
            codes[offset_code_ + d / 8] |= (1 << (d % 8));
        }
    }

    // 4. compute encode error
    error_type error = RaBitQFloatBinaryIP(data, codes, this->dim_);

    // 5. store norm and error
    *(norm_type*)(codes + offset_norm_) = norm;
    *(error_type*)(codes + offset_error_) = error;

    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        // TODO(ZXY): use batch optimize
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(this->dim_));
    for (uint64_t d = 0; d < this->dim_; ++d) {
        bool bit = ((codes[d / 8] >> (d % 8)) & 1) != 0;
        data[d] = bit ? inv_sqrt_d : -inv_sqrt_d;
    }
    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        // TODO(ZXY): use batch optimize
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

template <MetricType metric>
inline float
RaBitQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    // codes1 -> query (fp32, sq8, sq4...) + norm
    // codes2 -> base  (binary) + norm + error
    error_type base_error = *((error_type*)(codes2 + offset_error_));
    if (base_error < 1e-5) {
        base_error = 1.0f;
    }
    norm_type base_norm = *((norm_type*)(codes2 + offset_norm_));
    norm_type query_norm = *((norm_type*)(codes1 + query_offset_norm_));

    float ip_bq_1_32 = RaBitQFloatBinaryIP((DataType*)codes1, codes2, this->dim_);
    float ip_bb_1_32 = base_error;
    float ip_est = ip_bq_1_32 / ip_bb_1_32;

    float result = L2_UBE(base_norm, query_norm, ip_est);

    if (result < 0) {
        result = 0;
    }
    return result;
}

template <MetricType metric>
void
RaBitQuantizer<metric>::ProcessQueryImpl(const DataType* query,
                                         Computer<RaBitQuantizer>& computer) const {
    try {
        // TODO(ZXY): allow process query with SQ4 or SQ8, implement in ComputeDist and Param
        size_t align_size = std::max(sizeof(error_type), sizeof(norm_type));

        computer.buf_ =
            reinterpret_cast<uint8_t*>(this->allocator_->Allocate(query_code_size_ + align_size));
        std::fill(computer.buf_, computer.buf_ + query_code_size_ + align_size, 0);

        // 1. transform
        // TODO(ZXY) use random projection

        // 2. norm
        float query_norm =
            NormalizeWithCentroid(query, centroid_.data(), (DataType*)computer.buf_, this->dim_);

        // 3. store norm
        *(norm_type*)(computer.buf_ + query_offset_norm_) = query_norm;
    } catch (std::bad_alloc& e) {
        logger::error("bad alloc when init computer buf");
        throw e;
    }
}

template <MetricType metric>
void
RaBitQuantizer<metric>::ComputeDistImpl(Computer<RaBitQuantizer>& computer,
                                        const uint8_t* codes,
                                        float* dists) const {
    dists[0] = this->ComputeImpl(computer.buf_, codes);
}

template <MetricType metric>
void
RaBitQuantizer<metric>::ComputeBatchDistImpl(Computer<RaBitQuantizer<metric>>& computer,
                                             uint64_t count,
                                             const uint8_t* codes,
                                             float* dists) const {
    for (uint64_t i = 0; i < count; ++i) {
        // TODO(ZXY): use batch optimize
        this->ComputeDistImpl(computer, codes + i * this->code_size_, dists + i);
    }
}

template <MetricType metric>
void
RaBitQuantizer<metric>::ReleaseComputerImpl(Computer<RaBitQuantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

template <MetricType metric>
void
RaBitQuantizer<metric>::SerializeImpl(StreamWriter& writer) {
    StreamWriter::WriteObj(writer, this->offset_code_);
    StreamWriter::WriteObj(writer, this->offset_norm_);
    StreamWriter::WriteObj(writer, this->offset_error_);
    StreamWriter::WriteObj(writer, this->query_offset_norm_);
    StreamWriter::WriteObj(writer, this->query_code_size_);
    StreamWriter::WriteVector(writer, this->centroid_);
}

template <MetricType metric>
void
RaBitQuantizer<metric>::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->offset_code_);
    StreamReader::ReadObj(reader, this->offset_norm_);
    StreamReader::ReadObj(reader, this->offset_error_);
    StreamReader::ReadObj(reader, this->query_offset_norm_);
    StreamReader::ReadObj(reader, this->query_code_size_);
    StreamReader::ReadVector(reader, this->centroid_);
}

}  // namespace vsag
