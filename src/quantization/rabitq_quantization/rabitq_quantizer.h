
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

#include "impl/principal_component_analysis.h"
#include "impl/random_orthogonal_matrix.h"
#include "index/index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "quantization/scalar_quantization/sq4_uniform_quantizer.h"
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
    using sum_type = float;

    explicit RaBitQuantizer(int dim,
                            uint64_t pca_dim,
                            uint64_t num_bits_per_dim_query,
                            Allocator* allocator);

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

    inline float
    ComputeQueryBaseImpl(const uint8_t* query_codes, const uint8_t* base_codes) const;

    inline void
    ProcessQueryImpl(const DataType* query, Computer<RaBitQuantizer>& computer) const;

    inline void
    ComputeDistImpl(Computer<RaBitQuantizer>& computer, const uint8_t* codes, float* dists) const;

    inline void
    ScanBatchDistImpl(Computer<RaBitQuantizer<metric>>& computer,
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

public:
    void
    ReOrderSQ4(const uint8_t* input, uint8_t* output) const;

    void
    RecoverOrderSQ4(const uint8_t* output, uint8_t* input) const;

    inline float
    L2_UBE(float norm_base_raw, float norm_query_raw, float est_ip_norm) const {
        float p1 = norm_base_raw * norm_base_raw;
        float p2 = norm_query_raw * norm_query_raw;
        float p3 = -2 * norm_base_raw * norm_query_raw * est_ip_norm;
        float ret = p1 + p2 + p3;
        return ret;
    }

    inline float
    RecoverDistBetweenSQ4UandFP32(
        uint32_t ip_bq_1_4, float base_sum, float query_sum, float lower_bound, float delta) const {
        // reference: RaBitQ equation 19-20
        float p1 = inv_sqrt_d_ * delta * 2 * ip_bq_1_4;
        float p2 = inv_sqrt_d_ * lower_bound * 2 * base_sum;
        float p3 = inv_sqrt_d_ * delta * query_sum;
        float p4 = inv_sqrt_d_ * lower_bound * this->dim_;
        float ret = p1 + p2 - p3 - p4;
        return ret;
    }

private:
    // compute related
    float inv_sqrt_d_{0};

    // random projection related
    std::shared_ptr<RandomOrthogonalMatrix> rom_;
    std::vector<float> centroid_;  // TODO(ZXY): use centroids (e.g., IVF or Graph) outside

    // pca related
    std::shared_ptr<PrincipalComponentAnalysis> pca_;
    std::uint64_t original_dim_{0};
    std::uint64_t pca_dim_{0};

    /***
     * query layout: sq-code(required) + lower_bound(sq4) + delta(sq4) + sum(sq4) + norm(required)
     */
    uint64_t aligned_dim_{0};
    uint64_t num_bits_per_dim_query_{32};
    uint64_t query_code_size_{0};
    uint64_t query_offset_lb_{0};
    uint64_t query_offset_delta_{0};
    uint64_t query_offset_sum_{0};
    uint64_t query_offset_norm_{0};

    /***
     * code layout: bq-code(required) + norm(required) + error(required) + sum(sq4)
     */
    uint64_t offset_code_{0};
    uint64_t offset_norm_{0};
    uint64_t offset_error_{0};
    uint64_t offset_sum_{0};
};

template <MetricType metric>
RaBitQuantizer<metric>::RaBitQuantizer(int dim,
                                       uint64_t pca_dim,
                                       uint64_t num_bits_per_dim_query,
                                       Allocator* allocator)
    : Quantizer<RaBitQuantizer<metric>>(dim, allocator) {
    static_assert(metric == MetricType::METRIC_TYPE_L2SQR, "Unsupported metric type");

    // dim
    pca_dim_ = pca_dim;
    original_dim_ = dim;
    if (0 < pca_dim_ and pca_dim_ < dim) {
        pca_.reset(new PrincipalComponentAnalysis(dim, pca_dim_, allocator));
        this->dim_ = pca_dim_;
    } else {
        pca_dim_ = dim;
    }

    // bits query
    num_bits_per_dim_query_ = num_bits_per_dim_query;

    // centroid
    centroid_.resize(this->dim_, 0);

    // random orthogonal matrix
    rom_.reset(new RandomOrthogonalMatrix(this->dim_, allocator));

    // distance function related variable
    inv_sqrt_d_ = 1.0f / sqrt(this->dim_);

    // base code layout
    size_t align_size = std::max(std::max(sizeof(error_type), sizeof(norm_type)), sizeof(DataType));
    size_t code_original_size = (this->dim_ + 7) / 8;

    this->code_size_ = 0;

    offset_code_ = this->code_size_;
    this->code_size_ += ((code_original_size + align_size - 1) / align_size) * align_size;

    offset_norm_ = this->code_size_;
    this->code_size_ += ((sizeof(norm_type) + align_size - 1) / align_size) * align_size;

    offset_error_ = this->code_size_;
    this->code_size_ += ((sizeof(error_type) + align_size - 1) / align_size) * align_size;

    if (num_bits_per_dim_query_ != 32) {
        offset_sum_ = this->code_size_;
        this->code_size_ += ((sizeof(sum_type) + align_size - 1) / align_size) * align_size;
    }

    // query code layout
    if (num_bits_per_dim_query_ == 4) {
        // Re-order the SQ4U Code Layout (align with 8 bits)
        // e.g., for a float query with dim == 4:   [1, 2, 4, 8]
        //       suppose original SQ4U code is:     [0001 0010, 0100 1000]  (0001 is 4)
        //       then, the re-ordered code is:      [1000 0100, 0010 0001]
        aligned_dim_ = ((this->dim_ + 511) / 512) * 512;
        auto sq_code_size = aligned_dim_ / 8 * num_bits_per_dim_query_;
        this->query_code_size_ = (sq_code_size / align_size) * align_size;

        query_offset_lb_ = this->query_code_size_;
        this->query_code_size_ += ((sizeof(DataType) + align_size - 1) / align_size) * align_size;

        query_offset_delta_ = this->query_code_size_;
        this->query_code_size_ += ((sizeof(DataType) + align_size - 1) / align_size) * align_size;

        query_offset_sum_ = this->query_code_size_;
        this->query_code_size_ += ((sizeof(sum_type) + align_size - 1) / align_size) * align_size;
    } else {
        this->query_code_size_ = ((sizeof(DataType) * this->dim_) / align_size) * align_size;
    }

    query_offset_norm_ = this->query_code_size_;
    this->query_code_size_ += ((sizeof(norm_type) + align_size - 1) / align_size) * align_size;
}

template <MetricType metric>
RaBitQuantizer<metric>::RaBitQuantizer(const RaBitQuantizerParamPtr& param,
                                       const IndexCommonParam& common_param)
    : RaBitQuantizer<metric>(common_param.dim_,
                             param->pca_dim_,
                             param->num_bits_per_dim_query_,
                             common_param.allocator_.get()){};

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

    // pca
    if (pca_dim_ != this->original_dim_) {
        bool pca_result = pca_->Train(data, count);
        if (not pca_result) {
            return false;
        }
    }

    // get centroid
    for (int d = 0; d < this->dim_; d++) {
        centroid_[d] = 0;
    }
    for (uint64_t i = 0; i < count; ++i) {
        Vector<DataType> pca_data(this->dim_, 0, this->allocator_);
        if (pca_dim_ != this->original_dim_) {
            pca_->Transform(data + i * original_dim_, pca_data.data());
        } else {
            pca_data.assign(data + i * original_dim_, data + (i + 1) * original_dim_);
        }

        for (uint64_t d = 0; d < this->dim_; d++) {
            centroid_[d] += pca_data[d];
        }
    }
    for (uint64_t d = 0; d < this->dim_; d++) {
        centroid_[d] = centroid_[d] / (float)count;
    }

    // generate rom
    rom_->GenerateRandomOrthogonalMatrixWithRetry();

    // validate rom
    int retries = MAX_RETRIES;
    bool successful_gen = true;
    double det = rom_->ComputeDeterminant();
    if (std::fabs(det - 1) > 1e-4) {
        for (uint64_t i = 0; i < retries; i++) {
            successful_gen = rom_->GenerateRandomOrthogonalMatrix();
            if (successful_gen) {
                break;
            }
        }
    }
    if (not successful_gen) {
        return false;
    }

    // transform centroid
    Vector<DataType> rp_centroids(this->dim_, 0, this->allocator_);
    rom_->Transform(centroid_.data(), rp_centroids.data());
    centroid_.assign(rp_centroids.begin(), rp_centroids.end());

    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    // 0. init
    Vector<DataType> pca_data(this->dim_, 0, this->allocator_);
    Vector<DataType> transformed_data(this->dim_, 0, this->allocator_);
    Vector<DataType> normed_data(this->dim_, 0, this->allocator_);

    // 1. pca
    if (pca_dim_ != this->original_dim_) {
        pca_->Transform(data, pca_data.data());
    } else {
        pca_data.assign(data, data + original_dim_);
    }

    // 2. random projection
    rom_->Transform(pca_data.data(), transformed_data.data());

    // 3. normalize
    norm_type norm = NormalizeWithCentroid(
        transformed_data.data(), centroid_.data(), normed_data.data(), this->dim_);

    // 4. encode with BQ
    sum_type sum = 0;
    memset(codes, 0, this->code_size_);
    for (uint64_t d = 0; d < this->dim_; ++d) {
        if (normed_data[d] >= 0.0f) {
            sum += 1;
            codes[offset_code_ + d / 8] |= (1 << (d % 8));
        }
    }

    // 5. compute encode error
    error_type error =
        RaBitQFloatBinaryIP(normed_data.data(), codes + offset_code_, this->dim_, inv_sqrt_d_);

    // 6. store norm, error, sum
    *(norm_type*)(codes + offset_norm_) = norm;
    *(error_type*)(codes + offset_error_) = error;

    if (num_bits_per_dim_query_ != 32) {
        *(sum_type*)(codes + offset_sum_) = sum;
    }

    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        // TODO(ZXY): use batch optimize
        this->EncodeOneImpl(data + i * this->original_dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    if (pca_dim_ != this->original_dim_) {
        return false;
    }

    // 1. init
    Vector<DataType> normed_data(this->dim_, 0, this->allocator_);
    Vector<DataType> transformed_data(this->dim_, 0, this->allocator_);

    // 2. decode with BQ
    for (uint64_t d = 0; d < this->dim_; ++d) {
        bool bit = ((codes[d / 8] >> (d % 8)) & 1) != 0;
        normed_data[d] = bit ? inv_sqrt_d_ : -inv_sqrt_d_;
    }

    // 3. inverse normalize
    InverseNormalizeWithCentroid(normed_data.data(),
                                 centroid_.data(),
                                 transformed_data.data(),
                                 this->dim_,
                                 *(norm_type*)(codes + offset_norm_));

    // 4. inverse random projection
    // Note that the value may be much different between original since inv_sqrt_d is small
    rom_->InverseTransform(transformed_data.data(), data);
    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    if (pca_dim_ != this->original_dim_) {
        return false;
    }

    for (uint64_t i = 0; i < count; ++i) {
        // TODO(ZXY): use batch optimize
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

template <MetricType metric>
inline float
RaBitQuantizer<metric>::ComputeQueryBaseImpl(const uint8_t* query_codes,
                                             const uint8_t* base_codes) const {
    // codes1 -> query (fp32, sq8, sq4...) + norm
    // codes2 -> base  (binary) + norm + error
    float ip_bq_estimate;
    if (num_bits_per_dim_query_ == 4) {
        std::vector<uint8_t> tmp(aligned_dim_ / 8, 0);
        memcpy(tmp.data(), base_codes, offset_norm_);

        ip_bq_estimate = RaBitQSQ4UBinaryIP(query_codes, tmp.data(), aligned_dim_);

        sum_type base_sum = *((sum_type*)(base_codes + offset_sum_));
        sum_type query_sum = *((sum_type*)(query_codes + query_offset_sum_));
        DataType lower_bound = *((DataType*)(query_codes + query_offset_lb_));
        DataType delta = *((DataType*)(query_codes + query_offset_delta_));

        ip_bq_estimate =
            RecoverDistBetweenSQ4UandFP32(ip_bq_estimate, base_sum, query_sum, lower_bound, delta);
    } else {
        ip_bq_estimate =
            RaBitQFloatBinaryIP((DataType*)query_codes, base_codes, this->dim_, inv_sqrt_d_);
    }

    norm_type query_norm = *((norm_type*)(query_codes + query_offset_norm_));
    norm_type base_norm = *((norm_type*)(base_codes + offset_norm_));
    error_type base_error = *((error_type*)(base_codes + offset_error_));
    if (std::abs(base_error) < 1e-5) {
        base_error = (base_error > 0) ? 1.0f : -1.0f;
    }

    float ip_bb_1_32 = base_error;
    float ip_est = ip_bq_estimate / ip_bb_1_32;

    float result = L2_UBE(base_norm, query_norm, ip_est);

    return result;
}

template <MetricType metric>
inline float
RaBitQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    throw VsagException(ErrorType::INTERNAL_ERROR,
                        "building the index is not supported using RabbitQ alone");
}

template <MetricType metric>
void
RaBitQuantizer<metric>::ReOrderSQ4(const uint8_t* input, uint8_t* output) const {
    // note that the codesize of input is different from output
    // output: align dim bits with 8 bits (1 byte)
    for (uint64_t bit_pos = 0; bit_pos < num_bits_per_dim_query_; ++bit_pos) {
        for (uint64_t d = 0; d < this->dim_; d++) {
            // extract the bit
            uint8_t bit_value = (input[d / 2] >> ((d % 2) * 4 + bit_pos)) & 0x1;

            // calculate the position
            uint64_t output_bit_pos = bit_pos * aligned_dim_ + d;
            uint64_t output_byte_i = output_bit_pos / 8;
            uint64_t output_bit_i = output_bit_pos % 8;

            // set the bit
            output[output_byte_i] |= (bit_value << output_bit_i);
        }
    }
}

template <MetricType metric>
void
RaBitQuantizer<metric>::RecoverOrderSQ4(const uint8_t* output, uint8_t* input) const {
    // note that the codesize of input is different from output
    // output: align dim bits with 8 bits (1 byte)
    for (uint64_t d = 0; d < this->dim_; ++d) {
        for (uint64_t bit_pos = 0; bit_pos < num_bits_per_dim_query_; ++bit_pos) {
            // calculate the position in the reordered output
            uint64_t output_bit_pos = bit_pos * aligned_dim_ + d;
            uint64_t output_byte_i = output_bit_pos / 8;
            uint64_t output_bit_i = output_bit_pos % 8;

            // extract the bit
            uint8_t bit_value = (output[output_byte_i] >> output_bit_i) & 0x1;

            // calculate the position
            uint64_t input_byte_i = d / 2;
            uint64_t input_bit_i = (d % 2) * 4 + bit_pos;

            // set the bit
            input[input_byte_i] |= (bit_value << input_bit_i);
        }
    }
}

template <MetricType metric>
void
RaBitQuantizer<metric>::ProcessQueryImpl(const DataType* query,
                                         Computer<RaBitQuantizer>& computer) const {
    try {
        computer.buf_ = reinterpret_cast<uint8_t*>(this->allocator_->Allocate(query_code_size_));
        std::fill(computer.buf_, computer.buf_ + query_code_size_, 0);

        Vector<DataType> pca_data(this->dim_, 0, this->allocator_);
        Vector<DataType> transformed_data(this->dim_, 0, this->allocator_);
        Vector<DataType> normed_data(this->dim_, 0, this->allocator_);

        // 1. pca
        if (pca_dim_ != this->original_dim_) {
            pca_->Transform(query, pca_data.data());
        } else {
            pca_data.assign(query, query + original_dim_);
        }

        // 2. random projection
        rom_->Transform(pca_data.data(), transformed_data.data());

        // 3. norm
        float query_norm = NormalizeWithCentroid(
            transformed_data.data(), centroid_.data(), normed_data.data(), this->dim_);

        // 4. query quantization
        if (num_bits_per_dim_query_ == 4) {
            // sq4 quantization
            Vector<uint8_t> tmp_codes(this->query_code_size_, 0, this->allocator_);
            SQ4UniformQuantizer<MetricType::METRIC_TYPE_IP> sq4_quantizer(
                this->dim_, this->allocator_, 0.0f);
            sq4_quantizer.Train(normed_data.data(), 1);
            sq4_quantizer.EncodeOneImpl(normed_data.data(), tmp_codes.data());

            // re-order and store codes
            ReOrderSQ4(tmp_codes.data(), computer.buf_);

            // store info
            auto lb_and_diff = sq4_quantizer.GetLBandDiff();
            DataType lower_bound = lb_and_diff.first;
            DataType delta = lb_and_diff.second / 15.0;
            *(DataType*)(computer.buf_ + query_offset_lb_) = lower_bound;
            *(DataType*)(computer.buf_ + query_offset_delta_) = delta;
            *(sum_type*)(computer.buf_ + query_offset_sum_) =
                sq4_quantizer.GetCodesSum(tmp_codes.data());
        } else {
            // store codes
            memcpy(computer.buf_, normed_data.data(), normed_data.size() * sizeof(DataType));
        }

        // 5. store norm
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
    dists[0] = this->ComputeQueryBaseImpl(computer.buf_, codes);
}

template <MetricType metric>
void
RaBitQuantizer<metric>::ScanBatchDistImpl(Computer<RaBitQuantizer<metric>>& computer,
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
    StreamWriter::WriteVector(writer, this->centroid_);
    this->rom_->Serialize(writer);
    if (pca_dim_ != this->original_dim_) {
        this->pca_->Serialize(writer);
    }
}

template <MetricType metric>
void
RaBitQuantizer<metric>::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadVector(reader, this->centroid_);
    this->rom_->Deserialize(reader);
    if (pca_dim_ != this->original_dim_) {
        this->pca_->Deserialize(reader);
    }
}

}  // namespace vsag
