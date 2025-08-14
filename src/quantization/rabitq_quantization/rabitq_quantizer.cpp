
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

#include "rabitq_quantizer.h"

#include "impl/transform/transformer_headers.h"
#include "simd/fp32_simd.h"
#include "simd/normalize.h"
#include "simd/rabitq_simd.h"
#include "typing.h"
#include "utils/util_functions.h"

namespace vsag {

template <MetricType metric>
RaBitQuantizer<metric>::RaBitQuantizer(int dim,
                                       uint64_t pca_dim,
                                       uint64_t num_bits_per_dim_query,
                                       bool use_fht,
                                       bool use_mrq,
                                       Allocator* allocator)
    : Quantizer<RaBitQuantizer<metric>>(dim, allocator) {
    // dim
    use_mrq_ = use_mrq;
    pca_dim_ = pca_dim;
    original_dim_ = dim;
    if (0 < pca_dim_ and pca_dim_ < dim) {
        if (use_mrq_) {
            pca_.reset(new PCATransformer(allocator, dim, dim));
        } else {
            pca_.reset(new PCATransformer(allocator, dim, pca_dim_));
        }
        this->dim_ = pca_dim_;
    } else {
        pca_dim_ = dim;
    }

    // bits query
    num_bits_per_dim_query_ = num_bits_per_dim_query;

    // centroid
    centroid_.resize(this->dim_, 0);

    // random orthogonal matrix
    use_fht_ = use_fht;
    if (use_fht_) {
        rom_.reset(new FhtKacRotator(allocator, this->dim_));
    } else {
        rom_.reset(new RandomOrthogonalMatrix(allocator, this->dim_));
    }
    // distance function related variable
    inv_sqrt_d_ = 1.0F / sqrt(this->dim_);

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

    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        offset_raw_norm_ = this->code_size_;
        this->code_size_ += ((sizeof(norm_type) + align_size - 1) / align_size) * align_size;
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

    // MRQ residual term
    if (pca_dim_ != original_dim_ and use_mrq_) {
        offset_mrq_norm_ = this->code_size_;
        this->code_size_ += ((sizeof(norm_type) + align_size - 1) / align_size) * align_size;

        query_offset_mrq_norm_ = this->query_code_size_;
        this->query_code_size_ += ((sizeof(norm_type) + align_size - 1) / align_size) * align_size;
    }

    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        query_offset_raw_norm_ = this->query_code_size_;
        this->query_code_size_ += ((sizeof(norm_type) + align_size - 1) / align_size) * align_size;
    }
}

template <MetricType metric>
RaBitQuantizer<metric>::RaBitQuantizer(const RaBitQuantizerParamPtr& param,
                                       const IndexCommonParam& common_param)
    : RaBitQuantizer<metric>(common_param.dim_,
                             param->pca_dim_,
                             param->num_bits_per_dim_query_,
                             param->use_fht_,
                             false,
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
        pca_->Train(data, count);
    }

    // get centroid
    for (int d = 0; d < this->dim_; d++) {
        centroid_[d] = 0;
    }
    for (uint64_t i = 0; i < count; ++i) {
        Vector<DataType> pca_data(this->original_dim_, 0, this->allocator_);
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

    rom_->Train(data, count);

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
    Vector<DataType> pca_data(this->original_dim_, 0, this->allocator_);
    Vector<DataType> transformed_data(this->dim_, 0, this->allocator_);
    Vector<DataType> normed_data(this->dim_, 0, this->allocator_);
    memset(codes, 0, this->code_size_);

    float raw_norm = 0;
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        for (uint64_t d = 0; d < this->dim_; ++d) {
            raw_norm += data[d] * data[d];
        }
    }
    raw_norm = std::sqrt(raw_norm);
    // 1. pca
    if (pca_dim_ != this->original_dim_) {
        pca_->Transform(data, pca_data.data());
        if (use_mrq_) {
            norm_type mrq_norm_sqr = FP32ComputeIP(pca_data.data() + this->dim_,
                                                   pca_data.data() + this->dim_,
                                                   this->original_dim_ - this->dim_);
            *(norm_type*)(codes + offset_mrq_norm_) = mrq_norm_sqr;
        }
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
    for (uint64_t d = 0; d < this->dim_; ++d) {
        if (normed_data[d] >= 0.0F) {
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
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        *(norm_type*)(codes + offset_raw_norm_) = raw_norm;
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

static float
l2_ube(float norm_base_raw, float norm_query_raw, float est_ip_norm) {
    float p1 = norm_base_raw * norm_base_raw;
    float p2 = norm_query_raw * norm_query_raw;
    float p3 = -2 * norm_base_raw * norm_query_raw * est_ip_norm;
    float ret = p1 + p2 + p3;
    return ret;
}

float
recover_dist_between_sq4u_and_fp32(uint32_t ip_bq_1_4,
                                   float base_sum,
                                   float query_sum,
                                   float lower_bound,
                                   float delta,
                                   float inv_sqrt_d,
                                   uint64_t dim) {
    // reference: RaBitQ equation 19-20
    float p1 = inv_sqrt_d * delta * 2 * static_cast<float>(ip_bq_1_4);
    float p2 = inv_sqrt_d * lower_bound * 2 * base_sum;
    float p3 = inv_sqrt_d * delta * query_sum;
    float p4 = inv_sqrt_d * lower_bound * static_cast<float>(dim);
    float ret = p1 + p2 - p3 - p4;
    return ret;
}

template <MetricType metric>
float
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

        ip_bq_estimate = recover_dist_between_sq4u_and_fp32(
            ip_bq_estimate, base_sum, query_sum, lower_bound, delta, this->inv_sqrt_d_, this->dim_);
    } else {
        ip_bq_estimate =
            RaBitQFloatBinaryIP((DataType*)query_codes, base_codes, this->dim_, inv_sqrt_d_);
    }

    norm_type query_norm = *((norm_type*)(query_codes + query_offset_norm_));
    norm_type base_norm = *((norm_type*)(base_codes + offset_norm_));

    norm_type query_raw_norm = 0;
    norm_type base_raw_norm = 0;
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        query_raw_norm = *((norm_type*)(query_codes + query_offset_raw_norm_));
        base_raw_norm = *((norm_type*)(base_codes + offset_raw_norm_));
    }

    error_type base_error = *((error_type*)(base_codes + offset_error_));
    if (std::abs(base_error) < 1e-5) {
        base_error = (base_error > 0) ? 1.0F : -1.0F;
    }

    float ip_bb_1_32 = base_error;
    float ip_est = ip_bq_estimate / ip_bb_1_32;

    float result = l2_ube(base_norm, query_norm, ip_est);

    if (pca_dim_ != this->original_dim_ and use_mrq_) {
        norm_type query_mrq_norm_sqr = *(norm_type*)(query_codes + query_offset_mrq_norm_);
        norm_type base_mrq_norm_sqr = *(norm_type*)(base_codes + offset_mrq_norm_);

        result += (query_mrq_norm_sqr + base_mrq_norm_sqr);
    }
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        if (is_approx_zero(query_raw_norm) or is_approx_zero(base_raw_norm)) {
            result = 1;
        } else {
            result =
                1 - (query_raw_norm * query_raw_norm + base_raw_norm * base_raw_norm - result) *
                        0.5F / (query_raw_norm * base_raw_norm);
        }
    }
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        if (is_approx_zero(query_raw_norm) or is_approx_zero(base_raw_norm)) {
            result = 1;
        } else {
            result =
                1 -
                (query_raw_norm * query_raw_norm + base_raw_norm * base_raw_norm - result) * 0.5F;
        }
    }

    return result;
}

template <MetricType metric>
float
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
RaBitQuantizer<metric>::EncodeSQ(const DataType* normed_data,
                                 uint8_t* quantized_data,
                                 float& upper_bound,
                                 float& lower_bound,
                                 float& delta,
                                 sum_type& query_sum) const {
    lower_bound = std::numeric_limits<float>::max();
    upper_bound = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < this->dim_; i++) {
        const float val = normed_data[i];
        if (val < lower_bound) {
            lower_bound = val;
        }
        if (val > upper_bound) {
            upper_bound = val;
        }
    }
    delta = (upper_bound - lower_bound) / ((1 << num_bits_per_dim_query_) - 1);
    const float inv_delta = is_approx_zero(delta) ? 0.0F : 1.0F / delta;
    query_sum = 0;
    for (int32_t i = 0; i < this->dim_; i++) {
        const auto val = std::round((normed_data[i] - lower_bound) * inv_delta);
        quantized_data[i] = static_cast<uint8_t>(val);
        query_sum += static_cast<float>(val);
    }
}

template <MetricType metric>
void
RaBitQuantizer<metric>::ReOrderSQ(const uint8_t* quantized_data, uint8_t* reorder_data) const {
    size_t offset = aligned_dim_ / 8;
    for (size_t d = 0; d < this->dim_; d++) {
        for (size_t bit_pos = 0; bit_pos < num_bits_per_dim_query_; bit_pos++) {
            const bool bit = ((quantized_data[d] & (1 << bit_pos)) != 0);
            reorder_data[bit_pos * offset + d / 8] |= (static_cast<int32_t>(bit) * (1 << (d % 8)));
        }
    }
}

template <MetricType metric>
void
RaBitQuantizer<metric>::DecodeSQ(const uint8_t* codes,
                                 DataType* data,
                                 const float upper_bound,
                                 const float lower_bound) const {
    for (uint64_t d = 0; d < this->dim_; d++) {
        data[d] = static_cast<float>(codes[d]) /
                      static_cast<float>((1 << num_bits_per_dim_query_) - 1) *
                      (upper_bound - lower_bound) +
                  lower_bound;
    }
}

template <MetricType metric>
void
RaBitQuantizer<metric>::RecoverOrderSQ(const uint8_t* output, uint8_t* input) const {
    // note that the codesize of input is different from output
    // output: align dim bits with 8 bits (1 byte)
    size_t offset = aligned_dim_ / 8;
    for (uint64_t d = 0; d < this->dim_; ++d) {
        for (uint64_t bit_pos = 0; bit_pos < num_bits_per_dim_query_; ++bit_pos) {
            // calculate the position in the reordered output
            uint64_t output_bit_pos = bit_pos * aligned_dim_ + d;
            uint64_t output_byte_i = output_bit_pos / 8;
            uint64_t output_bit_i = output_bit_pos % 8;

            // extract the bit
            uint8_t bit_value = (output[output_byte_i] >> output_bit_i) & 0x1;

            // calculate the position
            uint64_t input_byte_i = d;
            uint64_t input_bit_i = bit_pos;

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
        if (computer.buf_ == nullptr) {
            computer.buf_ =
                reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        }
        std::fill(computer.buf_, computer.buf_ + this->query_code_size_, 0);

        // use residual term in pca, so it's this->original_dim_
        Vector<DataType> pca_data(this->original_dim_, 0, this->allocator_);
        Vector<DataType> transformed_data(this->dim_, 0, this->allocator_);
        Vector<DataType> normed_data(this->dim_, 0, this->allocator_);

        float query_raw_norm = 0;
        if constexpr (metric == MetricType::METRIC_TYPE_IP or
                      metric == MetricType::METRIC_TYPE_COSINE) {
            for (uint64_t d = 0; d < this->dim_; ++d) {
                query_raw_norm += query[d] * query[d];
            }
        }
        query_raw_norm = std::sqrt(query_raw_norm);
        // 1. pca
        if (pca_dim_ != this->original_dim_) {
            pca_->Transform(query, pca_data.data());
            if (use_mrq_) {
                norm_type mrq_norm_sqr = FP32ComputeIP(pca_data.data() + this->dim_,
                                                       pca_data.data() + this->dim_,
                                                       this->original_dim_ - this->dim_);

                *(norm_type*)(computer.buf_ + query_offset_mrq_norm_) = mrq_norm_sqr;
            }
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
            Vector<uint8_t> quantized_data(this->dim_, 0, this->allocator_);
            float lower_bound = std::numeric_limits<float>::max();
            float upper_bound = std::numeric_limits<float>::lowest();
            float delta = 0.0F;
            sum_type query_sum = 0;
            EncodeSQ(normed_data.data(),
                     quantized_data.data(),
                     upper_bound,
                     lower_bound,
                     delta,
                     query_sum);
            ReOrderSQ(quantized_data.data(), reinterpret_cast<uint8_t*>(computer.buf_));
            // store info
            *(DataType*)(computer.buf_ + query_offset_lb_) = lower_bound;
            *(DataType*)(computer.buf_ + query_offset_delta_) = delta;
            *(sum_type*)(computer.buf_ + query_offset_sum_) = query_sum;
        } else {
            // store codes
            memcpy(computer.buf_, normed_data.data(), normed_data.size() * sizeof(DataType));
        }

        // 5. store norm
        *(norm_type*)(computer.buf_ + query_offset_norm_) = query_norm;
        if constexpr (metric == MetricType::METRIC_TYPE_IP or
                      metric == MetricType::METRIC_TYPE_COSINE) {
            *(norm_type*)(computer.buf_ + query_offset_raw_norm_) = query_raw_norm;
        }
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

TEMPLATE_QUANTIZER(RaBitQuantizer)

}  // namespace vsag
