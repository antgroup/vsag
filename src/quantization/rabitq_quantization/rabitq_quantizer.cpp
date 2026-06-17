
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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <utility>

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
                                       uint64_t num_bits_per_dim_base,
                                       bool use_fht,
                                       bool use_mrq,
                                       Allocator* allocator,
                                       std::string rabitq_version,
                                       float rabitq_error_rate,
                                       uint64_t num_bits_per_dim_filter)
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
    num_bits_per_dim_base_ = num_bits_per_dim_base;
    num_bits_per_dim_filter_ = static_cast<uint32_t>(num_bits_per_dim_filter);
    if (num_bits_per_dim_query_ == 4 and num_bits_per_dim_base_ != 1) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            "not support num_bits_per_dim_query_ == 4 with num_bits_per_dim_base_ != 1");
    }
    if (num_bits_per_dim_filter_ < 1 or num_bits_per_dim_filter_ > num_bits_per_dim_base_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "rabitq_bits_per_dim_filter must be in [1, rabitq_bits_per_dim_base]");
    }

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
    rabitq_version_ = std::move(rabitq_version);
    rabitq_error_rate_ = rabitq_error_rate;

    // base code layout
    uint64_t align_size = std::max(std::max(sizeof(error_type), sizeof(norm_type)), sizeof(float));

    uint64_t code_original_size = (this->dim_ + 7) / 8;
    code_original_size *= num_bits_per_dim_base_;

    this->code_size_ = 0;

    offset_code_ = this->code_size_;
    this->code_size_ += ((code_original_size + align_size - 1) / align_size) * align_size;

    if (num_bits_per_dim_base_ != 1) {
        offset_norm_code_ = this->code_size_;
        this->code_size_ += ((sizeof(norm_type) + align_size - 1) / align_size) * align_size;
    }

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
        this->query_code_size_ += ((sizeof(float) + align_size - 1) / align_size) * align_size;

        query_offset_delta_ = this->query_code_size_;
        this->query_code_size_ += ((sizeof(float) + align_size - 1) / align_size) * align_size;
    } else {
        this->query_code_size_ =
            ((sizeof(float) * this->dim_ + align_size - 1) / align_size) * align_size;
    }

    if (num_bits_per_dim_query_ == 4 or num_bits_per_dim_base_ != 1) {
        query_offset_sum_ = this->query_code_size_;
        this->query_code_size_ += ((sizeof(sum_type) + align_size - 1) / align_size) * align_size;
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

    if (HasFilterQueryLookupTable()) {
        query_offset_filter_lut_ = this->query_code_size_;
        this->query_code_size_ += AlignCodeField(FilterQueryLookupTableSize());
    }

    if (SupportSplitCodeStorage()) {
        offset_low_bound_error_ = this->code_size_;
        this->code_size_ += ((sizeof(error_type) + align_size - 1) / align_size) * align_size;

        offset_one_bit_error_ = this->code_size_;
        this->code_size_ += ((sizeof(error_type) + align_size - 1) / align_size) * align_size;
    }
}

template <MetricType metric>
RaBitQuantizer<metric>::RaBitQuantizer(const RaBitQuantizerParamPtr& param,
                                       const IndexCommonParam& common_param)
    : RaBitQuantizer<metric>(common_param.dim_,
                             param->pca_dim_,
                             param->num_bits_per_dim_query_,
                             param->num_bits_per_dim_base_,
                             param->use_fht_,
                             false,
                             common_param.allocator_.get(),
                             param->rabitq_version_,
                             param->rabitq_error_rate_,
                             param->num_bits_per_dim_filter_){};

template <MetricType metric>
RaBitQuantizer<metric>::RaBitQuantizer(const QuantizerParamPtr& param,
                                       const IndexCommonParam& common_param)
    : RaBitQuantizer<metric>(std::dynamic_pointer_cast<RaBitQuantizerParameter>(param),
                             common_param){};

template <MetricType metric>
bool
RaBitQuantizer<metric>::TrainImpl(const float* data, uint64_t count) {
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
        Vector<float> pca_data(this->original_dim_, 0, this->allocator_);
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
    Vector<float> rp_centroids(this->dim_, 0, this->allocator_);
    rom_->Transform(centroid_.data(), rp_centroids.data());
    centroid_.assign(rp_centroids.begin(), rp_centroids.end());

    this->is_trained_ = true;
    return true;
}

inline float
ip_obar_q(float ip_yu_q, float q_prime_sum, float y_norm, int B) {
    // used for recover distance from ip_yu_q
    const float c = 0.5F * float((1U << B) - 1U);

    if (y_norm <= 0.0F) {
        return 0.0F;
    }
    auto ret = (ip_yu_q - c * q_prime_sum);
    ret /= y_norm;
    return ret;
}

template <MetricType metric>
typename RaBitQuantizer<metric>::norm_type
RaBitQuantizer<metric>::ComputeScalarCodeNorm(const uint8_t* scalar_codes,
                                              uint32_t code_bits,
                                              uint32_t dropped_bits) const {
    if (code_bits == 0) {
        return 1.0F;
    }

    const float center = 0.5F * static_cast<float>((1U << code_bits) - 1U);
    double norm_sqr = 0.0;
    for (uint64_t d = 0; d < this->dim_; ++d) {
        const auto code = static_cast<uint32_t>(scalar_codes[d]) >> dropped_bits;
        const float centered = static_cast<float>(code) - center;
        norm_sqr += static_cast<double>(centered) * centered;
    }

    auto norm = static_cast<norm_type>(std::sqrt(norm_sqr));
    if (not std::isfinite(norm) or norm <= 0.0F) {
        norm = 1.0F;
    }
    return norm;
}

template <MetricType metric>
typename RaBitQuantizer<metric>::norm_type
RaBitQuantizer<metric>::ComputeFilterCodeNorm(const uint8_t* filter_code,
                                              uint32_t filter_bits) const {
    if (filter_bits == 0) {
        return 1.0F;
    }

    const uint64_t plane_bytes = PlaneBytes();
    const float center = 0.5F * static_cast<float>((1U << filter_bits) - 1U);
    double norm_sqr = 0.0;
    for (uint64_t d = 0; d < this->dim_; ++d) {
        const uint64_t byte_idx = d >> 3;
        const auto bit_mask = static_cast<uint8_t>(1U << (d & 7));
        uint32_t code = 0;
        for (uint32_t bit = 0; bit < filter_bits; ++bit) {
            const auto* plane = filter_code + static_cast<uint64_t>(bit) * plane_bytes;
            if ((plane[byte_idx] & bit_mask) != 0U) {
                code += 1U << (filter_bits - bit - 1);
            }
        }
        const float centered = static_cast<float>(code) - center;
        norm_sqr += static_cast<double>(centered) * centered;
    }

    auto norm = static_cast<norm_type>(std::sqrt(norm_sqr));
    if (not std::isfinite(norm) or norm <= 0.0F) {
        norm = 1.0F;
    }
    return norm;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::StoredPlaneIndex(uint32_t logical_bit) const {
    if (not SupportSplitCodeStorage()) {
        return logical_bit;
    }
    const auto first_filter_bit = num_bits_per_dim_base_ - FilterBits();
    if (logical_bit >= first_filter_bit) {
        return static_cast<uint64_t>(num_bits_per_dim_base_ - 1 - logical_bit);
    }
    return static_cast<uint64_t>(FilterBits()) + logical_bit;
}

template <MetricType metric>
const uint8_t*
RaBitQuantizer<metric>::GetStoredPlane(const uint8_t* planes,
                                       uint32_t logical_bit,
                                       uint64_t plane_bytes) const {
    return planes + StoredPlaneIndex(logical_bit) * plane_bytes;
}

template <MetricType metric>
uint8_t*
RaBitQuantizer<metric>::GetStoredPlane(uint8_t* planes,
                                       uint32_t logical_bit,
                                       uint64_t plane_bytes) const {
    return planes + StoredPlaneIndex(logical_bit) * plane_bytes;
}

template <MetricType metric>
float
RaBitQuantizer<metric>::RaBitQFloatSQIPByPlanes(const float* query, const uint8_t* planes) const {
    uint64_t plane_bytes = (this->dim_ + 7) / 8;
    const auto filter_bits = SupportSplitCodeStorage() ? FilterBits() : static_cast<uint32_t>(1);
    const auto supplement_bits = SupportSplitCodeStorage()
                                     ? ReorderBits()
                                     : static_cast<uint32_t>(num_bits_per_dim_base_ - 1);
    const auto* one_bit_code =
        GetStoredPlane(planes, static_cast<uint32_t>(num_bits_per_dim_base_ - 1), plane_bytes);
    const auto* supplement_code =
        supplement_bits == 0 ? nullptr : GetStoredPlane(planes, 0, plane_bytes);
    if (filter_bits == 1) {
        return RaBitQFloatSplitCodeIP(
            query, one_bit_code, supplement_code, this->dim_, supplement_bits);
    }
    return RaBitQFloatSQIPBySplitCode(
        query, one_bit_code, supplement_code, filter_bits, supplement_bits);
}

template <MetricType metric>
float
RaBitQuantizer<metric>::RaBitQFloatSQIPBySplitCode(const float* query,
                                                   const uint8_t* one_bit_code,
                                                   const uint8_t* supplement_code) const {
    return RaBitQFloatSplitCodeIP(query, one_bit_code, supplement_code, this->dim_, ReorderBits());
}

template <MetricType metric>
float
RaBitQuantizer<metric>::RaBitQFloatSQIPBySplitCode(const float* query,
                                                   const uint8_t* filter_code,
                                                   const uint8_t* supplement_code,
                                                   uint32_t filter_bits,
                                                   uint32_t supplement_bits) const {
    if (this->dim_ == 0) {
        return 0.0F;
    }

    const uint64_t plane_bytes = PlaneBytes();
    float result = 0.0F;
    for (uint64_t d = 0; d < this->dim_; ++d) {
        const uint64_t byte_idx = d >> 3;
        const auto bit_mask = static_cast<uint8_t>(1U << (d & 7));
        uint32_t code = 0;
        for (uint32_t bit = 0; bit < filter_bits; ++bit) {
            const auto* plane = filter_code + static_cast<uint64_t>(bit) * plane_bytes;
            if ((plane[byte_idx] & bit_mask) != 0) {
                code += 1U << (supplement_bits + filter_bits - bit - 1);
            }
        }
        if (supplement_code != nullptr) {
            for (uint32_t bit = 0; bit < supplement_bits; ++bit) {
                const auto* plane = supplement_code + static_cast<uint64_t>(bit) * plane_bytes;
                if ((plane[byte_idx] & bit_mask) != 0) {
                    code += 1U << bit;
                }
            }
        }
        result += query[d] * static_cast<float>(code);
    }
    return result;
}

template <MetricType metric>
void
RaBitQuantizer<metric>::PackIntoPlanes(const uint8_t* src, uint8_t* dst) const {
    size_t plane_size = (this->dim_ + 7) / 8;
    memset(dst, 0, plane_size * num_bits_per_dim_base_);

    const uint8_t mask_n =
        (num_bits_per_dim_base_ == 8) ? 0xFFU : uint8_t((1U << num_bits_per_dim_base_) - 1U);

    for (uint64_t i = 0; i < this->dim_; ++i) {
        uint8_t v = src[i] & mask_n;
        const auto byte_idx = (i >> 3);
        const auto bit_in_byte = uint8_t(i & 7);
        const auto bitmask = uint8_t(1U << bit_in_byte);

        for (int b = 0; b < num_bits_per_dim_base_; ++b) {
            if ((v & (1U << b)) != 0U) {
                auto* plane = GetStoredPlane(dst, static_cast<uint32_t>(b), plane_size);
                plane[byte_idx] |= bitmask;
            }
        }
    }
}

template <MetricType metric>
void
RaBitQuantizer<metric>::EncodeExtendRaBitQ(const float* o_prime,
                                           uint8_t* code,
                                           float& y_norm) const {
    // used for encode float into multi-bit rabitq
    // we use y2 means 2 * y to avoid operations on 0.5
    constexpr double eps = 1e-12;  // for stability at boundaries
    const int y2_max = int((1U << this->num_bits_per_dim_base_) - 1U);  // e.g. 15
    const double c = 0.5 * double(y2_max);                              // e.g. 7.5
    const int step = 2;                                                 // y2 grid step

    auto clamp_int = [](int x, int lo, int hi) -> int { return x < lo ? lo : (x > hi ? hi : x); };

    auto round_clamp_parity = [&](double val) -> int {
        int lo = -y2_max;
        int hi = +y2_max;
        auto r = llround(val);
        int ri = clamp_int(r, lo, hi);

        if ((ri & 1) == (y2_max & 1)) {
            return ri;
        }

        int step = (val >= 0.0) ? +1 : -1;
        int cand = ri + step;

        if (cand < lo or cand > hi) {
            cand = ri - step;
        }
        return clamp_int(cand, lo, hi);
    };

    double max_o = 0.0;
    for (size_t i = 0; i < this->dim_; ++i) {
        max_o = std::max(max_o, std::fabs(double(o_prime[i])));
    }

    if (max_o <= 0.0) {
        for (size_t i = 0; i < this->dim_; ++i) {
            code[i] = uint8_t(y2_max / 2);
        }
        y_norm = 1.F;
        return;
    }

    // [step 1]: enumerate t
    std::vector<int> y2_cur(this->dim_, 0);
    const double t_start = 0.0;
    const double t_end = (double(y2_max) + 2.0) / (2.0 * max_o);
    double ip_y2_o = 0.0;
    double norm_y2 = 0.0;

    std::priority_queue<std::pair<double, std::size_t>,
                        std::vector<std::pair<double, std::size_t>>,
                        std::greater<>>
        pq;

    auto compute_next_t_for_dim = [&](size_t i) -> double {
        auto oi = double(o_prime[i]);
        if (std::fabs(oi) < 1e-3) {
            return std::numeric_limits<double>::infinity();
        }

        auto sign = (oi > 0.0) ? +1 : -1;
        auto y2_next = y2_cur[i] + sign * step;
        if (y2_next < -y2_max or y2_next > +y2_max) {
            return std::numeric_limits<double>::infinity();
        }

        auto t = double(y2_cur[i] + sign) / (2.0 * oi);

        if (t < 0.0) {
            t = 0.0;
        }
        return t;
    };

    for (size_t i = 0; i < this->dim_; ++i) {
        auto oi = double(o_prime[i]);
        if (oi == 0.0) {
            continue;
        }
        double t0 = compute_next_t_for_dim(i);
        if (std::isfinite(t0) and t0 <= t_end) {
            pq.emplace(t0, i);
        }
    }

    // [step 2]: choose a best t
    double best_ip = eps;
    double best_t = t_start;

    while (not pq.empty()) {
        auto cur_t = pq.top().first;
        auto k = pq.top().second;
        pq.pop();

        if (cur_t >= t_end) {
            break;
        }

        const int sign = (o_prime[k] > 0.0) ? +1 : -1;

        const int y2_old = y2_cur[k];
        const int y2_new = y2_old + sign * step;
        if (y2_new < -y2_max or y2_new > +y2_max) {
            // shouldn't happen because compute_next_t_for_dim filtered it
            continue;
        }
        y2_cur[k] = y2_new;

        ip_y2_o += (double(y2_new) - double(y2_old)) * o_prime[k];
        norm_y2 += double(y2_new) * double(y2_new) - double(y2_old) * double(y2_old);

        auto cur_ip = (norm_y2 > 0.0) ? (ip_y2_o / std::sqrt(norm_y2)) : 0.0;

        if (cur_ip > best_ip) {
            best_ip = cur_ip;
            best_t = cur_t;
        }

        double t_next = compute_next_t_for_dim(k);
        if (t_next <= cur_t) {
            t_next = std::nextafter(cur_t, std::numeric_limits<double>::infinity());
        }
        if (std::isfinite(t_next) and t_next < t_end) {
            pq.emplace(t_next, k);
        }
    }

    // [step 3]: encode the data according to best t
    std::vector<int> y2_bar(this->dim_, 0);
    for (size_t i = 0; i < this->dim_; ++i) {
        const double val = 2.0 * best_t * double(o_prime[i]);
        int y2 = round_clamp_parity(val + ((val >= 0) ? eps : -eps));
        y2_bar[i] = y2;

        int u = (y2 + y2_max) / 2;
        u = clamp_int(u, 0, y2_max);
        code[i] = uint8_t(u);
    }

    double sum_y2 = 0.0;
    for (size_t i = 0; i < this->dim_; ++i) {
        const double y = double(code[i]) - c;
        sum_y2 += y * y;
    }
    y_norm = float(std::sqrt(sum_y2));
    if (not std::isfinite(y_norm) or y_norm <= 0.F) {
        y_norm = 1.F;
    }
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::EncodeOneImpl(const float* data, uint8_t* codes) const {
    // 0. init
    Vector<float> pca_data(this->original_dim_, 0, this->allocator_);
    Vector<float> transformed_data(this->dim_, 0, this->allocator_);
    Vector<float> normed_data(this->dim_, 0, this->allocator_);
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

    if (num_bits_per_dim_base_ != 1) {
        float norm_code = 0;

        Vector<uint8_t> scalar_codes(this->dim_, 0, this->allocator_);
        EncodeExtendRaBitQ(normed_data.data(), scalar_codes.data(), norm_code);
        norm_type filter_norm_code = norm_code;
        if (SupportSplitCodeStorage() and HasMultiBitFilter()) {
            filter_norm_code =
                ComputeScalarCodeNorm(scalar_codes.data(), FilterBits(), ReorderBits());
        }
        PackIntoPlanes(scalar_codes.data(), codes + offset_code_);

        *(norm_type*)(codes + offset_norm_code_) = norm_code;

        // 5. compute encode error
        float o_sum = 0;
        for (auto i = 0; i < this->dim_; i++) {
            o_sum += normed_data[i];
        }

        error_type one_bit_error = 0.0F;
        if (SupportSplitCodeStorage()) {
            uint64_t plane_bytes = (this->dim_ + 7) / 8;
            const auto* one_bit_plane =
                GetStoredPlane(codes + offset_code_,
                               static_cast<uint32_t>(num_bits_per_dim_base_ - 1),
                               plane_bytes);
            if (FilterBits() == 1) {
                one_bit_error =
                    RaBitQFloatBinaryIP(normed_data.data(), one_bit_plane, this->dim_, inv_sqrt_d_);
            } else {
                const float filter_ip_yu_q = RaBitQFloatSQIPBySplitCode(
                    normed_data.data(), one_bit_plane, nullptr, FilterBits(), 0);
                one_bit_error = ip_obar_q(filter_ip_yu_q, o_sum, filter_norm_code, FilterBits());
            }
        }

        float ip_yu_q = RaBitQFloatSQIPByPlanes(normed_data.data(), codes + offset_code_);

        error_type error = ip_obar_q(ip_yu_q, o_sum, norm_code, num_bits_per_dim_base_);

        // 6. store norm, error, sum
        *(norm_type*)(codes + offset_norm_) = norm;
        *(error_type*)(codes + offset_error_) = error;
        if (SupportSplitCodeStorage()) {
            one_bit_error = std::fabs(one_bit_error);
            const float safe_one_bit_error = std::clamp(one_bit_error, 1e-5F, 1.0F);
            error_type low_bound_error =
                std::sqrt(std::max(0.0F, 1.0F - safe_one_bit_error * safe_one_bit_error) /
                          std::max(1.0F, float(this->dim_ - 1)));
            *(error_type*)(codes + offset_one_bit_error_) = one_bit_error;
            *(error_type*)(codes + offset_low_bound_error_) = low_bound_error;
        }
    } else {
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

        if (SupportSplitCodeStorage()) {
            *(error_type*)(codes + offset_one_bit_error_) = error;
            *(error_type*)(codes + offset_low_bound_error_) = 0.0F;
        }

        if (num_bits_per_dim_query_ != 32) {
            *(sum_type*)(codes + offset_sum_) = sum;
        }
    }

    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        *(norm_type*)(codes + offset_raw_norm_) = raw_norm;
    }
    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        // TODO(ZXY): use batch optimize
        this->EncodeOneImpl(data + i * this->original_dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, float* data) {
    if (pca_dim_ != this->original_dim_) {
        return false;
    }

    // 1. init
    Vector<float> normed_data(this->dim_, 0, this->allocator_);
    Vector<float> transformed_data(this->dim_, 0, this->allocator_);

    // 2. decode with BQ
    if (num_bits_per_dim_base_ == 1) {
        for (uint64_t d = 0; d < this->dim_; ++d) {
            bool bit = ((codes[d / 8] >> (d % 8)) & 1) != 0;
            normed_data[d] = bit ? inv_sqrt_d_ : -inv_sqrt_d_;
        }
    } else {
        return false;
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
RaBitQuantizer<metric>::DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count) {
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
uint64_t
RaBitQuantizer<metric>::AlignCodeField(uint64_t size) const {
    uint64_t align_size = std::max(std::max(sizeof(error_type), sizeof(norm_type)), sizeof(float));
    return ((size + align_size - 1) / align_size) * align_size;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::HasFilterQueryLookupTable() const {
    return SupportSplitCodeStorage() && (FilterBits() == 2 or FilterBits() == 3) &&
           num_bits_per_dim_query_ == 32;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::FilterQueryLookupTableSize() const {
    return PlaneBytes() * 256 * sizeof(float);
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::PlaneBytes() const {
    return (this->dim_ + 7) / 8;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::CodePlanesSize() const {
    return AlignCodeField(PlaneBytes() * num_bits_per_dim_base_);
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::CodeMetaOffset() const {
    return offset_code_ + CodePlanesSize();
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::SupplementPlanesSize() const {
    if (ReorderBits() == 0) {
        return 0;
    }
    return PlaneBytes() * ReorderBits();
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::SupplementMetaOffset() const {
    return SupplementPlanesSize();
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::SupportSplitCodeStorage() const {
    return RaBitQuantizerParameter::IsSplitVersion(rabitq_version_) &&
           num_bits_per_dim_query_ == 32 && num_bits_per_dim_base_ >= 1 &&
           num_bits_per_dim_filter_ >= 1 && num_bits_per_dim_filter_ <= num_bits_per_dim_base_;
}

template <MetricType metric>
uint32_t
RaBitQuantizer<metric>::FilterBits() const {
    return SupportSplitCodeStorage() ? num_bits_per_dim_filter_ : 1;
}

template <MetricType metric>
uint32_t
RaBitQuantizer<metric>::ReorderBits() const {
    if (not SupportSplitCodeStorage()) {
        return 0;
    }
    return num_bits_per_dim_base_ - num_bits_per_dim_filter_;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::FilterPlanesSize() const {
    return PlaneBytes() * FilterBits();
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::HasMultiBitFilter() const {
    return SupportSplitCodeStorage() && FilterBits() > 1 && num_bits_per_dim_base_ > 1;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::OneBitRecordNormOffset() const {
    return AlignCodeField(FilterPlanesSize());
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::OneBitRecordNormCodeOffset() const {
    return OneBitRecordNormOffset() + AlignCodeField(sizeof(norm_type));
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::OneBitRecordMrqNormOffset() const {
    uint64_t offset = OneBitRecordNormOffset() + AlignCodeField(sizeof(norm_type));
    if (HasMultiBitFilter()) {
        offset += AlignCodeField(sizeof(norm_type));
    }
    return offset;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::OneBitRecordRawNormOffset() const {
    uint64_t offset = OneBitRecordNormOffset() + AlignCodeField(sizeof(norm_type));
    if (HasMultiBitFilter()) {
        offset += AlignCodeField(sizeof(norm_type));
    }
    if (pca_dim_ != original_dim_ && use_mrq_) {
        offset += AlignCodeField(sizeof(norm_type));
    }
    return offset;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::OneBitRecordLowBoundErrorOffset() const {
    uint64_t offset = OneBitRecordNormOffset() + AlignCodeField(sizeof(norm_type));
    if (HasMultiBitFilter()) {
        offset += AlignCodeField(sizeof(norm_type));
    }
    if (pca_dim_ != original_dim_ && use_mrq_) {
        offset += AlignCodeField(sizeof(norm_type));
    }
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        offset += AlignCodeField(sizeof(norm_type));
    }
    return offset;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::OneBitRecordOneBitErrorOffset() const {
    return OneBitRecordLowBoundErrorOffset() + AlignCodeField(sizeof(error_type));
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::OneBitRecordSize() const {
    return OneBitRecordOneBitErrorOffset() + AlignCodeField(sizeof(error_type));
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::GetOneBitCodeSize() const {
    if (not SupportSplitCodeStorage()) {
        return this->code_size_;
    }
    return OneBitRecordSize();
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::GetSupplementCodeSize() const {
    if (not SupportSplitCodeStorage()) {
        return 0;
    }
    return SupplementMetaOffset() + (this->code_size_ - CodeMetaOffset());
}

template <MetricType metric>
void
RaBitQuantizer<metric>::SplitCode(const uint8_t* full_code,
                                  uint8_t* one_bit_code,
                                  uint8_t* supplement_code) const {
    if (not SupportSplitCodeStorage()) {
        return;
    }

    memset(one_bit_code, 0, GetOneBitCodeSize());
    memset(supplement_code, 0, GetSupplementCodeSize());

    const auto filter_planes_size = FilterPlanesSize();
    memcpy(one_bit_code, full_code + offset_code_, filter_planes_size);
    memcpy(one_bit_code + OneBitRecordNormOffset(), full_code + offset_norm_, sizeof(norm_type));
    if (HasMultiBitFilter()) {
        const norm_type filter_norm_code = ComputeFilterCodeNorm(one_bit_code, FilterBits());
        memcpy(one_bit_code + OneBitRecordNormCodeOffset(), &filter_norm_code, sizeof(norm_type));
    }
    if (pca_dim_ != original_dim_ && use_mrq_) {
        memcpy(one_bit_code + OneBitRecordMrqNormOffset(),
               full_code + offset_mrq_norm_,
               sizeof(norm_type));
    }
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        memcpy(one_bit_code + OneBitRecordRawNormOffset(),
               full_code + offset_raw_norm_,
               sizeof(norm_type));
    }
    memcpy(one_bit_code + OneBitRecordLowBoundErrorOffset(),
           full_code + offset_low_bound_error_,
           sizeof(error_type));
    memcpy(one_bit_code + OneBitRecordOneBitErrorOffset(),
           full_code + offset_one_bit_error_,
           sizeof(error_type));

    memcpy(supplement_code, full_code + offset_code_ + filter_planes_size, SupplementPlanesSize());
    memcpy(supplement_code + SupplementMetaOffset(),
           full_code + CodeMetaOffset(),
           this->code_size_ - CodeMetaOffset());
}

template <MetricType metric>
void
RaBitQuantizer<metric>::MergeSplitCode(const uint8_t* one_bit_code,
                                       const uint8_t* supplement_code,
                                       uint8_t* full_code) const {
    if (not SupportSplitCodeStorage()) {
        return;
    }

    memset(full_code, 0, this->code_size_);

    const auto filter_planes_size = FilterPlanesSize();
    memcpy(full_code + offset_code_, one_bit_code, filter_planes_size);
    memcpy(full_code + offset_code_ + filter_planes_size, supplement_code, SupplementPlanesSize());
    memcpy(full_code + CodeMetaOffset(),
           supplement_code + SupplementMetaOffset(),
           this->code_size_ - CodeMetaOffset());
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::ComputeDistWithOneBitLowerBound(Computer<RaBitQuantizer>& computer,
                                                        const uint8_t* one_bit_code,
                                                        float* dists,
                                                        float* lower_bound,
                                                        float runtime_rabitq_error_rate) const {
    if (lower_bound != nullptr) {
        *lower_bound = std::numeric_limits<float>::max();
    }
    if (not SupportSplitCodeStorage()) {
        return false;
    }

    const auto* query = computer.buf_;
    const error_type one_bit_error =
        std::fabs(*((error_type*)(one_bit_code + OneBitRecordOneBitErrorOffset())));
    if (one_bit_error <= 1e-5F) {
        return false;
    }

    float filter_ip_estimate = 0.0F;
    float filter_ip_yu_q = 0.0F;
    norm_type base_norm_code = 0.0F;
    if (FilterBits() == 1) {
        filter_ip_estimate = RaBitQFloatBinaryIP(
            reinterpret_cast<const float*>(query), one_bit_code, this->dim_, inv_sqrt_d_);
    } else {
        sum_type query_raw_sum = *((sum_type*)(query + query_offset_sum_));
        if (HasFilterQueryLookupTable()) {
            filter_ip_yu_q = RaBitQFloatMultiBitIPByLookup(
                reinterpret_cast<const float*>(query + query_offset_filter_lut_),
                one_bit_code,
                this->dim_,
                0,
                FilterBits());
        } else {
            filter_ip_yu_q = RaBitQFloatSQIPBySplitCode(
                reinterpret_cast<const float*>(query), one_bit_code, nullptr, FilterBits(), 0);
        }
        memcpy(
            &base_norm_code, one_bit_code + OneBitRecordNormCodeOffset(), sizeof(base_norm_code));
        filter_ip_estimate = ip_obar_q(filter_ip_yu_q, query_raw_sum, base_norm_code, FilterBits());
    }
    float ip_est = filter_ip_estimate / one_bit_error;

    norm_type query_norm = *((norm_type*)(query + query_offset_norm_));
    norm_type base_norm = *((norm_type*)(one_bit_code + OneBitRecordNormOffset()));
    float result = l2_ube(base_norm, query_norm, ip_est);

    if (pca_dim_ != this->original_dim_ && use_mrq_) {
        norm_type query_mrq_norm_sqr = *(norm_type*)(query + query_offset_mrq_norm_);
        norm_type base_mrq_norm_sqr = *(norm_type*)(one_bit_code + OneBitRecordMrqNormOffset());
        result += (query_mrq_norm_sqr + base_mrq_norm_sqr);
    }

    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        norm_type query_raw_norm = *((norm_type*)(query + query_offset_raw_norm_));
        norm_type base_raw_norm = *((norm_type*)(one_bit_code + OneBitRecordRawNormOffset()));
        if (is_approx_zero(query_raw_norm) or is_approx_zero(base_raw_norm)) {
            result = 1;
        } else {
            result =
                1 - (query_raw_norm * query_raw_norm + base_raw_norm * base_raw_norm - result) *
                        0.5F / (query_raw_norm * base_raw_norm);
        }
    }
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        norm_type query_raw_norm = *((norm_type*)(query + query_offset_raw_norm_));
        norm_type base_raw_norm = *((norm_type*)(one_bit_code + OneBitRecordRawNormOffset()));
        if (is_approx_zero(query_raw_norm) or is_approx_zero(base_raw_norm)) {
            result = 1;
        } else {
            result =
                1 -
                (query_raw_norm * query_raw_norm + base_raw_norm * base_raw_norm - result) * 0.5F;
        }
    }

    if (not std::isfinite(result)) {
        return false;
    }

    *dists = result;
    if (lower_bound == nullptr) {
        return true;
    }

    error_type low_bound_error = *((error_type*)(one_bit_code + OneBitRecordLowBoundErrorOffset()));
    const float effective_error_rate =
        std::isfinite(runtime_rabitq_error_rate) and runtime_rabitq_error_rate > 0.0F
            ? runtime_rabitq_error_rate
            : rabitq_error_rate_;
    float lower_bound_error_term =
        2.0F * base_norm * query_norm * effective_error_rate * low_bound_error / one_bit_error;
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        norm_type query_raw_norm = *((norm_type*)(query + query_offset_raw_norm_));
        norm_type base_raw_norm = *((norm_type*)(one_bit_code + OneBitRecordRawNormOffset()));
        if (not is_approx_zero(query_raw_norm) and not is_approx_zero(base_raw_norm)) {
            lower_bound_error_term *= 0.5F / (query_raw_norm * base_raw_norm);
        }
    }
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        lower_bound_error_term *= 0.5F;
    }

    float lower_bound_result = result - lower_bound_error_term;
    if (std::isfinite(lower_bound_result)) {
        *lower_bound = lower_bound_result - 1e-5F * std::max(1.0F, std::fabs(lower_bound_result));
    }
    return true;
}

template <MetricType metric>
void
RaBitQuantizer<metric>::ComputeDistsWithOneBitLowerBoundBatch4(
    Computer<RaBitQuantizer>& computer,
    const uint8_t* one_bit_code1,
    const uint8_t* one_bit_code2,
    const uint8_t* one_bit_code3,
    const uint8_t* one_bit_code4,
    float& dist1,
    float& dist2,
    float& dist3,
    float& dist4,
    float* lower_bound1,
    float* lower_bound2,
    float* lower_bound3,
    float* lower_bound4,
    bool& computed1,
    bool& computed2,
    bool& computed3,
    bool& computed4,
    float runtime_rabitq_error_rate) const {
    if constexpr (metric != MetricType::METRIC_TYPE_L2SQR) {
        computed1 = this->ComputeDistWithOneBitLowerBound(
            computer, one_bit_code1, &dist1, lower_bound1, runtime_rabitq_error_rate);
        computed2 = this->ComputeDistWithOneBitLowerBound(
            computer, one_bit_code2, &dist2, lower_bound2, runtime_rabitq_error_rate);
        computed3 = this->ComputeDistWithOneBitLowerBound(
            computer, one_bit_code3, &dist3, lower_bound3, runtime_rabitq_error_rate);
        computed4 = this->ComputeDistWithOneBitLowerBound(
            computer, one_bit_code4, &dist4, lower_bound4, runtime_rabitq_error_rate);
        return;
    }

    if (FilterBits() != 1 and not HasFilterQueryLookupTable()) {
        computed1 = this->ComputeDistWithOneBitLowerBound(
            computer, one_bit_code1, &dist1, lower_bound1, runtime_rabitq_error_rate);
        computed2 = this->ComputeDistWithOneBitLowerBound(
            computer, one_bit_code2, &dist2, lower_bound2, runtime_rabitq_error_rate);
        computed3 = this->ComputeDistWithOneBitLowerBound(
            computer, one_bit_code3, &dist3, lower_bound3, runtime_rabitq_error_rate);
        computed4 = this->ComputeDistWithOneBitLowerBound(
            computer, one_bit_code4, &dist4, lower_bound4, runtime_rabitq_error_rate);
        return;
    }

    const auto* query = computer.buf_;
    const auto* query_data = reinterpret_cast<const float*>(query);
    const norm_type query_norm = *((norm_type*)(query + query_offset_norm_));
    const norm_type query_mrq_norm_sqr = pca_dim_ != this->original_dim_ and use_mrq_
                                             ? *(norm_type*)(query + query_offset_mrq_norm_)
                                             : 0.0F;

    if (not SupportSplitCodeStorage()) {
        computed1 = false;
        computed2 = false;
        computed3 = false;
        computed4 = false;
        return;
    }

    float filter_ip_values[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    if (FilterBits() == 1) {
        RaBitQFloatBinaryIPBatch4(query_data,
                                  one_bit_code1,
                                  one_bit_code2,
                                  one_bit_code3,
                                  one_bit_code4,
                                  this->dim_,
                                  inv_sqrt_d_,
                                  filter_ip_values);
    } else {
        RaBitQFloatMultiBitIPBatch4ByLookup(
            reinterpret_cast<const float*>(query + query_offset_filter_lut_),
            one_bit_code1,
            one_bit_code2,
            one_bit_code3,
            one_bit_code4,
            this->dim_,
            0,
            FilterBits(),
            filter_ip_values);
    }

    auto compute_one = [&](const uint8_t* one_bit_code,
                           float filter_ip,
                           float& dist,
                           float* lower_bound) {
        if (lower_bound != nullptr) {
            *lower_bound = std::numeric_limits<float>::max();
        }

        const error_type one_bit_error =
            std::fabs(*((error_type*)(one_bit_code + OneBitRecordOneBitErrorOffset())));
        if (one_bit_error <= 1e-5F) {
            return false;
        }

        float filter_ip_yu_q = filter_ip;
        float filter_ip_estimate = filter_ip;
        norm_type base_norm_code = 0.0F;
        if (FilterBits() > 1) {
            const sum_type query_raw_sum = *((sum_type*)(query + query_offset_sum_));
            memcpy(&base_norm_code,
                   one_bit_code + OneBitRecordNormCodeOffset(),
                   sizeof(base_norm_code));
            filter_ip_estimate = ip_obar_q(filter_ip, query_raw_sum, base_norm_code, FilterBits());
        }

        const float ip_est = filter_ip_estimate / one_bit_error;
        const norm_type base_norm = *((norm_type*)(one_bit_code + OneBitRecordNormOffset()));
        float result = l2_ube(base_norm, query_norm, ip_est);

        if (pca_dim_ != this->original_dim_ and use_mrq_) {
            const norm_type base_mrq_norm_sqr =
                *(norm_type*)(one_bit_code + OneBitRecordMrqNormOffset());
            result += (query_mrq_norm_sqr + base_mrq_norm_sqr);
        }

        if (not std::isfinite(result)) {
            return false;
        }

        dist = result;
        if (lower_bound == nullptr) {
            return true;
        }

        const error_type low_bound_error =
            *((error_type*)(one_bit_code + OneBitRecordLowBoundErrorOffset()));
        const float effective_error_rate =
            std::isfinite(runtime_rabitq_error_rate) and runtime_rabitq_error_rate > 0.0F
                ? runtime_rabitq_error_rate
                : rabitq_error_rate_;
        const float lower_bound_error_term =
            2.0F * base_norm * query_norm * effective_error_rate * low_bound_error / one_bit_error;
        const float lower_bound_result = result - lower_bound_error_term;
        if (std::isfinite(lower_bound_result)) {
            *lower_bound =
                lower_bound_result - 1e-5F * std::max(1.0F, std::fabs(lower_bound_result));
        }
        return true;
    };

    computed1 = compute_one(one_bit_code1, filter_ip_values[0], dist1, lower_bound1);
    computed2 = compute_one(one_bit_code2, filter_ip_values[1], dist2, lower_bound2);
    computed3 = compute_one(one_bit_code3, filter_ip_values[2], dist3, lower_bound3);
    computed4 = compute_one(one_bit_code4, filter_ip_values[3], dist4, lower_bound4);
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::ComputeDistWithSplitCode(Computer<RaBitQuantizer>& computer,
                                                 const uint8_t* one_bit_code,
                                                 const uint8_t* supplement_code,
                                                 float* dists) const {
    if (not SupportSplitCodeStorage()) {
        return false;
    }

    const auto* query = computer.buf_;
    const auto* split_meta = supplement_code + SupplementMetaOffset();
    const auto code_meta_offset = CodeMetaOffset();
    auto meta_field = [split_meta, code_meta_offset](uint64_t offset) {
        return split_meta + (offset - code_meta_offset);
    };

    float ip_bq_estimate = 0.0F;
    if (num_bits_per_dim_base_ == 1) {
        ip_bq_estimate = RaBitQFloatBinaryIP(
            reinterpret_cast<const float*>(query), one_bit_code, this->dim_, inv_sqrt_d_);
    } else if (FilterBits() == 1) {
        sum_type query_raw_sum = *((sum_type*)(query + query_offset_sum_));
        float ip_yu_q = RaBitQFloatSplitCodeIP(reinterpret_cast<const float*>(query),
                                               one_bit_code,
                                               supplement_code,
                                               this->dim_,
                                               ReorderBits());

        norm_type base_norm_code = 0;
        memcpy(&base_norm_code, meta_field(offset_norm_code_), sizeof(base_norm_code));
        ip_bq_estimate = ip_obar_q(ip_yu_q, query_raw_sum, base_norm_code, num_bits_per_dim_base_);
    } else {
        sum_type query_raw_sum = *((sum_type*)(query + query_offset_sum_));
        float ip_yu_q = 0.0F;
        if (HasFilterQueryLookupTable()) {
            ip_yu_q = RaBitQFloatMultiBitIPByLookup(
                reinterpret_cast<const float*>(query + query_offset_filter_lut_),
                one_bit_code,
                this->dim_,
                ReorderBits(),
                FilterBits());
            ip_yu_q += RaBitQFloatSupplementCodeIP(
                reinterpret_cast<const float*>(query), supplement_code, this->dim_, ReorderBits());
        } else {
            ip_yu_q = RaBitQFloatSQIPBySplitCode(reinterpret_cast<const float*>(query),
                                                 one_bit_code,
                                                 supplement_code,
                                                 FilterBits(),
                                                 ReorderBits());
        }

        norm_type base_norm_code = 0;
        memcpy(&base_norm_code, meta_field(offset_norm_code_), sizeof(base_norm_code));
        ip_bq_estimate = ip_obar_q(ip_yu_q, query_raw_sum, base_norm_code, num_bits_per_dim_base_);
    }

    norm_type query_norm = *((norm_type*)(query + query_offset_norm_));
    norm_type base_norm = 0;
    memcpy(&base_norm, meta_field(offset_norm_), sizeof(base_norm));

    norm_type query_raw_norm = 0;
    norm_type base_raw_norm = 0;
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        query_raw_norm = *((norm_type*)(query + query_offset_raw_norm_));
        memcpy(&base_raw_norm, meta_field(offset_raw_norm_), sizeof(base_raw_norm));
    }

    error_type base_error = 0;
    memcpy(&base_error, meta_field(offset_error_), sizeof(base_error));
    if (std::abs(base_error) < 1e-5) {
        base_error = (base_error >= 0) ? 1.0F : -1.0F;
    }

    float ip_est = ip_bq_estimate / base_error;
    float result = l2_ube(base_norm, query_norm, ip_est);

    if (pca_dim_ != this->original_dim_ and use_mrq_) {
        norm_type query_mrq_norm_sqr = *(norm_type*)(query + query_offset_mrq_norm_);
        norm_type base_mrq_norm_sqr = 0;
        memcpy(&base_mrq_norm_sqr, meta_field(offset_mrq_norm_), sizeof(base_mrq_norm_sqr));
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

    *dists = result;
    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::ComputeDistWithSplitCodeAndFilterDist(Computer<RaBitQuantizer>& computer,
                                                              const uint8_t* one_bit_code,
                                                              const uint8_t* supplement_code,
                                                              float filter_dist,
                                                              float* dists) const {
    if constexpr (metric != MetricType::METRIC_TYPE_L2SQR) {
        return false;
    }

    if (not SupportSplitCodeStorage() or FilterBits() <= 1 or not HasFilterQueryLookupTable() or
        not std::isfinite(filter_dist)) {
        return false;
    }

    const auto* query = computer.buf_;
    const auto* split_meta = supplement_code + SupplementMetaOffset();
    const auto code_meta_offset = CodeMetaOffset();
    auto meta_field = [split_meta, code_meta_offset](uint64_t offset) {
        return split_meta + (offset - code_meta_offset);
    };

    const norm_type query_norm = *((norm_type*)(query + query_offset_norm_));
    const norm_type base_norm = *((norm_type*)(one_bit_code + OneBitRecordNormOffset()));
    const error_type filter_error =
        std::fabs(*((error_type*)(one_bit_code + OneBitRecordOneBitErrorOffset())));
    if (filter_error <= 1e-5F or is_approx_zero(query_norm) or is_approx_zero(base_norm)) {
        return false;
    }

    float filter_l2 = filter_dist;
    if (pca_dim_ != this->original_dim_ and use_mrq_) {
        const norm_type query_mrq_norm_sqr = *(norm_type*)(query + query_offset_mrq_norm_);
        const norm_type base_mrq_norm_sqr =
            *(norm_type*)(one_bit_code + OneBitRecordMrqNormOffset());
        filter_l2 -= query_mrq_norm_sqr + base_mrq_norm_sqr;
    }

    const float filter_ip_est = (base_norm * base_norm + query_norm * query_norm - filter_l2) /
                                (2.0F * base_norm * query_norm) * filter_error;

    norm_type filter_norm_code = 0;
    memcpy(
        &filter_norm_code, one_bit_code + OneBitRecordNormCodeOffset(), sizeof(filter_norm_code));
    if (filter_norm_code <= 0.0F) {
        return false;
    }

    norm_type full_norm_code = 0;
    memcpy(&full_norm_code, meta_field(offset_norm_code_), sizeof(full_norm_code));
    if (full_norm_code <= 0.0F) {
        return false;
    }

    const sum_type query_raw_sum = *((sum_type*)(query + query_offset_sum_));
    const float filter_center = 0.5F * static_cast<float>((1U << FilterBits()) - 1U);
    const float filter_ip_yu_q = filter_ip_est * filter_norm_code + filter_center * query_raw_sum;
    const float shifted_filter_ip_yu_q = filter_ip_yu_q * static_cast<float>(1U << ReorderBits());
    const float supplement_ip_yu_q = RaBitQFloatSupplementCodeIP(
        reinterpret_cast<const float*>(query), supplement_code, this->dim_, ReorderBits());
    const float ip_bq_estimate = ip_obar_q(shifted_filter_ip_yu_q + supplement_ip_yu_q,
                                           query_raw_sum,
                                           full_norm_code,
                                           num_bits_per_dim_base_);

    error_type base_error = 0;
    memcpy(&base_error, meta_field(offset_error_), sizeof(base_error));
    if (std::abs(base_error) < 1e-5F) {
        base_error = (base_error >= 0) ? 1.0F : -1.0F;
    }

    float result = l2_ube(base_norm, query_norm, ip_bq_estimate / base_error);
    if (pca_dim_ != this->original_dim_ and use_mrq_) {
        const norm_type query_mrq_norm_sqr = *(norm_type*)(query + query_offset_mrq_norm_);
        norm_type base_mrq_norm_sqr = 0;
        memcpy(&base_mrq_norm_sqr, meta_field(offset_mrq_norm_), sizeof(base_mrq_norm_sqr));
        result += query_mrq_norm_sqr + base_mrq_norm_sqr;
    }

    if (not std::isfinite(result)) {
        return false;
    }
    *dists = result;
    return true;
}

template <MetricType metric>
float
RaBitQuantizer<metric>::ComputeQueryBaseImpl(const uint8_t* query_codes,
                                             const uint8_t* base_codes) const {
    // codes1 -> query (fp32, sq8, sq4...) + norm
    // codes2 -> base  (binary) + norm + error
    float ip_bq_estimate = 0;
    if (num_bits_per_dim_query_ == 4 and num_bits_per_dim_base_ == 1) {
        //
        std::vector<uint8_t> tmp(aligned_dim_ / 8, 0);
        memcpy(tmp.data(), base_codes, offset_norm_);

        ip_bq_estimate = RaBitQSQ4UBinaryIP(query_codes, tmp.data(), aligned_dim_);

        sum_type base_sum = *reinterpret_cast<const sum_type*>(base_codes + offset_sum_);
        sum_type query_sum = *((sum_type*)(query_codes + query_offset_sum_));
        float lower_bound = *((float*)(query_codes + query_offset_lb_));
        float delta = *((float*)(query_codes + query_offset_delta_));

        ip_bq_estimate = recover_dist_between_sq4u_and_fp32(
            ip_bq_estimate, base_sum, query_sum, lower_bound, delta, this->inv_sqrt_d_, this->dim_);
    } else if (num_bits_per_dim_query_ == 32 and num_bits_per_dim_base_ == 1) {
        ip_bq_estimate = RaBitQFloatBinaryIP(
            reinterpret_cast<const float*>(query_codes), base_codes, this->dim_, inv_sqrt_d_);
    } else if (num_bits_per_dim_query_ == 32 and num_bits_per_dim_base_ != 1) {
        sum_type query_raw_sum = *((sum_type*)(query_codes + query_offset_sum_));

        float ip_yu_q = RaBitQFloatSQIPByPlanes(reinterpret_cast<const float*>(query_codes),
                                                base_codes + offset_code_);

        ip_bq_estimate = ip_obar_q(ip_yu_q,
                                   query_raw_sum,
                                   *(norm_type*)(base_codes + offset_norm_code_),
                                   num_bits_per_dim_base_);
    } else {
        // num_bits_per_dim_query_ == 4 and num_bits_per_dim_base_ != 1: not support for now
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
        base_error = (base_error >= 0) ? 1.0F : -1.0F;
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
RaBitQuantizer<metric>::EncodeSQ(const float* normed_data,
                                 uint8_t* quantized_data,
                                 float& upper_bound,
                                 float& lower_bound,
                                 float& delta,
                                 sum_type& query_sum) const {
    lower_bound = std::numeric_limits<float>::max();
    upper_bound = std::numeric_limits<float>::lowest();
    for (uint64_t i = 0; i < this->dim_; i++) {
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
    uint64_t offset = aligned_dim_ / 8;
    for (uint64_t d = 0; d < this->dim_; d++) {
        for (uint64_t bit_pos = 0; bit_pos < num_bits_per_dim_query_; bit_pos++) {
            const bool bit = ((quantized_data[d] & (1 << bit_pos)) != 0);
            reorder_data[bit_pos * offset + d / 8] |= (static_cast<int32_t>(bit) * (1 << (d % 8)));
        }
    }
}

template <MetricType metric>
void
RaBitQuantizer<metric>::DecodeSQ(const uint8_t* codes,
                                 float* data,
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
    uint64_t offset = aligned_dim_ / 8;
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
RaBitQuantizer<metric>::ProcessQueryImpl(const float* query,
                                         Computer<RaBitQuantizer>& computer) const {
    try {
        if (computer.buf_ == nullptr) {
            computer.buf_ =
                reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        }
        std::fill(computer.buf_, computer.buf_ + this->query_code_size_, 0);

        // use residual term in pca, so it's this->original_dim_
        Vector<float> pca_data(this->original_dim_, 0, this->allocator_);
        Vector<float> transformed_data(this->dim_, 0, this->allocator_);
        Vector<float> normed_data(this->dim_, 0, this->allocator_);

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
            *(float*)(computer.buf_ + query_offset_lb_) = lower_bound;
            *(float*)(computer.buf_ + query_offset_delta_) = delta;
            *(sum_type*)(computer.buf_ + query_offset_sum_) = query_sum;
        } else {
            // store codes
            memcpy(computer.buf_, normed_data.data(), normed_data.size() * sizeof(float));
        }

        if (num_bits_per_dim_base_ != 1) {
            float query_raw_sum = 0;
            for (uint32_t d = 0; d < this->dim_; d++) {
                query_raw_sum += normed_data[d];
            }
            *(sum_type*)(computer.buf_ + query_offset_sum_) = query_raw_sum;
        }

        if (HasFilterQueryLookupTable()) {
            generic::RaBitQFloatBuildByteIPLookupTable(
                normed_data.data(),
                this->dim_,
                reinterpret_cast<float*>(computer.buf_ + query_offset_filter_lut_));
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
