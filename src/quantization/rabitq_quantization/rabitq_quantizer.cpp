
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

namespace {

[[nodiscard]] bool
is_normal_ra_bit_q_value(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    constexpr uint32_t k_exponent_mask = 0x7F800000U;
    const uint32_t exponent = bits & k_exponent_mask;
    return exponent != 0U and exponent != k_exponent_mask;
}

}  // namespace

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
                                       uint64_t num_bits_per_dim_filter,
                                       bool fast_encode_rabitq,
                                       uint64_t fast_encode_rabitq_rounds)
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
    const bool support_split_code_storage =
        RaBitQuantizerParameter::IsSplitVersion(rabitq_version_) && num_bits_per_dim_query_ == 32 &&
        num_bits_per_dim_base_ >= 1 && num_bits_per_dim_filter_ >= 1 &&
        num_bits_per_dim_filter_ <= num_bits_per_dim_base_;
    rabitq_error_rate_ = rabitq_error_rate;
    fast_encode_rabitq_ = fast_encode_rabitq;
    fast_encode_rabitq_rounds_ = fast_encode_rabitq_rounds;
    if (fast_encode_rabitq_rounds_ < RaBitQuantizerParameter::MIN_FAST_ENCODE_RABITQ_ROUNDS or
        fast_encode_rabitq_rounds_ > RaBitQuantizerParameter::MAX_FAST_ENCODE_RABITQ_ROUNDS) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("fast_encode_rabitq_rounds must be in [{}, {}], but got {}",
                                        RaBitQuantizerParameter::MIN_FAST_ENCODE_RABITQ_ROUNDS,
                                        RaBitQuantizerParameter::MAX_FAST_ENCODE_RABITQ_ROUNDS,
                                        fast_encode_rabitq_rounds_));
    }

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

    if (support_split_code_storage) {
        offset_low_bound_error_ = this->code_size_;
        this->code_size_ += ((sizeof(error_type) + align_size - 1) / align_size) * align_size;

        offset_one_bit_error_ = this->code_size_;
        this->code_size_ += ((sizeof(error_type) + align_size - 1) / align_size) * align_size;
    }
    RefreshSplitLayout(support_split_code_storage);
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
                             param->num_bits_per_dim_filter_,
                             param->fast_encode_rabitq_,
                             param->fast_encode_rabitq_rounds_){};

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

template <MetricType metric>
void
RaBitQuantizer<metric>::SetCentroid(const float* centroid) {
    CHECK_ARGUMENT(centroid != nullptr, "RaBitQ centroid must not be null");
    Vector<float> pca_centroid(this->original_dim_, 0.0F, this->allocator_);
    Vector<float> rotated_centroid(this->dim_, 0.0F, this->allocator_);
    if (pca_dim_ != this->original_dim_) {
        pca_->Transform(centroid, pca_centroid.data());
    } else {
        std::copy(centroid, centroid + original_dim_, pca_centroid.begin());
    }
    rom_->Transform(pca_centroid.data(), rotated_centroid.data());
    centroid_.assign(rotated_centroid.begin(), rotated_centroid.end());
}

template <MetricType metric>
void
RaBitQuantizer<metric>::ShareFusedModelFrom(const RaBitQuantizer& source) {
    CHECK_ARGUMENT(this->dim_ == source.dim_ and this->original_dim_ == source.original_dim_ and
                       this->pca_dim_ == source.pca_dim_ and this->use_fht_ == source.use_fht_,
                   "incompatible RaBitQ fused model");
    this->rom_ = source.rom_;
    this->pca_ = source.pca_;
    this->is_trained_ = source.is_trained_;
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

    const float center = 0.5F * static_cast<float>((1U << filter_bits) - 1U);
    double norm_sqr = 0.0;
    for (uint64_t d = 0; d < this->dim_; ++d) {
        const uint64_t byte_idx = d >> 3;
        const auto bit_mask = static_cast<uint8_t>(1U << (d & 7));
        uint32_t code = 0;
        for (uint32_t bit = 0; bit < filter_bits; ++bit) {
            const auto* plane = filter_code + split_layout_.sequential_plane_offsets[bit];
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
    return split_layout_.stored_plane_indices[logical_bit];
}

template <MetricType metric>
const uint8_t*
RaBitQuantizer<metric>::GetStoredPlane(const uint8_t* planes,
                                       uint32_t logical_bit,
                                       uint64_t plane_bytes) const {
    const auto plane_offset = plane_bytes == split_layout_.plane_bytes
                                  ? split_layout_.stored_plane_offsets[logical_bit]
                                  : split_layout_.stored_plane_indices[logical_bit] * plane_bytes;
    return planes + plane_offset;
}

template <MetricType metric>
uint8_t*
RaBitQuantizer<metric>::GetStoredPlane(uint8_t* planes,
                                       uint32_t logical_bit,
                                       uint64_t plane_bytes) const {
    const auto plane_offset = plane_bytes == split_layout_.plane_bytes
                                  ? split_layout_.stored_plane_offsets[logical_bit]
                                  : split_layout_.stored_plane_indices[logical_bit] * plane_bytes;
    return planes + plane_offset;
}

template <MetricType metric>
float
RaBitQuantizer<metric>::RaBitQFloatSQIPByPlanes(const float* query,
                                                const uint8_t* planes,
                                                float query_sum) const {
    const uint64_t plane_bytes = PlaneBytes();
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
        query, one_bit_code, supplement_code, filter_bits, supplement_bits, query_sum);
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
                                                   uint32_t supplement_bits,
                                                   float query_sum) const {
    if (this->dim_ == 0) {
        return 0.0F;
    }

    if (filter_bits >= 2 and filter_bits <= 4) {
        float centered_filter_ip = 0.0F;
        if (filter_bits == 2) {
            centered_filter_ip = RaBitQFloatTwoBitCenteredIP(query, filter_code, this->dim_);
        } else if (filter_bits == 3) {
            centered_filter_ip = RaBitQFloatThreeBitCenteredIP(query, filter_code, this->dim_);
        } else {
            centered_filter_ip = RaBitQFloatFourBitCenteredIP(query, filter_code, this->dim_);
        }
        const float filter_center = 0.5F * static_cast<float>((1U << filter_bits) - 1U);
        const auto filter_scale = static_cast<float>(1U << supplement_bits);
        const float supplement_ip =
            RaBitQFloatSupplementCodeIP(query, supplement_code, this->dim_, supplement_bits);
        return filter_scale * (centered_filter_ip + filter_center * query_sum) + supplement_ip;
    }
    float result = 0.0F;
    for (uint64_t d = 0; d < this->dim_; ++d) {
        const uint64_t byte_idx = d >> 3;
        const auto bit_mask = static_cast<uint8_t>(1U << (d & 7));
        uint32_t code = 0;
        for (uint32_t bit = 0; bit < filter_bits; ++bit) {
            const auto* plane = filter_code + split_layout_.sequential_plane_offsets[bit];
            if ((plane[byte_idx] & bit_mask) != 0) {
                code += 1U << (supplement_bits + filter_bits - bit - 1);
            }
        }
        if (supplement_code != nullptr) {
            for (uint32_t bit = 0; bit < supplement_bits; ++bit) {
                const auto* plane = supplement_code + split_layout_.sequential_plane_offsets[bit];
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
    const uint64_t plane_size = PlaneBytes();
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
void
RaBitQuantizer<metric>::FastEncodeRaBitQ(const float* o_prime, uint8_t* code, float& y_norm) const {
    // CAQ starts from an LVQ grid and improves cosine alignment with coordinate adjustment.
    // Each coordinate moves by at most one level per round, keeping the complexity O(rounds * D).
    constexpr double adjustment_epsilon = 1e-8;
    const uint32_t code_max = (1U << num_bits_per_dim_base_) - 1U;
    const double center = 0.5 * static_cast<double>(code_max);

    double max_abs = 0.0;
    for (uint64_t d = 0; d < this->dim_; ++d) {
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(o_prime[d])));
    }

    if (max_abs <= 0.0) {
        std::fill_n(code, this->dim_, static_cast<uint8_t>(code_max / 2U));
        y_norm = 1.0F;
        return;
    }

    const double delta = 2.0 * max_abs / static_cast<double>(code_max + 1U);
    const double inv_delta = 1.0 / delta;
    double ip = 0.0;
    double norm_sqr = 0.0;
    for (uint64_t d = 0; d < this->dim_; ++d) {
        const double scaled = (static_cast<double>(o_prime[d]) + max_abs) * inv_delta;
        auto quantized = static_cast<int64_t>(std::floor(scaled));
        quantized = std::clamp<int64_t>(quantized, 0, code_max);
        code[d] = static_cast<uint8_t>(quantized);

        const double centered = static_cast<double>(quantized) - center;
        ip += static_cast<double>(o_prime[d]) * centered;
        norm_sqr += centered * centered;
    }

    auto alignment_score = [](double inner_product, double squared_norm) {
        return squared_norm > 0.0 ? inner_product * inner_product / squared_norm : 0.0;
    };

    for (uint64_t round = 0; round < fast_encode_rabitq_rounds_; ++round) {
        bool adjusted = false;
        for (uint64_t d = 0; d < this->dim_; ++d) {
            const int32_t current_code = code[d];
            const double current_value = static_cast<double>(current_code) - center;
            int32_t best_code = current_code;
            double best_ip = ip;
            double best_norm_sqr = norm_sqr;
            double best_score = alignment_score(ip, norm_sqr);

            for (int32_t direction = -1; direction <= 1; direction += 2) {
                const int32_t candidate_code = current_code + direction;
                if (candidate_code < 0 or candidate_code > static_cast<int32_t>(code_max)) {
                    continue;
                }

                const double candidate_ip =
                    ip + static_cast<double>(direction) * static_cast<double>(o_prime[d]);
                if (candidate_ip < 0.0) {
                    continue;
                }
                const double candidate_norm_sqr =
                    norm_sqr + 2.0 * current_value * static_cast<double>(direction) + 1.0;
                const double candidate_score = alignment_score(candidate_ip, candidate_norm_sqr);
                const double tolerance = adjustment_epsilon * std::max(1.0, std::fabs(best_score));
                if (candidate_score > best_score + tolerance) {
                    best_code = candidate_code;
                    best_ip = candidate_ip;
                    best_norm_sqr = candidate_norm_sqr;
                    best_score = candidate_score;
                }
            }

            if (best_code != current_code) {
                code[d] = static_cast<uint8_t>(best_code);
                ip = best_ip;
                norm_sqr = best_norm_sqr;
                adjusted = true;
            }
        }

        if (not adjusted) {
            break;
        }
    }

    norm_sqr = 0.0;
    for (uint64_t d = 0; d < this->dim_; ++d) {
        const double centered = static_cast<double>(code[d]) - center;
        norm_sqr += centered * centered;
    }

    y_norm = static_cast<float>(std::sqrt(norm_sqr));
    if (not std::isfinite(y_norm) or y_norm <= 0.0F) {
        y_norm = 1.0F;
    }
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::EncodeOneImpl(const float* data, uint8_t* codes) const {
    return EncodeOneInternal(data, codes, nullptr, nullptr);
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::EncodeOneToScalarCode(const float* data,
                                              uint8_t* scalar_code,
                                              uint64_t& code_sum) const {
    if (not SupportScalarCodeBuild()) {
        return false;
    }
    return EncodeOneInternal(data, nullptr, scalar_code, &code_sum);
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::EncodeOneInternal(const float* data,
                                          uint8_t* codes,
                                          uint8_t* build_scalar_code,
                                          uint64_t* code_sum) const {
    Vector<float> pca_data(this->original_dim_, 0, this->allocator_);
    Vector<float> transformed_data(this->dim_, 0, this->allocator_);
    Vector<float> normed_data(this->dim_, 0, this->allocator_);
    if (codes != nullptr) {
        memset(codes, 0, this->code_size_);
    }
    if (build_scalar_code != nullptr) {
        memset(build_scalar_code, 0, GetScalarCodeSize());
    }
    if (code_sum != nullptr) {
        *code_sum = 0;
    }

    auto metadata_field = [&](uint64_t full_code_offset) {
        if (build_scalar_code == nullptr) {
            return codes + full_code_offset;
        }
        return build_scalar_code + ScalarCodeMetaOffset() + (full_code_offset - CodeMetaOffset());
    };

    float raw_norm = 0;
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        for (uint64_t d = 0; d < this->dim_; ++d) {
            raw_norm += data[d] * data[d];
        }
    }
    raw_norm = std::sqrt(raw_norm);

    if (pca_dim_ != this->original_dim_) {
        pca_->Transform(data, pca_data.data());
        if (use_mrq_) {
            const norm_type mrq_norm_sqr = FP32ComputeIP(pca_data.data() + this->dim_,
                                                         pca_data.data() + this->dim_,
                                                         this->original_dim_ - this->dim_);
            memcpy(metadata_field(offset_mrq_norm_), &mrq_norm_sqr, sizeof(mrq_norm_sqr));
        }
    } else {
        pca_data.assign(data, data + original_dim_);
    }

    rom_->Transform(pca_data.data(), transformed_data.data());
    const norm_type norm = NormalizeWithCentroid(
        transformed_data.data(), centroid_.data(), normed_data.data(), this->dim_);

    if (num_bits_per_dim_base_ != 1) {
        Vector<uint8_t> local_scalar_code(this->allocator_);
        uint8_t* scalar_code = build_scalar_code;
        if (scalar_code == nullptr) {
            local_scalar_code.resize(this->dim_, 0);
            scalar_code = local_scalar_code.data();
        }

        norm_type norm_code = 0;
        if (fast_encode_rabitq_) {
            FastEncodeRaBitQ(normed_data.data(), scalar_code, norm_code);
        } else {
            EncodeExtendRaBitQ(normed_data.data(), scalar_code, norm_code);
        }

        uint64_t scalar_sum = 0;
        float query_sum = 0.0F;
        float filter_ip = 0.0F;
        for (uint64_t d = 0; d < this->dim_; ++d) {
            scalar_sum += scalar_code[d];
            query_sum += normed_data[d];
            filter_ip += normed_data[d] * static_cast<float>(scalar_code[d] >> ReorderBits());
        }
        if (code_sum != nullptr) {
            *code_sum = scalar_sum;
        }
        if (codes != nullptr) {
            PackIntoPlanes(scalar_code, codes + offset_code_);
        }

        memcpy(metadata_field(offset_norm_code_), &norm_code, sizeof(norm_code));
        const float scalar_ip = RaBitQFloatSQIP(normed_data.data(), scalar_code, this->dim_);
        const error_type error = ip_obar_q(scalar_ip, query_sum, norm_code, num_bits_per_dim_base_);
        memcpy(metadata_field(offset_norm_), &norm, sizeof(norm));
        memcpy(metadata_field(offset_error_), &error, sizeof(error));

        if (SupportSplitCodeStorage()) {
            const norm_type filter_norm_code =
                ComputeScalarCodeNorm(scalar_code, FilterBits(), ReorderBits());
            error_type filter_error =
                ip_obar_q(filter_ip, query_sum, filter_norm_code, FilterBits());
            filter_error = std::fabs(filter_error);
            const float safe_filter_error = std::clamp(filter_error, 1e-5F, 1.0F);
            const error_type low_bound_error =
                std::sqrt(std::max(0.0F, 1.0F - safe_filter_error * safe_filter_error) /
                          std::max(1.0F, static_cast<float>(this->dim_ - 1)));
            memcpy(metadata_field(offset_one_bit_error_), &filter_error, sizeof(filter_error));
            memcpy(
                metadata_field(offset_low_bound_error_), &low_bound_error, sizeof(low_bound_error));
        }
    } else {
        uint64_t binary_sum = 0;
        for (uint64_t d = 0; d < this->dim_; ++d) {
            if (normed_data[d] >= 0.0F) {
                ++binary_sum;
                codes[offset_code_ + d / 8] |= static_cast<uint8_t>(1U << (d % 8));
            }
        }
        if (code_sum != nullptr) {
            *code_sum = binary_sum;
        }

        const error_type error =
            RaBitQFloatBinaryIP(normed_data.data(), codes + offset_code_, this->dim_, inv_sqrt_d_);
        memcpy(metadata_field(offset_norm_), &norm, sizeof(norm));
        memcpy(metadata_field(offset_error_), &error, sizeof(error));
        if (SupportSplitCodeStorage()) {
            const error_type low_bound_error = 0.0F;
            memcpy(metadata_field(offset_one_bit_error_), &error, sizeof(error));
            memcpy(
                metadata_field(offset_low_bound_error_), &low_bound_error, sizeof(low_bound_error));
        }
        if (num_bits_per_dim_query_ != 32) {
            const auto sum = static_cast<sum_type>(binary_sum);
            memcpy(metadata_field(offset_sum_), &sum, sizeof(sum));
        }
    }

    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        memcpy(metadata_field(offset_raw_norm_), &raw_norm, sizeof(raw_norm));
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
void
RaBitQuantizer<metric>::RefreshSplitLayout(bool is_split) {
    SplitLayout layout;
    layout.is_split = is_split;
    layout.filter_bits = layout.is_split ? num_bits_per_dim_filter_ : 1;
    layout.reorder_bits = layout.is_split ? num_bits_per_dim_base_ - layout.filter_bits : 0;
    layout.has_multi_bit_filter =
        layout.is_split && layout.filter_bits > 1 && num_bits_per_dim_base_ > 1;

    layout.plane_bytes = (this->dim_ + 7) / 8;
    layout.code_planes_size = AlignCodeField(layout.plane_bytes * num_bits_per_dim_base_);
    layout.code_meta_offset = offset_code_ + layout.code_planes_size;
    layout.code_meta_size = this->code_size_ - layout.code_meta_offset;
    layout.filter_planes_size = layout.plane_bytes * layout.filter_bits;
    layout.supplement_planes_size = layout.plane_bytes * layout.reorder_bits;
    layout.supplement_meta_offset = layout.supplement_planes_size;

    const auto aligned_norm_size = AlignCodeField(sizeof(norm_type));
    const auto aligned_error_size = AlignCodeField(sizeof(error_type));
    layout.one_bit_record_norm_offset = AlignCodeField(layout.filter_planes_size);
    uint64_t one_bit_record_offset = layout.one_bit_record_norm_offset + aligned_norm_size;
    layout.one_bit_record_norm_code_offset = one_bit_record_offset;
    if (layout.has_multi_bit_filter) {
        one_bit_record_offset += aligned_norm_size;
    }
    layout.one_bit_record_mrq_norm_offset = one_bit_record_offset;
    if (pca_dim_ != original_dim_ && use_mrq_) {
        one_bit_record_offset += aligned_norm_size;
    }
    layout.one_bit_record_raw_norm_offset = one_bit_record_offset;
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        one_bit_record_offset += aligned_norm_size;
    }
    layout.one_bit_record_low_bound_error_offset = one_bit_record_offset;
    layout.one_bit_record_one_bit_error_offset = one_bit_record_offset + aligned_error_size;
    layout.one_bit_record_size = layout.one_bit_record_one_bit_error_offset + aligned_error_size;
    layout.one_bit_code_size = layout.is_split ? layout.one_bit_record_size : this->code_size_;
    layout.supplement_code_size =
        layout.is_split ? layout.supplement_meta_offset + layout.code_meta_size : 0;

    for (uint32_t bit = 0; bit < layout.sequential_plane_offsets.size(); ++bit) {
        layout.sequential_plane_offsets[bit] = static_cast<uint64_t>(bit) * layout.plane_bytes;
    }
    for (uint32_t logical_bit = 0; logical_bit < num_bits_per_dim_base_; ++logical_bit) {
        uint64_t stored_plane_index = logical_bit;
        if (layout.is_split) {
            const auto first_filter_bit = num_bits_per_dim_base_ - layout.filter_bits;
            stored_plane_index = logical_bit >= first_filter_bit
                                     ? num_bits_per_dim_base_ - 1 - logical_bit
                                     : layout.filter_bits + logical_bit;
        }
        layout.stored_plane_indices[logical_bit] = stored_plane_index;
        layout.stored_plane_offsets[logical_bit] = stored_plane_index * layout.plane_bytes;
    }

    auto supplement_field_offset = [&layout](uint64_t full_code_offset) {
        return layout.supplement_meta_offset + full_code_offset - layout.code_meta_offset;
    };
    if (num_bits_per_dim_base_ != 1) {
        layout.supplement_norm_code_offset = supplement_field_offset(offset_norm_code_);
    }
    layout.supplement_norm_offset = supplement_field_offset(offset_norm_);
    layout.supplement_error_offset = supplement_field_offset(offset_error_);
    if (pca_dim_ != original_dim_ && use_mrq_) {
        layout.supplement_mrq_norm_offset = supplement_field_offset(offset_mrq_norm_);
    }
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        layout.supplement_raw_norm_offset = supplement_field_offset(offset_raw_norm_);
    }

    split_layout_ = layout;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::PlaneBytes() const {
    return split_layout_.plane_bytes;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::CodePlanesSize() const {
    return split_layout_.code_planes_size;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::CodeMetaOffset() const {
    return split_layout_.code_meta_offset;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::ScalarCodeMetaOffset() const {
    return AlignCodeField(this->dim_);
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::GetScalarCodeSize() const {
    return ScalarCodeMetaOffset() + this->code_size_ - CodeMetaOffset();
}

template <MetricType metric>
void
RaBitQuantizer<metric>::PackScalarCode(const uint8_t* scalar_code, uint8_t* full_code) const {
    memset(full_code, 0, this->code_size_);
    PackIntoPlanes(scalar_code, full_code + offset_code_);
    memcpy(full_code + CodeMetaOffset(),
           scalar_code + ScalarCodeMetaOffset(),
           this->code_size_ - CodeMetaOffset());
}

template <MetricType metric>
void
RaBitQuantizer<metric>::PackScalarCodeToSplitCode(const uint8_t* scalar_code,
                                                  uint8_t* one_bit_code,
                                                  uint8_t* supplement_code) const {
    if (not SupportSplitCodeStorage()) {
        return;
    }

    memset(one_bit_code, 0, GetOneBitCodeSize());
    memset(supplement_code, 0, GetSupplementCodeSize());
    RaBitQPackScalarToSplitPlanes(scalar_code,
                                  one_bit_code,
                                  supplement_code,
                                  this->dim_,
                                  num_bits_per_dim_base_,
                                  FilterBits());

    auto scalar_metadata = [&](uint64_t full_code_offset) {
        return scalar_code + ScalarCodeMetaOffset() + (full_code_offset - CodeMetaOffset());
    };
    memcpy(
        one_bit_code + OneBitRecordNormOffset(), scalar_metadata(offset_norm_), sizeof(norm_type));
    if (HasMultiBitFilter()) {
        const norm_type filter_norm_code =
            ComputeScalarCodeNorm(scalar_code, FilterBits(), ReorderBits());
        memcpy(one_bit_code + OneBitRecordNormCodeOffset(),
               &filter_norm_code,
               sizeof(filter_norm_code));
    }
    if (pca_dim_ != original_dim_ && use_mrq_) {
        memcpy(one_bit_code + OneBitRecordMrqNormOffset(),
               scalar_metadata(offset_mrq_norm_),
               sizeof(norm_type));
    }
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        memcpy(one_bit_code + OneBitRecordRawNormOffset(),
               scalar_metadata(offset_raw_norm_),
               sizeof(norm_type));
    }
    memcpy(one_bit_code + OneBitRecordLowBoundErrorOffset(),
           scalar_metadata(offset_low_bound_error_),
           sizeof(error_type));
    memcpy(one_bit_code + OneBitRecordOneBitErrorOffset(),
           scalar_metadata(offset_one_bit_error_),
           sizeof(error_type));

    memcpy(supplement_code + SupplementMetaOffset(),
           scalar_code + ScalarCodeMetaOffset(),
           this->code_size_ - CodeMetaOffset());
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::UnpackScalarCode(const uint8_t* codes, uint8_t* scalar_code) const {
    memset(scalar_code, 0, GetScalarCodeSize());
    const uint64_t plane_bytes = PlaneBytes();
    const auto* planes = codes + offset_code_;
    uint64_t code_sum = 0;
    for (uint64_t d = 0; d < this->dim_; ++d) {
        const uint64_t byte_idx = d >> 3;
        const auto bit_mask = static_cast<uint8_t>(1U << (d & 7));
        uint8_t value = 0;
        for (uint32_t bit = 0; bit < num_bits_per_dim_base_; ++bit) {
            const auto* plane = GetStoredPlane(planes, bit, plane_bytes);
            if ((plane[byte_idx] & bit_mask) != 0) {
                value |= static_cast<uint8_t>(1U << bit);
            }
        }
        scalar_code[d] = value;
        code_sum += value;
    }
    memcpy(scalar_code + ScalarCodeMetaOffset(),
           codes + CodeMetaOffset(),
           this->code_size_ - CodeMetaOffset());
    return code_sum;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::SupplementPlanesSize() const {
    return split_layout_.supplement_planes_size;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::SupplementMetaOffset() const {
    return split_layout_.supplement_meta_offset;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::SupportSplitCodeStorage() const {
    return split_layout_.is_split;
}

template <MetricType metric>
uint32_t
RaBitQuantizer<metric>::FilterBits() const {
    return split_layout_.filter_bits;
}

template <MetricType metric>
uint32_t
RaBitQuantizer<metric>::ReorderBits() const {
    return split_layout_.reorder_bits;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::FilterPlanesSize() const {
    return split_layout_.filter_planes_size;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::HasMultiBitFilter() const {
    return split_layout_.has_multi_bit_filter;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::OneBitRecordNormOffset() const {
    return split_layout_.one_bit_record_norm_offset;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::OneBitRecordNormCodeOffset() const {
    return split_layout_.one_bit_record_norm_code_offset;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::OneBitRecordMrqNormOffset() const {
    return split_layout_.one_bit_record_mrq_norm_offset;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::OneBitRecordRawNormOffset() const {
    return split_layout_.one_bit_record_raw_norm_offset;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::OneBitRecordLowBoundErrorOffset() const {
    return split_layout_.one_bit_record_low_bound_error_offset;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::OneBitRecordOneBitErrorOffset() const {
    return split_layout_.one_bit_record_one_bit_error_offset;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::OneBitRecordSize() const {
    return split_layout_.one_bit_record_size;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::GetOneBitCodeSize() const {
    return split_layout_.one_bit_code_size;
}

template <MetricType metric>
uint64_t
RaBitQuantizer<metric>::GetSupplementCodeSize() const {
    return split_layout_.supplement_code_size;
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
    return ComputeDistWithOneBitLowerBoundAndFilterIP(
        computer, one_bit_code, dists, lower_bound, nullptr, runtime_rabitq_error_rate);
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::ComputeDistWithOneBitLowerBoundAndFilterIP(
    Computer<RaBitQuantizer>& computer,
    const uint8_t* one_bit_code,
    float* dists,
    float* lower_bound,
    float* filter_inner_product,
    float runtime_rabitq_error_rate) const {
    if (lower_bound != nullptr) {
        *lower_bound = std::numeric_limits<float>::max();
    }
    if (filter_inner_product != nullptr) {
        *filter_inner_product = std::numeric_limits<float>::quiet_NaN();
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
        memcpy(
            &base_norm_code, one_bit_code + OneBitRecordNormCodeOffset(), sizeof(base_norm_code));
        if (FilterBits() == 2) {
            filter_ip_yu_q = RaBitQFloatTwoBitCenteredIP(
                reinterpret_cast<const float*>(query), one_bit_code, this->dim_);
            filter_ip_estimate = base_norm_code <= 0.0F ? 0.0F : filter_ip_yu_q / base_norm_code;
        } else if (FilterBits() == 3) {
            filter_ip_yu_q = RaBitQFloatThreeBitCenteredIP(
                reinterpret_cast<const float*>(query), one_bit_code, this->dim_);
            filter_ip_estimate = base_norm_code <= 0.0F ? 0.0F : filter_ip_yu_q / base_norm_code;
        } else if (FilterBits() == 4) {
            filter_ip_yu_q = RaBitQFloatFourBitCenteredIP(
                reinterpret_cast<const float*>(query), one_bit_code, this->dim_);
            filter_ip_estimate = base_norm_code <= 0.0F ? 0.0F : filter_ip_yu_q / base_norm_code;
        } else {
            filter_ip_yu_q = RaBitQFloatSQIPBySplitCode(reinterpret_cast<const float*>(query),
                                                        one_bit_code,
                                                        nullptr,
                                                        FilterBits(),
                                                        0,
                                                        query_raw_sum);
            filter_ip_estimate =
                ip_obar_q(filter_ip_yu_q, query_raw_sum, base_norm_code, FilterBits());
        }
    }
    // The x=1 path does not expose a reusable inner-product hint.
    if (filter_inner_product != nullptr and FilterBits() >= 2) {
        *filter_inner_product = filter_ip_estimate;
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

    if (not IsFiniteRaBitQValue(result)) {
        return false;
    }

    *dists = result;
    if (lower_bound == nullptr) {
        return true;
    }

    error_type low_bound_error = *((error_type*)(one_bit_code + OneBitRecordLowBoundErrorOffset()));
    const float effective_error_rate =
        IsFiniteRaBitQValue(runtime_rabitq_error_rate) and runtime_rabitq_error_rate > 0.0F
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
    if (IsFiniteRaBitQValue(lower_bound_result)) {
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
    if constexpr (metric != MetricType::METRIC_TYPE_L2SQR and
                  metric != MetricType::METRIC_TYPE_IP) {
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

    if (FilterBits() < 1 or FilterBits() > 4) {
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
    norm_type query_raw_norm = 0.0F;
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        std::memcpy(&query_raw_norm, query + query_offset_raw_norm_, sizeof(query_raw_norm));
    }
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
    bool filter_ip_values_are_centered = false;
    if (FilterBits() == 1) {
        RaBitQFloatBinaryIPBatch4(query_data,
                                  one_bit_code1,
                                  one_bit_code2,
                                  one_bit_code3,
                                  one_bit_code4,
                                  this->dim_,
                                  inv_sqrt_d_,
                                  filter_ip_values);
    } else if (FilterBits() == 2) {
        RaBitQFloatTwoBitCenteredIPBatch4(query_data,
                                          one_bit_code1,
                                          one_bit_code2,
                                          one_bit_code3,
                                          one_bit_code4,
                                          this->dim_,
                                          filter_ip_values);
        filter_ip_values_are_centered = true;
    } else if (FilterBits() == 3) {
        RaBitQFloatThreeBitCenteredIPBatch4(query_data,
                                            one_bit_code1,
                                            one_bit_code2,
                                            one_bit_code3,
                                            one_bit_code4,
                                            this->dim_,
                                            filter_ip_values);
        filter_ip_values_are_centered = true;
    } else if (FilterBits() == 4) {
        RaBitQFloatFourBitCenteredIPBatch4(query_data,
                                           one_bit_code1,
                                           one_bit_code2,
                                           one_bit_code3,
                                           one_bit_code4,
                                           this->dim_,
                                           filter_ip_values);
        filter_ip_values_are_centered = true;
    }

    auto compute_one =
        [&](const uint8_t* one_bit_code, float filter_ip, float& dist, float* lower_bound) {
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
                filter_ip_estimate =
                    filter_ip_values_are_centered
                        ? (base_norm_code <= 0.0F ? 0.0F : filter_ip / base_norm_code)
                        : ip_obar_q(filter_ip, query_raw_sum, base_norm_code, FilterBits());
            }

            const float ip_est = filter_ip_estimate / one_bit_error;
            const norm_type base_norm = *((norm_type*)(one_bit_code + OneBitRecordNormOffset()));
            float result = l2_ube(base_norm, query_norm, ip_est);

            if (pca_dim_ != this->original_dim_ and use_mrq_) {
                const norm_type base_mrq_norm_sqr =
                    *(norm_type*)(one_bit_code + OneBitRecordMrqNormOffset());
                result += (query_mrq_norm_sqr + base_mrq_norm_sqr);
            }

            if constexpr (metric == MetricType::METRIC_TYPE_IP) {
                norm_type base_raw_norm = 0.0F;
                std::memcpy(&base_raw_norm,
                            one_bit_code + OneBitRecordRawNormOffset(),
                            sizeof(base_raw_norm));
                if (is_approx_zero(query_raw_norm) or is_approx_zero(base_raw_norm)) {
                    result = 1.0F;
                } else {
                    result = 1.0F - (query_raw_norm * query_raw_norm +
                                     base_raw_norm * base_raw_norm - result) *
                                        0.5F;
                }
            }

            if (not IsFiniteRaBitQValue(result)) {
                return false;
            }

            dist = result;
            if (lower_bound == nullptr) {
                return true;
            }

            const error_type low_bound_error =
                *((error_type*)(one_bit_code + OneBitRecordLowBoundErrorOffset()));
            const float effective_error_rate =
                IsFiniteRaBitQValue(runtime_rabitq_error_rate) and runtime_rabitq_error_rate > 0.0F
                    ? runtime_rabitq_error_rate
                    : rabitq_error_rate_;
            float lower_bound_error_term = 2.0F * base_norm * query_norm * effective_error_rate *
                                           low_bound_error / one_bit_error;
            if constexpr (metric == MetricType::METRIC_TYPE_IP) {
                lower_bound_error_term *= 0.5F;
            }
            const float lower_bound_result = result - lower_bound_error_term;
            if (IsFiniteRaBitQValue(lower_bound_result)) {
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
        memcpy(&base_norm_code,
               supplement_code + split_layout_.supplement_norm_code_offset,
               sizeof(base_norm_code));
        ip_bq_estimate = ip_obar_q(ip_yu_q, query_raw_sum, base_norm_code, num_bits_per_dim_base_);
    } else {
        sum_type query_raw_sum = *((sum_type*)(query + query_offset_sum_));
        float ip_yu_q = 0.0F;
        if (FilterBits() >= 2 and FilterBits() <= 4) {
            const auto* query_data = reinterpret_cast<const float*>(query);
            float filter_centered_ip = 0.0F;
            if (FilterBits() == 2) {
                filter_centered_ip =
                    RaBitQFloatTwoBitCenteredIP(query_data, one_bit_code, this->dim_);
            } else if (FilterBits() == 3) {
                filter_centered_ip =
                    RaBitQFloatThreeBitCenteredIP(query_data, one_bit_code, this->dim_);
            } else {
                filter_centered_ip =
                    RaBitQFloatFourBitCenteredIP(query_data, one_bit_code, this->dim_);
            }
            const float filter_center = 0.5F * static_cast<float>((1U << FilterBits()) - 1U);
            const float filter_ip_yu_q = filter_centered_ip + filter_center * query_raw_sum;
            ip_yu_q = filter_ip_yu_q * static_cast<float>(1U << ReorderBits());
            ip_yu_q +=
                RaBitQFloatSupplementCodeIP(query_data, supplement_code, this->dim_, ReorderBits());
        } else {
            ip_yu_q = RaBitQFloatSQIPBySplitCode(reinterpret_cast<const float*>(query),
                                                 one_bit_code,
                                                 supplement_code,
                                                 FilterBits(),
                                                 ReorderBits(),
                                                 query_raw_sum);
        }

        norm_type base_norm_code = 0;
        memcpy(&base_norm_code,
               supplement_code + split_layout_.supplement_norm_code_offset,
               sizeof(base_norm_code));
        ip_bq_estimate = ip_obar_q(ip_yu_q, query_raw_sum, base_norm_code, num_bits_per_dim_base_);
    }

    norm_type query_norm = *((norm_type*)(query + query_offset_norm_));
    norm_type base_norm = 0;
    memcpy(&base_norm, supplement_code + split_layout_.supplement_norm_offset, sizeof(base_norm));

    norm_type query_raw_norm = 0;
    norm_type base_raw_norm = 0;
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        query_raw_norm = *((norm_type*)(query + query_offset_raw_norm_));
        memcpy(&base_raw_norm,
               supplement_code + split_layout_.supplement_raw_norm_offset,
               sizeof(base_raw_norm));
    }

    error_type base_error = 0;
    memcpy(
        &base_error, supplement_code + split_layout_.supplement_error_offset, sizeof(base_error));
    if (std::abs(base_error) < 1e-5) {
        base_error = (base_error >= 0) ? 1.0F : -1.0F;
    }

    float ip_est = ip_bq_estimate / base_error;
    float result = l2_ube(base_norm, query_norm, ip_est);

    if (pca_dim_ != this->original_dim_ and use_mrq_) {
        norm_type query_mrq_norm_sqr = *(norm_type*)(query + query_offset_mrq_norm_);
        norm_type base_mrq_norm_sqr = 0;
        memcpy(&base_mrq_norm_sqr,
               supplement_code + split_layout_.supplement_mrq_norm_offset,
               sizeof(base_mrq_norm_sqr));
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
    if constexpr (metric != MetricType::METRIC_TYPE_L2SQR and
                  metric != MetricType::METRIC_TYPE_IP) {
        return false;
    }

    if (not SupportSplitCodeStorage() or ReorderBits() == 0 or
        not IsFiniteRaBitQValue(filter_dist) or FilterBits() == 1) {
        return false;
    }

    const auto* query = computer.buf_;

    const norm_type query_norm = *((norm_type*)(query + query_offset_norm_));
    const norm_type base_norm = *((norm_type*)(one_bit_code + OneBitRecordNormOffset()));
    const error_type filter_error =
        std::fabs(*((error_type*)(one_bit_code + OneBitRecordOneBitErrorOffset())));
    if (filter_error <= 1e-5F or is_approx_zero(query_norm) or is_approx_zero(base_norm)) {
        return false;
    }

    norm_type query_raw_norm = 0.0F;
    norm_type base_raw_norm = 0.0F;
    float filter_l2 = filter_dist;
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        std::memcpy(&query_raw_norm, query + query_offset_raw_norm_, sizeof(query_raw_norm));
        std::memcpy(
            &base_raw_norm, one_bit_code + OneBitRecordRawNormOffset(), sizeof(base_raw_norm));
        if (is_approx_zero(query_raw_norm) or is_approx_zero(base_raw_norm)) {
            *dists = 1.0F;
            return true;
        }
        filter_l2 = query_raw_norm * query_raw_norm + base_raw_norm * base_raw_norm -
                    2.0F * (1.0F - filter_dist);
    }
    if (pca_dim_ != this->original_dim_ and use_mrq_) {
        const norm_type query_mrq_norm_sqr = *(norm_type*)(query + query_offset_mrq_norm_);
        const norm_type base_mrq_norm_sqr =
            *(norm_type*)(one_bit_code + OneBitRecordMrqNormOffset());
        filter_l2 -= query_mrq_norm_sqr + base_mrq_norm_sqr;
    }

    const float filter_ip_est = (base_norm * base_norm + query_norm * query_norm - filter_l2) /
                                (2.0F * base_norm * query_norm) * filter_error;

    return ComputeDistWithSplitCodeAndFilterIP(
        computer, one_bit_code, supplement_code, filter_ip_est, dists);
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::ComputeDistWithSplitCodeAndFilterIP(Computer<RaBitQuantizer>& computer,
                                                            const uint8_t* one_bit_code,
                                                            const uint8_t* supplement_code,
                                                            float filter_inner_product,
                                                            float* dists) const {
    if constexpr (metric != MetricType::METRIC_TYPE_L2SQR and
                  metric != MetricType::METRIC_TYPE_IP) {
        return false;
    }
    if (not SupportSplitCodeStorage() or ReorderBits() == 0 or
        not IsFiniteRaBitQValue(filter_inner_product) or FilterBits() == 1) {
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

    norm_type filter_norm_code = 0.0F;
    if (FilterBits() == 1) {
        if (inv_sqrt_d_ <= 0.0F) {
            return false;
        }
        filter_norm_code = 0.5F / inv_sqrt_d_;
    } else {
        memcpy(&filter_norm_code,
               one_bit_code + OneBitRecordNormCodeOffset(),
               sizeof(filter_norm_code));
    }
    if (filter_norm_code <= 0.0F) {
        return false;
    }

    norm_type full_norm_code = 0;
    memcpy(&full_norm_code,
           supplement_code + split_layout_.supplement_norm_code_offset,
           sizeof(full_norm_code));
    if (full_norm_code <= 0.0F) {
        return false;
    }

    const sum_type query_raw_sum = *((sum_type*)(query + query_offset_sum_));
    const float filter_center = 0.5F * static_cast<float>((1U << FilterBits()) - 1U);
    const float filter_ip_yu_q =
        filter_inner_product * filter_norm_code + filter_center * query_raw_sum;
    const float shifted_filter_ip_yu_q = filter_ip_yu_q * static_cast<float>(1U << ReorderBits());
    const float supplement_ip_yu_q = RaBitQFloatSupplementCodeIP(
        reinterpret_cast<const float*>(query), supplement_code, this->dim_, ReorderBits());
    const float ip_bq_estimate = ip_obar_q(shifted_filter_ip_yu_q + supplement_ip_yu_q,
                                           query_raw_sum,
                                           full_norm_code,
                                           num_bits_per_dim_base_);

    error_type base_error = 0;
    memcpy(
        &base_error, supplement_code + split_layout_.supplement_error_offset, sizeof(base_error));
    if (std::abs(base_error) < 1e-5F) {
        base_error = (base_error >= 0) ? 1.0F : -1.0F;
    }

    float result = l2_ube(base_norm, query_norm, ip_bq_estimate / base_error);
    if (pca_dim_ != this->original_dim_ and use_mrq_) {
        const norm_type query_mrq_norm_sqr = *(norm_type*)(query + query_offset_mrq_norm_);
        norm_type base_mrq_norm_sqr = 0;
        memcpy(&base_mrq_norm_sqr,
               supplement_code + split_layout_.supplement_mrq_norm_offset,
               sizeof(base_mrq_norm_sqr));
        result += query_mrq_norm_sqr + base_mrq_norm_sqr;
    }

    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        norm_type query_raw_norm = 0.0F;
        norm_type base_raw_norm = 0.0F;
        std::memcpy(&query_raw_norm, query + query_offset_raw_norm_, sizeof(query_raw_norm));
        std::memcpy(
            &base_raw_norm, one_bit_code + OneBitRecordRawNormOffset(), sizeof(base_raw_norm));
        if (is_approx_zero(query_raw_norm) or is_approx_zero(base_raw_norm)) {
            *dists = 1.0F;
            return true;
        }
        result = 1.0F -
                 (query_raw_norm * query_raw_norm + base_raw_norm * base_raw_norm - result) * 0.5F;
    }

    if (not IsFiniteRaBitQValue(result)) {
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

        float ip_yu_q = RaBitQFloatSQIPByPlanes(
            reinterpret_cast<const float*>(query_codes), base_codes + offset_code_, query_raw_sum);

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
void
RaBitQuantizer<metric>::ComputeDistWithScalarCode(Computer<RaBitQuantizer>& computer,
                                                  const uint8_t* scalar_code,
                                                  float* dist) const {
    const auto* query = computer.buf_;
    const auto* scalar_meta = scalar_code + ScalarCodeMetaOffset();
    auto metadata_field = [&](uint64_t full_code_offset) {
        return scalar_meta + (full_code_offset - CodeMetaOffset());
    };

    const sum_type query_sum = *reinterpret_cast<const sum_type*>(query + query_offset_sum_);
    const float scalar_ip =
        RaBitQFloatSQIP(reinterpret_cast<const float*>(query), scalar_code, this->dim_);

    norm_type base_norm_code = 0;
    memcpy(&base_norm_code, metadata_field(offset_norm_code_), sizeof(base_norm_code));
    const float code_ip = ip_obar_q(scalar_ip, query_sum, base_norm_code, num_bits_per_dim_base_);

    error_type base_error = 0;
    memcpy(&base_error, metadata_field(offset_error_), sizeof(base_error));
    if (std::abs(base_error) < 1e-5F) {
        base_error = base_error >= 0.0F ? 1.0F : -1.0F;
    }

    const norm_type query_norm = *reinterpret_cast<const norm_type*>(query + query_offset_norm_);
    norm_type base_norm = 0;
    memcpy(&base_norm, metadata_field(offset_norm_), sizeof(base_norm));
    float result = l2_ube(base_norm, query_norm, code_ip / base_error);

    if (pca_dim_ != original_dim_ and use_mrq_) {
        const norm_type query_mrq_norm =
            *reinterpret_cast<const norm_type*>(query + query_offset_mrq_norm_);
        norm_type base_mrq_norm = 0;
        memcpy(&base_mrq_norm, metadata_field(offset_mrq_norm_), sizeof(base_mrq_norm));
        result += query_mrq_norm + base_mrq_norm;
    }

    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        const norm_type query_raw_norm =
            *reinterpret_cast<const norm_type*>(query + query_offset_raw_norm_);
        norm_type base_raw_norm = 0;
        memcpy(&base_raw_norm, metadata_field(offset_raw_norm_), sizeof(base_raw_norm));
        if (is_approx_zero(query_raw_norm) or is_approx_zero(base_raw_norm)) {
            *dist = 1.0F;
            return;
        }
        const float inner_product =
            (query_raw_norm * query_raw_norm + base_raw_norm * base_raw_norm - result) * 0.5F;
        if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
            result = 1.0F - inner_product / (query_raw_norm * base_raw_norm);
        } else {
            result = 1.0F - inner_product;
        }
    }
    *dist = result;
}

template <MetricType metric>
float
RaBitQuantizer<metric>::ComputeScalarCodesDistance(const uint8_t* scalar_code1,
                                                   uint64_t code_sum1,
                                                   const uint8_t* scalar_code2,
                                                   uint64_t code_sum2) const {
    const uint64_t raw_dot = RaBitQCodeCodeIP(scalar_code1, scalar_code2, this->dim_);
    const auto max_code = static_cast<double>((1U << num_bits_per_dim_base_) - 1U);
    const double centered_dot = static_cast<double>(raw_dot) -
                                0.5 * max_code * static_cast<double>(code_sum1 + code_sum2) +
                                0.25 * static_cast<double>(this->dim_) * max_code * max_code;

    const auto* metadata1 = scalar_code1 + ScalarCodeMetaOffset();
    const auto* metadata2 = scalar_code2 + ScalarCodeMetaOffset();
    auto metadata_field = [&](const uint8_t* metadata, uint64_t full_code_offset) {
        return metadata + (full_code_offset - CodeMetaOffset());
    };

    norm_type norm_code1 = 0;
    norm_type norm_code2 = 0;
    if (num_bits_per_dim_base_ == 1) {
        norm_code1 = norm_code2 = 0.5F * std::sqrt(static_cast<float>(this->dim_));
    } else {
        memcpy(&norm_code1, metadata_field(metadata1, offset_norm_code_), sizeof(norm_code1));
        memcpy(&norm_code2, metadata_field(metadata2, offset_norm_code_), sizeof(norm_code2));
    }
    float code_ip = 0.0F;
    if (norm_code1 > 0.0F and norm_code2 > 0.0F) {
        code_ip = std::clamp(
            static_cast<float>(centered_dot / (static_cast<double>(norm_code1) * norm_code2)),
            -1.0F,
            1.0F);
    }

    norm_type norm1 = 0;
    norm_type norm2 = 0;
    memcpy(&norm1, metadata_field(metadata1, offset_norm_), sizeof(norm1));
    memcpy(&norm2, metadata_field(metadata2, offset_norm_), sizeof(norm2));
    float result = l2_ube(norm1, norm2, code_ip);
    if (pca_dim_ != original_dim_ and use_mrq_) {
        norm_type mrq_norm1 = 0;
        norm_type mrq_norm2 = 0;
        memcpy(&mrq_norm1, metadata_field(metadata1, offset_mrq_norm_), sizeof(mrq_norm1));
        memcpy(&mrq_norm2, metadata_field(metadata2, offset_mrq_norm_), sizeof(mrq_norm2));
        result += mrq_norm1 + mrq_norm2;
    }

    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        norm_type raw_norm1 = 0;
        norm_type raw_norm2 = 0;
        memcpy(&raw_norm1, metadata_field(metadata1, offset_raw_norm_), sizeof(raw_norm1));
        memcpy(&raw_norm2, metadata_field(metadata2, offset_raw_norm_), sizeof(raw_norm2));
        if (is_approx_zero(raw_norm1) or is_approx_zero(raw_norm2)) {
            return 1.0F;
        }
        const float inner_product = (raw_norm1 * raw_norm1 + raw_norm2 * raw_norm2 - result) * 0.5F;
        if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
            return 1.0F - inner_product / (raw_norm1 * raw_norm2);
        }
        return 1.0F - inner_product;
    }
    return result;
}

template <MetricType metric>
float
RaBitQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    if (not SupportSplitCodeStorage()) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "building the index is not supported using RabbitQ alone");
    }
    Vector<uint8_t> scalar_code1(this->allocator_);
    Vector<uint8_t> scalar_code2(this->allocator_);
    scalar_code1.resize(GetScalarCodeSize());
    scalar_code2.resize(GetScalarCodeSize());
    const uint64_t code_sum1 = UnpackScalarCode(codes1, scalar_code1.data());
    const uint64_t code_sum2 = UnpackScalarCode(codes2, scalar_code2.data());
    return ComputeScalarCodesDistance(
        scalar_code1.data(), code_sum1, scalar_code2.data(), code_sum2);
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
RaBitQuantizer<metric>::TransformFusedQuery(const float* query,
                                            Vector<float>& transformed_query,
                                            float& query_raw_norm,
                                            norm_type& mrq_norm_sqr) const {
    Vector<float> pca_data(this->original_dim_, 0, this->allocator_);
    query_raw_norm = 0.0F;
    mrq_norm_sqr = 0.0F;
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        for (uint64_t d = 0; d < this->dim_; ++d) {
            query_raw_norm += query[d] * query[d];
        }
        query_raw_norm = std::sqrt(query_raw_norm);
    }
    if (pca_dim_ != this->original_dim_) {
        pca_->Transform(query, pca_data.data());
        if (use_mrq_) {
            mrq_norm_sqr = FP32ComputeIP(pca_data.data() + this->dim_,
                                         pca_data.data() + this->dim_,
                                         this->original_dim_ - this->dim_);
        }
    } else {
        pca_data.assign(query, query + original_dim_);
    }
    rom_->Transform(pca_data.data(), transformed_query.data());
}

template <MetricType metric>
void
RaBitQuantizer<metric>::PrepareHnswFourBitQuery(const float* transformed_query,
                                                Vector<uint8_t>& query_planes,
                                                float& delta,
                                                float& vl,
                                                float& query_sum) const {
    constexpr uint32_t k_query_bits = 4;
    constexpr uint32_t k_ex_bits = k_query_bits - 1;
    constexpr uint32_t k_ex_mask = (1U << k_ex_bits) - 1U;
    constexpr float k_center = 7.5F;
    const uint64_t plane_bytes = PlaneBytes();
    query_planes.assign(k_query_bits * plane_bytes, 0);

    double norm_sqr = 0.0;
    for (uint64_t i = 0; i < this->dim_; ++i) {
        CHECK_ARGUMENT(IsFiniteRaBitQValue(transformed_query[i]),
                       "RaBitQ query must contain only finite values");
        norm_sqr += static_cast<double>(transformed_query[i]) * transformed_query[i];
    }
    const auto norm = static_cast<float>(std::sqrt(norm_sqr));
    const float query_scale =
        norm > 0.0F ? 3.21F * std::sqrt(static_cast<float>(this->dim_)) / norm : 0.0F;
    double reconstruction_ip = 0.0;
    double reconstruction_norm_sqr = 0.0;
    query_sum = 0.0F;
    for (uint64_t i = 0; i < this->dim_; ++i) {
        auto magnitude =
            static_cast<uint32_t>(query_scale * std::fabs(transformed_query[i]) + 1e-5F);
        magnitude = std::min(magnitude, k_ex_mask);
        const auto supplement = transformed_query[i] < 0.0F ? k_ex_mask - magnitude : magnitude;
        const auto quantized = static_cast<uint8_t>(
            supplement | (static_cast<uint32_t>(transformed_query[i] > 0.0F) << k_ex_bits));
        const float centered = static_cast<float>(quantized) - k_center;
        reconstruction_ip += static_cast<double>(transformed_query[i]) * centered;
        reconstruction_norm_sqr += static_cast<double>(centered) * centered;
        query_sum += transformed_query[i];
        for (uint32_t bit = 0; bit < k_query_bits; ++bit) {
            if ((quantized & (1U << bit)) != 0U) {
                query_planes[static_cast<uint64_t>(bit) * plane_bytes + i / 8] |=
                    static_cast<uint8_t>(1U << (i & 7));
            }
        }
    }
    delta = reconstruction_norm_sqr > 0.0
                ? static_cast<float>(reconstruction_ip / reconstruction_norm_sqr)
                : 0.0F;
    vl = -k_center * delta;
}

template <MetricType metric>
void
RaBitQuantizer<metric>::EncodeHnswOneBitMetadata(const float* data, uint8_t* one_bit_code) const {
    Vector<float> transformed_data(this->dim_, 0.0F, this->allocator_);
    float raw_norm = 0.0F;
    norm_type mrq_norm_sqr = 0.0F;
    TransformFusedQuery(data, transformed_data, raw_norm, mrq_norm_sqr);

    double residual_norm_sqr = 0.0;
    double residual_code_ip = 0.0;
    double centroid_code_ip = 0.0;
    double code_norm_sqr = 0.0;
    for (uint64_t i = 0; i < this->dim_; ++i) {
        const float residual = transformed_data[i] - centroid_[i];
        const float centered_code = residual > 0.0F ? 0.5F : -0.5F;
        residual_norm_sqr += static_cast<double>(residual) * residual;
        residual_code_ip += static_cast<double>(residual) * centered_code;
        centroid_code_ip += static_cast<double>(centroid_[i]) * centered_code;
        code_norm_sqr += static_cast<double>(centered_code) * centered_code;
    }

    const auto l2_sqr = static_cast<float>(residual_norm_sqr);
    const float l2_norm = std::sqrt(l2_sqr);
    const float safe_ip = std::fabs(residual_code_ip) > 1e-20
                              ? static_cast<float>(residual_code_ip)
                              : std::numeric_limits<float>::infinity();
    float f_add = 0.0F;
    float f_rescale = 0.0F;
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        const float residual_centroid_ip =
            FP32ComputeIP(transformed_data.data(), centroid_.data(), this->dim_) -
            FP32ComputeIP(centroid_.data(), centroid_.data(), this->dim_);
        f_add =
            1.0F - residual_centroid_ip + l2_sqr * static_cast<float>(centroid_code_ip) / safe_ip;
        f_rescale = -l2_sqr / safe_ip;
    } else {
        f_add = l2_sqr + 2.0F * l2_sqr * static_cast<float>(centroid_code_ip) / safe_ip;
        f_rescale = -2.0F * l2_sqr / safe_ip;
    }
    const double ratio = residual_norm_sqr * code_norm_sqr /
                             (static_cast<double>(safe_ip) * static_cast<double>(safe_ip)) -
                         1.0;
    const float error_scale = metric == MetricType::METRIC_TYPE_IP ? 1.0F : 2.0F;
    const float f_error =
        error_scale * l2_norm * RaBitQuantizerParameter::DEFAULT_RABITQ_ERROR_RATE *
        std::sqrt(static_cast<float>(std::max(0.0, ratio) / std::max<uint64_t>(1, this->dim_ - 1)));
    std::memcpy(one_bit_code + PlaneBytes(), &f_add, sizeof(float));
    std::memcpy(one_bit_code + PlaneBytes() + sizeof(float), &f_rescale, sizeof(float));
    std::memcpy(one_bit_code + PlaneBytes() + 2 * sizeof(float), &f_error, sizeof(float));
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::EncodeHnswSupplement(const float* data, uint8_t* supplement_code) const {
    if (data == nullptr or supplement_code == nullptr or centroid_.size() != this->dim_) {
        return false;
    }
    for (uint64_t i = 0; i < this->original_dim_; ++i) {
        if (not IsFiniteRaBitQValue(data[i])) {
            return false;
        }
    }
    constexpr uint32_t k_ex_bits = 7;
    constexpr uint32_t k_ex_mask = (1U << k_ex_bits) - 1U;
    constexpr float k_center = 127.5F;
    constexpr float k_scale_960 = 1180.0F;
    Vector<float> transformed_data(this->dim_, 0.0F, this->allocator_);
    Vector<float> residual(this->dim_, 0.0F, this->allocator_);
    Vector<uint8_t> ex_codes(this->dim_, 0, this->allocator_);
    float raw_norm = 0.0F;
    norm_type mrq_norm_sqr = 0.0F;
    TransformFusedQuery(data, transformed_data, raw_norm, mrq_norm_sqr);

    double residual_norm_sqr = 0.0;
    for (uint64_t i = 0; i < this->dim_; ++i) {
        if (not IsFiniteRaBitQValue(transformed_data[i]) or not IsFiniteRaBitQValue(centroid_[i])) {
            return false;
        }
        residual[i] = transformed_data[i] - centroid_[i];
        if (not IsFiniteRaBitQValue(residual[i])) {
            return false;
        }
        residual_norm_sqr += static_cast<double>(residual[i]) * residual[i];
    }
    const auto residual_norm = static_cast<float>(std::sqrt(residual_norm_sqr));
    if (not IsFiniteRaBitQValue(residual_norm)) {
        return false;
    }
    const float scale =
        residual_norm > 0.0F
            ? k_scale_960 * std::sqrt(static_cast<float>(this->dim_) / 960.0F) / residual_norm
            : 0.0F;
    if (not IsFiniteRaBitQValue(scale)) {
        return false;
    }
    double ipnorm = 0.0;
    for (uint64_t i = 0; i < this->dim_; ++i) {
        const float scaled_magnitude = scale * std::fabs(residual[i]) + 1e-5F;
        if (not IsFiniteRaBitQValue(scaled_magnitude)) {
            return false;
        }
        const auto magnitude =
            static_cast<uint32_t>(std::min(scaled_magnitude, static_cast<float>(k_ex_mask)));
        ex_codes[i] = static_cast<uint8_t>(residual[i] < 0.0F ? k_ex_mask - magnitude : magnitude);
        ipnorm += (static_cast<double>(magnitude) + 0.5) *
                  std::fabs(static_cast<double>(residual[i])) /
                  std::max<double>(residual_norm, 1e-30);
    }

    std::fill(supplement_code, supplement_code + SupplementPlanesSize(), 0);
    if ((this->dim_ & 63U) == 0U) {
        auto* output = supplement_code;
        for (uint64_t block = 0; block < this->dim_; block += 64) {
            const auto* input = ex_codes.data() + block;
            for (uint64_t lane = 0; lane < 16; ++lane) {
                output[lane] = static_cast<uint8_t>((input[lane] & 0x3FU) |
                                                    ((input[48 + lane] & 0x03U) << 6U));
                output[16 + lane] = static_cast<uint8_t>((input[16 + lane] & 0x3FU) |
                                                         ((input[48 + lane] & 0x0CU) << 4U));
                output[32 + lane] = static_cast<uint8_t>((input[32 + lane] & 0x3FU) |
                                                         ((input[48 + lane] & 0x30U) << 2U));
            }
            uint64_t top_bits = 0;
            constexpr uint64_t k_top_mask = 0x0101010101010101ULL;
            for (uint64_t lane = 0; lane < 64; lane += 8) {
                uint64_t codes = 0;
                std::memcpy(&codes, input + lane, sizeof(codes));
                top_bits |= ((codes >> 6U) & k_top_mask) << (lane / 8U);
            }
            std::memcpy(output + 48, &top_bits, sizeof(top_bits));
            output += 56;
        }
    } else {
        const uint64_t plane_bytes = PlaneBytes();
        for (uint64_t i = 0; i < this->dim_; ++i) {
            for (uint32_t bit = 0; bit < k_ex_bits; ++bit) {
                if ((ex_codes[i] & (1U << bit)) != 0U) {
                    supplement_code[static_cast<uint64_t>(bit) * plane_bytes + i / 8] |=
                        static_cast<uint8_t>(1U << (i & 7));
                }
            }
        }
    }

    double residual_code_ip = 0.0;
    double centroid_code_ip = 0.0;
    for (uint64_t i = 0; i < this->dim_; ++i) {
        const float total_code =
            static_cast<float>(ex_codes[i]) + (residual[i] >= 0.0F ? 128.0F : 0.0F);
        const float centered = total_code - k_center;
        residual_code_ip += static_cast<double>(residual[i]) * centered;
        centroid_code_ip += static_cast<double>(centroid_[i]) * centered;
    }
    const float safe_ip = std::fabs(residual_code_ip) > 1e-20
                              ? static_cast<float>(residual_code_ip)
                              : std::numeric_limits<float>::infinity();
    const auto inverse_ipnorm = static_cast<float>(1.0 / ipnorm);
    const float ipnorm_inv = is_normal_ra_bit_q_value(inverse_ipnorm) ? inverse_ipnorm : 1.0F;
    float f_add = 0.0F;
    float f_rescale = 0.0F;
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        const float residual_centroid_ip =
            FP32ComputeIP(residual.data(), centroid_.data(), this->dim_);
        f_add = 1.0F - residual_centroid_ip +
                static_cast<float>(residual_norm_sqr * centroid_code_ip / safe_ip);
        f_rescale = -ipnorm_inv * residual_norm;
    } else {
        f_add = static_cast<float>(residual_norm_sqr +
                                   2.0 * residual_norm_sqr * centroid_code_ip / safe_ip);
        f_rescale = -2.0F * ipnorm_inv * residual_norm;
    }
    if (not IsFiniteRaBitQValue(f_add) or not IsFiniteRaBitQValue(f_rescale)) {
        return false;
    }
    std::memcpy(supplement_code + SupplementMetaOffset(), &f_add, sizeof(float));
    std::memcpy(
        supplement_code + SupplementMetaOffset() + sizeof(float), &f_rescale, sizeof(float));
    return true;
}

template <MetricType metric>
void
RaBitQuantizer<metric>::ComputeHnswCentroidTerms(const float* transformed_query,
                                                 float& g_add,
                                                 float& g_error) const {
    const float centroid_distance_sqr =
        FP32ComputeL2Sqr(transformed_query, centroid_.data(), this->dim_);
    g_error = std::sqrt(centroid_distance_sqr);
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        g_add = -FP32ComputeIP(transformed_query, centroid_.data(), this->dim_);
    } else {
        g_add = centroid_distance_sqr;
    }
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::ComputeHnswOneBit(const uint8_t* query_planes,
                                          float query_delta,
                                          float query_vl,
                                          float query_sum,
                                          float g_add,
                                          float g_error,
                                          const uint8_t* one_bit_code,
                                          const uint8_t* supplement_code,
                                          float* distance,
                                          float* lower_bound,
                                          float* filter_inner_product,
                                          float runtime_rabitq_error_rate) const {
    (void)supplement_code;
    const auto packed_ip = RaBitQSQ4UBinaryIPWithBaseSum(query_planes, one_bit_code, this->dim_);
    const auto raw_ip = static_cast<uint32_t>(packed_ip);
    const auto base_sum = static_cast<uint32_t>(packed_ip >> 32U);
    const float ip_x0_q =
        query_delta * static_cast<float>(raw_ip) + query_vl * static_cast<float>(base_sum);

    float f_add = 0.0F;
    float f_rescale = 0.0F;
    float f_error = 0.0F;
    std::memcpy(&f_add, one_bit_code + PlaneBytes(), sizeof(float));
    std::memcpy(&f_rescale, one_bit_code + PlaneBytes() + sizeof(float), sizeof(float));
    std::memcpy(&f_error, one_bit_code + PlaneBytes() + 2 * sizeof(float), sizeof(float));

    *distance = f_add + g_add + f_rescale * (ip_x0_q - 0.5F * query_sum);
    const float effective_error_rate =
        IsFiniteRaBitQValue(runtime_rabitq_error_rate) and runtime_rabitq_error_rate > 0.0F
            ? runtime_rabitq_error_rate
            : rabitq_error_rate_;
    const float error_rate_scale =
        effective_error_rate / RaBitQuantizerParameter::DEFAULT_RABITQ_ERROR_RATE;
    *lower_bound = *distance - error_rate_scale * f_error * g_error;
    *filter_inner_product = ip_x0_q;
    return IsFiniteRaBitQValue(*distance) and IsFiniteRaBitQValue(*lower_bound);
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::ComputeHnswFull(const float* transformed_query,
                                        float query_sum,
                                        float g_add,
                                        float g_error,
                                        const uint8_t* one_bit_code,
                                        const uint8_t* supplement_code,
                                        float filter_inner_product,
                                        float* distance,
                                        float* lower_bound) const {
    if (not IsFiniteRaBitQValue(filter_inner_product)) {
        return false;
    }
    float f_add = 0.0F;
    float f_rescale = 0.0F;
    float f_error = 0.0F;
    std::memcpy(&f_add, supplement_code + SupplementMetaOffset(), sizeof(float));
    std::memcpy(
        &f_rescale, supplement_code + SupplementMetaOffset() + sizeof(float), sizeof(float));
    std::memcpy(&f_error, one_bit_code + PlaneBytes() + 2 * sizeof(float), sizeof(float));
    const float supplement_ip =
        (this->dim_ & 63U) == 0U
            ? RaBitQFloatExCode7IP(transformed_query, supplement_code, this->dim_)
            : RaBitQFloatSupplementCodeIP(transformed_query, supplement_code, this->dim_, 7);
    *distance = f_add + g_add +
                f_rescale * (128.0F * filter_inner_product + supplement_ip - 127.5F * query_sum);
    *lower_bound = *distance - f_error * g_error / 128.0F;
    return IsFiniteRaBitQValue(*distance) and IsFiniteRaBitQValue(*lower_bound);
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::EncodeFusedAffineMetadata(const float* data,
                                                  uint8_t* one_bit_code,
                                                  uint8_t* supplement_code) const {
    if constexpr (metric != MetricType::METRIC_TYPE_L2SQR and
                  metric != MetricType::METRIC_TYPE_IP) {
        return false;
    }
    if (data == nullptr or one_bit_code == nullptr or supplement_code == nullptr or
        not SupportSplitCodeStorage() or FilterBits() < 1 or FilterBits() > 4 or
        ReorderBits() == 0) {
        return false;
    }

    const auto read_float = [](const uint8_t* address) {
        float value = 0.0F;
        std::memcpy(&value, address, sizeof(value));
        return value;
    };
    const auto write_float = [](uint8_t* address, float value) {
        std::memcpy(address, &value, sizeof(value));
    };

    const uint32_t filter_bits = FilterBits();
    const uint32_t supplement_bits = ReorderBits();
    const uint32_t base_bits = filter_bits + supplement_bits;
    const uint64_t plane_bytes = PlaneBytes();
    const uint64_t filter_meta_offset = OneBitRecordNormOffset();
    const uint64_t supplement_meta_offset = SupplementMetaOffset();
    const auto* supplement_meta = supplement_code + supplement_meta_offset;
    const auto full_meta_field = [this, supplement_meta](uint64_t full_code_offset) {
        return supplement_meta + (full_code_offset - CodeMetaOffset());
    };

    const float base_norm = read_float(one_bit_code + OneBitRecordNormOffset());
    const float filter_norm_code = filter_bits == 1
                                       ? 0.5F / inv_sqrt_d_
                                       : read_float(one_bit_code + OneBitRecordNormCodeOffset());
    const float filter_error =
        std::fabs(read_float(one_bit_code + OneBitRecordOneBitErrorOffset()));
    const float filter_epsilon = read_float(one_bit_code + OneBitRecordLowBoundErrorOffset());
    const float full_norm_code = read_float(full_meta_field(offset_norm_code_));
    float full_error = read_float(full_meta_field(offset_error_));
    if (std::fabs(full_error) < 1e-5F) {
        full_error = full_error >= 0.0F ? 1.0F : -1.0F;
    }

    Vector<float> transformed_data(this->dim_, 0.0F, this->allocator_);
    float raw_norm = 0.0F;
    norm_type mrq_norm_sqr = 0.0F;
    TransformFusedQuery(data, transformed_data, raw_norm, mrq_norm_sqr);

    const float filter_center = 0.5F * static_cast<float>((1U << filter_bits) - 1U);
    const float full_center = 0.5F * static_cast<float>((1U << base_bits) - 1U);
    double centroid_filter_ip = 0.0;
    double centroid_full_ip = 0.0;
    double centroid_residual_ip = 0.0;
    double residual_norm_sqr = 0.0;
    for (uint64_t d = 0; d < this->dim_; ++d) {
        const uint64_t byte_idx = d >> 3U;
        const auto bit_mask = static_cast<uint8_t>(1U << (d & 7U));

        uint32_t filter_code = 0;
        for (uint32_t bit = 0; bit < filter_bits; ++bit) {
            const auto* plane = one_bit_code + static_cast<uint64_t>(bit) * plane_bytes;
            if ((plane[byte_idx] & bit_mask) != 0U) {
                filter_code |= 1U << (filter_bits - bit - 1U);
            }
        }

        uint32_t supplement = 0;
        for (uint32_t bit = 0; bit < supplement_bits; ++bit) {
            const auto* plane = supplement_code + static_cast<uint64_t>(bit) * plane_bytes;
            if ((plane[byte_idx] & bit_mask) != 0U) {
                supplement |= 1U << bit;
            }
        }

        const uint32_t full_code = (filter_code << supplement_bits) | supplement;
        const auto centroid = static_cast<double>(centroid_[d]);
        const double residual =
            static_cast<double>(transformed_data[d]) - static_cast<double>(centroid_[d]);
        centroid_filter_ip += centroid * (static_cast<double>(filter_code) - filter_center);
        centroid_full_ip += centroid * (static_cast<double>(full_code) - full_center);
        centroid_residual_ip += centroid * residual;
        residual_norm_sqr += residual * residual;
    }

    constexpr float metric_scale = metric == MetricType::METRIC_TYPE_IP ? 1.0F : 2.0F;
    const float base_norm_sqr = base_norm * base_norm;
    float filter_add = std::numeric_limits<float>::quiet_NaN();
    float filter_rescale = std::numeric_limits<float>::quiet_NaN();
    float filter_error_unit = std::numeric_limits<float>::quiet_NaN();
    constexpr double k_normalize_zero_threshold = 1e-5;
    const bool degenerate_residual =
        std::isfinite(residual_norm_sqr) and residual_norm_sqr < k_normalize_zero_threshold;
    if (degenerate_residual) {
        const auto residual_norm = static_cast<float>(std::sqrt(std::max(0.0, residual_norm_sqr)));
        filter_rescale = 0.0F;
        if constexpr (metric == MetricType::METRIC_TYPE_IP) {
            filter_add = 1.0F - static_cast<float>(centroid_residual_ip);
        } else {
            filter_add = static_cast<float>(residual_norm_sqr);
            if (pca_dim_ != original_dim_ and use_mrq_) {
                filter_add += mrq_norm_sqr;
            }
        }
        filter_error_unit = metric_scale * residual_norm;
    } else if (IsFiniteRaBitQValue(base_norm) and IsFiniteRaBitQValue(filter_norm_code) and
               filter_norm_code > 0.0F and IsFiniteRaBitQValue(filter_error) and
               filter_error > 1e-5F and IsFiniteRaBitQValue(filter_epsilon)) {
        filter_rescale = -metric_scale * base_norm / (filter_error * filter_norm_code);
        if constexpr (metric == MetricType::METRIC_TYPE_IP) {
            filter_add = 1.0F - static_cast<float>(centroid_residual_ip) -
                         filter_rescale * static_cast<float>(centroid_filter_ip);
        } else {
            filter_add = base_norm_sqr - filter_rescale * static_cast<float>(centroid_filter_ip);
            if (pca_dim_ != original_dim_ and use_mrq_) {
                filter_add += mrq_norm_sqr;
            }
        }
        filter_error_unit = metric_scale * base_norm * filter_epsilon / filter_error;
    }
    if (not IsFiniteRaBitQValue(filter_add) or not IsFiniteRaBitQValue(filter_rescale) or
        not IsFiniteRaBitQValue(filter_error_unit)) {
        filter_add = std::numeric_limits<float>::quiet_NaN();
        filter_rescale = std::numeric_limits<float>::quiet_NaN();
        filter_error_unit = std::numeric_limits<float>::quiet_NaN();
    }

    float full_add = 0.0F;
    float full_rescale = 0.0F;
    if (degenerate_residual) {
        full_add = filter_add;
    } else {
        if (not IsFiniteRaBitQValue(base_norm) or not IsFiniteRaBitQValue(full_norm_code) or
            full_norm_code <= 0.0F or not IsFiniteRaBitQValue(full_error)) {
            return false;
        }
        full_rescale = -metric_scale * base_norm / (full_error * full_norm_code);
        if constexpr (metric == MetricType::METRIC_TYPE_IP) {
            full_add = 1.0F - static_cast<float>(centroid_residual_ip) -
                       full_rescale * static_cast<float>(centroid_full_ip);
        } else {
            full_add = base_norm_sqr - full_rescale * static_cast<float>(centroid_full_ip);
            if (pca_dim_ != original_dim_ and use_mrq_) {
                full_add += mrq_norm_sqr;
            }
        }
    }
    if (not IsFiniteRaBitQValue(full_add) or not IsFiniteRaBitQValue(full_rescale)) {
        return false;
    }

    write_float(one_bit_code + filter_meta_offset, filter_add);
    write_float(one_bit_code + filter_meta_offset + sizeof(float), filter_rescale);
    write_float(one_bit_code + filter_meta_offset + 2U * sizeof(float), filter_error_unit);
    write_float(supplement_code + supplement_meta_offset, full_add);
    write_float(supplement_code + supplement_meta_offset + sizeof(float), full_rescale);
    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::DecodeFusedSplitCode(const uint8_t* one_bit_code,
                                             const uint8_t* supplement_code,
                                             bool legacy_hnsw_codec,
                                             float* data) const {
    if constexpr (metric != MetricType::METRIC_TYPE_L2SQR and
                  metric != MetricType::METRIC_TYPE_IP) {
        return false;
    }
    if (one_bit_code == nullptr or supplement_code == nullptr or data == nullptr or
        not SupportSplitCodeStorage() or pca_dim_ != original_dim_ or rom_ == nullptr or
        centroid_.size() != this->dim_) {
        return false;
    }

    const uint32_t filter_bits = FilterBits();
    const uint32_t supplement_bits = ReorderBits();
    const uint32_t base_bits = filter_bits + supplement_bits;
    if (filter_bits < 1 or filter_bits > 4 or supplement_bits == 0 or base_bits >= 32 or
        (legacy_hnsw_codec and (filter_bits != 1 or supplement_bits != 7))) {
        return false;
    }

    float full_rescale = 0.0F;
    std::memcpy(&full_rescale,
                supplement_code + SupplementMetaOffset() + sizeof(float),
                sizeof(full_rescale));
    if (not IsFiniteRaBitQValue(full_rescale)) {
        return false;
    }

    constexpr float metric_scale = metric == MetricType::METRIC_TYPE_IP ? 1.0F : 2.0F;
    const float residual_scale = -full_rescale / metric_scale;
    if (not IsFiniteRaBitQValue(residual_scale)) {
        return false;
    }

    const uint64_t plane_bytes = PlaneBytes();
    const float full_center = 0.5F * static_cast<float>((1U << base_bits) - 1U);
    Vector<float> transformed_data(this->dim_, 0.0F, this->allocator_);
    for (uint64_t d = 0; d < this->dim_; ++d) {
        const uint64_t byte_idx = d >> 3U;
        const auto bit_mask = static_cast<uint8_t>(1U << (d & 7U));
        uint32_t filter_code = 0;
        for (uint32_t bit = 0; bit < filter_bits; ++bit) {
            const auto* plane = one_bit_code + static_cast<uint64_t>(bit) * plane_bytes;
            if ((plane[byte_idx] & bit_mask) != 0U) {
                filter_code |= 1U << (filter_bits - bit - 1U);
            }
        }

        uint32_t supplement = 0;
        if (legacy_hnsw_codec and (this->dim_ & 63U) == 0U) {
            constexpr uint64_t legacy_block_size = 56;
            constexpr uint64_t legacy_low_dimension_count = 48;
            const uint64_t lane = d & 63U;
            const auto* block = supplement_code + (d >> 6U) * legacy_block_size;
            const uint32_t top = (block[48U + (lane & 7U)] >> (lane >> 3U)) & 1U;
            uint32_t low = 0;
            if (lane < legacy_low_dimension_count) {
                low = block[lane] & 0x3FU;
            } else {
                const uint64_t packed_lane = lane - legacy_low_dimension_count;
                low = ((block[packed_lane] >> 6U) & 0x3U) |
                      (((block[16U + packed_lane] >> 6U) & 0x3U) << 2U) |
                      (((block[32U + packed_lane] >> 6U) & 0x3U) << 4U);
            }
            supplement = low | (top << 6U);
        } else {
            for (uint32_t bit = 0; bit < supplement_bits; ++bit) {
                const auto* plane = supplement_code + static_cast<uint64_t>(bit) * plane_bytes;
                if ((plane[byte_idx] & bit_mask) != 0U) {
                    supplement |= 1U << bit;
                }
            }
        }

        const uint32_t full_code = (filter_code << supplement_bits) | supplement;
        transformed_data[d] =
            centroid_[d] + residual_scale * (static_cast<float>(full_code) - full_center);
        if (not IsFiniteRaBitQValue(transformed_data[d])) {
            return false;
        }
    }

    rom_->InverseTransform(transformed_data.data(), data);
    for (uint64_t d = 0; d < original_dim_; ++d) {
        if (not IsFiniteRaBitQValue(data[d])) {
            return false;
        }
    }
    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::ComputeFusedExactCenteredFilterIP(
    const float* transformed_query,
    const uint8_t* one_bit_code,
    float* centered_filter_inner_product) const {
    if (transformed_query == nullptr or one_bit_code == nullptr or
        centered_filter_inner_product == nullptr or FilterBits() < 1 or FilterBits() > 4) {
        return false;
    }

    if (FilterBits() == 1) {
        if (inv_sqrt_d_ <= 0.0F) {
            return false;
        }
        const float normalized_ip =
            RaBitQFloatBinaryIP(transformed_query, one_bit_code, this->dim_, inv_sqrt_d_);
        *centered_filter_inner_product = normalized_ip * (0.5F / inv_sqrt_d_);
    } else if (FilterBits() == 2) {
        *centered_filter_inner_product =
            RaBitQFloatTwoBitCenteredIP(transformed_query, one_bit_code, this->dim_);
    } else if (FilterBits() == 3) {
        *centered_filter_inner_product =
            RaBitQFloatThreeBitCenteredIP(transformed_query, one_bit_code, this->dim_);
    } else {
        *centered_filter_inner_product =
            RaBitQFloatFourBitCenteredIP(transformed_query, one_bit_code, this->dim_);
    }
    return IsFiniteRaBitQValue(*centered_filter_inner_product);
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::ComputeFusedAffineFilter(const float* transformed_query,
                                                 const uint8_t* query_planes,
                                                 float query_delta,
                                                 float query_vl,
                                                 float query_sum,
                                                 float g_add,
                                                 float g_error,
                                                 const uint8_t* one_bit_code,
                                                 float runtime_rabitq_error_rate,
                                                 float* distance,
                                                 float* lower_bound,
                                                 float* centered_filter_inner_product,
                                                 RaBitQFusedIPPrecision* precision) const {
    if (distance != nullptr) {
        *distance = std::numeric_limits<float>::max();
    }
    if (lower_bound != nullptr) {
        *lower_bound = std::numeric_limits<float>::max();
    }
    if (centered_filter_inner_product != nullptr) {
        *centered_filter_inner_product = std::numeric_limits<float>::quiet_NaN();
    }
    if (precision != nullptr) {
        *precision = RaBitQFusedIPPrecision::INVALID;
    }
    if constexpr (metric != MetricType::METRIC_TYPE_L2SQR and
                  metric != MetricType::METRIC_TYPE_IP) {
        return false;
    }
    if (transformed_query == nullptr or one_bit_code == nullptr or distance == nullptr or
        lower_bound == nullptr or centered_filter_inner_product == nullptr or
        precision == nullptr or not SupportSplitCodeStorage() or FilterBits() < 1 or
        FilterBits() > 4 or not IsFiniteRaBitQValue(query_sum) or not IsFiniteRaBitQValue(g_add) or
        not IsFiniteRaBitQValue(g_error)) {
        return false;
    }

    float filter_ip = 0.0F;
    auto filter_precision = RaBitQFusedIPPrecision::EXACT;
    if (FilterBits() == 1 and query_planes != nullptr) {
        if (not IsFiniteRaBitQValue(query_delta) or not IsFiniteRaBitQValue(query_vl)) {
            return false;
        }
        const uint64_t packed_ip =
            RaBitQSQ4UBinaryIPWithBaseSum(query_planes, one_bit_code, this->dim_);
        const auto raw_ip = static_cast<uint32_t>(packed_ip);
        const auto base_sum = static_cast<uint32_t>(packed_ip >> 32U);
        const float uncentered_filter_ip =
            query_delta * static_cast<float>(raw_ip) + query_vl * static_cast<float>(base_sum);
        filter_ip = uncentered_filter_ip - 0.5F * query_sum;
        filter_precision = RaBitQFusedIPPrecision::APPROXIMATE;
    } else if (not ComputeFusedExactCenteredFilterIP(transformed_query, one_bit_code, &filter_ip)) {
        return false;
    }

    const uint64_t metadata_offset = OneBitRecordNormOffset();
    float filter_add = 0.0F;
    float filter_rescale = 0.0F;
    float filter_error_unit = 0.0F;
    std::memcpy(&filter_add, one_bit_code + metadata_offset, sizeof(filter_add));
    std::memcpy(
        &filter_rescale, one_bit_code + metadata_offset + sizeof(float), sizeof(filter_rescale));
    std::memcpy(&filter_error_unit,
                one_bit_code + metadata_offset + 2U * sizeof(float),
                sizeof(filter_error_unit));
    if (not IsFiniteRaBitQValue(filter_ip) or not IsFiniteRaBitQValue(filter_add) or
        not IsFiniteRaBitQValue(filter_rescale) or not IsFiniteRaBitQValue(filter_error_unit)) {
        return false;
    }

    const float result = filter_add + g_add + filter_rescale * filter_ip;
    const float effective_error_rate =
        IsFiniteRaBitQValue(runtime_rabitq_error_rate) and runtime_rabitq_error_rate > 0.0F
            ? runtime_rabitq_error_rate
            : rabitq_error_rate_;
    const float lower_bound_result = result - effective_error_rate * filter_error_unit * g_error;
    if (not IsFiniteRaBitQValue(result)) {
        return false;
    }

    // Traversal can still use a valid coarse distance when its error bound overflows and
    // supplement reranking is disabled.
    *distance = result;
    if (not IsFiniteRaBitQValue(lower_bound_result)) {
        return false;
    }
    *lower_bound = lower_bound_result - 1e-5F * std::max(1.0F, std::fabs(lower_bound_result));
    *centered_filter_inner_product = filter_ip;
    *precision = filter_precision;
    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::ComputeFusedAffineFullWithFilterIP(
    const float* transformed_query,
    float query_sum,
    float g_add,
    const uint8_t* one_bit_code,
    const uint8_t* supplement_code,
    float exact_centered_filter_inner_product,
    float* distance) const {
    if constexpr (metric != MetricType::METRIC_TYPE_L2SQR and
                  metric != MetricType::METRIC_TYPE_IP) {
        return false;
    }
    if (transformed_query == nullptr or one_bit_code == nullptr or supplement_code == nullptr or
        distance == nullptr or not SupportSplitCodeStorage() or ReorderBits() == 0 or
        not IsFiniteRaBitQValue(query_sum) or not IsFiniteRaBitQValue(g_add) or
        not IsFiniteRaBitQValue(exact_centered_filter_inner_product)) {
        return false;
    }

    float full_add = 0.0F;
    float full_rescale = 0.0F;
    std::memcpy(&full_add, supplement_code + SupplementMetaOffset(), sizeof(full_add));
    std::memcpy(&full_rescale,
                supplement_code + SupplementMetaOffset() + sizeof(float),
                sizeof(full_rescale));
    if (not IsFiniteRaBitQValue(full_add) or not IsFiniteRaBitQValue(full_rescale)) {
        return false;
    }

    const uint32_t supplement_bits = ReorderBits();
    const float supplement_ip = RaBitQFloatSupplementCodeIP(
        transformed_query, supplement_code, this->dim_, supplement_bits);
    const float supplement_center = 0.5F * static_cast<float>((1U << supplement_bits) - 1U);
    const float full_centered_ip =
        std::ldexp(exact_centered_filter_inner_product, supplement_bits) + supplement_ip -
        supplement_center * query_sum;
    const float result = full_add + g_add + full_rescale * full_centered_ip;
    if (not IsFiniteRaBitQValue(result)) {
        return false;
    }
    *distance = result;
    return true;
}

template <MetricType metric>
bool
RaBitQuantizer<metric>::ComputeFusedAffineFullDirect(const float* transformed_query,
                                                     float query_sum,
                                                     float g_add,
                                                     const uint8_t* one_bit_code,
                                                     const uint8_t* supplement_code,
                                                     float* distance) const {
    float exact_filter_ip = 0.0F;
    if (not ComputeFusedExactCenteredFilterIP(transformed_query, one_bit_code, &exact_filter_ip)) {
        return false;
    }
    return ComputeFusedAffineFullWithFilterIP(transformed_query,
                                              query_sum,
                                              g_add,
                                              one_bit_code,
                                              supplement_code,
                                              exact_filter_ip,
                                              distance);
}

template <MetricType metric>
void
RaBitQuantizer<metric>::ProcessTransformedFusedQuery(const float* transformed_query,
                                                     float query_raw_norm,
                                                     norm_type mrq_norm_sqr,
                                                     Computer<RaBitQuantizer>& computer) const {
    if (computer.buf_ == nullptr) {
        computer.buf_ =
            reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
    }
    std::fill(computer.buf_, computer.buf_ + this->query_code_size_, 0);
    Vector<float> normed_data(this->dim_, 0, this->allocator_);
    const float query_norm =
        NormalizeWithCentroid(transformed_query, centroid_.data(), normed_data.data(), this->dim_);

    if (num_bits_per_dim_query_ == 4) {
        Vector<uint8_t> quantized_data(this->dim_, 0, this->allocator_);
        float lower_bound = std::numeric_limits<float>::max();
        float upper_bound = std::numeric_limits<float>::lowest();
        float delta = 0.0F;
        sum_type query_sum = 0;
        EncodeSQ(
            normed_data.data(), quantized_data.data(), upper_bound, lower_bound, delta, query_sum);
        ReOrderSQ(quantized_data.data(), reinterpret_cast<uint8_t*>(computer.buf_));
        *(float*)(computer.buf_ + query_offset_lb_) = lower_bound;
        *(float*)(computer.buf_ + query_offset_delta_) = delta;
        *(sum_type*)(computer.buf_ + query_offset_sum_) = query_sum;
    } else {
        memcpy(computer.buf_, normed_data.data(), normed_data.size() * sizeof(float));
    }

    if (num_bits_per_dim_base_ != 1) {
        float query_raw_sum = 0;
        for (uint32_t d = 0; d < this->dim_; d++) {
            query_raw_sum += normed_data[d];
        }
        *(sum_type*)(computer.buf_ + query_offset_sum_) = query_raw_sum;
    }

    *(norm_type*)(computer.buf_ + query_offset_norm_) = query_norm;
    if (use_mrq_) {
        *(norm_type*)(computer.buf_ + query_offset_mrq_norm_) = mrq_norm_sqr;
    }
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        *(norm_type*)(computer.buf_ + query_offset_raw_norm_) = query_raw_norm;
    }
}

template <MetricType metric>
void
RaBitQuantizer<metric>::ProcessQueryImpl(const float* query,
                                         Computer<RaBitQuantizer>& computer) const {
    try {
        Vector<float> transformed_data(this->dim_, 0, this->allocator_);
        float query_raw_norm = 0.0F;
        norm_type mrq_norm_sqr = 0.0F;
        TransformFusedQuery(query, transformed_data, query_raw_norm, mrq_norm_sqr);
        ProcessTransformedFusedQuery(
            transformed_data.data(), query_raw_norm, mrq_norm_sqr, computer);
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
    RefreshSplitLayout(RaBitQuantizerParameter::IsSplitVersion(rabitq_version_) &&
                       num_bits_per_dim_query_ == 32 && num_bits_per_dim_base_ >= 1 &&
                       num_bits_per_dim_filter_ >= 1 &&
                       num_bits_per_dim_filter_ <= num_bits_per_dim_base_);
}

TEMPLATE_QUANTIZER(RaBitQuantizer)

}  // namespace vsag
