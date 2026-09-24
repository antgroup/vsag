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

#include "saq_quantizer.h"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

#include "impl/blas/blas_function.h"
#include "impl/transform/pca_transformer.h"
#include "simd/fp32_simd.h"
#include "simd/normalize.h"
#include "simd/rabitq_simd.h"
#include "utils/util_functions.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

struct DynamicPlanState {
    double error{std::numeric_limits<double>::infinity()};
    uint32_t previous_position{0};
    uint32_t previous_budget{0};
    uint16_t segment_count{0};
    uint8_t bits{0};
};

uint64_t
round_code_bits(uint64_t length, uint64_t bits) {
    return ((length + 7) / 8) * 8 * bits;
}

void
normalize_for_cosine(const float* source, float* destination, uint64_t dim) {
    const float squared_norm = FP32ComputeIP(source, source, dim);
    const float norm = std::sqrt(squared_norm);
    if (not std::isfinite(norm) or is_approx_zero(norm)) {
        std::fill(destination, destination + dim, 0.0F);
        return;
    }
    Normalize(source, destination, dim);
}

SAQQuantizerParamPtr
checked_saq_parameter(const SAQQuantizerParamPtr& param) {
    CHECK_ARGUMENT(param != nullptr, "SAQ quantizer requires SAQQuantizerParameter");
    return param;
}

SAQQuantizerParamPtr
checked_saq_parameter(const QuantizerParamPtr& param) {
    return checked_saq_parameter(std::dynamic_pointer_cast<SAQQuantizerParameter>(param));
}

}  // namespace

template <MetricType metric>
SAQQuantizer<metric>::SAQQuantizer(int dim,
                                   float avg_bits,
                                   uint64_t segment_count,
                                   uint64_t adjustment_rounds,
                                   bool use_pca,
                                   bool random_rotation,
                                   Allocator* allocator)
    : Quantizer<SAQQuantizer<metric>>(dim, allocator),
      avg_bits_(avg_bits),
      requested_segment_count_(segment_count),
      adjustment_rounds_(adjustment_rounds),
      use_pca_(use_pca),
      random_rotation_(random_rotation),
      projection_matrix_(allocator),
      projected_mean_(allocator) {
    CHECK_ARGUMENT(dim > 0, fmt::format("SAQ dimension must be positive, got {}", dim));
    const bool valid_avg_bits =
        std::isfinite(avg_bits_) and avg_bits_ >= 1.0F and avg_bits_ <= 8.0F;
    CHECK_ARGUMENT(valid_avg_bits,
                   fmt::format("saq_avg_bits must be finite and in [1, 8], got {}", avg_bits_));
    CHECK_ARGUMENT(adjustment_rounds_ <= SAQQuantizerParameter::MAX_ADJUSTMENT_ROUNDS,
                   fmt::format("saq_adjustment_rounds must not exceed {}, got {}",
                               SAQQuantizerParameter::MAX_ADJUSTMENT_ROUNDS,
                               adjustment_rounds_));

    const uint64_t requested_budget_bits =
        static_cast<uint64_t>(std::llround(avg_bits_ * this->dim_)) + RECORD_OVERHEAD_BITS;
    this->code_size_ = (requested_budget_bits + 7) / 8;
    budget_bits_ = this->code_size_ * 8;
    const uint64_t maximum_segment_count = (this->dim_ + SEGMENT_ALIGNMENT - 1) / SEGMENT_ALIGNMENT;
    const uint64_t lookup_size = ((this->dim_ + 7) / 8) * 256;
    this->query_code_size_ = (this->dim_ + 2 * maximum_segment_count + lookup_size) * sizeof(float);
    this->metric_ = metric;
}

template <MetricType metric>
SAQQuantizer<metric>::SAQQuantizer(const SAQQuantizerParamPtr& param,
                                   const IndexCommonParam& common_param)
    : SAQQuantizer(common_param.dim_,
                   checked_saq_parameter(param)->avg_bits_,
                   checked_saq_parameter(param)->segment_count_,
                   checked_saq_parameter(param)->adjustment_rounds_,
                   checked_saq_parameter(param)->use_pca_,
                   checked_saq_parameter(param)->random_rotation_,
                   common_param.allocator_.get()) {
}

template <MetricType metric>
SAQQuantizer<metric>::SAQQuantizer(const QuantizerParamPtr& param,
                                   const IndexCommonParam& common_param)
    : SAQQuantizer(checked_saq_parameter(param), common_param) {
}

template <MetricType metric>
bool
SAQQuantizer<metric>::TrainImpl(const float* data, uint64_t count) {
    if (data == nullptr or count < 2) {
        return false;
    }
    if (this->is_trained_) {
        return true;
    }

    const uint64_t training_count = std::min(count, MAX_TRAINING_SAMPLES);
    const float* training_data = data;
    Vector<float> sampled_data(this->allocator_);
    if (count > training_count or metric == MetricType::METRIC_TYPE_COSINE) {
        sampled_data.resize(training_count * this->dim_);
        for (uint64_t i = 0; i < training_count; ++i) {
            const uint64_t index =
                i * (count / training_count) + i * (count % training_count) / training_count;
            const float* source = data + index * this->dim_;
            float* destination = sampled_data.data() + i * this->dim_;
            if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
                normalize_for_cosine(source, destination, this->dim_);
            } else {
                std::copy(source, source + this->dim_, destination);
            }
        }
        training_data = sampled_data.data();
    }

    if (use_pca_) {
        PCATransformer pca(this->allocator_, this->dim_, this->dim_);
        pca.Train(training_data, training_count);
        projection_matrix_.resize(this->dim_ * this->dim_);
        pca.CopyPCAMatrix(projection_matrix_.data());
        if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
            Vector<float> mean(this->dim_, this->allocator_);
            pca.CopyMean(mean.data());
            projected_mean_.resize(this->dim_);
            BlasFunction::Sgemv(BlasFunction::RowMajor,
                                BlasFunction::NoTrans,
                                static_cast<int32_t>(this->dim_),
                                static_cast<int32_t>(this->dim_),
                                1.0F,
                                projection_matrix_.data(),
                                static_cast<int32_t>(this->dim_),
                                mean.data(),
                                1,
                                0.0F,
                                projected_mean_.data(),
                                1);
        } else {
            projected_mean_.clear();
        }
    } else {
        projection_matrix_.clear();
        if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
            projected_mean_.resize(this->dim_, 0.0F);
            for (uint64_t i = 0; i < training_count; ++i) {
                const auto sample_count = static_cast<float>(i + 1);
                for (uint64_t d = 0; d < this->dim_; ++d) {
                    const float value = training_data[i * this->dim_ + d];
                    projected_mean_[d] += (value - projected_mean_[d]) / sample_count;
                }
            }
        } else {
            projected_mean_.clear();
        }
    }

    Vector<float> means(this->dim_, this->allocator_);
    Vector<float> squared_differences(this->dim_, this->allocator_);
    Vector<float> projected(this->dim_, this->allocator_);
    std::fill(means.begin(), means.end(), 0.0F);
    std::fill(squared_differences.begin(), squared_differences.end(), 0.0F);
    for (uint64_t i = 0; i < training_count; ++i) {
        ProjectGlobal(training_data + i * this->dim_, projected.data());
        const auto sample_count = static_cast<float>(i + 1);
        for (uint64_t d = 0; d < this->dim_; ++d) {
            const float delta = projected[d] - means[d];
            means[d] += delta / sample_count;
            squared_differences[d] += delta * (projected[d] - means[d]);
        }
    }

    Vector<float> variances(this->dim_, this->allocator_);
    for (uint64_t d = 0; d < this->dim_; ++d) {
        variances[d] = squared_differences[d] / static_cast<float>(training_count - 1);
    }
    BuildPlan(variances);
    InitializeSegmentLayout();
    InitializeRotations(true);
    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
void
SAQQuantizer<metric>::ProjectGlobal(const float* data, float* projected) const {
    if (not use_pca_) {
        std::copy(data, data + this->dim_, projected);
    } else {
        BlasFunction::Sgemv(BlasFunction::RowMajor,
                            BlasFunction::NoTrans,
                            static_cast<int32_t>(this->dim_),
                            static_cast<int32_t>(this->dim_),
                            1.0F,
                            projection_matrix_.data(),
                            static_cast<int32_t>(this->dim_),
                            data,
                            1,
                            0.0F,
                            projected,
                            1);
    }
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        for (uint64_t d = 0; d < this->dim_; ++d) {
            projected[d] -= projected_mean_[d];
        }
    }
}

template <MetricType metric>
void
SAQQuantizer<metric>::Project(const float* data, float* projected) const {
    Vector<float> normalized(this->allocator_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        normalized.resize(this->dim_);
        normalize_for_cosine(data, normalized.data(), this->dim_);
        data = normalized.data();
    }

    Vector<float> global_projected(this->dim_, this->allocator_);
    ProjectGlobal(data, global_projected.data());
    for (uint64_t i = 0; i < segments_.size(); ++i) {
        const auto& segment = segments_[i];
        if (rotations_[i] != nullptr) {
            rotations_[i]->Transform(global_projected.data() + segment.begin,
                                     projected + segment.begin);
        } else {
            std::copy(global_projected.data() + segment.begin,
                      global_projected.data() + segment.begin + segment.length,
                      projected + segment.begin);
        }
    }
}

template <MetricType metric>
void
SAQQuantizer<metric>::InverseProject(const float* projected, float* data) const {
    Vector<float> global_projected(this->dim_, this->allocator_);
    for (uint64_t i = 0; i < segments_.size(); ++i) {
        const auto& segment = segments_[i];
        if (rotations_[i] != nullptr) {
            rotations_[i]->InverseTransform(projected + segment.begin,
                                            global_projected.data() + segment.begin);
        } else {
            std::copy(projected + segment.begin,
                      projected + segment.begin + segment.length,
                      global_projected.data() + segment.begin);
        }
    }

    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        for (uint64_t d = 0; d < this->dim_; ++d) {
            global_projected[d] += projected_mean_[d];
        }
    }
    if (not use_pca_) {
        std::copy(global_projected.begin(), global_projected.end(), data);
        return;
    }
    BlasFunction::Sgemv(BlasFunction::RowMajor,
                        BlasFunction::Trans,
                        static_cast<int32_t>(this->dim_),
                        static_cast<int32_t>(this->dim_),
                        1.0F,
                        projection_matrix_.data(),
                        static_cast<int32_t>(this->dim_),
                        global_projected.data(),
                        1,
                        0.0F,
                        data,
                        1);
}

template <MetricType metric>
void
SAQQuantizer<metric>::BuildPlan(const Vector<float>& variances) {
    if (requested_segment_count_ == 0) {
        BuildDynamicPlan(variances);
    } else {
        BuildFixedSegmentPlan(variances);
    }
    CHECK_ARGUMENT(not segments_.empty(), "SAQ could not construct a quantization plan");
}

template <MetricType metric>
void
SAQQuantizer<metric>::BuildDynamicPlan(const Vector<float>& variances) {
    std::vector<uint64_t> positions{0};
    for (uint64_t d = SEGMENT_ALIGNMENT; d < this->dim_; d += SEGMENT_ALIGNMENT) {
        positions.push_back(d);
    }
    positions.push_back(this->dim_);

    std::vector<double> prefix(this->dim_ + 1, 0.0);
    for (uint64_t d = 0; d < this->dim_; ++d) {
        prefix[d + 1] = prefix[d] + variances[d];
    }

    const uint64_t stride = budget_bits_ + 1;
    std::vector<DynamicPlanState> states(positions.size() * stride);
    states[0].error = 0.0;
    for (uint64_t end = 1; end < positions.size(); ++end) {
        for (uint64_t start = 0; start < end; ++start) {
            const uint64_t length = positions[end] - positions[start];
            const double variance = prefix[positions[end]] - prefix[positions[start]];
            for (uint64_t used = 0; used <= budget_bits_; ++used) {
                const auto& from = states[start * stride + used];
                if (not std::isfinite(from.error)) {
                    continue;
                }
                for (uint64_t bits = 1; bits <= MAX_BITS; ++bits) {
                    const uint64_t charge = METADATA_BITS + round_code_bits(length, bits);
                    if (used + charge > budget_bits_) {
                        break;
                    }
                    auto& to = states[end * stride + used + charge];
                    const double error =
                        from.error + variance / static_cast<double>(uint64_t{1} << bits);
                    if (error < to.error) {
                        to.error = error;
                        to.previous_position = static_cast<uint32_t>(start);
                        to.previous_budget = static_cast<uint32_t>(used);
                        to.segment_count = from.segment_count + 1;
                        to.bits = static_cast<uint8_t>(bits);
                    }
                }
            }
        }
    }

    const uint64_t last = positions.size() - 1;
    uint64_t best_budget = 0;
    double best_error = std::numeric_limits<double>::infinity();
    uint16_t best_segments = std::numeric_limits<uint16_t>::max();
    for (uint64_t used = 0; used <= budget_bits_; ++used) {
        const auto& state = states[last * stride + used];
        const bool materially_better = state.error < best_error * 0.99;
        const bool comparable_and_simpler =
            state.error <= best_error * 1.01 and state.segment_count < best_segments;
        if (materially_better or comparable_and_simpler) {
            best_error = state.error;
            best_budget = used;
            best_segments = state.segment_count;
        }
    }

    CHECK_ARGUMENT(
        std::isfinite(best_error),
        fmt::format("SAQ bit budget {} is too small for dimension {}", budget_bits_, this->dim_));
    segments_.clear();
    uint64_t position = last;
    uint64_t used = best_budget;
    while (position > 0) {
        const auto& state = states[position * stride + used];
        const uint64_t previous = state.previous_position;
        segments_.push_back(
            {positions[previous], positions[position] - positions[previous], state.bits, 0, 0, 0});
        used = state.previous_budget;
        position = previous;
    }
    std::reverse(segments_.begin(), segments_.end());
}

template <MetricType metric>
void
SAQQuantizer<metric>::BuildFixedSegmentPlan(const Vector<float>& variances) {
    std::vector<uint64_t> positions{0};
    for (uint64_t d = SEGMENT_ALIGNMENT; d < this->dim_; d += SEGMENT_ALIGNMENT) {
        positions.push_back(d);
    }
    positions.push_back(this->dim_);
    const uint64_t block_count = positions.size() - 1;
    CHECK_ARGUMENT(requested_segment_count_ <= block_count,
                   fmt::format("saq_segment_count {} exceeds the {} aligned dimension blocks",
                               requested_segment_count_,
                               block_count));

    segments_.clear();
    for (uint64_t s = 0; s < requested_segment_count_; ++s) {
        const uint64_t first_block = s * block_count / requested_segment_count_;
        const uint64_t last_block = (s + 1) * block_count / requested_segment_count_;
        segments_.push_back(
            {positions[first_block], positions[last_block] - positions[first_block], 0, 0, 0, 0});
    }

    const uint64_t stride = budget_bits_ + 1;
    std::vector<DynamicPlanState> states((segments_.size() + 1) * stride);
    states[0].error = 0.0;
    for (uint64_t s = 0; s < segments_.size(); ++s) {
        double variance = 0.0;
        for (uint64_t d = segments_[s].begin; d < segments_[s].begin + segments_[s].length; ++d) {
            variance += variances[d];
        }
        for (uint64_t used = 0; used <= budget_bits_; ++used) {
            const auto& from = states[s * stride + used];
            if (not std::isfinite(from.error)) {
                continue;
            }
            for (uint64_t bits = 1; bits <= MAX_BITS; ++bits) {
                const uint64_t charge = METADATA_BITS + round_code_bits(segments_[s].length, bits);
                if (used + charge > budget_bits_) {
                    break;
                }
                auto& to = states[(s + 1) * stride + used + charge];
                const double error =
                    from.error + variance / static_cast<double>(uint64_t{1} << bits);
                if (error < to.error) {
                    to.error = error;
                    to.previous_budget = static_cast<uint32_t>(used);
                    to.bits = static_cast<uint8_t>(bits);
                }
            }
        }
    }

    uint64_t used = 0;
    double best_error = std::numeric_limits<double>::infinity();
    for (uint64_t candidate = 0; candidate <= budget_bits_; ++candidate) {
        const auto& state = states[segments_.size() * stride + candidate];
        if (state.error < best_error) {
            best_error = state.error;
            used = candidate;
        }
    }
    CHECK_ARGUMENT(
        std::isfinite(best_error),
        fmt::format("SAQ budget cannot represent {} requested segments", requested_segment_count_));
    for (uint64_t s = segments_.size(); s > 0; --s) {
        const auto& state = states[s * stride + used];
        segments_[s - 1].bits = state.bits;
        used = state.previous_budget;
    }
}

template <MetricType metric>
void
SAQQuantizer<metric>::InitializeSegmentLayout() {
    uint64_t offset = 0;
    for (auto& segment : segments_) {
        segment.metadata_offset = offset;
        offset += METADATA_BYTES;
        segment.code_offset = offset;
        segment.code_bytes = ((segment.length + 7) / 8) * segment.bits;
        offset += segment.code_bytes;
    }
    CHECK_ARGUMENT(
        offset <= this->code_size_,
        fmt::format("SAQ plan needs {} bytes but record budget is {}", offset, this->code_size_));
}

template <MetricType metric>
void
SAQQuantizer<metric>::InitializeRotations(bool train) {
    rotations_.clear();
    rotations_.reserve(segments_.size());
    for (const auto& segment : segments_) {
        if (not random_rotation_) {
            rotations_.push_back(nullptr);
            continue;
        }
        auto rotation = std::make_shared<RandomOrthogonalMatrix>(
            this->allocator_,
            segment.length,
            RandomOrthogonalMatrix::MAX_RETRIES,
            SAQQuantizerParameter::DEFAULT_RANDOM_ROTATION_SEED + segment.begin);
        if (train) {
            rotation->Train(nullptr, 0);
        }
        rotations_.push_back(std::move(rotation));
    }
}

template <MetricType metric>
void
SAQQuantizer<metric>::WritePackedCode(
    uint8_t* codes, uint64_t index, uint64_t length, uint64_t bits, uint16_t value) {
    const uint64_t plane_bytes = (length + 7) / 8;
    const uint64_t byte_offset = index >> 3U;
    const auto mask = static_cast<uint8_t>(1U << (index & 7U));
    for (uint64_t value_bit = 0; value_bit < bits; ++value_bit) {
        const uint64_t plane = bits - 1 - value_bit;
        auto& target = codes[plane * plane_bytes + byte_offset];
        if ((value & (uint16_t{1} << value_bit)) != 0U) {
            target |= mask;
        } else {
            target &= static_cast<uint8_t>(~mask);
        }
    }
}

template <MetricType metric>
uint16_t
SAQQuantizer<metric>::ReadPackedCode(const uint8_t* codes,
                                     uint64_t index,
                                     uint64_t length,
                                     uint64_t bits) {
    const uint64_t plane_bytes = (length + 7) / 8;
    const uint64_t byte_offset = index >> 3U;
    const auto mask = static_cast<uint8_t>(1U << (index & 7U));
    uint16_t value = 0;
    for (uint64_t plane = 0; plane < bits; ++plane) {
        if ((codes[plane * plane_bytes + byte_offset] & mask) != 0U) {
            value |= static_cast<uint16_t>(uint16_t{1} << (bits - 1 - plane));
        }
    }
    return value;
}

template <MetricType metric>
void
SAQQuantizer<metric>::EncodeSegment(const float* data,
                                    const Segment& segment,
                                    uint8_t* codes) const {
    const auto code_max = static_cast<uint16_t>((uint16_t{1} << segment.bits) - 1);
    float max_abs = 0.0F;
    double original_norm = 0.0;
    for (uint64_t d = 0; d < segment.length; ++d) {
        max_abs = std::max(max_abs, std::abs(data[d]));
        original_norm += static_cast<double>(data[d]) * data[d];
    }
    const double delta = max_abs == 0.0F ? 0.0 : 2.0 * max_abs / (code_max + 1.0);
    std::vector<uint16_t> scalar_codes(segment.length, 0);
    double inner_product = 0.0;
    double quantized_norm = 0.0;
    for (uint64_t d = 0; d < segment.length; ++d) {
        const double scaled = delta == 0.0 ? 0.0 : std::floor((data[d] + max_abs) / delta);
        scalar_codes[d] =
            static_cast<uint16_t>(std::clamp(scaled, 0.0, static_cast<double>(code_max)));
        const double quantized = (scalar_codes[d] + 0.5) * delta - max_abs;
        inner_product += data[d] * quantized;
        quantized_norm += quantized * quantized;
    }

    for (uint64_t round = 0; round < adjustment_rounds_ and quantized_norm > 0.0; ++round) {
        uint64_t adjusted = 0;
        const double epsilon = 1e-8 * quantized_norm;
        for (uint64_t d = 0; d < segment.length; ++d) {
            uint16_t current = scalar_codes[d];
            double quantized = (current + 0.5) * delta - max_abs;
            const double remaining_norm = quantized_norm - quantized * quantized;
            const double inner_delta = delta * data[d];
            while (current < code_max) {
                const double next_quantized = quantized + delta;
                const double next_norm = remaining_norm + next_quantized * next_quantized;
                const double next_inner = inner_product + inner_delta;
                if ((inner_product * inner_product + epsilon) * next_norm >=
                    next_inner * next_inner * quantized_norm) {
                    break;
                }
                ++current;
                ++adjusted;
                quantized = next_quantized;
                quantized_norm = next_norm;
                inner_product = next_inner;
            }
            while (current > 0) {
                const double next_quantized = quantized - delta;
                const double next_norm = remaining_norm + next_quantized * next_quantized;
                const double next_inner = inner_product - inner_delta;
                if ((inner_product * inner_product + epsilon) * next_norm >=
                    next_inner * next_inner * quantized_norm) {
                    break;
                }
                --current;
                ++adjusted;
                quantized = next_quantized;
                quantized_norm = next_norm;
                inner_product = next_inner;
            }
            scalar_codes[d] = current;
        }
        if (adjusted == 0) {
            break;
        }
        inner_product = 0.0;
        quantized_norm = 0.0;
        for (uint64_t d = 0; d < segment.length; ++d) {
            const double quantized = (scalar_codes[d] + 0.5) * delta - max_abs;
            inner_product += data[d] * quantized;
            quantized_norm += quantized * quantized;
        }
    }

    // Zero is the canonical scale for zero or numerically degenerate segments.
    const float adjusted_scale = inner_product > std::numeric_limits<double>::epsilon()
                                     ? static_cast<float>(original_norm / inner_product * max_abs)
                                     : 0.0F;
    auto stored_norm = static_cast<float>(original_norm);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        double reconstructed_norm = 0.0;
        const auto level_count = static_cast<double>(uint32_t{1} << segment.bits);
        for (const uint16_t scalar_code : scalar_codes) {
            const double reconstructed =
                adjusted_scale * (2.0 * (scalar_code + 0.5) / level_count - 1.0);
            reconstructed_norm += reconstructed * reconstructed;
        }
        stored_norm = static_cast<float>(reconstructed_norm);
    }
    std::memcpy(codes + segment.metadata_offset, &adjusted_scale, sizeof(float));
    std::memcpy(codes + segment.metadata_offset + sizeof(float), &stored_norm, sizeof(float));
    for (uint64_t d = 0; d < segment.length; ++d) {
        WritePackedCode(
            codes + segment.code_offset, d, segment.length, segment.bits, scalar_codes[d]);
    }
}

template <MetricType metric>
void
SAQQuantizer<metric>::DecodeSegment(const uint8_t* codes,
                                    const Segment& segment,
                                    float* data) const {
    float adjusted_scale = 0.0F;
    std::memcpy(&adjusted_scale, codes + segment.metadata_offset, sizeof(float));
    const auto level_count = static_cast<float>(uint32_t{1} << segment.bits);
    for (uint64_t d = 0; d < segment.length; ++d) {
        const uint16_t value =
            ReadPackedCode(codes + segment.code_offset, d, segment.length, segment.bits);
        data[d] = adjusted_scale * (2.0F * (static_cast<float>(value) + 0.5F) / level_count - 1.0F);
    }
}

template <MetricType metric>
bool
SAQQuantizer<metric>::EncodeOneImpl(const float* data, uint8_t* codes) const {
    if (not this->is_trained_ or data == nullptr or codes == nullptr) {
        return false;
    }
    std::fill(codes, codes + this->code_size_, 0);
    Vector<float> projected(this->dim_, this->allocator_);
    Project(data, projected.data());
    for (const auto& segment : segments_) {
        EncodeSegment(projected.data() + segment.begin, segment, codes);
    }
    return true;
}

template <MetricType metric>
bool
SAQQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, float* data) const {
    if (not this->is_trained_ or codes == nullptr or data == nullptr) {
        return false;
    }
    Vector<float> projected(this->dim_, this->allocator_);
    for (const auto& segment : segments_) {
        DecodeSegment(codes, segment, projected.data() + segment.begin);
    }
    InverseProject(projected.data(), data);
    return true;
}

template <MetricType metric>
float
SAQQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    // EncodeOneImpl and DecodeOneImpl expose boolean failure results, but ComputeImpl returns a
    // distance. Returning false here would silently report a valid zero distance.
    if (not this->is_trained_ or codes1 == nullptr or codes2 == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "cannot compute distance from untrained or null SAQ codes");
    }

    double distance = 0.0;
    double inner = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    for (const auto& segment : segments_) {
        float adjusted_scale1 = 0.0F;
        float adjusted_scale2 = 0.0F;
        std::memcpy(&adjusted_scale1, codes1 + segment.metadata_offset, sizeof(float));
        std::memcpy(&adjusted_scale2, codes2 + segment.metadata_offset, sizeof(float));

        const uint64_t plane_bytes = (segment.length + 7) / 8;
        const uint64_t word_count = (segment.length + 63) / 64;
        double code_sum1 = 0.0;
        double code_sum2 = 0.0;
        [[maybe_unused]] double code_square_sum1 = 0.0;
        [[maybe_unused]] double code_square_sum2 = 0.0;
        double code_product_sum = 0.0;
        for (uint64_t word_index = 0; word_index < word_count; ++word_index) {
            std::array<uint64_t, MAX_BITS> words1{};
            std::array<uint64_t, MAX_BITS> words2{};
            const uint64_t byte_offset = word_index * sizeof(uint64_t);
            const uint64_t bytes_to_copy =
                std::min<uint64_t>(sizeof(uint64_t), plane_bytes - byte_offset);
            const uint64_t valid_tail_bits = segment.length % 64;
            for (uint64_t plane = 0; plane < segment.bits; ++plane) {
                std::memcpy(&words1[plane],
                            codes1 + segment.code_offset + plane * plane_bytes + byte_offset,
                            bytes_to_copy);
                std::memcpy(&words2[plane],
                            codes2 + segment.code_offset + plane * plane_bytes + byte_offset,
                            bytes_to_copy);
                if (word_index + 1 == word_count and valid_tail_bits != 0) {
                    const uint64_t tail_mask = (uint64_t{1} << valid_tail_bits) - 1;
                    words1[plane] &= tail_mask;
                    words2[plane] &= tail_mask;
                }

                const auto weight = static_cast<double>(uint64_t{1} << (segment.bits - 1 - plane));
                code_sum1 += weight * __builtin_popcountll(words1[plane]);
                code_sum2 += weight * __builtin_popcountll(words2[plane]);
                if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
                    code_square_sum1 += weight * weight * __builtin_popcountll(words1[plane]);
                    code_square_sum2 += weight * weight * __builtin_popcountll(words2[plane]);
                }
            }

            for (uint64_t plane1 = 0; plane1 < segment.bits; ++plane1) {
                const auto weight1 =
                    static_cast<double>(uint64_t{1} << (segment.bits - 1 - plane1));
                for (uint64_t plane2 = plane1 + 1; plane2 < segment.bits; ++plane2) {
                    const auto weight2 =
                        static_cast<double>(uint64_t{1} << (segment.bits - 1 - plane2));
                    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
                        code_square_sum1 += 2.0 * weight1 * weight2 *
                                            __builtin_popcountll(words1[plane1] & words1[plane2]);
                        code_square_sum2 += 2.0 * weight1 * weight2 *
                                            __builtin_popcountll(words2[plane1] & words2[plane2]);
                    }
                }
                for (uint64_t plane2 = 0; plane2 < segment.bits; ++plane2) {
                    const auto weight2 =
                        static_cast<double>(uint64_t{1} << (segment.bits - 1 - plane2));
                    code_product_sum +=
                        weight1 * weight2 * __builtin_popcountll(words1[plane1] & words2[plane2]);
                }
            }
        }

        const auto level_count = static_cast<double>(uint32_t{1} << segment.bits);
        const double code_scale = 2.0 / level_count;
        const double code_offset = 1.0 / level_count - 1.0;
        const double segment_inner = static_cast<double>(adjusted_scale1) * adjusted_scale2 *
                                     (code_scale * code_scale * code_product_sum +
                                      code_scale * code_offset * (code_sum1 + code_sum2) +
                                      code_offset * code_offset * segment.length);
        if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
            const double segment_norm1 = static_cast<double>(adjusted_scale1) * adjusted_scale1 *
                                         (code_scale * code_scale * code_square_sum1 +
                                          2.0 * code_scale * code_offset * code_sum1 +
                                          code_offset * code_offset * segment.length);
            const double segment_norm2 = static_cast<double>(adjusted_scale2) * adjusted_scale2 *
                                         (code_scale * code_scale * code_square_sum2 +
                                          2.0 * code_scale * code_offset * code_sum2 +
                                          code_offset * code_offset * segment.length);
            distance += std::max(0.0, segment_norm1 + segment_norm2 - 2.0 * segment_inner);
        } else {
            inner += segment_inner;
            if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
                float reconstructed_norm1 = 0.0F;
                float reconstructed_norm2 = 0.0F;
                std::memcpy(&reconstructed_norm1,
                            codes1 + segment.metadata_offset + sizeof(float),
                            sizeof(float));
                std::memcpy(&reconstructed_norm2,
                            codes2 + segment.metadata_offset + sizeof(float),
                            sizeof(float));
                norm1 += reconstructed_norm1;
                norm2 += reconstructed_norm2;
            }
        }
    }

    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        return static_cast<float>(distance);
    } else if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        norm1 = std::max(0.0, norm1);
        norm2 = std::max(0.0, norm2);
        if (is_approx_zero(static_cast<float>(norm1)) or
            is_approx_zero(static_cast<float>(norm2))) {
            return 1.0F;
        }
        return static_cast<float>(1.0 - inner / std::sqrt(norm1 * norm2));
    } else {
        return static_cast<float>(1.0 - inner);
    }
}

template <MetricType metric>
void
SAQQuantizer<metric>::ProcessQueryImpl(const float* query, Computer<SAQQuantizer>& computer) const {
    if (not this->is_trained_) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "SAQ quantizer is not trained");
    }
    if (computer.buf_ == nullptr) {
        computer.buf_ =
            reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        if (computer.buf_ == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "cannot allocate SAQ query buffer");
        }
    }
    auto* projected = reinterpret_cast<float*>(computer.buf_);
    Project(query, projected);
    const uint64_t maximum_segment_count = (this->dim_ + SEGMENT_ALIGNMENT - 1) / SEGMENT_ALIGNMENT;
    auto* query_norms = projected + this->dim_;
    auto* query_sums = query_norms + maximum_segment_count;
    auto* query_lookup = query_sums + maximum_segment_count;
    for (uint64_t i = 0; i < segments_.size(); ++i) {
        const auto& segment = segments_[i];
        query_norms[i] =
            FP32ComputeIP(projected + segment.begin, projected + segment.begin, segment.length);
        query_sums[i] = 0.0F;
        for (uint64_t d = 0; d < segment.length; ++d) {
            query_sums[i] += projected[segment.begin + d];
        }
        generic::RaBitQFloatBuildByteIPLookupTable(
            projected + segment.begin, segment.length, query_lookup + (segment.begin / 8) * 256);
    }
}

template <MetricType metric>
float
SAQQuantizer<metric>::ComputeSegmentInnerProduct(const float* query_lookup,
                                                 float query_sum,
                                                 const uint8_t* codes,
                                                 const Segment& segment) const {
    float adjusted_scale = 0.0F;
    std::memcpy(&adjusted_scale, codes + segment.metadata_offset, sizeof(float));
    const auto level_count = static_cast<float>(uint32_t{1} << segment.bits);
    const float raw_inner = RaBitQFloatMultiBitIPByLookup(query_lookup + (segment.begin / 8) * 256,
                                                          codes + segment.code_offset,
                                                          segment.length,
                                                          0,
                                                          segment.bits);
    return adjusted_scale *
           (2.0F * raw_inner / level_count + (1.0F / level_count - 1.0F) * query_sum);
}

template <MetricType metric>
float
SAQQuantizer<metric>::ComputeProjectedDistance(const float* query_lookup,
                                               const float* query_norms,
                                               const float* query_sums,
                                               const uint8_t* codes,
                                               float threshold,
                                               bool enable_threshold,
                                               bool* stopped_early) const {
    float distance = 0.0F;
    float query_norm = 0.0F;
    float candidate_norm = 0.0F;
    for (uint64_t i = 0; i < segments_.size(); ++i) {
        const auto& segment = segments_[i];
        const float inner = ComputeSegmentInnerProduct(query_lookup, query_sums[i], codes, segment);
        if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
            float original_norm = 0.0F;
            std::memcpy(
                &original_norm, codes + segment.metadata_offset + sizeof(float), sizeof(float));
            distance += std::max(0.0F, query_norms[i] + original_norm - 2.0F * inner);
            if (enable_threshold and distance > threshold) {
                if (stopped_early != nullptr) {
                    *stopped_early = true;
                }
                return distance;
            }
        } else if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
            distance += inner;
            query_norm += query_norms[i];
            float reconstructed_norm = 0.0F;
            std::memcpy(&reconstructed_norm,
                        codes + segment.metadata_offset + sizeof(float),
                        sizeof(float));
            candidate_norm += reconstructed_norm;
        } else {
            distance += inner;
        }
    }
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        query_norm = std::max(0.0F, query_norm);
        candidate_norm = std::max(0.0F, candidate_norm);
        distance = is_approx_zero(query_norm) or is_approx_zero(candidate_norm)
                       ? 1.0F
                       : 1.0F - distance / std::sqrt(query_norm * candidate_norm);
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        distance = 1.0F - distance;
    }
    if (stopped_early != nullptr) {
        *stopped_early = false;
    }
    return distance;
}

template <MetricType metric>
void
SAQQuantizer<metric>::ComputeDistImpl(Computer<SAQQuantizer>& computer,
                                      const uint8_t* codes,
                                      float* dists) const {
    auto* query = reinterpret_cast<float*>(computer.buf_);
    const uint64_t maximum_segment_count = (this->dim_ + SEGMENT_ALIGNMENT - 1) / SEGMENT_ALIGNMENT;
    const float* query_norms = query + this->dim_;
    const float* query_sums = query_norms + maximum_segment_count;
    const float* query_lookup = query_sums + maximum_segment_count;
    dists[0] = ComputeProjectedDistance(
        query_lookup, query_norms, query_sums, codes, 0.0F, false, nullptr);
}

template <MetricType metric>
void
SAQQuantizer<metric>::ComputeDistsBatch4Impl(Computer<SAQQuantizer>& computer,
                                             const uint8_t* codes1,
                                             const uint8_t* codes2,
                                             const uint8_t* codes3,
                                             const uint8_t* codes4,
                                             float& dist1,
                                             float& dist2,
                                             float& dist3,
                                             float& dist4) const {
    auto* query = reinterpret_cast<float*>(computer.buf_);
    const uint64_t maximum_segment_count = (this->dim_ + SEGMENT_ALIGNMENT - 1) / SEGMENT_ALIGNMENT;
    const float* query_norms = query + this->dim_;
    const float* query_sums = query_norms + maximum_segment_count;
    const float* query_lookup = query_sums + maximum_segment_count;
    const uint8_t* candidates[4] = {codes1, codes2, codes3, codes4};
    float distances[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    float query_norm = 0.0F;
    float candidate_norms[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (uint64_t i = 0; i < segments_.size(); ++i) {
        const auto& segment = segments_[i];
        float raw_inner[4];
        RaBitQFloatMultiBitIPBatch4ByLookup(query_lookup + (segment.begin / 8) * 256,
                                            codes1 + segment.code_offset,
                                            codes2 + segment.code_offset,
                                            codes3 + segment.code_offset,
                                            codes4 + segment.code_offset,
                                            segment.length,
                                            0,
                                            segment.bits,
                                            raw_inner);
        const auto level_count = static_cast<float>(uint32_t{1} << segment.bits);
        if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
            query_norm += query_norms[i];
        }
        for (uint64_t candidate = 0; candidate < 4; ++candidate) {
            float adjusted_scale = 0.0F;
            std::memcpy(
                &adjusted_scale, candidates[candidate] + segment.metadata_offset, sizeof(float));
            const float inner = adjusted_scale * (2.0F * raw_inner[candidate] / level_count +
                                                  (1.0F / level_count - 1.0F) * query_sums[i]);
            if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
                float original_norm = 0.0F;
                std::memcpy(&original_norm,
                            candidates[candidate] + segment.metadata_offset + sizeof(float),
                            sizeof(float));
                distances[candidate] +=
                    std::max(0.0F, query_norms[i] + original_norm - 2.0F * inner);
            } else if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
                distances[candidate] += inner;
                float reconstructed_norm = 0.0F;
                std::memcpy(&reconstructed_norm,
                            candidates[candidate] + segment.metadata_offset + sizeof(float),
                            sizeof(float));
                candidate_norms[candidate] += reconstructed_norm;
            } else {
                distances[candidate] += inner;
            }
        }
    }
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        query_norm = std::max(0.0F, query_norm);
        for (uint64_t candidate = 0; candidate < 4; ++candidate) {
            candidate_norms[candidate] = std::max(0.0F, candidate_norms[candidate]);
            distances[candidate] =
                is_approx_zero(query_norm) or is_approx_zero(candidate_norms[candidate])
                    ? 1.0F
                    : 1.0F -
                          distances[candidate] / std::sqrt(query_norm * candidate_norms[candidate]);
        }
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        for (auto& distance : distances) {
            distance = 1.0F - distance;
        }
    }
    dist1 = distances[0];
    dist2 = distances[1];
    dist3 = distances[2];
    dist4 = distances[3];
}

template <MetricType metric>
bool
SAQQuantizer<metric>::ComputeDistWithThresholdImpl(Computer<SAQQuantizer>& computer,
                                                   const uint8_t* codes,
                                                   float threshold,
                                                   float* dists) const {
    bool stopped_early = false;
    auto* query = reinterpret_cast<float*>(computer.buf_);
    const uint64_t maximum_segment_count = (this->dim_ + SEGMENT_ALIGNMENT - 1) / SEGMENT_ALIGNMENT;
    const float* query_norms = query + this->dim_;
    const float* query_sums = query_norms + maximum_segment_count;
    const float* query_lookup = query_sums + maximum_segment_count;
    // Only L2 accumulates non-negative segment distances, so threshold early termination is safe.
    // Inner product and cosine compute the full distance and therefore return false.
    constexpr bool enable_threshold = metric == MetricType::METRIC_TYPE_L2SQR;
    dists[0] = ComputeProjectedDistance(
        query_lookup, query_norms, query_sums, codes, threshold, enable_threshold, &stopped_early);
    return stopped_early;
}

template <MetricType metric>
void
SAQQuantizer<metric>::SerializeImpl(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, avg_bits_);
    StreamWriter::WriteObj(writer, requested_segment_count_);
    StreamWriter::WriteObj(writer, adjustment_rounds_);
    StreamWriter::WriteObj(writer, use_pca_);
    StreamWriter::WriteObj(writer, random_rotation_);
    StreamWriter::WriteObj(writer, budget_bits_);
    StreamWriter::WriteVector(writer, projection_matrix_);
    StreamWriter::WriteVector(writer, projected_mean_);
    const uint64_t segment_count = segments_.size();
    StreamWriter::WriteObj(writer, segment_count);
    for (const auto& segment : segments_) {
        StreamWriter::WriteObj(writer, segment.begin);
        StreamWriter::WriteObj(writer, segment.length);
        StreamWriter::WriteObj(writer, segment.bits);
    }
    if (random_rotation_) {
        for (const auto& rotation : rotations_) {
            rotation->Serialize(writer);
        }
    }
}

template <MetricType metric>
void
SAQQuantizer<metric>::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadObj(reader, avg_bits_);
    StreamReader::ReadObj(reader, requested_segment_count_);
    StreamReader::ReadObj(reader, adjustment_rounds_);
    StreamReader::ReadObj(reader, use_pca_);
    StreamReader::ReadObj(reader, random_rotation_);
    StreamReader::ReadObj(reader, budget_bits_);
    StreamReader::ReadVector(reader, projection_matrix_);
    StreamReader::ReadVector(reader, projected_mean_);
    CHECK_ARGUMENT(this->dim_ > 0 and this->metric_ == metric and this->is_trained_,
                   "serialized SAQ base state is invalid");
    const bool valid_parameters =
        std::isfinite(avg_bits_) and avg_bits_ >= 1.0F and avg_bits_ <= 8.0F and
        adjustment_rounds_ <= SAQQuantizerParameter::MAX_ADJUSTMENT_ROUNDS;
    CHECK_ARGUMENT(valid_parameters, "serialized SAQ parameters are invalid");
    const uint64_t requested_budget_bits =
        static_cast<uint64_t>(std::llround(avg_bits_ * this->dim_)) + RECORD_OVERHEAD_BITS;
    const uint64_t expected_budget_bits = ((requested_budget_bits + 7) / 8) * 8;
    CHECK_ARGUMENT(
        budget_bits_ == expected_budget_bits and this->code_size_ == (budget_bits_ + 7) / 8,
        "serialized SAQ configuration or record budget is invalid");
    uint64_t segment_count = 0;
    StreamReader::ReadObj(reader, segment_count);
    const uint64_t maximum_segment_count = (this->dim_ + SEGMENT_ALIGNMENT - 1) / SEGMENT_ALIGNMENT;
    const bool valid_segment_count = segment_count > 0 and segment_count <= maximum_segment_count;
    CHECK_ARGUMENT(valid_segment_count, "serialized SAQ segment count is invalid");
    segments_.resize(segment_count);
    uint64_t covered_dimensions = 0;
    for (auto& segment : segments_) {
        StreamReader::ReadObj(reader, segment.begin);
        StreamReader::ReadObj(reader, segment.length);
        StreamReader::ReadObj(reader, segment.bits);
        CHECK_ARGUMENT(segment.begin == covered_dimensions and segment.length > 0 and
                           segment.length <= this->dim_ - covered_dimensions and
                           segment.begin % SEGMENT_ALIGNMENT == 0 and segment.bits >= 1 and
                           segment.bits <= MAX_BITS,
                       "invalid serialized SAQ segment plan");
        covered_dimensions += segment.length;
    }
    CHECK_ARGUMENT(
        covered_dimensions == this->dim_,
        fmt::format(
            "serialized SAQ plan covers {} of {} dimensions", covered_dimensions, this->dim_));
    const uint64_t expected_projection_size = use_pca_ ? this->dim_ * this->dim_ : 0;
    CHECK_ARGUMENT(projection_matrix_.size() == expected_projection_size,
                   "serialized SAQ PCA matrix has invalid dimensions");
    const uint64_t expected_mean_size = metric == MetricType::METRIC_TYPE_L2SQR ? this->dim_ : 0;
    CHECK_ARGUMENT(projected_mean_.size() == expected_mean_size,
                   "serialized SAQ projected mean has invalid dimensions");
    const bool valid_fixed_segment_count =
        requested_segment_count_ == 0 or requested_segment_count_ == segment_count;
    CHECK_ARGUMENT(valid_fixed_segment_count,
                   "serialized SAQ fixed segment count does not match its plan");
    const uint64_t lookup_size = ((this->dim_ + 7) / 8) * 256;
    this->query_code_size_ = (this->dim_ + 2 * maximum_segment_count + lookup_size) * sizeof(float);
    InitializeSegmentLayout();
    InitializeRotations(false);
    if (random_rotation_) {
        for (auto& rotation : rotations_) {
            rotation->Deserialize(reader);
        }
    }
}

TEMPLATE_QUANTIZER(SAQQuantizer)

}  // namespace vsag
