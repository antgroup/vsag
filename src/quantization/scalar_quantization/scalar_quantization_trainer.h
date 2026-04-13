
/**
 * @file scalar_quantization_trainer.h
 * @brief Trainer class for computing scalar quantization bounds and parameters.
 */

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

#include <cstdint>

#include "typing.h"

namespace vsag {

/**
 * @brief Enumeration of scalar quantization training modes.
 */
enum SQTrainMode {
    CLASSIC = 1,      /// Classic training using min/max bounds
    K_MEANS = 2,      /// K-means based training
    TRUNC_BOUND = 3,  /// Training with truncated bounds to handle outliers
};

/**
 * @brief Trainer class for computing scalar quantization parameters.
 *
 * This class provides methods to compute upper and lower bounds
 * for scalar quantization using various training strategies.
 */
class ScalarQuantizationTrainer {
public:
    /**
     * @brief Constructs a trainer with specified dimension and bit width.
     * @param dim Vector dimension.
     * @param bits Number of bits per dimension (default: 8).
     */
    explicit ScalarQuantizationTrainer(int32_t dim, int bits = 8);

    /**
     * @brief Trains bounds for per-dimension scalar quantization.
     * @param data Training data array.
     * @param count Number of vectors in training data.
     * @param upper_bound Output upper bounds per dimension.
     * @param lower_bound Output lower bounds per dimension.
     * @param need_normalize Whether to normalize training data first.
     * @param mode Training mode (default: TRUNC_BOUND).
     */
    void
    Train(const float* data,
          uint64_t count,
          float* upper_bound,
          float* lower_bound,
          bool need_normalize = false,
          SQTrainMode mode = SQTrainMode::TRUNC_BOUND);

    /**
     * @brief Trains uniform bounds for scalar quantization.
     * @param data Training data array.
     * @param count Number of vectors in training data.
     * @param upper_bound Output upper bound (single value).
     * @param lower_bound Output lower bound (single value).
     * @param need_normalize Whether to normalize training data first.
     * @param mode Training mode (default: TRUNC_BOUND).
     */
    void
    TrainUniform(const float* data,
                 uint64_t count,
                 float& upper_bound,
                 float& lower_bound,
                 bool need_normalize = false,
                 SQTrainMode mode = SQTrainMode::TRUNC_BOUND);

    /**
     * @brief Sets the maximum sample count for training.
     * @param sample Maximum number of samples to use.
     */
    inline void
    SetSampleCount(uint64_t sample) {
        this->max_sample_count_ = sample;
    }

    /**
     * @brief Sets the truncation rate for SQ4 uniform quantizer.
     * @param trunc_rate Truncation rate for outlier handling.
     */
    inline void
    SetSQ4UniformTruncRate(float trunc_rate) {
        this->trunc_rate_ = trunc_rate;
    }

private:
    /**
     * @brief Classic training using min/max bounds.
     * @param data Training data array.
     * @param count Number of vectors.
     * @param upper_bound Output upper bounds.
     * @param lower_bound Output lower bounds.
     */
    void
    classic_train(const float* data, uint64_t count, float* upper_bound, float* lower_bound) const;

    /**
     * @brief Training with truncated bounds to handle outliers.
     * @param data Training data array.
     * @param count Number of vectors.
     * @param upper_bound Output upper bounds.
     * @param lower_bound Output lower bounds.
     */
    void
    trunc_bound_train(const float* data,
                      uint64_t count,
                      float* upper_bound,
                      float* lower_bound) const;

    /**
     * @brief Samples training data for efficiency.
     * @param data Original training data.
     * @param count Number of vectors.
     * @param sample_datas Output sampled data.
     * @param need_normalize Whether to normalize samples.
     * @return Number of sampled vectors.
     */
    uint64_t
    sample_train_data(const float* data,
                      uint64_t count,
                      std::vector<float>& sample_datas,
                      bool need_normalize = false) const;

private:
    int dim_{0};                                     /// Vector dimension
    int bits_{8};                                    /// Bits per dimension
    float trunc_rate_{0.05F};                        /// Truncation rate for outlier handling
    uint64_t max_sample_count_{MAX_DEFAULT_SAMPLE};  /// Maximum samples for training

    static constexpr uint64_t MAX_DEFAULT_SAMPLE{100000};  /// Default max sample count
};

}  // namespace vsag
