
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

#include "scalar_quantization_trainer.h"

#include <cstring>
#include <queue>
#include <random>
#include <stdexcept>

#include "simd/bf16_simd.h"
#include "simd/normalize.h"
#include "utils/half_input.h"

namespace vsag {

ScalarQuantizationTrainer::ScalarQuantizationTrainer(int32_t dim, int bits)
    : dim_(dim), bits_(bits) {
}

void
ScalarQuantizationTrainer::Train(const float* data,
                                 uint64_t count,
                                 float* upper_bound,
                                 float* lower_bound,
                                 bool need_normalize,
                                 SQTrainMode mode) {
    std::vector<float> sample_datas;
    auto sample_count = this->sample_train_data(data, count, sample_datas, need_normalize);
    if (mode == CLASSIC) {
        this->classic_train(sample_datas.data(), sample_count, upper_bound, lower_bound);
    } else if (mode == TRUNC_BOUND) {
        this->trunc_bound_train(sample_datas.data(), sample_count, upper_bound, lower_bound);
    }
}

void
ScalarQuantizationTrainer::TrainHalf(const uint16_t* data,
                                     uint64_t count,
                                     bool bf16,
                                     float* upper,
                                     float* lower,
                                     bool normalize,
                                     SQTrainMode mode) {
    if (dim_ <= 0 || count > static_cast<uint64_t>(PTRDIFF_MAX) / sizeof(uint16_t) / dim_) {
        throw std::invalid_argument("invalid half scalar training extent");
    }
    const auto row_bytes = static_cast<uint64_t>(dim_) * sizeof(uint16_t);
    const TrainingRows input(data,
                             count,
                             dim_,
                             bf16 ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16,
                             row_bytes,
                             count * row_bytes);
    TrainRows(input, upper, lower, normalize, mode);
}

void
ScalarQuantizationTrainer::TrainRows(
    const TrainingRows& input, float* upper, float* lower, bool normalize, SQTrainMode mode) {
    if (upper == nullptr || lower == nullptr || dim_ <= 0 || input.Dim() != dim_ ||
        max_sample_count_ == 0) {
        throw std::invalid_argument("invalid scalar row training input");
    }
    // Preserve existing FP32 entry point and sampling/normalization implementation.
    if (const auto* contiguous = input.ContiguousFP32()) {
        Train(contiguous, input.Count(), upper, lower, normalize, mode);
        return;
    }
    const uint64_t count = input.Count();
    const uint64_t samples = std::min(count, max_sample_count_);
    const uint64_t step = count <= max_sample_count_ ? 1 : 2147483647UL % count;
    const uint64_t width = std::min<uint64_t>(32, dim_);
    if (samples > static_cast<uint64_t>(PTRDIFF_MAX) / sizeof(float) / width ||
        (step != 0 && samples > UINT64_MAX / step)) {
        throw std::overflow_error("row training extent overflow");
    }
    if (mode != CLASSIC && mode != TRUNC_BOUND) {
        throw std::invalid_argument("unsupported row scalar training mode");
    }
    std::vector<float> tile(samples * width);
    // A full row only when normalization or a provider lacking ReadColumns requires it.
    std::vector<float> row((normalize || !input.HasColumnReader()) ? dim_ : width);
    std::vector<float> normalized(normalize ? dim_ : 0);
    std::vector<float> norms(normalize ? samples : 0);
    if (normalize) {
        for (uint64_t j = 0; j < samples; ++j) {
            input.Read((j * step) % count, row.data());
            norms[j] = Normalize(row.data(), normalized.data(), dim_);
        }
    }
    for (uint64_t begin = 0; begin < static_cast<uint64_t>(dim_); begin += width) {
        const auto cols = std::min<uint64_t>(width, dim_ - begin);
        for (uint64_t j = 0; j < samples; ++j) {
            const auto id = (j * step) % count;
            const float* values;
            if (input.HasColumnReader()) {
                input.ReadColumns(id, begin, cols, row.data());
                values = row.data();
            } else {
                input.Read(id, row.data());
                values = row.data() + begin;
            }
            if (normalize) {
                DivScalar(values, tile.data() + j * cols, cols, norms[j]);
            } else {
                std::memcpy(tile.data() + j * cols, values, cols * sizeof(float));
            }
        }
        ScalarQuantizationTrainer block(static_cast<int32_t>(cols), bits_);
        block.trunc_rate_ = trunc_rate_;
        if (mode == CLASSIC) {
            block.classic_train(tile.data(), samples, upper + begin, lower + begin);
        } else {
            block.trunc_bound_train(tile.data(), samples, upper + begin, lower + begin);
        }
    }
}

void
ScalarQuantizationTrainer::TrainUniformRows(
    const TrainingRows& input, float& upper, float& lower, bool normalize, SQTrainMode mode) {
    if (dim_ <= 0 || input.Dim() != dim_) {
        throw std::invalid_argument("uniform row dimension mismatch");
    }
    if (const auto* contiguous = input.ContiguousFP32()) {
        TrainUniform(contiguous, input.Count(), upper, lower, normalize, mode);
        return;
    }
    std::vector<float> uppers(dim_);
    std::vector<float> lowers(dim_);
    TrainRows(input, uppers.data(), lowers.data(), normalize, mode);
    if (mode == CLASSIC) {
        upper = *std::max_element(uppers.begin(), uppers.end());
        lower = *std::min_element(lowers.begin(), lowers.end());
    } else {
        upper = *std::min_element(uppers.begin(), uppers.end());
        lower = *std::max_element(lowers.begin(), lowers.end());
        if (lower > upper) {
            std::swap(lower, upper);
        }
    }
}

void
ScalarQuantizationTrainer::TrainUniformHalf(const uint16_t* data,
                                            uint64_t count,
                                            bool bf16,
                                            float& upper,
                                            float& lower,
                                            bool normalize,
                                            SQTrainMode mode) {
    if (dim_ <= 0 || count > static_cast<uint64_t>(PTRDIFF_MAX) / sizeof(uint16_t) / dim_) {
        throw std::invalid_argument("invalid uniform half training extent");
    }
    const auto bytes = static_cast<uint64_t>(dim_) * sizeof(uint16_t);
    const TrainingRows input(data,
                             count,
                             dim_,
                             bf16 ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16,
                             bytes,
                             count * bytes);
    TrainUniformRows(input, upper, lower, normalize, mode);
}

void
ScalarQuantizationTrainer::TrainUniform(const float* data,
                                        uint64_t count,
                                        float& upper_bound,
                                        float& lower_bound,
                                        bool need_normalize,
                                        SQTrainMode mode) {
    std::vector<float> sample_datas;
    auto sample_count = this->sample_train_data(data, count, sample_datas, need_normalize);
    std::vector<float> upper(dim_);
    std::vector<float> lower(dim_);
    if (mode == CLASSIC) {
        this->classic_train(sample_datas.data(), sample_count, upper.data(), lower.data());
        upper_bound = *std::max_element(upper.begin(), upper.end());
        lower_bound = *std::min_element(lower.begin(), lower.end());
    } else if (mode == TRUNC_BOUND) {
        this->trunc_bound_train(sample_datas.data(), sample_count, upper.data(), lower.data());
        upper_bound = *std::min_element(upper.begin(), upper.end());
        lower_bound = *std::max_element(lower.begin(), lower.end());
        if (lower_bound > upper_bound) {
            // case for count == 1 or trunc_rate > 0.5
            std::swap(lower_bound, upper_bound);
        }
    }
}

void
ScalarQuantizationTrainer::classic_train(const float* data,
                                         uint64_t count,
                                         float* upper_bound,
                                         float* lower_bound) const {
    for (uint64_t i = 0; i < dim_; ++i) {
        upper_bound[i] = std::numeric_limits<float>::lowest();
        lower_bound[i] = std::numeric_limits<float>::max();
        for (uint64_t j = 0; j < count; ++j) {
            auto value = data[j * dim_ + i];
            upper_bound[i] = std::max(upper_bound[i], value);
            lower_bound[i] = std::min(lower_bound[i], value);
        }
    }
}

void
ScalarQuantizationTrainer::trunc_bound_train(const float* data,
                                             uint64_t count,
                                             float* upper_bound,
                                             float* lower_bound) const {
    double ignore_rate = 0.001;
    if (this->bits_ == 4) {
        ignore_rate = this->trunc_rate_;
    }
    auto ignore_count = static_cast<uint64_t>(static_cast<double>(count - 1) * ignore_rate);
    ignore_count = ignore_count < 1 ? 1 : ignore_count;
    for (uint64_t i = 0; i < dim_; ++i) {
        std::priority_queue<float, std::vector<float>, std::greater<>> heap_max;
        std::priority_queue<float, std::vector<float>, std::less<>> heap_min;
        heap_max.emplace(std::numeric_limits<float>::lowest());
        heap_min.emplace(std::numeric_limits<float>::max());
        for (uint64_t j = 0; j < count; ++j) {
            auto value = data[j * dim_ + i];
            if (value > heap_max.top() || heap_max.size() < ignore_count) {
                heap_max.emplace(value);
            }
            if (heap_max.size() > ignore_count) {
                heap_max.pop();
            }
            if (value < heap_min.top() || heap_min.size() < ignore_count) {
                heap_min.emplace(value);
            }
            if (heap_min.size() > ignore_count) {
                heap_min.pop();
            }
        }
        upper_bound[i] = heap_max.top();
        lower_bound[i] = heap_min.top();
    }
}

uint64_t
ScalarQuantizationTrainer::sample_train_data(const float* data,
                                             uint64_t count,
                                             std::vector<float>& sample_datas,
                                             bool need_normalize) const {
    uint64_t step = 2147483647UL % count;
    auto sample_count = max_sample_count_;
    if (count <= max_sample_count_) {
        step = 1;
        sample_count = count;
    }

    sample_datas.resize(sample_count * dim_);
    for (uint64_t j = 0; j < sample_count; ++j) {
        auto new_index = (j * step) % count;
        if (need_normalize) {
            Normalize(data + new_index * dim_, sample_datas.data() + j * dim_, dim_);
        } else {
            memcpy(sample_datas.data() + j * dim_, data + new_index * dim_, dim_ * sizeof(float));
        }
    }
    return sample_count;
}
}  // namespace vsag
