
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

#include "simd/normalize.h"

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
