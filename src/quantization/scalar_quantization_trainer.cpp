
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

#include <random>
#include <unordered_set>

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
                                 ScalarQuantizationTrainer::SQTrainMode mode) {
    std::vector<float> sample_datas;
    auto sample_count = this->sample_train_data(data, count, sample_datas, need_normalize);
    if (mode == CLASSIC) {
        this->classic_train(sample_datas.data(), sample_count, upper_bound, lower_bound);
    }
}

void
ScalarQuantizationTrainer::TrainUniform(const float* data,
                                        uint64_t count,
                                        float& upper_bound,
                                        float& lower_bound,
                                        bool need_normalize,
                                        ScalarQuantizationTrainer::SQTrainMode mode) {
    std::vector<float> sample_datas;
    auto sample_count = this->sample_train_data(data, count, sample_datas, need_normalize);
    std::vector<float> upper(dim_);
    std::vector<float> lower(dim_);
    if (mode == CLASSIC) {
        this->classic_train(sample_datas.data(), sample_count, upper.data(), lower.data());
        upper_bound = *std::max_element(upper.begin(), upper.end());
        lower_bound = *std::min_element(lower.begin(), lower.end());
    }
}

void
ScalarQuantizationTrainer::classic_train(const float* data,
                                         uint64_t count,
                                         float* upper_bound,
                                         float* lower_bound) {
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

uint64_t
ScalarQuantizationTrainer::sample_train_data(const float* data,
                                             uint64_t count,
                                             std::vector<float>& sample_datas,
                                             bool need_normalize) {
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
