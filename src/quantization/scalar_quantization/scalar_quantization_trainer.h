
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

enum SQTrainMode {
    CLASSIC = 1,
    K_MEANS = 2,
    TRUNC_BOUND = 3,
};

class ScalarQuantizationTrainer {
public:
    explicit ScalarQuantizationTrainer(int32_t dim, int bits = 8);

    void
    Train(const float* data,
          uint64_t count,
          float* upper_bound,
          float* lower_bound,
          bool need_normalize = false,
          SQTrainMode mode = SQTrainMode::TRUNC_BOUND);

    void
    TrainUniform(const float* data,
                 uint64_t count,
                 float& upper_bound,
                 float& lower_bound,
                 bool need_normalize = false,
                 SQTrainMode mode = SQTrainMode::TRUNC_BOUND);

    inline void
    SetSampleCount(uint64_t sample) {
        this->max_sample_count_ = sample;
    }

    inline void
    SetSQ4UniformTruncRate(float trunc_rate) {
        this->trunc_rate_ = trunc_rate;
    }

private:
    void
    classic_train(const float* data, uint64_t count, float* upper_bound, float* lower_bound) const;

    void
    trunc_bound_train(const float* data,
                      uint64_t count,
                      float* upper_bound,
                      float* lower_bound) const;

    uint64_t
    sample_train_data(const float* data,
                      uint64_t count,
                      std::vector<float>& sample_datas,
                      bool need_normalize = false) const;

private:
    int dim_{0};

    int bits_{8};

    float trunc_rate_{0.05F};

    uint64_t max_sample_count_{MAX_DEFAULT_SAMPLE};

    constexpr static uint64_t MAX_DEFAULT_SAMPLE{100000};
};

}  // namespace vsag
