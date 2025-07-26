
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

#include <algorithm>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "vsag/vsag.h"

using namespace vsag;

float
compute_mse(const std::vector<float>& data, float lower, float upper, int bits) {
    float div = (1 << bits) - 1;
    float step = (upper - lower) / div;
    float mse = 0.0f;
    for (float v : data) {
        float code = std::round((v - lower) / step);
        code = std::min(std::max(code, 0.0f), div);
        float recon = lower + code * step;
        mse += (v - recon) * (v - recon);
    }
    return mse / data.size();
}

TEST_CASE("ScalarQuantizationTrainer", "[ft][scalar_quantization_trainer]") {
    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < 1000; ++i) {
        data.push_back(dist(gen));
    }
    int bits = 4;
    float lower_c, upper_c, lower_t, upper_t, lower_p, upper_p;

    vsag::ScalarQuantizationTrainer trainer(1, bits);

    // CLASSIC
    trainer.Train(data.data(), data.size(), &lower_c, &upper_c, vsag::SQTrainMode::CLASSIC);
    float mse_classic = compute_mse(data, lower_c, upper_c, bits);

    // TRUNC_BOUND
    trainer.Train(data.data(), data.size(), &lower_t, &upper_t, vsag::SQTrainMode::TRUNC_BOUND);
    float mse_trunc = compute_mse(data, lower_t, upper_t, bits);

    // PSO
    trainer.Train(data.data(), data.size(), &lower_p, &upper_p, vsag::SQTrainMode::PSO);
    float mse_pso = compute_mse(data, lower_p, upper_p, bits);

    REQUIRE(lower_c <= upper_c);
    REQUIRE(lower_t <= upper_t);
    REQUIRE(lower_p <= upper_p);

    REQUIRE(mse_pso < mse_classic * 0.95);
    REQUIRE(mse_pso < mse_trunc);
}
