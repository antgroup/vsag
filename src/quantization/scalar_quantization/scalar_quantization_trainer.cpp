
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
    } else if (mode == PSO) {
        this->pso_train(sample_datas.data(), sample_count, upper_bound, lower_bound);
    }
}

void
ScalarQuantizationTrainer::pso_train(const float* data,
                                     uint64_t count,
                                     float* upper_bound,
                                     float* lower_bound) const {
    return pso_train_impl(data, count, upper_bound, lower_bound);
}

void
ScalarQuantizationTrainer::pso_train_impl(const float* data,
                                          uint64_t count,
                                          float* upper_bound,
                                          float* lower_bound) const {
    this->classic_train(data, count, upper_bound, lower_bound);
    float div = (1 << this->bits_) - 1;

#pragma omp parallel for
    for (uint64_t i = 0; i < dim_; ++i) {
        float init_upper_bound = upper_bound[i];
        float init_lower_bound = lower_bound[i];
        const float init_range_width = init_upper_bound - init_lower_bound;
        const float init_range_center = (init_lower_bound + init_upper_bound) * 0.5f;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> v_dis(-init_range_width * 0.1f,
                                                    init_range_width * 0.1f);
        std::uniform_real_distribution<float> p_dis(0.0f, 1.0f);

        constexpr size_t max_iter = 128;
        constexpr size_t grid_side_length = 8;
        constexpr float grid_scale_factor = 0.1f;
        constexpr float init_inertia = 0.9f;
        constexpr float final_inertia = 0.4f;
        constexpr float c1 = 1.8f;
        constexpr float c2 = 1.8f;

        auto loss = [=](float center, float width) {
            float step_size = width / div;
            step_size = std::max(step_size, 1e-8f);
            float total_loss = 0.0f;
            for (uint64_t j = 0; j < count; ++j) {
                float value = data[j * dim_ + i];
                float quantized_code = std::round((value - center) / step_size);
                quantized_code = std::min(quantized_code, div);
                quantized_code = std::max(quantized_code, 0.0f);
                float error = (value - (center + quantized_code * step_size)) *
                              (value - (center + quantized_code * step_size));
                total_loss += error;
            }
            return total_loss * step_size * step_size;
        };

        struct Particle {
            float center;
            float width;
            float v_center;
            float v_width;
            float best_center;
            float best_width;
            float min_loss;

            Particle(const float c_val, const float w_val, const float vc_val, const float vw_val)
                : center(c_val),
                  width(w_val),
                  v_center(vc_val),
                  v_width(vw_val),
                  best_center(c_val),
                  best_width(w_val),
                  min_loss(std::numeric_limits<float>::max()) {
            }
        };

        std::vector<Particle> swarm;
        swarm.reserve(grid_side_length * grid_side_length);
        for (size_t m = 0; m < grid_side_length; ++m) {
            for (size_t n = 0; n < grid_side_length; ++n) {
                float lower = init_lower_bound - grid_scale_factor * init_range_width +
                              static_cast<float>(m) * 2 * grid_scale_factor * init_range_width /
                                  grid_side_length;
                float upper = init_upper_bound - grid_scale_factor * init_range_width +
                              static_cast<float>(n) * 2 * grid_scale_factor * init_range_width /
                                  grid_side_length;
                float particle_center = (lower + upper) * 0.5f;
                float particle_width = upper - lower;
                swarm.emplace_back(particle_center, particle_width, v_dis(gen), v_dis(gen));
            }
        }

        float global_best_center = init_range_center;
        float global_best_width = init_range_width;
        float global_min_loss = loss(init_range_center, init_range_width);
        for (auto& particle : swarm) {
            float curr_lower = particle.center - particle.width * 0.5f;
            float curr_loss = loss(particle.center, particle.width);
            particle.min_loss = curr_loss;
            if (curr_loss < global_min_loss) {
                global_min_loss = curr_loss;
                global_best_center = particle.center;
                global_best_width = particle.width;
            }
        }

        for (size_t iter = 0; iter < max_iter; ++iter) {
            float inertia =
                init_inertia - (init_inertia - final_inertia) * static_cast<float>(iter) / max_iter;
            for (auto& particle : swarm) {
                float r1 = p_dis(gen);
                float r2 = p_dis(gen);
                particle.v_center = inertia * particle.v_center +
                                    c1 * r1 * (particle.best_center - particle.center) +
                                    c2 * r2 * (global_best_center - particle.center);
                particle.v_width = inertia * particle.v_width +
                                   c1 * r1 * (particle.best_width - particle.width) +
                                   c2 * r2 * (global_best_width - particle.width);
                particle.center += particle.v_center;
                particle.width += particle.v_width;
                if (particle.width <= std::numeric_limits<float>::epsilon()) {
                    particle.width = std::numeric_limits<float>::epsilon();
                }
                float curr_loss = loss(particle.center, particle.width);
                if (curr_loss < particle.min_loss) {
                    particle.min_loss = curr_loss;
                    particle.best_center = particle.center;
                    particle.best_width = particle.width;
                }
                if (curr_loss < global_min_loss) {
                    global_min_loss = curr_loss;
                    global_best_center = particle.center;
                    global_best_width = particle.width;
                }
            }
        }
        lower_bound[i] = global_best_center - global_best_width * 0.5f;
        upper_bound[i] = global_best_center + global_best_width * 0.5f;
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
    } else if (mode == PSO) {
        this->pso_train(sample_datas.data(), sample_count, upper.data(), lower.data());
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
