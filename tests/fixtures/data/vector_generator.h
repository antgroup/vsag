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

/**
 * @file vector_generator.h
 * @brief Utilities for generating test vectors and datasets.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "simd/normalize.h"
#include "typing.h"
#include "vsag/vsag.h"

namespace fixtures {

template <typename T, typename RT = typename std::enable_if<std::is_integral_v<T>, T>::type>
std::vector<RT>
GenerateVectors(uint64_t count,
                uint32_t dim,
                int seed = 47,
                T min = std::numeric_limits<T>::lowest(),
                T max = std::numeric_limits<T>::max()) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<T> distrib_real(min, max);
    std::vector<T> vectors(dim * count);
    for (int64_t i = 0; i < dim * count; ++i) {
        vectors[i] = distrib_real(rng);
    }
    return vectors;
}

template <typename T, typename RT = typename std::enable_if<std::is_floating_point_v<T>, T>::type>
std::vector<RT>
GenerateVectors(uint64_t count, uint32_t dim, int seed = 47, bool need_normalize = true) {
    using EngineType = std::conditional_t<(sizeof(T) > 4), std::mt19937_64, std::mt19937>;
    EngineType rng(seed);
    std::uniform_real_distribution<T> distrib_real(0.1, 0.9);
    std::vector<T> vectors(dim * count);
    for (int64_t i = 0; i < dim * count; ++i) {
        vectors[i] = distrib_real(rng);
    }
    if (need_normalize) {
        for (int64_t i = 0; i < count; ++i) {
            vsag::Normalize(vectors.data() + i * dim, vectors.data() + i * dim, dim);
        }
    }
    return vectors;
}

std::vector<vsag::SparseVector>
GenerateSparseVectors(uint32_t count,
                      uint32_t max_dim = 100,
                      uint32_t max_id = 1000,
                      float min_val = -1,
                      float max_val = 1,
                      int seed = 47);

vsag::Vector<vsag::SparseVector>
GenerateSparseVectors(vsag::Allocator* allocator,
                      uint32_t count,
                      uint32_t max_dim = 100,
                      uint32_t max_id = 1000,
                      float min_val = -1,
                      float max_val = 1,
                      int seed = 47);

std::pair<std::vector<float>, std::vector<uint8_t>>
GenerateBinaryVectorsAndCodes(uint32_t count, uint32_t dim, int seed = 47);

std::vector<float>
generate_vectors(uint64_t count, uint32_t dim, bool need_normalize = true, int seed = 47);

std::vector<float>
generate_normal_vectors(uint64_t count,
                        uint32_t dim,
                        bool need_normalize = true,
                        int seed = 47,
                        float mean = 0.0f,
                        float stddev = 1.0f);

std::vector<uint8_t>
generate_int4_codes(uint64_t count, uint32_t dim, int seed = 47);

std::vector<int8_t>
generate_int8_codes(uint64_t count, uint32_t dim, int seed = 47);

std::vector<uint8_t>
generate_uint8_codes(uint64_t count, uint32_t dim, int seed = 47);

// Generate FP16/BF16 codes from float values in [0.1, 0.9] range
std::vector<uint16_t>
generate_fp16_codes(uint64_t count, uint32_t dim, int seed = 47);

std::vector<uint16_t>
generate_bf16_codes(uint64_t count, uint32_t dim, int seed = 47);

std::pair<std::vector<uint16_t>, std::vector<float>>
generate_fp16_codes_with_floats(uint64_t count, uint32_t dim, int seed = 47);

std::pair<std::vector<uint16_t>, std::vector<float>>
generate_bf16_codes_with_floats(uint64_t count, uint32_t dim, int seed = 47);

std::tuple<std::vector<int64_t>, std::vector<float>>
generate_ids_and_vectors(int64_t num_elements,
                         int64_t dim,
                         bool need_normalize = true,
                         int seed = 47);

std::vector<char>
generate_extra_infos(uint64_t count, uint32_t size, int seed = 47);

vsag::AttributeSet*
generate_attributes(uint64_t count,
                    uint32_t max_term_count = 10,
                    uint32_t max_value_count = 10,
                    int seed = 97);

vsag::IndexPtr
generate_index(const std::string& name,
               const std::string& metric_type,
               int64_t num_vectors,
               int64_t dim,
               std::vector<int64_t>& ids,
               std::vector<float>& vectors,
               bool use_conjugate_graph = false);

std::string
generate_hnsw_build_parameters_string(const std::string& metric_type, int64_t dim);

vsag::DatasetPtr
generate_one_dataset(int64_t dim, uint64_t count);

float
GetSparseDistance(const vsag::SparseVector& vec1, const vsag::SparseVector& vec2);

}  // namespace fixtures
