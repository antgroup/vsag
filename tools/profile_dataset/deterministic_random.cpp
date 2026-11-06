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

#include "deterministic_random.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace vsag::dataset_profile {

namespace {

uint64_t
splitmix64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

uint64_t
rotate_left(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

}  // namespace

DeterministicRng::DeterministicRng(uint64_t seed) {
    uint64_t state = seed;
    for (auto& word : state_) {
        word = splitmix64(state);
    }
}

uint64_t
DeterministicRng::Next() {
    const uint64_t result = rotate_left(state_[1] * 5U, 7U) * 9U;
    const uint64_t shifted = state_[1] << 17U;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= shifted;
    state_[3] = rotate_left(state_[3], 45U);
    return result;
}

uint64_t
DeterministicRng::Uniform(uint64_t bound) {
    if (bound == 0) {
        throw std::invalid_argument("DeterministicRng::Uniform requires a positive bound");
    }
    // Reject the 2^64 mod bound smallest values so that the accepted range is an exact
    // multiple of bound.
    const uint64_t threshold = (0 - bound) % bound;
    uint64_t value = Next();
    while (value < threshold) {
        value = Next();
    }
    return value % bound;
}

double
DeterministicRng::UniformDouble() {
    constexpr double scale = 1.0 / static_cast<double>(1ULL << 53U);
    return static_cast<double>(Next() >> 11U) * scale;
}

uint64_t
derive_seed(uint64_t seed, const std::string& stream) {
    uint64_t hash = 0xCBF29CE484222325ULL;
    for (const char character : stream) {
        hash ^= static_cast<uint8_t>(character);
        hash *= 0x100000001B3ULL;
    }
    uint64_t state = seed ^ hash;
    return splitmix64(state);
}

std::vector<uint64_t>
sample_without_replacement(uint64_t population, uint64_t count, DeterministicRng& rng) {
    std::vector<uint64_t> result;
    if (count >= population) {
        result.resize(population);
        for (uint64_t i = 0; i < population; ++i) {
            result[i] = i;
        }
        return result;
    }
    std::unordered_set<uint64_t> chosen;
    chosen.reserve(count * 2);
    for (uint64_t j = population - count; j < population; ++j) {
        const uint64_t candidate = rng.Uniform(j + 1);
        if (!chosen.insert(candidate).second) {
            chosen.insert(j);
        }
    }
    result.assign(chosen.begin(), chosen.end());
    std::sort(result.begin(), result.end());
    return result;
}

void
shuffle_in_place(std::vector<uint64_t>& values, DeterministicRng& rng) {
    for (uint64_t i = values.size(); i > 1; --i) {
        const uint64_t j = rng.Uniform(i);
        std::swap(values[i - 1], values[j]);
    }
}

}  // namespace vsag::dataset_profile
