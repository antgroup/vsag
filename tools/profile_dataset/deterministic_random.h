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
#include <string>
#include <vector>

namespace vsag::dataset_profile {

/// xoshiro256** seeded through SplitMix64. Unlike std::uniform_int_distribution, every
/// operation here is fully specified, so a seed yields the same samples with any compiler,
/// standard library or platform.
class DeterministicRng {
public:
    explicit DeterministicRng(uint64_t seed);

    uint64_t
    Next();

    /// Unbiased integer in [0, bound). Throws std::invalid_argument when bound is 0.
    uint64_t
    Uniform(uint64_t bound);

    /// Double in [0, 1) built from the top 53 bits of Next().
    double
    UniformDouble();

private:
    uint64_t state_[4]{};
};

/// Derives an independent seed for one named purpose ("sample/train", "pairs/train", ...),
/// so changing one sample size never changes the rows drawn for another analysis.
uint64_t
derive_seed(uint64_t seed, const std::string& stream);

/// Floyd's algorithm: `count` distinct indices drawn uniformly from [0, population),
/// returned in ascending order. Returns every index when count >= population.
std::vector<uint64_t>
sample_without_replacement(uint64_t population, uint64_t count, DeterministicRng& rng);

/// Fisher-Yates shuffle driven by DeterministicRng.
void
shuffle_in_place(std::vector<uint64_t>& values, DeterministicRng& rng);

}  // namespace vsag::dataset_profile
