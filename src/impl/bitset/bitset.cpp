
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

#include "vsag/bitset.h"

#include <cstdint>
#include <functional>
#include <random>

#include "computable_bitset.h"

namespace vsag {

BitsetPtr
Bitset::Random(int64_t length) {
    auto bitset = ComputableBitset::MakeInstance(ComputableBitsetType::SparseBitset);
    static auto gen =
        std::bind(std::uniform_int_distribution<>(0, 1),  // NOLINT(modernize-avoid-bind)
                  std::default_random_engine());
    for (int64_t i = 0; i < length; ++i) {
        bitset->Set(i, gen() != 0);
    }
    return bitset;
}

BitsetPtr
Bitset::Make() {
    return ComputableBitset::MakeInstance(ComputableBitsetType::SparseBitset);
}
}  // namespace vsag
