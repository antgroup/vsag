
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

#include "elias_fano_encoder.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <random>

#include "impl/allocator/safe_allocator.h"

namespace vsag {

TEST_CASE("EliasFanoEncoder, original seq equal to decoded seq", "[ut][EliasFanoEncoder]") {
    const size_t max_size = 250;
    const int max_id = 1000000;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<InnerIdType> dist(0, max_id);

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::shared_ptr<EliasFanoEncoder> encoder = std::make_shared<EliasFanoEncoder>(allocator.get());

    for (int size = 0; size <= max_size; size++) {
        Vector<InnerIdType> values(allocator.get());
        values.reserve(size);

        // generate an ordered seq with length ${size}
        for (size_t i = 0; i < size; i++) {
            values.push_back(dist(gen));
        }
        std::sort(values.begin(), values.end());

        encoder->Encode(values, max_id);
        REQUIRE(encoder->Size() == values.size());

        // check if original seq equal to decoded seq
        Vector<InnerIdType> decompressed(allocator.get());
        encoder->DecompressAll(decompressed);
        REQUIRE(decompressed.size() == values.size());
        for (size_t i = 0; i < values.size(); i++) {
            REQUIRE(decompressed[i] == values[i]);
        }
    }
}

}  // namespace vsag
