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

#include "sparse_dmq_quantizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"

namespace vsag {

TEST_CASE("SparseDmqQuantizer basic operations", "[ut][SparseDmqQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::vector<uint32_t> ids_0{1, 4, 9};
    std::vector<float> values_0{0.0F, 0.5F, 1.0F};
    std::vector<uint32_t> ids_1{1, 4, 9};
    std::vector<float> values_1{1.0F, 0.5F, 0.0F};
    std::vector<SparseVector> vectors{
        {3, ids_0.data(), values_0.data()},
        {3, ids_1.data(), values_1.data()},
    };
    SparseDmqQuantizer quantizer(16, allocator.get());
    REQUIRE(quantizer.TrainImpl(reinterpret_cast<const float*>(vectors.data()), vectors.size()));
    REQUIRE(quantizer.Name() == "dmq8");
    REQUIRE(quantizer.Metric() == MetricType::METRIC_TYPE_IP);

    std::vector<uint8_t> codes(quantizer.GetEncodedSize(vectors[0]));
    REQUIRE(quantizer.EncodeOne(reinterpret_cast<const float*>(&vectors[0]), codes.data()));
    auto computer = quantizer.FactoryComputer();
    computer->SetQuery(reinterpret_cast<const float*>(&vectors[0]));
    float distance = 0.0F;
    computer->ComputeDist(codes.data(), &distance);

    SparseVector decoded;
    REQUIRE(quantizer.DecodeOne(codes.data(), reinterpret_cast<float*>(&decoded)));
    float expected_distance = 1.0F;
    for (uint32_t index = 0; index < decoded.len_; ++index) {
        expected_distance -= decoded.vals_[index] * values_0[index];
    }
    REQUIRE(std::abs(distance - expected_distance) <= 1e-6F);

    std::vector<uint8_t> codes_1(quantizer.GetEncodedSize(vectors[1]));
    REQUIRE(quantizer.EncodeOne(reinterpret_cast<const float*>(&vectors[1]), codes_1.data()));
    SparseVector decoded_1;
    REQUIRE(quantizer.DecodeOne(codes_1.data(), reinterpret_cast<float*>(&decoded_1)));
    float expected_pair_distance = 1.0F;
    for (uint32_t index = 0; index < decoded.len_; ++index) {
        expected_pair_distance -= decoded.vals_[index] * decoded_1.vals_[index];
    }
    REQUIRE(std::abs(quantizer.Compute(codes.data(), codes_1.data()) - expected_pair_distance) <=
            1e-6F);

    std::vector<uint32_t> query_ids{1, 4, 9, std::numeric_limits<uint32_t>::max()};
    std::vector<float> query_values{0.0F, 0.5F, 1.0F, 100.0F};
    SparseVector query{
        static_cast<uint32_t>(query_ids.size()), query_ids.data(), query_values.data()};
    auto unknown_term_computer = quantizer.FactoryComputer();
    unknown_term_computer->SetQuery(reinterpret_cast<const float*>(&query));
    float distance_with_unknown_term = 0.0F;
    unknown_term_computer->ComputeDist(codes.data(), &distance_with_unknown_term);
    REQUIRE(distance_with_unknown_term == distance);

    allocator->Deallocate(decoded.ids_);
    allocator->Deallocate(decoded.vals_);
    allocator->Deallocate(decoded_1.ids_);
    allocator->Deallocate(decoded_1.vals_);

    SparseDmqQuantizer restored(16, allocator.get());
    test_serializion(quantizer, restored);
    auto restored_computer = restored.FactoryComputer();
    restored_computer->SetQuery(reinterpret_cast<const float*>(&vectors[0]));
    float restored_distance = 0.0F;
    restored_computer->ComputeDist(codes.data(), &restored_distance);
    REQUIRE(restored_distance == distance);
}

TEST_CASE("SparseDmqQuantizer encodes inclusive term id limit", "[ut][SparseDmqQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::vector<uint32_t> ids{16};
    std::vector<float> values{1.0F};
    SparseVector vector{1, ids.data(), values.data()};
    SparseDmqQuantizer quantizer(16, allocator.get());

    REQUIRE(quantizer.TrainImpl(reinterpret_cast<const float*>(&vector), 1));
    std::vector<uint8_t> codes(quantizer.GetEncodedSize(vector));
    REQUIRE(quantizer.EncodeOne(reinterpret_cast<const float*>(&vector), codes.data()));

    SparseVector decoded;
    REQUIRE(quantizer.DecodeOne(codes.data(), reinterpret_cast<float*>(&decoded)));
    REQUIRE(decoded.len_ == 1);
    REQUIRE(decoded.ids_[0] == 16);
    allocator->Deallocate(decoded.ids_);
    allocator->Deallocate(decoded.vals_);

    auto computer = quantizer.FactoryComputer();
    computer->SetQuery(reinterpret_cast<const float*>(&vector));
    float distance = 1.0F;
    computer->ComputeDist(codes.data(), &distance);
    REQUIRE(std::abs(distance) <= 1e-6F);
}

TEST_CASE("SparseDmqQuantizer packs compact term ids by distinct term count",
          "[ut][SparseDmqQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const std::vector<std::pair<uint32_t, uint32_t>> cases{
        {1, 1}, {2, 1}, {8, 3}, {9, 4}, {16, 4}, {17, 5}};

    for (const auto& [term_count, expected_bits] : cases) {
        std::vector<uint32_t> ids(term_count);
        std::vector<float> values(term_count);
        for (uint32_t index = 0; index < term_count; ++index) {
            ids[index] = std::numeric_limits<uint32_t>::max() - index * 100;
            values[index] = static_cast<float>(index + 1) / term_count;
        }
        SparseVector vector{term_count, ids.data(), values.data()};
        SparseDmqQuantizer quantizer(std::numeric_limits<uint32_t>::max(), allocator.get());

        REQUIRE(quantizer.TrainImpl(reinterpret_cast<const float*>(&vector), 1));
        REQUIRE(quantizer.GetIdBits() == expected_bits);
        REQUIRE(quantizer.GetEncodedSize(vector) ==
                sizeof(SparseDmqQuantizer::EncodedHeader) +
                    (static_cast<uint64_t>(term_count) * expected_bits + 7) / 8 + term_count);

        std::vector<uint8_t> codes(quantizer.GetEncodedSize(vector));
        REQUIRE(quantizer.EncodeOne(reinterpret_cast<const float*>(&vector), codes.data()));
        SparseVector decoded;
        REQUIRE(quantizer.DecodeOne(codes.data(), reinterpret_cast<float*>(&decoded)));
        auto sorted_ids = ids;
        std::sort(sorted_ids.begin(), sorted_ids.end());
        REQUIRE(std::equal(sorted_ids.begin(), sorted_ids.end(), decoded.ids_));
        allocator->Deallocate(decoded.ids_);
        allocator->Deallocate(decoded.vals_);
    }
}

TEST_CASE("SparseDmqQuantizer shares low frequency codebooks", "[ut][SparseDmqQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::vector<uint32_t> ids_0{1, 2, 3, 5};
    std::vector<float> values_0{0.1F, 0.2F, 0.3F, 0.4F};
    std::vector<uint32_t> ids_1{1, 2, 3, 5};
    std::vector<float> values_1{0.5F, 0.6F, 0.7F, 0.8F};
    std::vector<uint32_t> ids_2{1, 4, 5};
    std::vector<float> values_2{0.9F, 1.0F, 1.1F};
    std::vector<SparseVector> vectors{
        {4, ids_0.data(), values_0.data()},
        {4, ids_1.data(), values_1.data()},
        {3, ids_2.data(), values_2.data()},
    };

    SparseDmqQuantizer shared(10, allocator.get(), 2);
    REQUIRE(shared.TrainImpl(reinterpret_cast<const float*>(vectors.data()), vectors.size()));
    REQUIRE(shared.GetIdBits() == 3);
    REQUIRE(shared.GetCodebookCount() == 3);

    SparseDmqQuantizer disabled(10, allocator.get(), 0);
    REQUIRE(disabled.TrainImpl(reinterpret_cast<const float*>(vectors.data()), vectors.size()));
    REQUIRE(disabled.GetIdBits() == 3);
    REQUIRE(disabled.GetCodebookCount() == 5);

    SparseDmqQuantizer defaults(10, allocator.get());
    REQUIRE(defaults.TrainImpl(reinterpret_cast<const float*>(vectors.data()), vectors.size()));
    REQUIRE(defaults.GetCodebookCount() == 1);

    SparseDmqQuantizer exported(10, allocator.get(), 0);
    exported.ExportModel(shared);
    REQUIRE(exported.GetCodebookCount() == 3);
    REQUIRE(exported.GetSharedCodebookThreshold() == 2);

    SparseDmqQuantizer restored(10, allocator.get(), 2);
    test_serializion(shared, restored);
    REQUIRE(restored.GetCodebookCount() == 3);
    std::vector<uint8_t> codes(shared.GetEncodedSize(vectors[0]));
    REQUIRE(shared.EncodeOne(reinterpret_cast<const float*>(&vectors[0]), codes.data()));
    auto expected_computer = shared.FactoryComputer();
    expected_computer->SetQuery(reinterpret_cast<const float*>(&vectors[2]));
    auto restored_computer = restored.FactoryComputer();
    restored_computer->SetQuery(reinterpret_cast<const float*>(&vectors[2]));
    float expected_distance = 0.0F;
    float restored_distance = 0.0F;
    expected_computer->ComputeDist(codes.data(), &expected_distance);
    restored_computer->ComputeDist(codes.data(), &restored_distance);
    REQUIRE(restored_distance == expected_distance);
}

}  // namespace vsag
