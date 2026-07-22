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

#include <cmath>
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
    allocator->Deallocate(decoded.ids_);
    allocator->Deallocate(decoded.vals_);

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

}  // namespace vsag
