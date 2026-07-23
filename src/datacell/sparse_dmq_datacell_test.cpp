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

#include "sparse_dmq_datacell.h"

#include <cmath>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"

namespace vsag {
namespace {

float
sparse_distance(const SparseVector& left, const SparseVector& right) {
    float inner_product = 0.0F;
    for (uint32_t left_index = 0; left_index < left.len_; ++left_index) {
        for (uint32_t right_index = 0; right_index < right.len_; ++right_index) {
            if (left.ids_[left_index] == right.ids_[right_index]) {
                inner_product += left.vals_[left_index] * right.vals_[right_index];
            }
        }
    }
    return 1.0F - inner_product;
}

}  // namespace

TEST_CASE("SparseDmqDataCell implements FlattenInterface", "[ut][SparseDmqDataCell]") {
    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    common_param.metric_ = MetricType::METRIC_TYPE_IP;
    common_param.dim_ = 1024;

    std::vector<uint32_t> ids_0{9, 2, 5};
    std::vector<float> vals_0{0.8F, 0.2F, 0.5F};
    std::vector<uint32_t> ids_1{7, 2, 9};
    std::vector<float> vals_1{0.3F, 0.6F, 0.4F};
    std::vector<uint32_t> ids_2{5, 7, 2};
    std::vector<float> vals_2{0.9F, 0.1F, 0.7F};
    std::vector<SparseVector> vectors{
        {static_cast<uint32_t>(ids_0.size()), ids_0.data(), vals_0.data()},
        {static_cast<uint32_t>(ids_1.size()), ids_1.data(), vals_1.data()},
        {static_cast<uint32_t>(ids_2.size()), ids_2.data(), vals_2.data()},
    };

    auto cell = std::make_shared<SparseDmqDataCell>(1024, common_param);
    cell->BatchInsertVector(vectors.data(), vectors.size());
    REQUIRE(cell->TotalCount() == vectors.size());
    REQUIRE_THROWS_AS(cell->BatchInsertVector(vectors.data(), vectors.size()), VsagException);
    REQUIRE(cell->TotalCount() == vectors.size());
    REQUIRE(cell->GetQuantizerName() == "dmq8");
    REQUIRE(cell->GetMetricType() == MetricType::METRIC_TYPE_IP);

    std::vector<uint32_t> query_ids{9, 7, 2};
    std::vector<float> query_vals{0.25F, 0.5F, 0.75F};
    SparseVector query{
        static_cast<uint32_t>(query_ids.size()), query_ids.data(), query_vals.data()};
    auto computer = cell->FactoryComputer(&query);

    std::vector<InnerIdType> inner_ids{0, 1, 2};
    std::vector<float> distances(inner_ids.size());
    cell->Query(distances.data(), computer, inner_ids.data(), inner_ids.size());

    for (uint64_t index = 0; index < vectors.size(); ++index) {
        SparseVector decoded;
        cell->GetSparseVectorByInnerId(index, &decoded, common_param.allocator_.get());
        REQUIRE(std::abs(distances[index] - sparse_distance(query, decoded)) <= 1e-6F);
        common_param.allocator_->Deallocate(decoded.ids_);
        common_param.allocator_->Deallocate(decoded.vals_);
    }

    SECTION("serialize and deserialize through the datacell interface") {
        auto restored = std::make_shared<SparseDmqDataCell>(1024, common_param);
        test_serializion(*cell, *restored);
        auto restored_computer = restored->FactoryComputer(&query);
        std::vector<float> restored_distances(inner_ids.size());
        restored->Query(
            restored_distances.data(), restored_computer, inner_ids.data(), inner_ids.size());
        REQUIRE(restored_distances == distances);
    }
}

}  // namespace vsag
