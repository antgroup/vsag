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

#include "sparse_index.h"

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("SparseIndex GetVectorByIds Returns Sparse Vectors", "[ut][SparseIndex]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;
    common_param.dim_ = 100;

    auto param = std::make_shared<SparseIndexParameters>();
    auto index = std::make_unique<SparseIndex>(param, common_param);

    constexpr uint32_t count = 4;
    std::vector<int64_t> ids = {10, 11, 12, 13};
    auto vectors = fixtures::GenerateSparseVectors(count, common_param.dim_);
    auto base = Dataset::Make();
    base->NumElements(count)->SparseVectors(vectors.data())->Ids(ids.data())->Owner(false);
    REQUIRE(index->Build(base).empty());

    int64_t selected[] = {ids[1], ids[3]};
    auto result = index->GetVectorByIds(selected, 2, nullptr);
    REQUIRE(result->GetSparseVectors() != nullptr);
    REQUIRE(result->GetFloat32Vectors() == nullptr);
    REQUIRE(result->GetNumElements() == 2);
    REQUIRE(result->GetSparseVectors()[0].len_ == vectors[1].len_);
    REQUIRE(result->GetSparseVectors()[1].len_ == vectors[3].len_);

    for (auto& item : vectors) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}
