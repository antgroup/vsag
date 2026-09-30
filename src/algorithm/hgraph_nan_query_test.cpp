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

#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <vector>

#include "hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "index/index_impl.h"

namespace {

vsag::DatasetPtr
MakeFloatDataset(std::vector<float>& vectors,
                 std::vector<int64_t>& ids,
                 int64_t dim,
                 int64_t count) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    return dataset;
}

vsag::IndexCommonParam
MakeCommonParam(int64_t dim) {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    return common_param;
}

}  // namespace

// Regression test: a non-finite query makes every distance NaN, so the route layer
// yields no eligible seed. The route loop must not dereference the resulting empty
// heap (previously SIGSEGV in SearchWithRequest, si_addr = 0x4).
TEST_CASE("HGraph search with a NaN query and finite FP32 base", "[ut][hgraph][hgraph_nan_query]") {
    constexpr int64_t dim = 16;
    constexpr int64_t count = 64;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1
    })");
    auto index = std::make_shared<vsag::IndexImpl<vsag::HGraph>>(hgraph_json, common_param);

    std::vector<float> vectors(count * dim);
    std::vector<int64_t> ids(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = i;
        for (int64_t d = 0; d < dim; ++d) {
            vectors[i * dim + d] = static_cast<float>(i) + static_cast<float>(d) / dim;
        }
    }
    REQUIRE(index->Build(MakeFloatDataset(vectors, ids, dim, count)).has_value());

    auto base_distance = index->CalDistanceById(vectors.data(), ids.data(), 1);
    REQUIRE(base_distance.has_value());
    REQUIRE(std::isfinite(base_distance.value()->GetDistances()[0]));

    std::vector<float> query_vector(vectors.begin(), vectors.begin() + dim);
    query_vector[0] = std::numeric_limits<float>::quiet_NaN();
    auto query = MakeFloatDataset(query_vector, ids, dim, 1);

    // The route layer finds no eligible seed here; the resulting empty heap must not
    // be dereferenced.
    auto result = index->KnnSearch(query, 1, R"({"hgraph":{"ef_search":32}})");
    REQUIRE(result.has_value());
    // No finite distance exists, so the result set is empty; the point is that it is
    // returned instead of crashing.
    CHECK(result.value()->GetDim() == 0);
}
