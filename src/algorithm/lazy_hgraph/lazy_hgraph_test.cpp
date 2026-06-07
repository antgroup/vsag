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

#include "lazy_hgraph.h"

#include <memory>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "unittest.h"
#include "vsag/factory.h"

namespace {

constexpr int64_t DIM = 8;

vsag::IndexCommonParam
MakeCommonParam() {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = DIM;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    return common_param;
}

vsag::JsonType
MakeLazyParam(uint64_t threshold) {
    auto param = vsag::JsonType::Parse(R"({
        "transition_threshold": 4,
        "hgraph": {
            "base_quantization_type": "fp32",
            "max_degree": 4,
            "ef_construction": 8,
            "build_thread_count": 1
        }
    })");
    param["transition_threshold"].SetInt(threshold);
    return param;
}

vsag::DatasetPtr
MakeDataset(int64_t count,
            int64_t first_id,
            std::vector<float>& vectors,
            std::vector<int64_t>& ids) {
    vectors.resize(count * DIM);
    ids.resize(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = first_id + i;
        for (int64_t j = 0; j < DIM; ++j) {
            vectors[i * DIM + j] = static_cast<float>((first_id + i) * 0.1 + j);
        }
    }
    return vsag::Dataset::Make()
        ->NumElements(count)
        ->Dim(DIM)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
}

vsag::DatasetPtr
MakeQuery(const std::vector<float>& vectors, int64_t row) {
    return vsag::Dataset::Make()
        ->NumElements(1)
        ->Dim(DIM)
        ->Float32Vectors(vectors.data() + row * DIM)
        ->Owner(false);
}

std::shared_ptr<vsag::LazyHGraph>
MakeLazyIndex(uint64_t threshold) {
    auto common_param = MakeCommonParam();
    auto param =
        vsag::LazyHGraph::CheckAndMappingExternalParam(MakeLazyParam(threshold), common_param);
    auto index = std::make_shared<vsag::LazyHGraph>(param, common_param);
    index->InitFeatures();
    return index;
}

std::string
MakeFactoryParam(uint64_t threshold) {
    auto root = vsag::JsonType::Parse(R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 8,
        "lazy_hgraph": {
            "transition_threshold": 4,
            "hgraph": {
                "base_quantization_type": "fp32",
                "max_degree": 4,
                "ef_construction": 8,
                "build_thread_count": 1
            }
        }
    })");
    root["lazy_hgraph"]["transition_threshold"].SetInt(threshold);
    return root.Dump();
}

}  // namespace

TEST_CASE("LazyHGraph stays flat before threshold and searches exactly", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(4);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(3, 100, vectors, ids);

    auto failed_ids = index->Add(data);
    REQUIRE(failed_ids.empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::FLAT);

    auto result =
        index->KnnSearch(MakeQuery(vectors, 1), 1, R"({"hgraph":{"ef_search":40}})", nullptr);
    REQUIRE(result->GetIds()[0] == 101);
    REQUIRE(result->GetDistances()[0] == 0.0F);

    auto range = index->RangeSearch(MakeQuery(vectors, 2), 0.0F, "{}", nullptr);
    REQUIRE(range->GetNumElements() == 1);
    REQUIRE(range->GetIds()[0] == 102);
}

TEST_CASE("LazyHGraph transitions to graph and accepts more data", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(4);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(4, 200, vectors, ids);

    REQUIRE(index->Add(data).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);
    REQUIRE(index->GetNumElements() == 4);

    std::vector<float> more_vectors;
    std::vector<int64_t> more_ids;
    auto more = MakeDataset(1, 300, more_vectors, more_ids);
    REQUIRE(index->Add(more).empty());
    REQUIRE(index->GetNumElements() == 5);

    auto result =
        index->KnnSearch(MakeQuery(more_vectors, 0), 1, R"({"hgraph":{"ef_search":40}})", nullptr);
    REQUIRE(result->GetIds()[0] == 300);
}

TEST_CASE("LazyHGraph filters removed flat ids during transition", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(3);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(2, 400, vectors, ids);
    REQUIRE(index->Add(data).empty());
    REQUIRE(index->Remove({401}) == 1);

    std::vector<float> more_vectors;
    std::vector<int64_t> more_ids;
    auto more = MakeDataset(2, 500, more_vectors, more_ids);
    REQUIRE(index->Add(more).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);
    REQUIRE(index->GetNumElements() == 3);
    REQUIRE_FALSE(index->CheckIdExist(401));
}

TEST_CASE("LazyHGraph threshold one transitions immediately", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(1);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(1, 600, vectors, ids);

    REQUIRE(index->Add(data).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);
}

TEST_CASE("LazyHGraph factory and flat serialization round trip", "[ut][lazy_hgraph]") {
    auto index = vsag::Factory::CreateIndex("lazy_hgraph", MakeFactoryParam(10));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->GetIndexType() == vsag::IndexType::LAZY_HGRAPH);

    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(3, 700, vectors, ids);
    REQUIRE(index.value()->Add(data).has_value());

    auto binary = index.value()->Serialize();
    REQUIRE(binary.has_value());

    auto restored = vsag::Factory::CreateIndex("lazy_hgraph", MakeFactoryParam(10));
    REQUIRE(restored.has_value());
    REQUIRE(restored.value()->Deserialize(binary.value()).has_value());

    auto result = restored.value()->KnnSearch(
        MakeQuery(vectors, 0), 1, R"({"hgraph":{"ef_search":40}})", vsag::FilterPtr{});
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetIds()[0] == 700);
}

TEST_CASE("LazyHGraph rejects flat quantization parameters", "[ut][lazy_hgraph]") {
    auto common_param = MakeCommonParam();
    auto param = MakeLazyParam(4);
    param["flat_quantization_type"].SetString("sq8");

    REQUIRE_THROWS(vsag::LazyHGraph::CheckAndMappingExternalParam(param, common_param));
}
