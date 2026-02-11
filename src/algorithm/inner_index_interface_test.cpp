
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

#include "inner_index_interface.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <sstream>

#include "brute_force.h"
#include "hgraph.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

TEST_CASE("Fast Create Index", "[ut][InnerIndexInterface]") {
    IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.thread_pool_ = SafeThreadPool::FactoryDefaultThreadPool();
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

    SECTION("HGraph created with minimal parameters") {
        std::string index_fast_str = "hgraph|100|fp16";
        auto index = InnerIndexInterface::FastCreateIndex(index_fast_str, common_param);
        REQUIRE(index != nullptr);
        REQUIRE(dynamic_cast<HGraph*>(index.get()) != nullptr);
    }

    SECTION("HGraph created with optional parameters") {
        std::string index_fast_str = "hgraph|100|sq8|fp32";
        auto index = InnerIndexInterface::FastCreateIndex(index_fast_str, common_param);
        REQUIRE(index != nullptr);
        REQUIRE(dynamic_cast<HGraph*>(index.get()) != nullptr);
    }

    SECTION("BruteForce created") {
        std::string index_fast_str = "brute_force|fp32";
        auto index = InnerIndexInterface::FastCreateIndex(index_fast_str, common_param);
        REQUIRE(index != nullptr);
        REQUIRE(dynamic_cast<BruteForce*>(index.get()) != nullptr);
    }

    SECTION("Unsupported index type returns null") {
        std::string index_fast_str = "UNKNOWN|other";
        REQUIRE_THROWS(InnerIndexInterface::FastCreateIndex(index_fast_str, common_param));
    }

    SECTION("Invalid parameter count for HGraph (too few)") {
        std::string index_fast_str = "hgraph|100";
        REQUIRE_THROWS(InnerIndexInterface::FastCreateIndex(index_fast_str, common_param));
    }

    SECTION("Invalid parameter count for BruteForce (too few)") {
        std::string index_fast_str = "bruteforce";
        REQUIRE_THROWS(InnerIndexInterface::FastCreateIndex(index_fast_str, common_param));
    }
}

class EmptyInnerIndex : public InnerIndexInterface {
public:
    EmptyInnerIndex() : InnerIndexInterface() {
    }

    std::string
    GetName() const override {
        return "EmptyInnerIndex";
    }

    IndexType
    GetIndexType() const override {
        throw std::runtime_error("Index not support GetIndexType");
    }

    void
    InitFeatures() override {
        return;
    }

    std::vector<int64_t>
    Add(const DatasetPtr& base, AddMode mode) override {
        return {};
    }

    DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override {
        return nullptr;
    }

    [[nodiscard]] DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override {
        return nullptr;
    }

    void
    Serialize(StreamWriter& writer) const override {
        return;
    }

    void
    Deserialize(StreamReader& reader) override {
        return;
    }

    int64_t
    GetNumElements() const override {
        return 0;
    }
};

TEST_CASE("InnerIndexInterface NOT Implemented", "[ut][InnerIndexInterface]") {
    InnerIndexPtr empty_index = std::make_shared<EmptyInnerIndex>();
    IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.thread_pool_ = SafeThreadPool::FactoryDefaultThreadPool();
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

    BinarySet binary;
    std::vector<int64_t> pretrain_ids;
    std::vector<MergeUnit> merge_units;

    REQUIRE_THROWS(empty_index->Remove(0));
    REQUIRE_THROWS(empty_index->GetNumberRemoved());
    REQUIRE_THROWS(empty_index->EstimateMemory(1000));
    REQUIRE_THROWS(empty_index->GetEstimateBuildMemory(1000));
    REQUIRE_THROWS(empty_index->Feedback(nullptr, 10, ""));
    REQUIRE_THROWS(empty_index->GetStats());
    REQUIRE_THROWS(empty_index->UpdateId(0, 1));
    REQUIRE_THROWS(empty_index->UpdateVector(0, nullptr));
    REQUIRE_THROWS(empty_index->UpdateExtraInfo(nullptr));
    REQUIRE_THROWS(empty_index->ContinueBuild(nullptr, binary));
    REQUIRE_THROWS(empty_index->Pretrain(pretrain_ids, 10, ""));
    REQUIRE_THROWS(empty_index->CalcDistanceById(nullptr, 1));
    REQUIRE_THROWS(empty_index->ExportModel(common_param));
    REQUIRE_THROWS(empty_index->GetCodeByInnerId(1, nullptr));
    REQUIRE_THROWS(empty_index->GetMinAndMaxId());
    REQUIRE_THROWS(empty_index->GetMemoryUsageDetail());
    REQUIRE_THROWS(empty_index->Merge(merge_units));
    REQUIRE_THROWS(empty_index->GetExtraInfoByIds(nullptr, 1, nullptr));
    REQUIRE_THROWS(empty_index->GetVectorByInnerId(1, nullptr));
    REQUIRE_THROWS(empty_index->SetImmutable());

    AttributeSet old_attrs;
    AttributeSet new_attrs;
    REQUIRE_THROWS(empty_index->UpdateAttribute(0, new_attrs));
    REQUIRE_THROWS(empty_index->UpdateAttribute(1, new_attrs, old_attrs));

    REQUIRE_NOTHROW(empty_index->Train(nullptr));

    SearchRequest req;
    REQUIRE_THROWS(empty_index->AnalyzeIndexBySearch(req));
    REQUIRE_THROWS(empty_index->SearchWithRequest(req));
    REQUIRE_THROWS(empty_index->Fork(common_param));

    REQUIRE_THROWS(empty_index->RangeSearch(nullptr, 0.0F, "", nullptr, nullptr));
    REQUIRE_THROWS(empty_index->KnnSearch(nullptr, 0, "", nullptr, nullptr));

    SearchParam param(true, "", nullptr, nullptr);
    REQUIRE_THROWS(empty_index->KnnSearch(nullptr, 0, param));

    REQUIRE_NOTHROW(empty_index->GetIndexDetailInfos());
}
