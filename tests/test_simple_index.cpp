
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
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <fstream>

#include "functest.h"
#include "vsag/index.h"

using namespace vsag;
class SimpleIndex : public Index {
public:
    virtual tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override {
        return tl::expected<std::vector<int64_t>, Error>();
    }

    [[nodiscard]] virtual tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        return tl::expected<DatasetPtr, Error>();
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        return tl::expected<DatasetPtr, Error>();
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {
        return tl::expected<DatasetPtr, Error>();
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {
        return tl::expected<DatasetPtr, Error>();
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const override {
        return tl::expected<DatasetPtr, Error>();
    }

    tl::expected<BinarySet, Error>
    Serialize() const override {
        return tl::expected<BinarySet, Error>();
    }

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        return tl::expected<void, Error>();
    }

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        return tl::expected<void, Error>();
    }

    int64_t
    GetNumElements() const override {
        return 0;
    }

    uint64_t
    GetMemoryUsage() const override {
        return 0;
    }
};

TEST_CASE("Test Simple Index", "[ft][simple_index]") {
    IndexPtr index = std::make_shared<SimpleIndex>();
    auto pool = std::make_shared<fixtures::TestDatasetPool>();
    auto dim = 12;
    auto base_count = 100;
    auto dataset = pool->GetDatasetAndCreate(dim, base_count, "l2");
    BinarySet binary;
    std::vector<int64_t> pretrain_ids;
    FilterPtr filter = nullptr;
    SearchRequest req;
    IteratorContext* itex = nullptr;
    std::string search_param = "{}";
    SearchParam param(true, search_param, filter, nullptr);

    REQUIRE_FALSE(index->Add(dataset->base_).has_value());
    REQUIRE_FALSE(index->Remove(0).has_value());
    REQUIRE_FALSE(index->CheckFeature(IndexFeature::SUPPORT_ESTIMATE_MEMORY));
    REQUIRE_THROWS(index->EstimateMemory(1000));
    REQUIRE_THROWS(index->EstimateBuildMemory(1000));
    REQUIRE_FALSE(index->Feedback(dataset->query_, 10, "").has_value());
    REQUIRE_THROWS_MATCHES(index->GetStats(),
                           std::runtime_error,
                           Catch::Matchers::Message("Index does not support GetStats"));
    REQUIRE_FALSE(index->UpdateId(0, 1).has_value());
    REQUIRE_FALSE(index->UpdateVector(0, dataset->query_).has_value());
    REQUIRE_FALSE(index->ContinueBuild(dataset->base_, binary).has_value());
    REQUIRE_FALSE(index->Pretrain(pretrain_ids, 10, "").has_value());
    REQUIRE_THROWS(index->CheckIdExist(0));
    REQUIRE_FALSE(index->CalcDistanceById(dataset->base_->GetFloat32Vectors(), 1).has_value());
    REQUIRE_FALSE(index->CalcDistanceById(dataset->query_, 1).has_value());
    REQUIRE_FALSE(
        index->CalDistanceById(dataset->base_->GetFloat32Vectors(), nullptr, 1).has_value());
    REQUIRE_FALSE(index->GetMinAndMaxId().has_value());
    REQUIRE_FALSE(index->GetExtraInfoByIds(nullptr, 1, nullptr).has_value());
    REQUIRE_FALSE(index->UpdateExtraInfo(dataset->query_).has_value());
    REQUIRE_FALSE(index->GetRawVectorByIds(nullptr, 1).has_value());
    REQUIRE_FALSE(index->Clone().has_value());
    REQUIRE_FALSE(index->ExportModel().has_value());
    REQUIRE_FALSE(index->Train(dataset->base_).has_value());
    REQUIRE_FALSE(
        index->KnnSearch(dataset->query_, 10, search_param, filter, itex, true).has_value());
    REQUIRE_FALSE(index->KnnSearch(dataset->query_, 10, search_param, filter).has_value());
    REQUIRE_FALSE(index->KnnSearch(dataset->query_, 10, param).has_value());
    REQUIRE_FALSE(index->SearchWithRequest(req).has_value());
    REQUIRE_FALSE(index->RangeSearch(dataset->query_, 1.0F, search_param, filter).has_value());
    REQUIRE_THROWS(index->GetMemoryUsageDetail());
    REQUIRE_FALSE(index->SetImmutable().has_value());
    AttributeSet old_attrs;
    AttributeSet new_attrs;
    REQUIRE_FALSE(index->UpdateAttribute(0, new_attrs).has_value());
    REQUIRE_FALSE(index->UpdateAttribute(1, new_attrs, old_attrs).has_value());

    std::vector<MergeUnit> units;
    REQUIRE_FALSE(index->Merge(units).has_value());

    fixtures::TempDir dir("test_simple_index");
    std::ofstream o_file(dir.path + "1234", std::ios::binary);
    REQUIRE_FALSE(index->Serialize(o_file).has_value());

    std::ifstream i_file(dir.path + "1234", std::ios::binary);
    REQUIRE_FALSE(index->Deserialize(i_file).has_value());

    REQUIRE_FALSE(index->GetDataByIds(nullptr, 1).has_value());
    REQUIRE_FALSE(index->ExportIDs().has_value());
    REQUIRE_THROWS(index->AnalyzeIndexBySearch(req));
    REQUIRE_THROWS(index->GetIndexType());
    REQUIRE_FALSE(index->GetIndexDetailInfos().has_value());
    REQUIRE_FALSE(index->Serialize(WriteFuncType(nullptr)).has_value());

    IndexDetailInfo info;
    std::string name = INDEX_DETAIL_NAME_NUM_ELEMENTS;
    REQUIRE_FALSE(index->GetDetailDataByName(name, info).has_value());
}
