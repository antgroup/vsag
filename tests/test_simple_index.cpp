
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
#include <fstream>

#include "fixtures/test_dataset_pool.h"
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

    int64_t
    GetMemoryUsage() const override {
        return 0;
    }
};

TEST_CASE("Test Simple Index", "[ft][simple_index]") {
    IndexPtr index = std::make_shared<SimpleIndex>();
    auto pool = std::make_shared<fixtures::TestDatasetPool>();
    auto dim = 12;
    auto base_count = 100;
    auto dataset = pool->GetDatasetAndCreate(dim, base_count, "fp32");
    BinarySet binary;
    std::vector<int64_t> pretrain_ids;
    FilterPtr filter = nullptr;
    SearchRequest req;
    IteratorContext* itex = nullptr;
    std::string search_param = "{}";
    SearchParam param(true, search_param, filter, nullptr);

    REQUIRE_THROWS(index->Add(dataset->base_));
    REQUIRE_THROWS(index->Remove(0));
    REQUIRE_THROWS(index->CheckFeature(IndexFeature::SUPPORT_ESTIMATE_MEMORY));
    REQUIRE_THROWS(index->EstimateMemory(1000));
    REQUIRE_THROWS(index->GetEstimateBuildMemory(1000));
    REQUIRE_THROWS(index->Feedback(dataset->query_, 10, ""));
    REQUIRE_THROWS(index->GetStats());
    REQUIRE_THROWS(index->UpdateId(0, 1));
    REQUIRE_THROWS(index->UpdateVector(0, dataset->query_));
    REQUIRE_THROWS(index->ContinueBuild(dataset->base_, binary));
    REQUIRE_THROWS(index->Pretrain(pretrain_ids, 10, ""));
    REQUIRE_THROWS(index->CheckIdExist(0));
    REQUIRE_THROWS(index->CalcDistanceById(dataset->base_->GetFloat32Vectors(), 1));
    REQUIRE_THROWS(index->CalDistanceById(dataset->base_->GetFloat32Vectors(), nullptr, 1));
    REQUIRE_THROWS(index->GetMinAndMaxId());
    REQUIRE_THROWS(index->GetExtraInfoByIds(nullptr, 1, nullptr));
    REQUIRE_THROWS(index->GetRawVectorByIds(nullptr, 1));
    REQUIRE_THROWS(index->Clone());
    REQUIRE_THROWS(index->ExportModel());
    REQUIRE_THROWS(index->Train(dataset->base_));
    REQUIRE_THROWS(index->KnnSearch(dataset->query_, 10, search_param, filter, itex, true));
    REQUIRE_THROWS(index->KnnSearch(dataset->query_, 10, search_param, filter));
    REQUIRE_THROWS(index->KnnSearch(dataset->query_, 10, param));
    REQUIRE_THROWS(index->SearchWithRequest(req));
    REQUIRE_THROWS(index->RangeSearch(dataset->query_, 1.0F, search_param, filter));
    REQUIRE_THROWS(index->GetMemoryUsageDetail());
    REQUIRE_THROWS(index->SetImmutable());

    std::vector<MergeUnit> units;
    REQUIRE_THROWS(index->Merge(units));

    std::ofstream o_file("1234", std::ios::binary);
    REQUIRE_THROWS(index->Serialize(o_file));

    std::ifstream i_file("1234", std::ios::binary);
    REQUIRE_THROWS(index->Deserialize(i_file));
}
