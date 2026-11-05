
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

// Overrides the CANONICAL batch name. Deprecated alias forwards here.
class CanonicalBatchIndex : public SimpleIndex {
public:
    tl::expected<DatasetPtr, Error>
    CalcDistancesById(const float* query,
                      const int64_t* ids,
                      int64_t count,
                      bool precise = true,
                      int64_t topk = -1) const override {
        REQUIRE(ids != nullptr);
        REQUIRE(count == 3);
        REQUIRE_FALSE(precise);
        REQUIRE(topk == 2);
        return Dataset::Make()->Owner(false)->Distances(const_cast<float*>(query));
    }
    tl::expected<DatasetPtr, Error>
    CalcDistancesById(const DatasetPtr& query,
                      const int64_t* ids,
                      int64_t count,
                      bool precise = true,
                      int64_t topk = -1) const override {
        return CalcDistancesById(query->GetFloat32Vectors(), ids, count, precise, topk);
    }
};

// Defines only the deprecated batch alias (which is now a non-virtual
// forwarding helper on the base). After the layout change this does not
// participate in the canonical CalcDistancesById dispatch. The deprecated
// name hides the base and is unreachable when the caller sends a generic
// Index pointer to CalcDistancesById.
class DeprecatedOnlyBatchIndex : public SimpleIndex {
public:
    tl::expected<DatasetPtr, Error>
    CalDistanceById(const float* query,
                    const int64_t* ids,
                    int64_t count,
                    bool precise = true,
                    int64_t topk = -1) const {
        REQUIRE(ids != nullptr);
        REQUIRE(count == 3);
        REQUIRE(precise);
        REQUIRE(topk == -1);
        return Dataset::Make()->Owner(false)->Distances(const_cast<float*>(query));
    }
    tl::expected<DatasetPtr, Error>
    CalDistanceById(const DatasetPtr& query,
                    const int64_t* ids,
                    int64_t count,
                    bool precise = true,
                    int64_t topk = -1) const {
        return CalDistanceById(query->GetFloat32Vectors(), ids, count, precise, topk);
    }
};

TEST_CASE("Canonical batch virtual dispatch and deprecated aliases",
          "[ft][simple_index][distance_contract]") {
    // 1) Canonical override -> canonical + deprecated both work
    IndexPtr canonical_index = std::make_shared<CanonicalBatchIndex>();
    float distance = 1.0F;
    int64_t ids[] = {1, 2, 3};
    auto query = Dataset::Make()->Owner(false)->NumElements(1)->Float32Vectors(&distance);
    auto raw = canonical_index->CalcDistancesById(&distance, ids, 3, false, 2);
    auto native = canonical_index->CalcDistancesById(query, ids, 3, false, 2);
    REQUIRE(raw.has_value());
    REQUIRE(native.has_value());
    REQUIRE(raw.value()->GetDistances() == &distance);
    REQUIRE(native.value()->GetDistances() == &distance);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    auto legacy_raw = canonical_index->CalDistanceById(&distance, ids, 3, false, 2);
    auto legacy_native = canonical_index->CalDistanceById(query, ids, 3, false, 2);
#pragma GCC diagnostic pop
    REQUIRE(legacy_raw.has_value());
    REQUIRE(legacy_native.has_value());
    REQUIRE(legacy_raw.value()->GetDistances() == &distance);
    REQUIRE(legacy_native.value()->GetDistances() == &distance);

    // 2) Index that only defines the deprecated alias (now non-virtual).
    //    The deprecated name is reachable through the static type but the
    //    canonical CalcDistancesById dispatch never sees it.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    auto deprecated_index = std::make_shared<DeprecatedOnlyBatchIndex>();
    auto dep_legacy = deprecated_index->CalDistanceById(&distance, ids, 3, true);
    REQUIRE(dep_legacy.has_value());
#pragma GCC diagnostic pop
    // Canonical dispatch through the Index* pointer hits the base
    // UNSUPPORTED fallback because DeprecatedOnlyBatchIndex does not
    // override CalcDistancesById.
    IndexPtr base_ptr = deprecated_index;
    auto dep_canonical = base_ptr->CalcDistancesById(&distance, ids, 3, true);
    REQUIRE_FALSE(dep_canonical.has_value());
}

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
        index->CalcDistancesById(dataset->base_->GetFloat32Vectors(), nullptr, 1).has_value());
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
