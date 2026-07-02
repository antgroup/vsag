
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

#include <memory>
#include <sstream>
#include <utility>

#include "algorithm/bruteforce/bruteforce.h"
#include "algorithm/hgraph/hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "unittest.h"
#include "vsag_exception.h"

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
    EmptyInnerIndex() = default;

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
    }

    std::vector<int64_t>
    Add(const DatasetPtr& base) override {
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
    }

    void
    Deserialize(StreamReader& reader) override {
    }

    int64_t
    GetNumElements() const override {
        return 0;
    }
};

class ReadingInnerIndex : public EmptyInnerIndex {
public:
    std::string
    GetName() const override {
        return "ReadingInnerIndex";
    }

    void
    Deserialize(StreamReader& reader) override {
        uint64_t value = 0;
        StreamReader::ReadObj(reader, value);
    }
};

class NullDestinationInnerIndex : public EmptyInnerIndex {
public:
    std::string
    GetName() const override {
        return "NullDestinationInnerIndex";
    }

    void
    Deserialize(StreamReader& reader) override {
        reader.Read(nullptr, 1);
    }
};

class SeekOutOfRangeInnerIndex : public EmptyInnerIndex {
public:
    std::string
    GetName() const override {
        return "SeekOutOfRangeInnerIndex";
    }

    void
    Deserialize(StreamReader& reader) override {
        reader.Seek(2);
        uint8_t value = 0;
        reader.Read(reinterpret_cast<char*>(&value), 1);
    }
};

Binary
MakeBinary(const std::shared_ptr<int8_t[]>& data, uint64_t size) {
    Binary binary;
    binary.data = data;
    binary.size = size;
    return binary;
}

template <typename Func>
void
RequireReadError(Func&& func) {
    bool got_read_error = false;
    try {
        std::forward<Func>(func)();
    } catch (const VsagException& e) {
        got_read_error = true;
        REQUIRE(e.error_.type == ErrorType::READ_ERROR);
    } catch (const std::exception& e) {
        FAIL("Expected READ_ERROR VsagException, got std::exception: " << e.what());
    } catch (...) {
        FAIL("Expected READ_ERROR VsagException, got non-standard exception");
    }
    REQUIRE(got_read_error);
}

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
    REQUIRE_THROWS(empty_index->EstimateBuildMemory(1000));
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

TEST_CASE("InnerIndexInterface rejects malformed binary set", "[ut][InnerIndexInterface]") {
    InnerIndexPtr index = std::make_shared<ReadingInnerIndex>();

    SECTION("missing index binary") {
        BinarySet binary;

        RequireReadError([&index, &binary]() { index->Deserialize(binary); });
    }

    SECTION("null non-empty index binary") {
        BinarySet binary;
        binary.Set(index->GetName(), MakeBinary(std::shared_ptr<int8_t[]>(), sizeof(uint64_t)));

        RequireReadError([&index, &binary]() { index->Deserialize(binary); });
    }

    SECTION("truncated index binary") {
        auto data = std::shared_ptr<int8_t[]>(new int8_t[1]{});
        BinarySet binary;
        binary.Set(index->GetName(), MakeBinary(data, 1));

        RequireReadError([&index, &binary]() { index->Deserialize(binary); });
    }

    SECTION("null read destination") {
        index = std::make_shared<NullDestinationInnerIndex>();
        auto data = std::shared_ptr<int8_t[]>(new int8_t[1]{});
        BinarySet binary;
        binary.Set(index->GetName(), MakeBinary(data, 1));

        RequireReadError([&index, &binary]() { index->Deserialize(binary); });
    }

    SECTION("seek out of range") {
        index = std::make_shared<SeekOutOfRangeInnerIndex>();
        auto data = std::shared_ptr<int8_t[]>(new int8_t[1]{});
        BinarySet binary;
        binary.Set(index->GetName(), MakeBinary(data, 1));

        RequireReadError([&index, &binary]() { index->Deserialize(binary); });
    }
}
