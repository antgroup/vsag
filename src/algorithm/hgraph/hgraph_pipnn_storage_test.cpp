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

#include <algorithm>
#include <atomic>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "datacell/flatten_datacell.h"
#include "hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index/index_impl.h"
#include "quantization/fp32_quantizer.h"
#include "unittest.h"
#include "vsag/options.h"

namespace vsag {

struct HGraphPiPNNStorageTestAccess {
    static void
    MaterializeWithCodes(HGraph& index, const DatasetPtr& data, FlattenInterfacePtr codes) {
        auto original = index.basic_flatten_codes_;
        index.basic_flatten_codes_ = std::move(codes);
        Vector<InnerIdType> ids(index.allocator_);
        Vector<int64_t> rows(index.allocator_);
        for (int64_t row = 0; row < data->GetNumElements(); ++row) {
            ids.push_back(static_cast<InnerIdType>(row));
            rows.push_back(row);
        }
        try {
            index.materialize_pipnn_storage(data, ids, rows);
        } catch (...) {
            index.basic_flatten_codes_ = std::move(original);
            throw;
        }
        index.basic_flatten_codes_ = std::move(original);
    }
};

}  // namespace vsag

namespace {

class FailingStorageFlatten
    : public vsag::FlattenDataCell<vsag::FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>,
                                   vsag::MemoryIO> {
public:
    void
    InsertVector(const void* vector, vsag::InnerIdType id) override {
        ++attempts;
        throw std::runtime_error("injected storage encoding failure");
    }

    std::atomic<uint64_t> attempts{0};
};

using Index = vsag::IndexImpl<vsag::HGraph>;

class SmallStorageBlocks {
public:
    SmallStorageBlocks() : previous_(vsag::Options::Instance().block_size_limit()) {
        vsag::Options::Instance().set_block_size_limit(512ULL * 1024);
    }
    ~SmallStorageBlocks() {
        vsag::Options::Instance().set_block_size_limit(previous_);
    }

private:
    uint64_t previous_;
};

vsag::IndexCommonParam
make_common(int64_t dim, uint64_t threads = 1) {
    vsag::IndexCommonParam common;
    common.dim_ = dim;
    common.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    common.thread_pool_ = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    common.thread_pool_->SetPoolSize(threads);
    return common;
}

vsag::JsonType
make_parameters() {
    return vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32", "graph_type": "pipnn",
        "max_degree": 8, "ef_construction": 64, "build_thread_count": 1,
        "support_duplicate": true, "deduplicate_storage": true
    })");
}

vsag::DatasetPtr
make_data(std::vector<float>& vectors, std::vector<int64_t>& labels, int64_t dim) {
    return vsag::Dataset::Make()
        ->NumElements(static_cast<int64_t>(labels.size()))
        ->Dim(dim)
        ->Ids(labels.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
}

void
require_counts(const std::shared_ptr<Index>& index, uint32_t logical, uint32_t physical) {
    auto hgraph = std::dynamic_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    REQUIRE(hgraph != nullptr);
    REQUIRE(index->GetNumElements() == logical);
    REQUIRE(hgraph->GetCodeStorageCounts() ==
            std::pair<vsag::InnerIdType, vsag::CodeSlotIdType>{logical, physical});
}

void
require_labels(const std::shared_ptr<Index>& index,
               float* vector,
               int64_t dim,
               const std::set<int64_t>& labels) {
    auto query =
        vsag::Dataset::Make()->NumElements(1)->Dim(dim)->Float32Vectors(vector)->Owner(false);
    auto result = index->KnnSearch(query,
                                   static_cast<int64_t>(labels.size()),
                                   R"({"hgraph":{"ef_search":128,"brute_force_threshold":0.0}})");
    REQUIRE(result.has_value());
    REQUIRE(result->get()->GetDim() == static_cast<int64_t>(labels.size()));
    const auto* ids = result->get()->GetIds();
    REQUIRE(std::set<int64_t>(ids, ids + labels.size()) == labels);
}

}  // namespace

TEST_CASE("PiPNN storage dedup shares codes across quantizers and round trips",
          "[ut][hgraph][pipnn][pipnn_storage][duplicate][serialize]") {
    SmallStorageBlocks blocks;
    constexpr int64_t dim = 64;
    constexpr int64_t count = 128;
    constexpr int64_t unique = 16;
    const auto threads = GENERATE(1, 4);
    const auto adaptive = GENERATE(false, true);
    const auto* const encoding = GENERATE("fp32", "sq8_reorder", "rabitq_raw", "rabitq_deferred");
    CAPTURE(threads, adaptive, encoding);
    auto common = make_common(dim, threads);
    auto parameter = make_parameters();
    parameter["build_thread_count"].SetInt(threads);
    parameter["pipnn_adaptive_pruning"].SetBool(adaptive);
    const std::string kind = encoding;
    parameter["store_raw_vector"].SetBool(kind != "rabitq_deferred");
    if (kind == "sq8_reorder") {
        parameter["base_quantization_type"].SetString("sq8");
        parameter["use_reorder"].SetBool(true);
        parameter["precise_quantization_type"].SetString("fp32");
    } else if (kind == "rabitq_raw" or kind == "rabitq_deferred") {
        parameter["base_quantization_type"].SetString("rabitq");
        parameter["rabitq_bits_per_dim_base"].SetInt(1);
        parameter["store_raw_vector"].SetBool(kind == "rabitq_raw");
    }
    std::vector<float> vectors(count * dim);
    std::vector<int64_t> labels(count);
    for (int64_t row = 0; row < count; ++row) {
        labels[row] = 1000 + row * 7;
        for (int64_t d = 0; d < dim; ++d) {
            vectors[row * dim + d] = static_cast<float>(((row % unique) * 13 + d * 7) % 71);
        }
    }
    auto index = std::make_shared<Index>(parameter, common);
    auto built = index->Build(make_data(vectors, labels, dim));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    require_counts(index, count, unique);

    auto restored = std::make_shared<Index>(parameter, common);
    SECTION("binary set") {
        auto binary = index->Serialize();
        REQUIRE(binary.has_value());
        REQUIRE(restored->Deserialize(binary.value()).has_value());
    }
    SECTION("stream") {
        std::stringstream stream;
        REQUIRE(index->Serialize(stream).has_value());
        stream.seekg(0);
        REQUIRE(restored->Deserialize(stream).has_value());
    }
    require_counts(restored, count, unique);
    if (kind != "rabitq_deferred") {
        auto raw = restored->GetRawVectorByIds(labels.data(), count, nullptr);
        REQUIRE(raw.has_value());
        REQUIRE(std::equal(vectors.begin(), vectors.end(), raw.value()->GetFloat32Vectors()));
    }
    for (int64_t row = 0; row < count; ++row) {
        auto distance = restored->CalcDistanceById(vectors.data(), labels[row]);
        auto representative = restored->CalcDistanceById(vectors.data(), labels[row % unique]);
        REQUIRE(distance.has_value());
        REQUIRE(representative.has_value());
        REQUIRE(distance.value() == representative.value());
    }
    if (kind == "fp32" or kind == "sq8_reorder") {
        std::set<int64_t> expected;
        for (int64_t row = 0; row < count; row += unique) {
            expected.insert(labels[row]);
        }
        require_labels(restored, vectors.data(), dim, expected);
    }
}

TEST_CASE("PiPNN storage dedup preserves filtered rows and extends after restore",
          "[ut][hgraph][pipnn][pipnn_storage][duplicate][add]") {
    SmallStorageBlocks blocks;
    const auto first_add = GENERATE(false, true);
    const auto adaptive = GENERATE(false, true);
    auto common = make_common(2, 4);
    auto parameter = make_parameters();
    parameter["build_thread_count"].SetInt(4);
    parameter["pipnn_adaptive_pruning"].SetBool(adaptive);
    auto index = std::make_shared<Index>(parameter, common);
    std::vector<float> vectors{0, 0, 1, 0, 99, 99, 0, 0, 1, 0, 3, 0};
    std::vector<int64_t> labels{90, 10, 90, 30, 50, 70};
    auto data = make_data(vectors, labels, 2);
    auto built = first_add ? index->Add(data) : index->Build(data);
    REQUIRE(built.has_value());
    REQUIRE(built.value() == std::vector<int64_t>{90});
    require_counts(index, 5, 3);
    require_labels(index, vectors.data(), 2, {90, 30});
    require_labels(index, vectors.data() + 2, 2, {10, 50});

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = std::make_shared<Index>(parameter, common);
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    std::vector<float> added{1, 0, 7, 0};
    std::vector<int64_t> added_labels{110, 130};
    auto added_result = restored->Add(make_data(added, added_labels, 2));
    REQUIRE(added_result.has_value());
    REQUIRE(added_result.value().empty());
    require_counts(restored, 7, 4);
    require_labels(restored, added.data(), 2, {10, 50, 110});
    require_labels(restored, added.data() + 2, 2, {130});

    std::vector<int64_t> duplicate_label{150};
    std::vector<float> duplicate{7, 0};
    REQUIRE(restored->Add(make_data(duplicate, duplicate_label, 2)).has_value());
    require_counts(restored, 8, 4);
    require_labels(restored, duplicate.data(), 2, {130, 150});
}

TEST_CASE("PiPNN storage dedup handles one representative and unique-only input",
          "[ut][hgraph][pipnn][pipnn_storage][duplicate]") {
    SmallStorageBlocks blocks;
    const auto count = GENERATE(1, 48);
    const auto all_same = GENERATE(false, true);
    auto common = make_common(4, 4);
    auto parameter = make_parameters();
    parameter["build_thread_count"].SetInt(4);
    std::vector<float> vectors(static_cast<uint64_t>(count) * 4, 1.0F);
    std::vector<int64_t> labels(count);
    for (int64_t row = 0; row < count; ++row) {
        labels[row] = 100 + row;
        if (not all_same) {
            vectors[row * 4] = static_cast<float>(row);
        }
    }
    auto index = std::make_shared<Index>(parameter, common);
    REQUIRE(index->Build(make_data(vectors, labels, 4)).has_value());
    require_counts(index, count, all_same ? 1 : count);
    if (all_same) {
        require_labels(index, vectors.data(), 4, std::set<int64_t>(labels.begin(), labels.end()));
    }
}

TEST_CASE("PiPNN storage dedup retains per-label extra info and filtering",
          "[ut][hgraph][pipnn][pipnn_storage][duplicate][filter]") {
    SmallStorageBlocks blocks;
    auto common = make_common(2, 4);
    common.extra_info_size_ = 2;
    auto parameter = make_parameters();
    parameter["build_thread_count"].SetInt(4);
    auto index = std::make_shared<Index>(parameter, common);
    std::vector<float> vectors{1, 0, 1, 0, 1, 0, 9, 0};
    std::vector<int64_t> labels{10, 20, 30, 40};
    std::vector<char> extra{'a', 'A', 'b', 'B', 'c', 'C', 'd', 'D'};
    auto data = make_data(vectors, labels, 2)->ExtraInfoSize(2)->ExtraInfos(extra.data());
    REQUIRE(index->Build(data).has_value());
    require_counts(index, 4, 2);
    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = std::make_shared<Index>(parameter, common);
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    std::vector<char> fetched(extra.size());
    REQUIRE(restored->GetExtraInfoByIds(labels.data(), 4, fetched.data()).has_value());
    REQUIRE(fetched == extra);

    auto query =
        vsag::Dataset::Make()->NumElements(1)->Dim(2)->Float32Vectors(vectors.data())->Owner(false);
    const std::function<bool(int64_t)> exclude = [](int64_t label) { return label != 20; };
    auto result = restored->KnnSearch(query, 1, R"({"hgraph":{"ef_search":64}})", exclude);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 1);
    REQUIRE(result.value()->GetIds()[0] == 20);
}

TEST_CASE("PiPNN storage failure drains workers without consuming slots",
          "[ut][hgraph][pipnn][pipnn_storage][duplicate]") {
    SmallStorageBlocks blocks;
    const auto threads = GENERATE(1, 4);
    auto parameter = make_parameters();
    parameter["build_thread_count"].SetInt(threads);
    auto index = std::make_shared<Index>(parameter, make_common(2, threads));
    std::vector<float> vectors{1, 0, 1, 0, 2, 0, 3, 0};
    std::vector<int64_t> labels{10, 20, 30, 40};
    auto data = make_data(vectors, labels, 2);
    REQUIRE(index->Build(data).has_value());
    auto hgraph = std::dynamic_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    auto failing_codes = std::make_shared<FailingStorageFlatten>();
    REQUIRE_THROWS_AS(
        vsag::HGraphPiPNNStorageTestAccess::MaterializeWithCodes(*hgraph, data, failing_codes),
        std::runtime_error);
    REQUIRE(failing_codes->attempts.load() == static_cast<uint64_t>(threads == 1 ? 1 : 3));
    require_counts(index, 4, 3);
    require_labels(index, vectors.data(), 2, {10, 20});
}
