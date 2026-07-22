
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

#define VSAG_SINDI_TEST_ACCESS
#include "sindi.h"

#include <array>
#include <chrono>
#include <cstring>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include "algorithm/sindi/proximity_scorer.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "storage/serialization_tags.h"
#include "storage/serialization_template_test.h"
#include "storage/streaming_serialization_test_utils.h"
#include "unittest.h"
using namespace vsag;

namespace vsag {

class SINDITestAccess {
public:
    static bool
    UseTermListsHeapInsert(const SINDI& index, const SINDISearchParameter& search_param) {
        return index.UseTermListsHeapInsert(search_param);
    }

    static float
    TermListsHeapInsertPruneThreshold() {
        return SINDI::K_TERM_LISTS_HEAP_INSERT_PRUNE_THRESHOLD;
    }

    static void
    SerializeImmutableWindow(const SINDI& index,
                             StreamWriter& writer,
                             const ImmutableSINDIWindow& window) {
        index.serialize_immutable_window(writer, window);
    }

    static void
    DeserializeImmutableWindow(const SINDI& index,
                               StreamReader& reader,
                               ImmutableSINDIWindow& window) {
        index.deserialize_immutable_window(reader, window);
    }
};

}  // namespace vsag

namespace {

using vsag::test::EraseStreamingBlock;
using vsag::test::InsertUnknownStreamingBlock;

}  // namespace

class MockFilter : public Filter {
public:
    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        // return true if id is even, otherwise false
        return id % 2 == 0;
    }
};

class MockValidIdFilter : public Filter {
public:
    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return valid_ids_set_.find(id) != valid_ids_set_.end();
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        *valid_ids = valid_ids_.data();
        count = static_cast<int64_t>(valid_ids_.size());
    }

    void
    SetValidIds(std::vector<int64_t> valid_ids) {
        valid_ids_ = std::move(valid_ids);
        valid_ids_set_.clear();
        valid_ids_set_.reserve(valid_ids_.size());
        for (auto id : valid_ids_) {
            valid_ids_set_.insert(id);
        }
    }

private:
    std::vector<int64_t> valid_ids_;
    std::unordered_set<int64_t> valid_ids_set_;
};

TEST_CASE("SINDI Heap Insert Strategy Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    auto make_index = [&](float doc_prune_ratio) {
        auto param = std::make_shared<vsag::SINDIParameter>();
        param->term_id_limit = 30001;
        param->window_size = 10000;
        param->doc_prune_ratio = doc_prune_ratio;
        param->avg_doc_term_length = 100;
        return SINDI(param, common_param);
    };

    auto make_search_param = [](float query_prune_ratio) {
        SINDISearchParameter search_param;
        search_param.query_prune_ratio = query_prune_ratio;
        return search_param;
    };

    SECTION("uses distance insertion when both prune ratios are no greater than threshold") {
        std::array<float, 3> prune_ratios = {
            0.0F, 0.05F, SINDITestAccess::TermListsHeapInsertPruneThreshold()};
        for (auto doc_prune_ratio : prune_ratios) {
            auto index = make_index(doc_prune_ratio);
            for (auto query_prune_ratio : prune_ratios) {
                auto search_param = make_search_param(query_prune_ratio);
                REQUIRE_FALSE(SINDITestAccess::UseTermListsHeapInsert(index, search_param));
            }
        }
    }

    SECTION("matches threshold rule for distance and term-list insertion") {
        auto doc_prune_ratio = GENERATE(0.0F, 0.2F);
        auto query_prune_ratio = GENERATE(0.0F, 0.2F);
        auto index = make_index(doc_prune_ratio);
        auto search_param = make_search_param(query_prune_ratio);
        REQUIRE(SINDITestAccess::UseTermListsHeapInsert(index, search_param) ==
                (doc_prune_ratio > SINDITestAccess::TermListsHeapInsertPruneThreshold() ||
                 query_prune_ratio > SINDITestAccess::TermListsHeapInsertPruneThreshold()));
    }
}

SINDIParameterPtr
create_exact_sindi_param(uint32_t term_id_limit,
                         bool remap_term_ids = false,
                         uint32_t avg_doc_term_length = 100) {
    auto param_str = fmt::format(R"({{
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "term_prune_ratio": 0.0,
        "window_size": 50000,
        "term_id_limit": {},
        "remap_term_ids": {},
        "avg_doc_term_length": {}
    }})",
                                 term_id_limit,
                                 remap_term_ids ? "true" : "false",
                                 avg_doc_term_length);
    auto param_json = JsonType::Parse(param_str);
    auto index_param = std::make_shared<SINDIParameter>();
    index_param->FromJson(param_json);
    return index_param;
}

TEST_CASE("SINDI streaming compatibility", "[ut][SINDI][streaming][compatibility]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;
    common_param.dim_ = 64;

    constexpr uint32_t num_base = 128;
    constexpr int64_t max_dim = 64;
    constexpr int64_t max_id = 30000;
    std::vector<int64_t> ids(num_base);
    for (uint32_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base = fixtures::GenerateSparseVectors(num_base, max_dim, max_id, 0, 10, 114);
    auto base = Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    static constexpr auto param_str = R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "term_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 30001,
        "avg_doc_term_length": 100
    })";
    auto param_json = JsonType::Parse(param_str);
    auto index_param = std::make_shared<SINDIParameter>();
    index_param->FromJson(param_json);

    auto index = std::make_unique<SINDI>(index_param, common_param);
    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);

    std::stringstream stream;
    REQUIRE_NOTHROW(index->SerializeStreaming(stream));
    const auto bytes = stream.str();

    SECTION("skips unknown non-critical block") {
        auto mutated = InsertUnknownStreamingBlock(bytes, false);
        auto restored = std::make_unique<SINDI>(index_param, common_param);
        std::stringstream deserialize_stream(mutated);
        REQUIRE_NOTHROW(restored->DeserializeStreaming(deserialize_stream));
        REQUIRE(restored->GetNumElements() == num_base);

        std::stringstream load_stream(mutated);
        auto loaded = Index::Load(load_stream, "{}");
        REQUIRE(loaded.has_value());
        REQUIRE(loaded.value()->GetNumElements() == num_base);
    }

    SECTION("rejects unknown critical block") {
        auto mutated = InsertUnknownStreamingBlock(bytes, true);
        auto restored = std::make_unique<SINDI>(index_param, common_param);
        std::stringstream deserialize_stream(mutated);
        REQUIRE_THROWS(restored->DeserializeStreaming(deserialize_stream));

        std::stringstream load_stream(mutated);
        REQUIRE_FALSE(Index::Load(load_stream, "{}").has_value());
    }

    SECTION("rejects missing required block") {
        auto mutated = EraseStreamingBlock(bytes, StreamSerializationTag::SINDI_WINDOWS);
        auto restored = std::make_unique<SINDI>(index_param, common_param);
        std::stringstream deserialize_stream(mutated);
        REQUIRE_THROWS(restored->DeserializeStreaming(deserialize_stream));

        std::stringstream load_stream(mutated);
        REQUIRE_FALSE(Index::Load(load_stream, "{}").has_value());
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDI Basic Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    // Prepare Base and Query Dataset
    uint32_t num_base = 1000;
    uint32_t num_query = 100;
    int64_t max_dim = 128;
    common_param.dim_ = max_dim;
    int64_t max_id = 30000;
    float min_val = 0;
    float max_val = 10;
    int seed_base = 114;
    int64_t k = 10;

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base =
        fixtures::GenerateSparseVectors(num_base, max_dim, max_id, min_val, max_val, seed_base);
    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    static constexpr auto param_str = R"({{
        "use_reorder": true,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "term_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 30001,
        "avg_doc_term_length": 100
    }})";

    vsag::JsonType param_json = vsag::JsonType::Parse(fmt::format(param_str));
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);
    auto another_index = std::make_unique<SINDI>(index_param, common_param);
    auto exact_param = create_exact_sindi_param(30001);
    auto exact_index = std::make_unique<SINDI>(exact_param, common_param);

    // test build
    auto exact_build_res = exact_index->Build(base);
    REQUIRE(exact_build_res.size() == 0);
    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);
    REQUIRE(index->GetNumElements() == num_base);

    // test add failed
    SparseVector invalid_sv;
    int64_t tmp_id = 999999;
    uint32_t invalid_term_id = 30002;
    invalid_sv.ids_ = &invalid_term_id;
    invalid_sv.len_ = 1;
    auto invalid_data = vsag::Dataset::Make();
    invalid_data->NumElements(invalid_sv.len_)
        ->SparseVectors(&invalid_sv)
        ->Ids(&tmp_id)
        ->Owner(false);
    auto add_res = index->Add(invalid_data);
    REQUIRE(add_res.size() == 1);
    REQUIRE(index->GetNumElements() == num_base);

    // test serialize
    test_serializion(*index, *another_index);
    REQUIRE(another_index->GetNumElements() == num_base);
    auto streaming_index = std::make_unique<SINDI>(index_param, common_param);
    std::stringstream streaming_buffer;
    REQUIRE_NOTHROW(index->SerializeStreaming(streaming_buffer));
    const auto streaming_bytes = streaming_buffer.str();

    std::stringstream deserialize_stream(streaming_bytes);
    REQUIRE_NOTHROW(streaming_index->DeserializeStreaming(deserialize_stream));
    REQUIRE(streaming_index->GetNumElements() == num_base);

    std::stringstream load_stream(streaming_bytes);
    auto loaded_index = vsag::Index::Load(load_stream, "{}");
    REQUIRE(loaded_index.has_value());
    REQUIRE(loaded_index.value()->GetNumElements() == num_base);

    // test search process
    std::string search_param_str = R"(
    {
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20
        }
    }
    )";

    auto query = vsag::Dataset::Make();
    auto mock_filter = std::make_shared<MockFilter>();
    auto mock_valid_filter = std::make_shared<MockValidIdFilter>();
    query->NumElements(1)->SparseVectors(sv_base.data())->Owner(false);
    REQUIRE(index->CalcDistanceById(query, -1, true) == -1.0F);
    int64_t valid_count = static_cast<int64_t>(num_base * 0.5);
    std::vector<int64_t> valid_ids(valid_count, 0);
    valid_ids.push_back(invalid_term_id);
    for (int64_t i = 0; i < valid_count; i++) {
        valid_ids[i] = i;
    }
    mock_valid_filter->SetValidIds(valid_ids);

    for (int i = 0; i < num_query; ++i) {
        query->NumElements(1)->SparseVectors(sv_base.data() + i)->Owner(false);

        // gt
        auto bf_result = exact_index->KnnSearch(query, k, search_param_str, nullptr);

        // test basic performance
        auto result = index->KnnSearch(query, k, search_param_str, nullptr);
        auto loaded_result =
            loaded_index.value()->KnnSearch(query, k, search_param_str, vsag::BitsetPtr(nullptr));
        REQUIRE(loaded_result.has_value());
        REQUIRE(result->GetNumElements() == bf_result->GetNumElements());
        REQUIRE(result->GetDim() == bf_result->GetDim());
        REQUIRE(loaded_result.value()->GetNumElements() == result->GetNumElements());
        REQUIRE(loaded_result.value()->GetDim() == result->GetDim());
        for (int j = 0; j < k; j++) {
            REQUIRE(result->GetIds()[j] == bf_result->GetIds()[j]);
            REQUIRE(loaded_result.value()->GetIds()[j] == result->GetIds()[j]);
            REQUIRE(std::abs(result->GetDistances()[j] - bf_result->GetDistances()[j]) < 3e-3);
        }

        // test filter with knn
        auto filter_knn_result = index->KnnSearch(query, k, search_param_str, mock_filter);
        REQUIRE(filter_knn_result->GetDim() == k);
        auto cur = 0;
        for (int j = 0; j < k; j++) {
            if (mock_filter->CheckValid(result->GetIds()[j])) {
                REQUIRE(result->GetIds()[j] == filter_knn_result->GetIds()[cur]);
                cur++;
            }
        }

        auto valid_filter_knn_result =
            index->KnnSearch(query, k, search_param_str, mock_valid_filter);
        REQUIRE(valid_filter_knn_result->GetDim() == k);
        cur = 0;
        for (int j = 0; j < k; j++) {
            if (mock_valid_filter->CheckValid(result->GetIds()[j])) {
                REQUIRE(result->GetIds()[j] == valid_filter_knn_result->GetIds()[cur]);
                cur++;
            }
        }

        // test serialize
        auto another_result = another_index->KnnSearch(query, k, search_param_str, nullptr);
        for (int j = 0; j < another_result->GetDim(); j++) {
            REQUIRE(result->GetIds()[j] == another_result->GetIds()[j]);
            REQUIRE(std::abs(result->GetDistances()[j] - another_result->GetDistances()[j]) < 1e-3);
        }

        // test range search limit
        auto range_result_limit_3 = index->RangeSearch(query, 0, search_param_str, nullptr, 3);
        REQUIRE(range_result_limit_3->GetDim() == 3);
        for (int j = 0; j < 3; j++) {
            REQUIRE(result->GetIds()[j] == range_result_limit_3->GetIds()[j]);
            REQUIRE(std::abs(result->GetDistances()[j] - range_result_limit_3->GetDistances()[j]) <
                    1e-3);
        }

        // test filter with range limit
        auto filter_range_limit_result =
            index->RangeSearch(query, 0, search_param_str, mock_filter, 3);
        REQUIRE(filter_range_limit_result->GetDim() == 3);
        cur = 0;
        for (int j = 0; j < 3; j++) {
            if (mock_filter->CheckValid(range_result_limit_3->GetIds()[j])) {
                REQUIRE(range_result_limit_3->GetIds()[j] ==
                        filter_range_limit_result->GetIds()[cur]);
                cur++;
            }
        }

        // test range search radius
        auto target_radius = result->GetDistances()[5];
        auto range_result_radius_3 =
            index->RangeSearch(query, target_radius, search_param_str, nullptr);
        for (int j = 0; j < range_result_radius_3->GetDim(); j++) {
            REQUIRE(range_result_radius_3->GetDistances()[j] <= target_radius);
        }

        // test filter with range radius
        auto filter_range_radius_result =
            index->RangeSearch(query, target_radius, search_param_str, mock_filter);
        cur = 0;
        for (int j = 0; j < range_result_radius_3->GetDim(); j++) {
            if (mock_filter->CheckValid(range_result_radius_3->GetIds()[j])) {
                REQUIRE(range_result_radius_3->GetIds()[j] ==
                        filter_range_radius_result->GetIds()[cur]);
                cur++;
            }
        }
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDI Quantization Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    // Prepare Base and Query Dataset
    uint32_t num_base = 1000;
    uint32_t num_query = 100;
    int64_t max_dim = 128;
    common_param.dim_ = max_dim;
    int64_t max_id = 30000;
    float min_val = 0;
    float max_val = 10;
    int seed_base = 114;
    int64_t k = 10;

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base =
        fixtures::GenerateSparseVectors(num_base, max_dim, max_id, min_val, max_val, seed_base);
    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    static constexpr auto param_str = R"({{
        "use_reorder": true,
        "use_quantization": true,
        "doc_prune_ratio": 0.0,
        "term_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 30001,
        "avg_doc_term_length": 100
    }})";

    vsag::JsonType param_json = vsag::JsonType::Parse(fmt::format(param_str));
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);
    auto exact_param = create_exact_sindi_param(30001);
    auto exact_index = std::make_unique<SINDI>(exact_param, common_param);

    // test build
    auto exact_build_res = exact_index->Build(base);
    REQUIRE(exact_build_res.size() == 0);
    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);
    REQUIRE(index->GetNumElements() == num_base);

    // test search process
    std::string search_param_str = R"(
    {
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20
        }
    }
    )";

    auto query = vsag::Dataset::Make();
    int64_t correct_count = 0;

    for (int i = 0; i < num_query; ++i) {
        query->NumElements(1)->SparseVectors(sv_base.data() + i)->Owner(false);

        // gt
        auto bf_result = exact_index->KnnSearch(query, k, search_param_str, nullptr);

        // test basic performance
        auto result = index->KnnSearch(query, k, search_param_str, nullptr);
        REQUIRE(result->GetNumElements() == bf_result->GetNumElements());
        REQUIRE(result->GetDim() == bf_result->GetDim());

        std::unordered_set<int64_t> gt_ids;
        for (int j = 0; j < k; j++) {
            gt_ids.insert(bf_result->GetIds()[j]);
        }
        for (int j = 0; j < k; j++) {
            if (gt_ids.find(result->GetIds()[j]) != gt_ids.end()) {
                correct_count++;
            }
        }
    }

    float recall = static_cast<float>(correct_count) / (num_query * k);
    REQUIRE(recall > 0.99);

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDI Immutable Sparse Deserialize KNN Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    const char* sparse_value_quant_type = GENERATE("fp32", "sq8", "fp16");
    const bool remap_term_ids = GENERATE(false, true);
    const bool use_reorder = GENERATE(false, true);
    const bool use_term_lists_heap_insert = GENERATE(false, true);
    const std::string use_quantization =
        std::string(sparse_value_quant_type) == "fp16"
            ? R"("fp16")"
            : (std::string(sparse_value_quant_type) == "sq8" ? "true" : "false");

    uint32_t num_base = 300;
    uint32_t num_query = 20;
    int64_t max_dim = 64;
    int64_t max_id = 3000;
    float min_val = 0;
    float max_val = 10;
    int seed_base = 2024;
    int64_t k = 10;
    constexpr uint32_t id_offset = 1000000;

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base =
        fixtures::GenerateSparseVectors(num_base, max_dim, max_id, min_val, max_val, seed_base);
    std::set<uint32_t> unique_terms;
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            if (remap_term_ids) {
                sv_base[i].ids_[j] += id_offset;
            }
            unique_terms.insert(sv_base[i].ids_[j]);
        }
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    uint32_t term_id_limit = remap_term_ids ? static_cast<uint32_t>(unique_terms.size()) + 100
                                            : static_cast<uint32_t>(max_id) + 1;
    auto source_param_str = fmt::format(R"({{
        "use_reorder": {},
        "use_quantization": {},
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "remap_term_ids": {},
        "avg_doc_term_length": 64,
        "immutable": true
    }})",
                                        use_reorder,
                                        use_quantization,
                                        term_id_limit,
                                        remap_term_ids);
    auto target_param_str = fmt::format(R"({{
        "use_reorder": {},
        "use_quantization": {},
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "remap_term_ids": {},
        "avg_doc_term_length": 64,
        "immutable": true
    }})",
                                        use_reorder,
                                        use_quantization,
                                        term_id_limit,
                                        remap_term_ids);

    auto source_param = std::make_shared<vsag::SINDIParameter>();
    source_param->FromJson(vsag::JsonType::Parse(source_param_str));
    auto target_param = std::make_shared<vsag::SINDIParameter>();
    target_param->FromJson(vsag::JsonType::Parse(target_param_str));
    auto source = std::make_unique<SINDI>(source_param, common_param);
    auto immutable = std::make_unique<SINDI>(target_param, common_param);

    auto build_res = source->Build(base);
    REQUIRE(build_res.empty());
    test_serializion(*source, *immutable);
    REQUIRE(immutable->GetNumElements() == num_base);

    auto search_param_str = fmt::format(R"(
    {{
        "sindi": {{
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 30,
            "use_term_lists_heap_insert": {}
        }}
    }}
    )",
                                        use_term_lists_heap_insert);

    auto query = vsag::Dataset::Make();
    for (uint32_t i = 0; i < num_query; ++i) {
        query->NumElements(1)->SparseVectors(sv_base.data() + i)->Owner(false);
        auto source_result = source->KnnSearch(query, k, search_param_str, nullptr);
        auto immutable_result = immutable->KnnSearch(query, k, search_param_str, nullptr);
        REQUIRE(immutable_result->GetDim() == source_result->GetDim());
        for (int64_t j = 0; j < immutable_result->GetDim(); ++j) {
            REQUIRE(immutable_result->GetIds()[j] == source_result->GetIds()[j]);
            REQUIRE(std::abs(immutable_result->GetDistances()[j] -
                             source_result->GetDistances()[j]) < 1e-3);
        }

        auto source_range_result = source->RangeSearch(query, 0.0F, search_param_str, nullptr);
        auto immutable_range_result =
            immutable->RangeSearch(query, 0.0F, search_param_str, nullptr);
        REQUIRE(immutable_range_result->GetDim() == source_range_result->GetDim());
        for (int64_t j = 0; j < immutable_range_result->GetDim(); ++j) {
            REQUIRE(immutable_range_result->GetIds()[j] == source_range_result->GetIds()[j]);
            REQUIRE(std::abs(immutable_range_result->GetDistances()[j] -
                             source_range_result->GetDistances()[j]) < 1e-3);
        }

        constexpr int64_t range_limit = 3;
        auto limited_source_range_result =
            source->RangeSearch(query, 0.0F, search_param_str, nullptr, range_limit);
        auto limited_immutable_range_result =
            immutable->RangeSearch(query, 0.0F, search_param_str, nullptr, range_limit);
        REQUIRE(limited_immutable_range_result->GetDim() == limited_source_range_result->GetDim());
        for (int64_t j = 0; j < limited_immutable_range_result->GetDim(); ++j) {
            REQUIRE(limited_immutable_range_result->GetIds()[j] ==
                    limited_source_range_result->GetIds()[j]);
            REQUIRE(std::abs(limited_immutable_range_result->GetDistances()[j] -
                             limited_source_range_result->GetDistances()[j]) < 1e-3);
        }
    }

    REQUIRE_THROWS(immutable->GetSparseVectorByInnerId(0, nullptr, allocator.get()));
    REQUIRE_THROWS(immutable->CalcDistanceById(query, ids[0]));

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDI Immutable Runtime Rejects Mutable Operations", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    auto param = std::make_shared<vsag::SINDIParameter>();
    param->FromJson(vsag::JsonType::Parse(R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 100,
        "avg_doc_term_length": 10,
        "immutable": true
    })"));

    SINDI index(param, common_param);
    auto empty_base = vsag::Dataset::Make();
    REQUIRE_THROWS_AS(index.Add(empty_base), vsag::VsagException);

    std::stringstream ss;
    vsag::IOStreamWriter writer(ss);
    REQUIRE_NOTHROW(index.Serialize(writer));
}

TEST_CASE("SINDI Immutable Sparse Window Serialization Size", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    auto param = std::make_shared<vsag::SINDIParameter>();
    param->FromJson(vsag::JsonType::Parse(R"({
        "use_quantization": false,
        "window_size": 10000,
        "term_id_limit": 50000000,
        "remap_term_ids": true,
        "immutable": true
    })"));
    SINDI index(param, common_param);

    ImmutableSINDIWindow window(allocator.get());
    window.sorted_global_terms.push_back(24000000);
    window.offsets.push_back(0);
    window.offsets.push_back(1);
    window.id_payloads.push_back(7);
    window.value_payloads.resize(sizeof(float));

    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    SINDITestAccess::SerializeImmutableWindow(index, writer, window);

    constexpr uint64_t vector_header_bytes = sizeof(uint64_t) * 4;
    constexpr uint64_t vector_payload_bytes =
        sizeof(uint32_t) + sizeof(uint32_t) * 2 + sizeof(uint16_t) + sizeof(float);
    REQUIRE(writer.GetCursor() == vector_header_bytes + vector_payload_bytes);
}

TEST_CASE("SINDI Immutable Sparse Window Rejects Excessive Term Count", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    auto param = std::make_shared<vsag::SINDIParameter>();
    param->FromJson(vsag::JsonType::Parse(R"({
        "use_quantization": false,
        "window_size": 10000,
        "term_id_limit": 100,
        "remap_term_ids": true,
        "immutable": true
    })"));
    SINDI index(param, common_param);

    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    const uint64_t excessive_term_count = 101;
    StreamWriter::WriteObj(writer, excessive_term_count);

    vsag::IOStreamReader reader(stream);
    ImmutableSINDIWindow window(allocator.get());
    REQUIRE_THROWS_AS(SINDITestAccess::DeserializeImmutableWindow(index, reader, window),
                      vsag::VsagException);
}

TEST_CASE("SINDI Immutable Sparse Window Rejects Mismatched Offset Count", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    auto param = std::make_shared<vsag::SINDIParameter>();
    param->FromJson(vsag::JsonType::Parse(R"({
        "use_quantization": false,
        "window_size": 10000,
        "term_id_limit": 100,
        "remap_term_ids": true,
        "immutable": true
    })"));
    SINDI index(param, common_param);

    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    StreamWriter::WriteVector(writer, std::vector<uint32_t>{0});
    const uint64_t mismatched_offset_count = 101;
    StreamWriter::WriteObj(writer, mismatched_offset_count);

    vsag::IOStreamReader reader(stream);
    ImmutableSINDIWindow window(allocator.get());
    REQUIRE_THROWS_AS(SINDITestAccess::DeserializeImmutableWindow(index, reader, window),
                      vsag::VsagException);
}

TEST_CASE("SINDI Immutable Sparse Window Rejects Excessive Posting Count", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    auto param = std::make_shared<vsag::SINDIParameter>();
    param->FromJson(vsag::JsonType::Parse(R"({
        "use_quantization": false,
        "window_size": 10000,
        "term_id_limit": 100,
        "remap_term_ids": true,
        "immutable": true
    })"));
    SINDI index(param, common_param);

    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    StreamWriter::WriteVector(writer, std::vector<uint32_t>{0});
    StreamWriter::WriteVector(writer, std::vector<uint32_t>{0, 10001});

    vsag::IOStreamReader reader(stream);
    ImmutableSINDIWindow window(allocator.get());
    REQUIRE_THROWS_AS(SINDITestAccess::DeserializeImmutableWindow(index, reader, window),
                      vsag::VsagException);
}

TEST_CASE("SINDI Immutable Build Search And Serialize Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    auto param_tuple = GENERATE(std::make_tuple("fp32", false, false),
                                std::make_tuple("sq8", false, true),
                                std::make_tuple("fp16", true, false),
                                std::make_tuple("fp32", true, true));
    const auto* sparse_value_quant_type = std::get<0>(param_tuple);
    const bool remap_term_ids = std::get<1>(param_tuple);
    const bool use_reorder = std::get<2>(param_tuple);
    const std::string use_quantization =
        std::string(sparse_value_quant_type) == "fp16"
            ? R"("fp16")"
            : (std::string(sparse_value_quant_type) == "sq8" ? "true" : "false");

    constexpr uint32_t num_base = 10050;
    constexpr uint32_t num_query = 2;
    constexpr uint32_t window_size = 10000;
    constexpr int64_t max_dim = 48;
    constexpr int64_t max_id = 1000;
    constexpr uint32_t id_offset = 700000;

    std::vector<int64_t> ids(num_base);
    for (uint32_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base = fixtures::GenerateSparseVectors(num_base, max_dim, max_id, 0, 10, 2243);
    std::set<uint32_t> unique_terms;
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            if (remap_term_ids) {
                sv_base[i].ids_[j] += id_offset;
            }
            unique_terms.insert(sv_base[i].ids_[j]);
        }
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    uint32_t term_id_limit = remap_term_ids ? static_cast<uint32_t>(unique_terms.size()) + 10
                                            : static_cast<uint32_t>(max_id) + 1;
    auto mutable_param_str = fmt::format(R"({{
        "use_reorder": {},
        "use_quantization": {},
        "doc_prune_ratio": 0.0,
        "window_size": {},
        "term_id_limit": {},
        "remap_term_ids": {},
        "avg_doc_term_length": 48,
        "immutable": false
    }})",
                                         use_reorder,
                                         use_quantization,
                                         window_size,
                                         term_id_limit,
                                         remap_term_ids);
    auto immutable_param_str = fmt::format(R"({{
        "use_reorder": {},
        "use_quantization": {},
        "doc_prune_ratio": 0.0,
        "window_size": {},
        "term_id_limit": {},
        "remap_term_ids": {},
        "avg_doc_term_length": 48,
        "immutable": true
    }})",
                                           use_reorder,
                                           use_quantization,
                                           window_size,
                                           term_id_limit,
                                           remap_term_ids);

    auto mutable_param = std::make_shared<vsag::SINDIParameter>();
    mutable_param->FromJson(vsag::JsonType::Parse(mutable_param_str));
    auto immutable_param = std::make_shared<vsag::SINDIParameter>();
    immutable_param->FromJson(vsag::JsonType::Parse(immutable_param_str));
    auto mutable_index = std::make_unique<SINDI>(mutable_param, common_param);
    auto immutable_index = std::make_unique<SINDI>(immutable_param, common_param);

    REQUIRE(mutable_index->Build(base).empty());
    REQUIRE(immutable_index->Build(base).empty());
    REQUIRE(immutable_index->GetNumElements() == num_base);

    auto add_data = vsag::Dataset::Make();
    add_data->NumElements(1)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);
    REQUIRE_THROWS_AS(immutable_index->Add(add_data), vsag::VsagException);

    auto immutable_from_immutable = std::make_unique<SINDI>(immutable_param, common_param);
    auto mutable_from_immutable = std::make_unique<SINDI>(mutable_param, common_param);
    REQUIRE_THROWS_AS(test_serializion(*immutable_index, *mutable_from_immutable),
                      vsag::VsagException);
    test_serializion(*immutable_index, *immutable_from_immutable);

    auto immutable_from_mutable = std::make_unique<SINDI>(immutable_param, common_param);
    REQUIRE_THROWS_AS(test_serializion(*mutable_index, *immutable_from_mutable),
                      vsag::VsagException);

    static constexpr auto search_param_str = R"(
    {
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 40,
            "use_term_lists_heap_insert": true
        }
    }
    )";

    auto query = vsag::Dataset::Make();
    for (uint32_t i = 0; i < num_query; ++i) {
        query->NumElements(1)->SparseVectors(sv_base.data() + i)->Owner(false);
        auto expected = mutable_index->KnnSearch(query, 10, search_param_str, nullptr);
        auto built = immutable_index->KnnSearch(query, 10, search_param_str, nullptr);
        auto immutable_roundtrip =
            immutable_from_immutable->KnnSearch(query, 10, search_param_str, nullptr);
        REQUIRE(built->GetDim() == expected->GetDim());
        REQUIRE(immutable_roundtrip->GetDim() == expected->GetDim());
        for (int64_t j = 0; j < expected->GetDim(); ++j) {
            REQUIRE(built->GetIds()[j] == expected->GetIds()[j]);
            REQUIRE(immutable_roundtrip->GetIds()[j] == expected->GetIds()[j]);
            REQUIRE(std::abs(built->GetDistances()[j] - expected->GetDistances()[j]) < 1e-3);
            REQUIRE(std::abs(immutable_roundtrip->GetDistances()[j] - expected->GetDistances()[j]) <
                    1e-3);
        }

        auto expected_range = mutable_index->RangeSearch(query, 0.0F, search_param_str, nullptr);
        auto built_range = immutable_index->RangeSearch(query, 0.0F, search_param_str, nullptr);
        REQUIRE(built_range->GetDim() == expected_range->GetDim());
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDI Remap Basic Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    // Same density as original SINDI test but with large sparse term IDs
    uint32_t num_base = 1000;
    uint32_t num_query = 100;
    int64_t max_dim = 128;
    common_param.dim_ = max_dim;
    int64_t max_id = 30000;  // same as original test for good overlap
    float min_val = 0;
    float max_val = 10;
    int seed_base = 42;
    int64_t k = 10;

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base =
        fixtures::GenerateSparseVectors(num_base, max_dim, max_id, min_val, max_val, seed_base);

    // Shift all term IDs by a large offset to make them sparse in uint32 range
    // This simulates real-world vocabulary IDs that are non-contiguous
    constexpr uint32_t id_offset = 3000000;
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            sv_base[i].ids_[j] += id_offset;
        }
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    // Count unique terms to set term_id_limit
    std::set<uint32_t> unique_terms;
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            unique_terms.insert(sv_base[i].ids_[j]);
        }
    }
    uint32_t term_id_limit = static_cast<uint32_t>(unique_terms.size()) + 3000;

    auto param_str = fmt::format(R"({{
        "use_reorder": true,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "remap_term_ids": true,
        "avg_doc_term_length": 100
    }})",
                                 term_id_limit);

    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);
    auto another_index = std::make_unique<SINDI>(index_param, common_param);

    // Build a brute-force index for ground truth (uses original sparse IDs directly)
    auto exact_param = create_exact_sindi_param(term_id_limit, true);
    auto exact_index = std::make_unique<SINDI>(exact_param, common_param);

    auto exact_build_res = exact_index->Build(base);
    REQUIRE(exact_build_res.size() == 0);
    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);
    REQUIRE(index->GetNumElements() == num_base);

    // test serialize/deserialize
    test_serializion(*index, *another_index);
    REQUIRE(another_index->GetNumElements() == num_base);

    // test search
    std::string search_param_str = R"(
    {
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20
        }
    }
    )";

    auto query = vsag::Dataset::Make();
    for (int i = 0; i < num_query; ++i) {
        query->NumElements(1)->SparseVectors(sv_base.data() + i)->Owner(false);

        auto bf_result = exact_index->KnnSearch(query, k, search_param_str, nullptr);
        auto result = index->KnnSearch(query, k, search_param_str, nullptr);

        REQUIRE(result->GetDim() == bf_result->GetDim());
        for (int j = 0; j < result->GetDim(); j++) {
            REQUIRE(result->GetIds()[j] == bf_result->GetIds()[j]);
            REQUIRE(std::abs(result->GetDistances()[j] - bf_result->GetDistances()[j]) < 3e-3);
        }

        // test serialized index gives same results
        auto another_result = another_index->KnnSearch(query, k, search_param_str, nullptr);
        for (int j = 0; j < another_result->GetDim(); j++) {
            REQUIRE(result->GetIds()[j] == another_result->GetIds()[j]);
        }
    }

    // test unknown query terms
    {
        SparseVector unknown_query;
        uint32_t unknown_ids[] = {1, 2};  // IDs not in [id_offset, id_offset+max_id]
        float unknown_vals[] = {1.0f, 2.0f};
        unknown_query.len_ = 2;
        unknown_query.ids_ = unknown_ids;
        unknown_query.vals_ = unknown_vals;
        query->NumElements(1)->SparseVectors(&unknown_query)->Owner(false);
        auto result = index->KnnSearch(query, k, search_param_str, nullptr);
        REQUIRE(result->GetDim() == 0);
    }

    // test incremental add
    {
        uint32_t num_add = 100;
        std::vector<int64_t> add_ids(num_add);
        for (uint32_t i = 0; i < num_add; ++i) {
            add_ids[i] = num_base + i;
        }
        auto sv_add =
            fixtures::GenerateSparseVectors(num_add, max_dim, max_id, min_val, max_val, 99);
        for (uint32_t i = 0; i < num_add; ++i) {
            for (uint32_t j = 0; j < sv_add[i].len_; ++j) {
                sv_add[i].ids_[j] += id_offset;
            }
        }
        auto add_data = vsag::Dataset::Make();
        add_data->NumElements(num_add)
            ->SparseVectors(sv_add.data())
            ->Ids(add_ids.data())
            ->Owner(false);
        auto add_res = index->Add(add_data);
        REQUIRE(index->GetNumElements() == num_base + num_add);

        query->NumElements(1)->SparseVectors(sv_add.data())->Owner(false);
        auto result = index->KnnSearch(query, k, search_param_str, nullptr);
        REQUIRE(result->GetDim() == k);

        for (auto& item : sv_add) {
            delete[] item.vals_;
            delete[] item.ids_;
        }
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDI Remap with Reorder Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    uint32_t num_base = 1000;
    uint32_t num_query = 100;
    int64_t max_dim = 128;
    common_param.dim_ = max_dim;
    int64_t max_id = 30000;
    float min_val = 0;
    float max_val = 10;
    int seed_base = 77;
    int64_t k = 10;
    constexpr uint32_t id_offset = 2000000;

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base =
        fixtures::GenerateSparseVectors(num_base, max_dim, max_id, min_val, max_val, seed_base);
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            sv_base[i].ids_[j] += id_offset;
        }
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    std::set<uint32_t> unique_terms;
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            unique_terms.insert(sv_base[i].ids_[j]);
        }
    }
    uint32_t term_id_limit = static_cast<uint32_t>(unique_terms.size()) + 100;

    auto param_str = fmt::format(R"({{
        "use_reorder": true,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "remap_term_ids": true,
        "avg_doc_term_length": 100
    }})",
                                 term_id_limit);

    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);

    auto exact_param = create_exact_sindi_param(term_id_limit, true);
    auto exact_index = std::make_unique<SINDI>(exact_param, common_param);

    auto exact_build_res = exact_index->Build(base);
    REQUIRE(exact_build_res.size() == 0);
    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);

    std::string search_param_str = R"(
    {
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20
        }
    }
    )";

    auto query = vsag::Dataset::Make();
    for (int i = 0; i < num_query; ++i) {
        query->NumElements(1)->SparseVectors(sv_base.data() + i)->Owner(false);

        auto bf_result = exact_index->KnnSearch(query, k, search_param_str, nullptr);
        auto result = index->KnnSearch(query, k, search_param_str, nullptr);

        REQUIRE(result->GetDim() == bf_result->GetDim());
        for (int j = 0; j < result->GetDim(); j++) {
            REQUIRE(result->GetIds()[j] == bf_result->GetIds()[j]);
            REQUIRE(std::abs(result->GetDistances()[j] - bf_result->GetDistances()[j]) < 3e-3);
        }
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDI Remap Term ID Limit Exceeded", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    // Use small max_id so first doc has reasonable unique term count
    common_param.dim_ = 10;
    auto sv_base = fixtures::GenerateSparseVectors(10, 10, 50, 0, 10, 123);

    // Count unique terms in first doc to set a limit that allows first doc but not all
    std::set<uint32_t> first_doc_terms;
    for (uint32_t j = 0; j < sv_base[0].len_; ++j) {
        first_doc_terms.insert(sv_base[0].ids_[j]);
    }
    uint32_t term_id_limit = static_cast<uint32_t>(first_doc_terms.size()) + 2;

    auto param_str = fmt::format(R"({{
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "remap_term_ids": true,
        "avg_doc_term_length": 10
    }})",
                                 term_id_limit);

    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);

    std::vector<int64_t> ids(10);
    for (int64_t i = 0; i < 10; ++i) {
        ids[i] = i;
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(10)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    auto failed = index->Build(base);
    REQUIRE(failed.size() > 0);
    REQUIRE(index->GetNumElements() > 0);

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDI Remap with Quantization Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    uint32_t num_base = 1000;
    uint32_t num_query = 100;
    int64_t max_dim = 128;
    common_param.dim_ = max_dim;
    int64_t max_id = 30000;
    float min_val = 0;
    float max_val = 10;
    int seed_base = 55;
    int64_t k = 10;
    constexpr uint32_t id_offset = 2000000;

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base =
        fixtures::GenerateSparseVectors(num_base, max_dim, max_id, min_val, max_val, seed_base);
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            sv_base[i].ids_[j] += id_offset;
        }
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    std::set<uint32_t> unique_terms;
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            unique_terms.insert(sv_base[i].ids_[j]);
        }
    }
    uint32_t term_id_limit = static_cast<uint32_t>(unique_terms.size()) + 100;

    auto param_str = fmt::format(R"({{
        "use_reorder": true,
        "use_quantization": true,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "remap_term_ids": true,
        "avg_doc_term_length": 100
    }})",
                                 term_id_limit);

    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);

    auto exact_param = create_exact_sindi_param(term_id_limit, true);
    auto exact_index = std::make_unique<SINDI>(exact_param, common_param);

    auto exact_build_res = exact_index->Build(base);
    REQUIRE(exact_build_res.size() == 0);
    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);

    std::string search_param_str = R"(
    {
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20
        }
    }
    )";

    auto query = vsag::Dataset::Make();
    int64_t correct_count = 0;
    for (int i = 0; i < num_query; ++i) {
        query->NumElements(1)->SparseVectors(sv_base.data() + i)->Owner(false);

        auto bf_result = exact_index->KnnSearch(query, k, search_param_str, nullptr);
        auto result = index->KnnSearch(query, k, search_param_str, nullptr);

        REQUIRE(result->GetDim() == bf_result->GetDim());

        std::unordered_set<int64_t> gt_ids;
        for (int j = 0; j < k; j++) {
            gt_ids.insert(bf_result->GetIds()[j]);
        }
        for (int j = 0; j < k; j++) {
            if (gt_ids.find(result->GetIds()[j]) != gt_ids.end()) {
                correct_count++;
            }
        }
    }

    float recall = static_cast<float>(correct_count) / (num_query * k);
    REQUIRE(recall > 0.95);

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDI Remap with Filter Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    uint32_t num_base = 1000;
    uint32_t num_query = 100;
    int64_t max_dim = 128;
    common_param.dim_ = max_dim;
    int64_t max_id = 30000;
    float min_val = 0;
    float max_val = 10;
    int seed_base = 66;
    int64_t k = 10;
    constexpr uint32_t id_offset = 2000000;

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base =
        fixtures::GenerateSparseVectors(num_base, max_dim, max_id, min_val, max_val, seed_base);
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            sv_base[i].ids_[j] += id_offset;
        }
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    std::set<uint32_t> unique_terms;
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            unique_terms.insert(sv_base[i].ids_[j]);
        }
    }
    uint32_t term_id_limit = static_cast<uint32_t>(unique_terms.size()) + 100;

    auto param_str = fmt::format(R"({{
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "remap_term_ids": true,
        "avg_doc_term_length": 100
    }})",
                                 term_id_limit);

    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);

    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);

    std::string search_param_str = R"(
    {
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20
        }
    }
    )";

    auto mock_filter = std::make_shared<MockFilter>();
    auto query = vsag::Dataset::Make();

    for (int i = 0; i < num_query; ++i) {
        query->NumElements(1)->SparseVectors(sv_base.data() + i)->Owner(false);

        auto result = index->KnnSearch(query, k, search_param_str, nullptr);
        auto filter_result = index->KnnSearch(query, k, search_param_str, mock_filter);

        REQUIRE(filter_result->GetDim() == k);
        for (int j = 0; j < filter_result->GetDim(); j++) {
            REQUIRE(mock_filter->CheckValid(filter_result->GetIds()[j]));
        }

        auto cur = 0;
        for (int j = 0; j < k && cur < filter_result->GetDim(); j++) {
            if (mock_filter->CheckValid(result->GetIds()[j])) {
                REQUIRE(result->GetIds()[j] == filter_result->GetIds()[cur]);
                cur++;
            }
        }
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDI Remap GetSparseVectorByInnerId Reverse Mapping", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    uint32_t num_base = 50;
    int64_t max_dim = 32;
    common_param.dim_ = max_dim;
    int64_t max_id = 5000;
    float min_val = 0;
    float max_val = 10;
    int seed_base = 88;
    constexpr uint32_t id_offset = 4000000;

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base =
        fixtures::GenerateSparseVectors(num_base, max_dim, max_id, min_val, max_val, seed_base);
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            sv_base[i].ids_[j] += id_offset;
        }
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    std::set<uint32_t> unique_terms;
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            unique_terms.insert(sv_base[i].ids_[j]);
        }
    }
    uint32_t term_id_limit = static_cast<uint32_t>(unique_terms.size()) + 100;

    auto param_str = fmt::format(R"({{
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "remap_term_ids": true,
        "avg_doc_term_length": 32
    }})",
                                 term_id_limit);

    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);

    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);

    for (uint32_t i = 0; i < num_base; ++i) {
        SparseVector retrieved;
        index->GetSparseVectorByInnerId(i, &retrieved, allocator.get());

        std::set<uint32_t> original_ids;
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            original_ids.insert(sv_base[i].ids_[j]);
        }

        for (uint32_t j = 0; j < retrieved.len_; ++j) {
            REQUIRE(original_ids.count(retrieved.ids_[j]) > 0);
        }

        allocator->Deallocate(retrieved.ids_);
        allocator->Deallocate(retrieved.vals_);
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDI Remap UpdateVector Compatibility", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    uint32_t num_base = 50;
    int64_t max_dim = 32;
    common_param.dim_ = max_dim;
    int64_t max_id = 5000;
    float min_val = 0;
    float max_val = 10;
    int seed_base = 33;
    constexpr uint32_t id_offset = 4000000;

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base =
        fixtures::GenerateSparseVectors(num_base, max_dim, max_id, min_val, max_val, seed_base);
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            sv_base[i].ids_[j] += id_offset;
        }
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    std::set<uint32_t> unique_terms;
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            unique_terms.insert(sv_base[i].ids_[j]);
        }
    }
    uint32_t term_id_limit = static_cast<uint32_t>(unique_terms.size()) + 100;

    auto param_str = fmt::format(R"({{
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "remap_term_ids": true,
        "avg_doc_term_length": 32
    }})",
                                 term_id_limit);

    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);

    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);

    for (uint32_t i = 0; i < std::min(num_base, 10u); ++i) {
        auto update_data = vsag::Dataset::Make();
        update_data->NumElements(1)->SparseVectors(sv_base.data() + i)->Owner(false);
        bool result = index->UpdateVector(ids[i], update_data);
        REQUIRE(result == true);
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDI Remap Memory Comparison", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    // Generate data with dense overlap but large sparse term IDs
    uint32_t num_base = 500;
    int64_t max_dim = 64;
    common_param.dim_ = max_dim;
    int64_t max_id = 30000;
    float min_val = 0;
    float max_val = 10;
    int seed_base = 42;
    constexpr uint32_t id_offset = 5000000;  // shift IDs to simulate sparse vocab

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base =
        fixtures::GenerateSparseVectors(num_base, max_dim, max_id, min_val, max_val, seed_base);

    // Make a copy before shifting (for the no-remap index)
    std::vector<SparseVector> sv_base_shifted(num_base);
    for (uint32_t i = 0; i < num_base; ++i) {
        sv_base_shifted[i].len_ = sv_base[i].len_;
        sv_base_shifted[i].ids_ = new uint32_t[sv_base[i].len_];
        sv_base_shifted[i].vals_ = new float[sv_base[i].len_];
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            sv_base_shifted[i].ids_[j] = sv_base[i].ids_[j] + id_offset;
            sv_base_shifted[i].vals_[j] = sv_base[i].vals_[j];
        }
    }

    // Count unique terms
    std::set<uint32_t> unique_terms;
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base_shifted[i].len_; ++j) {
            unique_terms.insert(sv_base_shifted[i].ids_[j]);
        }
    }
    uint32_t unique_count = static_cast<uint32_t>(unique_terms.size());

    // Index WITHOUT remap: needs term_id_limit >= max_shifted_id
    uint32_t no_remap_limit = id_offset + max_id + 1;  // ~5030001
    auto no_remap_param_str = fmt::format(R"({{
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "remap_term_ids": false,
        "avg_doc_term_length": 64
    }})",
                                          no_remap_limit);

    auto no_remap_json = vsag::JsonType::Parse(no_remap_param_str);
    auto no_remap_param = std::make_shared<vsag::SINDIParameter>();
    no_remap_param->FromJson(no_remap_json);
    auto no_remap_index = std::make_unique<SINDI>(no_remap_param, common_param);

    auto base_no_remap = vsag::Dataset::Make();
    base_no_remap->NumElements(num_base)
        ->SparseVectors(sv_base_shifted.data())
        ->Ids(ids.data())
        ->Owner(false);
    auto res1 = no_remap_index->Build(base_no_remap);
    REQUIRE(res1.size() == 0);

    // Index WITH remap: term_id_limit = unique terms + headroom
    uint32_t remap_limit = unique_count + 100;
    auto remap_param_str = fmt::format(R"({{
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "remap_term_ids": true,
        "avg_doc_term_length": 64
    }})",
                                       remap_limit);

    auto remap_json = vsag::JsonType::Parse(remap_param_str);
    auto remap_param = std::make_shared<vsag::SINDIParameter>();
    remap_param->FromJson(remap_json);
    auto remap_index = std::make_unique<SINDI>(remap_param, common_param);

    auto base_remap = vsag::Dataset::Make();
    base_remap->NumElements(num_base)
        ->SparseVectors(sv_base_shifted.data())
        ->Ids(ids.data())
        ->Owner(false);
    auto res2 = remap_index->Build(base_remap);
    REQUIRE(res2.size() == 0);

    // Compare memory usage
    auto mem_no_remap = no_remap_index->EstimateMemory(num_base);
    auto mem_remap = remap_index->EstimateMemory(num_base);

    // Remap should use significantly less memory
    // no_remap: ~5M slots × 20B = ~100MB overhead
    // remap: ~30K slots × 20B + mapper = ~2MB overhead
    REQUIRE(mem_remap < mem_no_remap);
    float savings_ratio = 1.0f - static_cast<float>(mem_remap) / static_cast<float>(mem_no_remap);
    WARN("Memory comparison: no_remap=" << mem_no_remap << " remap=" << mem_remap << " savings="
                                        << savings_ratio << " unique_terms=" << unique_count);
    REQUIRE(savings_ratio > 0.9f);  // at least 90% memory reduction

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    for (auto& item : sv_base_shifted) {
        delete[] item.ids_;
        delete[] item.vals_;
    }
}

TEST_CASE("SINDI Remap Memory Comparison - MD5 Vocabulary", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    // Simulate MD5 hash-based tokenizer: term IDs scattered across uint32 range
    // Actual unique terms ~5M, but raw IDs could be anywhere in [0, 2^32)
    // Without remap: term_id_limit must be >= max_raw_id (impossible if > 10M)
    // With remap: term_id_limit = 5M (fits within 10M limit)

    // We can't actually test with 5M terms (too slow in QEMU), so we use
    // a scaled-down version that demonstrates the same principle:
    // 50K unique terms with raw IDs scattered in [0, 10M) range
    uint32_t num_base = 500;
    int64_t max_dim = 64;
    common_param.dim_ = max_dim;
    int64_t max_id = 10000;  // base range for generation
    float min_val = 0;
    float max_val = 10;
    int seed_base = 77;

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base =
        fixtures::GenerateSparseVectors(num_base, max_dim, max_id, min_val, max_val, seed_base);

    // Simulate MD5: scatter term IDs across a large range using a hash-like transform
    std::mt19937 rng(12345);
    std::unordered_map<uint32_t, uint32_t> id_scatter;
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            uint32_t orig = sv_base[i].ids_[j];
            if (id_scatter.find(orig) == id_scatter.end()) {
                // Map to a random ID in [0, 9999999] (simulating MD5 spread)
                id_scatter[orig] = rng() % 10000000;
            }
            sv_base[i].ids_[j] = id_scatter[orig];
        }
    }

    // Find max scattered ID
    uint32_t max_scattered_id = 0;
    std::set<uint32_t> unique_terms;
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            unique_terms.insert(sv_base[i].ids_[j]);
            max_scattered_id = std::max(max_scattered_id, sv_base[i].ids_[j]);
        }
    }
    uint32_t unique_count = static_cast<uint32_t>(unique_terms.size());

    // Without remap: needs term_id_limit >= max_scattered_id + 1
    uint32_t no_remap_limit = max_scattered_id + 1;

    // With remap: only needs unique_count
    uint32_t remap_limit = unique_count + 100;

    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    // Build without remap
    auto no_remap_param_str = fmt::format(R"({{
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "remap_term_ids": false,
        "avg_doc_term_length": 64
    }})",
                                          no_remap_limit);

    auto no_remap_json = vsag::JsonType::Parse(no_remap_param_str);
    auto no_remap_param = std::make_shared<vsag::SINDIParameter>();
    no_remap_param->FromJson(no_remap_json);
    auto no_remap_index = std::make_unique<SINDI>(no_remap_param, common_param);
    auto res1 = no_remap_index->Build(base);
    REQUIRE(res1.size() == 0);

    // Build with remap
    auto remap_param_str = fmt::format(R"({{
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "remap_term_ids": true,
        "avg_doc_term_length": 64
    }})",
                                       remap_limit);

    auto remap_json = vsag::JsonType::Parse(remap_param_str);
    auto remap_param = std::make_shared<vsag::SINDIParameter>();
    remap_param->FromJson(remap_json);
    auto remap_index = std::make_unique<SINDI>(remap_param, common_param);
    auto res2 = remap_index->Build(base);
    REQUIRE(res2.size() == 0);

    // Compare memory
    auto mem_no_remap = no_remap_index->EstimateMemory(num_base);
    auto mem_remap = remap_index->EstimateMemory(num_base);
    float savings_ratio = 1.0f - static_cast<float>(mem_remap) / static_cast<float>(mem_no_remap);
    WARN("MD5 vocab comparison: no_remap=" << mem_no_remap << " remap=" << mem_remap << " savings="
                                           << savings_ratio << " unique_terms=" << unique_count
                                           << " max_id=" << max_scattered_id);

    REQUIRE(mem_remap < mem_no_remap);
    REQUIRE(savings_ratio > 0.9f);

    // Verify search still works with remap
    std::string search_param_str = R"(
    {
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20
        }
    }
    )";

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(sv_base.data())->Owner(false);
    auto result = remap_index->KnnSearch(query, 5, search_param_str, nullptr);
    REQUIRE(result->GetDim() > 0);

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDI Proximity Basic Test", "[ut][SINDI][Proximity]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    // Create controlled data: 3 docs, same terms, different positions
    // Doc 0: terms [10, 20, 30] clustered at positions [0, 1, 2] → high proximity
    // Doc 1: terms [10, 20, 30] scattered at positions [0, 50, 100] → low proximity
    // Doc 2: terms [10, 20] at positions [0, 1] → partial match, high proximity
    uint32_t num_base = 3;

    SparseVector sv_base_prox[3];

    uint32_t ids0[] = {10, 20, 30};
    float vals0[] = {0.3f, 0.3f, 0.3f};
    uint32_t seq0[] = {10, 20, 30};
    sv_base_prox[0] = {3, ids0, vals0, 3, seq0};

    uint32_t ids1[] = {10, 20, 30};
    float vals1[] = {0.3f, 0.3f, 0.3f};
    std::vector<uint32_t> seq1_vec(101, 99);
    seq1_vec[0] = 10;
    seq1_vec[50] = 20;
    seq1_vec[100] = 30;
    sv_base_prox[1] = {3, ids1, vals1, 101, seq1_vec.data()};

    uint32_t ids2[] = {10, 20};
    float vals2[] = {0.3f, 0.3f};
    uint32_t seq2[] = {10, 20};
    sv_base_prox[2] = {2, ids2, vals2, 2, seq2};

    std::vector<int64_t> base_ids = {0, 1, 2};
    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base_prox)->Ids(base_ids.data())->Owner(false);

    auto param_str_prox = R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 200,
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 10
    })";

    vsag::JsonType param_json = vsag::JsonType::Parse(param_str_prox);
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);

    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);
    REQUIRE(index->GetNumElements() == 3);

    // Query: terms [10, 20, 30]
    uint32_t q_ids[] = {10, 20, 30};
    float q_vals[] = {0.3f, 0.3f, 0.3f};
    SparseVector query_sv = {3, q_ids, q_vals, 0, nullptr};
    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->SparseVectors(&query_sv)->Owner(false);

    // Search WITHOUT proximity boost
    std::string search_no_prox = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 10,
            "proximity_boost": {
                "weight": 0.0
            }
        }
    })";
    auto result_no_prox = index->KnnSearch(query_ds, 3, search_no_prox, nullptr);
    REQUIRE(result_no_prox->GetDim() == 3);
    // Doc 0 and Doc 1 have same IP score (both have terms 10,20,30 with weight 0.3)
    // IP = 0.3*0.3 * 3 = 0.27, distance = 1 - 0.27 = 0.73
    REQUIRE(result_no_prox->GetDistances()[0] == result_no_prox->GetDistances()[1]);

    // Search WITH proximity boost
    std::string search_with_prox = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 10,
            "proximity_boost": {
                "weight": 0.5,
                "ordered": false,
                "candidates": 10000
            }
        }
    })";
    auto result_prox = index->KnnSearch(query_ds, 3, search_with_prox, nullptr);
    REQUIRE(result_prox->GetDim() == 3);
    // Doc 0 (clustered) should rank before Doc 1 (scattered)
    REQUIRE(result_prox->GetIds()[0] == 0);
    REQUIRE(result_prox->GetDistances()[0] < result_prox->GetDistances()[1]);
}

TEST_CASE("SINDI Proximity Disabled Regression", "[ut][SINDI][Proximity]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    uint32_t num_base = 100;
    int64_t max_dim = 32;
    int64_t max_id = 500;
    int64_t k = 10;

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base_reg = fixtures::GenerateSparseVectors(num_base, max_dim, max_id, 0, 10, 42);

    // Add token sequences
    std::mt19937 rng(42);
    std::vector<std::vector<uint32_t>> token_seqs(num_base);
    for (uint32_t i = 0; i < num_base; ++i) {
        uint32_t seq_len = 50 + rng() % 100;
        token_seqs[i].resize(seq_len);
        for (uint32_t j = 0; j < seq_len; ++j) {
            token_seqs[i][j] = rng() % max_id;
        }
        sv_base_reg[i].token_seq_len_ = seq_len;
        sv_base_reg[i].token_sequence_ = token_seqs[i].data();
    }

    auto base_reg = vsag::Dataset::Make();
    base_reg->NumElements(num_base)
        ->SparseVectors(sv_base_reg.data())
        ->Ids(ids.data())
        ->Owner(false);

    auto param_str_reg = R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 501,
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 32
    })";

    vsag::JsonType param_json_reg = vsag::JsonType::Parse(param_str_reg);
    auto index_param_reg = std::make_shared<vsag::SINDIParameter>();
    index_param_reg->FromJson(param_json_reg);
    auto index_reg = std::make_unique<SINDI>(index_param_reg, common_param);
    auto build_res_reg = index_reg->Build(base_reg);
    REQUIRE(build_res_reg.size() == 0);

    // proximity_weight=0 should be deterministic and consistent
    std::string search_no_prox_reg = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20,
            "proximity_boost": {
                "weight": 0.0
            }
        }
    })";

    auto query_reg = vsag::Dataset::Make();
    for (int i = 0; i < 10; ++i) {
        query_reg->NumElements(1)->SparseVectors(sv_base_reg.data() + i)->Owner(false);
        auto result1 = index_reg->KnnSearch(query_reg, k, search_no_prox_reg, nullptr);
        auto result2 = index_reg->KnnSearch(query_reg, k, search_no_prox_reg, nullptr);
        REQUIRE(result1->GetDim() == result2->GetDim());
        for (int j = 0; j < result1->GetDim(); ++j) {
            REQUIRE(result1->GetIds()[j] == result2->GetIds()[j]);
        }
    }

    for (auto& item : sv_base_reg) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDI Proximity Score Verification", "[ut][SINDI][Proximity]") {
    // Verify exact scores for both multiplicative and additive modes
    // Use small weights so distance falls in [0, 1]
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    // 2 docs, both contain [10, 20, 30] with weight 0.3
    // Doc 0: clustered at [0, 1, 2]
    // Doc 1: scattered at [0, 50, 100]
    SparseVector sv_docs[2];
    uint32_t ids0[] = {10, 20, 30};
    float vals0[] = {0.3f, 0.3f, 0.3f};
    uint32_t seq0[] = {10, 20, 30};
    sv_docs[0] = {3, ids0, vals0, 3, seq0};

    uint32_t ids1[] = {10, 20, 30};
    float vals1[] = {0.3f, 0.3f, 0.3f};
    std::vector<uint32_t> seq1_vec(101, 99);
    seq1_vec[0] = 10;
    seq1_vec[50] = 20;
    seq1_vec[100] = 30;
    sv_docs[1] = {3, ids1, vals1, 101, seq1_vec.data()};

    std::vector<int64_t> base_ids = {0, 1};
    auto base = vsag::Dataset::Make();
    base->NumElements(2)->SparseVectors(sv_docs)->Ids(base_ids.data())->Owner(false);

    auto param_str = R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 200,
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 10
    })";

    vsag::JsonType pj = vsag::JsonType::Parse(param_str);
    auto ip = std::make_shared<vsag::SINDIParameter>();
    ip->FromJson(pj);
    auto index = std::make_unique<SINDI>(ip, common_param);
    index->Build(base);

    uint32_t q_ids[] = {10, 20, 30};
    float q_vals[] = {0.3f, 0.3f, 0.3f};
    SparseVector qsv = {3, q_ids, q_vals, 0, nullptr};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(&qsv)->Owner(false);

    // IP score for both docs: 0.3*0.3 + 0.3*0.3 + 0.3*0.3 = 0.27
    // Without proximity: distance = 1 - 0.27 = 0.73
    float ip_score = 0.27f;

    // All-pairs proximity (proximity_all_pairs=true): score all C(3,2)=3 pairs.
    // Doc 0: (10,20) dist=1→1/2, (10,30) dist=2→1/3, (20,30) dist=1→1/2
    float raw_boost_0_ap = 1.0f / 2.0f + 1.0f / 3.0f + 1.0f / 2.0f;
    // Doc 1: (10,20) dist=50→1/51, (10,30) dist=100→1/101, (20,30) dist=50→1/51
    float raw_boost_1_ap = 1.0f / 51.0f + 1.0f / 101.0f + 1.0f / 51.0f;
    float pair_count_ap = 3.0f;  // C(3,2)
    float norm_boost_0_ap = raw_boost_0_ap / pair_count_ap;
    float norm_boost_1_ap = raw_boost_1_ap / pair_count_ap;

    // Adjacent-only proximity (proximity_all_pairs=false, the default): score
    // only query-adjacent pairs (10,20) and (20,30) → n-1=2 pairs.
    float raw_boost_0_adj = 1.0f / 2.0f + 1.0f / 2.0f;
    float raw_boost_1_adj = 1.0f / 51.0f + 1.0f / 51.0f;
    float pair_count_adj = 2.0f;  // n-1
    float norm_boost_0_adj = raw_boost_0_adj / pair_count_adj;
    float norm_boost_1_adj = raw_boost_1_adj / pair_count_adj;

    float beta = 0.3f;

    SECTION("multiplicative mode, all-pairs") {
        std::string sp = R"({
            "sindi": {
                "query_prune_ratio": 0.0,
                "term_prune_ratio": 0.0,
                "n_candidate": 10,
                "proximity_boost": {
                    "weight": 0.3,
                    "boost_multiplicative": true,
                    "all_pairs": true
                }
            }
        })";
        auto result = index->KnnSearch(query, 2, sp, nullptr);
        // dists[doc] = -0.27 (negative IP)
        // multiplicative: dists *= (1 + beta * norm_boost)
        // final_distance = 1 + dists_after
        float expected_0 = 1.0f + (-ip_score) * (1.0f + beta * norm_boost_0_ap);
        float expected_1 = 1.0f + (-ip_score) * (1.0f + beta * norm_boost_1_ap);
        REQUIRE(result->GetIds()[0] == 0);
        REQUIRE(result->GetDistances()[0] > 0.0f);  // distance in [0,1]
        REQUIRE(result->GetDistances()[0] < 1.0f);
        REQUIRE(result->GetDistances()[1] > 0.0f);
        REQUIRE(result->GetDistances()[1] < 1.0f);
        REQUIRE(std::abs(result->GetDistances()[0] - expected_0) < 1e-4f);
        REQUIRE(std::abs(result->GetDistances()[1] - expected_1) < 1e-4f);
    }

    SECTION("additive mode, all-pairs") {
        std::string sp = R"({
            "sindi": {
                "query_prune_ratio": 0.0,
                "term_prune_ratio": 0.0,
                "n_candidate": 10,
                "proximity_boost": {
                    "weight": 0.3,
                    "boost_multiplicative": false,
                    "all_pairs": true
                }
            }
        })";
        auto result = index->KnnSearch(query, 2, sp, nullptr);
        // additive: dists -= beta * norm_boost
        // final_distance = 1 + (dists - beta * norm_boost) = 1 - ip - beta * norm_boost
        float expected_0 = 1.0f - ip_score - beta * norm_boost_0_ap;
        float expected_1 = 1.0f - ip_score - beta * norm_boost_1_ap;
        REQUIRE(result->GetIds()[0] == 0);
        REQUIRE(result->GetDistances()[0] > 0.0f);
        REQUIRE(result->GetDistances()[0] < 1.0f);
        REQUIRE(result->GetDistances()[1] > 0.0f);
        REQUIRE(result->GetDistances()[1] < 1.0f);
        REQUIRE(std::abs(result->GetDistances()[0] - expected_0) < 1e-4f);
        REQUIRE(std::abs(result->GetDistances()[1] - expected_1) < 1e-4f);
    }

    SECTION("multiplicative mode, adjacent-only (default)") {
        std::string sp = R"({
            "sindi": {
                "query_prune_ratio": 0.0,
                "term_prune_ratio": 0.0,
                "n_candidate": 10,
                "proximity_boost": {
                    "weight": 0.3,
                    "boost_multiplicative": true
                }
            }
        })";
        auto result = index->KnnSearch(query, 2, sp, nullptr);
        float expected_0 = 1.0f + (-ip_score) * (1.0f + beta * norm_boost_0_adj);
        float expected_1 = 1.0f + (-ip_score) * (1.0f + beta * norm_boost_1_adj);
        REQUIRE(result->GetIds()[0] == 0);
        REQUIRE(result->GetDistances()[0] > 0.0f);
        REQUIRE(result->GetDistances()[0] < 1.0f);
        REQUIRE(result->GetDistances()[1] > 0.0f);
        REQUIRE(result->GetDistances()[1] < 1.0f);
        REQUIRE(std::abs(result->GetDistances()[0] - expected_0) < 1e-4f);
        REQUIRE(std::abs(result->GetDistances()[1] - expected_1) < 1e-4f);
    }

    SECTION("additive mode, adjacent-only (default)") {
        std::string sp = R"({
            "sindi": {
                "query_prune_ratio": 0.0,
                "term_prune_ratio": 0.0,
                "n_candidate": 10,
                "proximity_boost": {
                    "weight": 0.3,
                    "boost_multiplicative": false
                }
            }
        })";
        auto result = index->KnnSearch(query, 2, sp, nullptr);
        float expected_0 = 1.0f - ip_score - beta * norm_boost_0_adj;
        float expected_1 = 1.0f - ip_score - beta * norm_boost_1_adj;
        REQUIRE(result->GetIds()[0] == 0);
        REQUIRE(result->GetDistances()[0] > 0.0f);
        REQUIRE(result->GetDistances()[0] < 1.0f);
        REQUIRE(result->GetDistances()[1] > 0.0f);
        REQUIRE(result->GetDistances()[1] < 1.0f);
        REQUIRE(std::abs(result->GetDistances()[0] - expected_0) < 1e-4f);
        REQUIRE(std::abs(result->GetDistances()[1] - expected_1) < 1e-4f);
    }
}

TEST_CASE("SINDI Proximity Additive Mode", "[ut][SINDI][Proximity]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    // Same setup as basic test with small weights
    SparseVector sv_docs[2];
    uint32_t ids0[] = {10, 20, 30};
    float vals0[] = {0.3f, 0.3f, 0.3f};
    uint32_t seq0[] = {10, 20, 30};
    sv_docs[0] = {3, ids0, vals0, 3, seq0};

    uint32_t ids1[] = {10, 20, 30};
    float vals1[] = {0.3f, 0.3f, 0.3f};
    std::vector<uint32_t> seq1_vec(101, 99);
    seq1_vec[0] = 10;
    seq1_vec[50] = 20;
    seq1_vec[100] = 30;
    sv_docs[1] = {3, ids1, vals1, 101, seq1_vec.data()};

    std::vector<int64_t> base_ids = {0, 1};
    auto base = vsag::Dataset::Make();
    base->NumElements(2)->SparseVectors(sv_docs)->Ids(base_ids.data())->Owner(false);

    auto param_str = R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 200,
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 10
    })";

    vsag::JsonType pj = vsag::JsonType::Parse(param_str);
    auto ip_param = std::make_shared<vsag::SINDIParameter>();
    ip_param->FromJson(pj);
    auto index = std::make_unique<SINDI>(ip_param, common_param);
    index->Build(base);

    uint32_t q_ids[] = {10, 20, 30};
    float q_vals[] = {0.3f, 0.3f, 0.3f};
    SparseVector qsv = {3, q_ids, q_vals, 0, nullptr};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(&qsv)->Owner(false);

    // Additive mode
    std::string sp = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 10,
            "proximity_boost": {
                "weight": 0.3,
                "boost_multiplicative": false,
                "candidates": 10000
            }
        }
    })";
    auto result = index->KnnSearch(query, 2, sp, nullptr);
    REQUIRE(result->GetIds()[0] == 0);
    REQUIRE(result->GetDistances()[0] < result->GetDistances()[1]);
    // Distances should be in [0, 1]
    REQUIRE(result->GetDistances()[0] > 0.0f);
    REQUIRE(result->GetDistances()[0] < 1.0f);
    REQUIRE(result->GetDistances()[1] > 0.0f);
    REQUIRE(result->GetDistances()[1] < 1.0f);
}

TEST_CASE("SINDI Proximity Serialize Round-Trip", "[ut][SINDI][Proximity]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    SparseVector sv_docs[2];
    uint32_t ids0[] = {10, 20, 30};
    float vals0[] = {1.0f, 1.0f, 1.0f};
    uint32_t seq0[] = {10, 20, 30};
    sv_docs[0] = {3, ids0, vals0, 3, seq0};

    uint32_t ids1[] = {10, 20, 30};
    float vals1[] = {1.0f, 1.0f, 1.0f};
    std::vector<uint32_t> seq1_vec(101, 99);
    seq1_vec[0] = 10;
    seq1_vec[50] = 20;
    seq1_vec[100] = 30;
    sv_docs[1] = {3, ids1, vals1, 101, seq1_vec.data()};

    std::vector<int64_t> base_ids = {0, 1};
    auto base = vsag::Dataset::Make();
    base->NumElements(2)->SparseVectors(sv_docs)->Ids(base_ids.data())->Owner(false);

    auto param_str = R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 200,
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 10
    })";

    vsag::JsonType pj = vsag::JsonType::Parse(param_str);
    auto ip1 = std::make_shared<vsag::SINDIParameter>();
    ip1->FromJson(pj);
    auto index1 = std::make_unique<SINDI>(ip1, common_param);
    index1->Build(base);

    // Search before serialize
    uint32_t q_ids[] = {10, 20, 30};
    float q_vals[] = {1.0f, 1.0f, 1.0f};
    SparseVector qsv = {3, q_ids, q_vals, 0, nullptr};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(&qsv)->Owner(false);

    std::string sp = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 10,
            "proximity_boost": {
                "weight": 0.5,
                "boost_multiplicative": true
            }
        }
    })";
    auto result_before = index1->KnnSearch(query, 2, sp, nullptr);

    // Serialize → Deserialize
    auto ip2 = std::make_shared<vsag::SINDIParameter>();
    ip2->FromJson(pj);
    auto index2 = std::make_unique<SINDI>(ip2, common_param);
    test_serializion(*index1, *index2);

    // Search after deserialize — results should match
    auto result_after = index2->KnnSearch(query, 2, sp, nullptr);
    REQUIRE(result_after->GetDim() == result_before->GetDim());
    for (int j = 0; j < result_after->GetDim(); ++j) {
        REQUIRE(result_after->GetIds()[j] == result_before->GetIds()[j]);
        REQUIRE(std::abs(result_after->GetDistances()[j] - result_before->GetDistances()[j]) <
                1e-6f);
    }
}

TEST_CASE("SINDI Proximity Streaming Serialize Round-Trip",
          "[ut][SINDI][Proximity][PhraseFilter][streaming]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    const bool remap_term_ids = GENERATE(false, true);
    constexpr uint32_t term_offset_remap = 5000000;
    const uint32_t term_offset = remap_term_ids ? term_offset_remap : 0;
    const uint32_t t10 = term_offset + 10;
    const uint32_t t20 = term_offset + 20;
    const uint32_t t30 = term_offset + 30;

    SparseVector sv_docs[3];
    uint32_t ids0[] = {t10, t20, t30};
    float vals0[] = {1.0f, 1.0f, 1.0f};
    uint32_t seq0[] = {t10, t20, t30};
    sv_docs[0] = {3, ids0, vals0, 3, seq0};

    uint32_t ids1[] = {t10, t20, t30};
    float vals1[] = {0.7f, 0.7f, 0.7f};
    std::vector<uint32_t> seq1_vec(101, term_offset + 99);
    seq1_vec[0] = t10;
    seq1_vec[50] = t20;
    seq1_vec[100] = t30;
    sv_docs[1] = {3, ids1, vals1, 101, seq1_vec.data()};

    uint32_t ids2[] = {t10, t20};
    float vals2[] = {0.8f, 0.8f};
    uint32_t seq2[] = {t10, t20};
    sv_docs[2] = {2, ids2, vals2, 2, seq2};

    std::vector<int64_t> base_ids = {0, 1, 2};
    auto base = Dataset::Make();
    base->NumElements(3)->SparseVectors(sv_docs)->Ids(base_ids.data())->Owner(false);

    auto param_str = fmt::format(R"({{
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 200,
        "remap_term_ids": {},
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 10
    }})",
                                 remap_term_ids);
    auto param_json = JsonType::Parse(param_str);
    auto param = std::make_shared<SINDIParameter>();
    param->FromJson(param_json);

    auto index = std::make_unique<SINDI>(param, common_param);
    REQUIRE(index->Build(base).empty());

    uint32_t q_ids[] = {t10, t20, t30};
    float q_vals[] = {1.0f, 1.0f, 1.0f};
    SparseVector qsv = {3, q_ids, q_vals, 0, nullptr};
    auto query = Dataset::Make();
    query->NumElements(1)->SparseVectors(&qsv)->Owner(false);

    auto search_param = fmt::format(R"({{
        "sindi": {{
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 10,
            "proximity_boost": {{
                "weight": 0.5,
                "candidates": 10000
            }},
            "phrase_terms": [{}, {}, {}],
            "phrase_slop": 5
        }}
    }})",
                                    t10,
                                    t20,
                                    t30);
    auto result_before = index->KnnSearch(query, 3, search_param, nullptr);

    std::stringstream stream;
    REQUIRE_NOTHROW(index->SerializeStreaming(stream));

    auto restored = std::make_unique<SINDI>(param, common_param);
    std::stringstream deserialize_stream(stream.str());
    REQUIRE_NOTHROW(restored->DeserializeStreaming(deserialize_stream));

    auto result_after = restored->KnnSearch(query, 3, search_param, nullptr);
    REQUIRE(result_after->GetDim() == result_before->GetDim());
    for (int64_t j = 0; j < result_after->GetDim(); ++j) {
        REQUIRE(result_after->GetIds()[j] == result_before->GetIds()[j]);
        REQUIRE(std::abs(result_after->GetDistances()[j] - result_before->GetDistances()[j]) <
                1e-6f);
    }
}

TEST_CASE("SINDI Proximity with Remap", "[ut][SINDI][Proximity]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    // Use large term IDs to trigger remap
    constexpr uint32_t offset = 2000000;

    SparseVector sv_docs[2];
    uint32_t ids0[] = {offset + 10, offset + 20, offset + 30};
    float vals0[] = {1.0f, 1.0f, 1.0f};
    uint32_t seq0[] = {offset + 10, offset + 20, offset + 30};
    sv_docs[0] = {3, ids0, vals0, 3, seq0};

    uint32_t ids1[] = {offset + 10, offset + 20, offset + 30};
    float vals1[] = {1.0f, 1.0f, 1.0f};
    std::vector<uint32_t> seq1_vec(101, offset + 99);
    seq1_vec[0] = offset + 10;
    seq1_vec[50] = offset + 20;
    seq1_vec[100] = offset + 30;
    sv_docs[1] = {3, ids1, vals1, 101, seq1_vec.data()};

    std::vector<int64_t> base_ids = {0, 1};
    auto base = vsag::Dataset::Make();
    base->NumElements(2)->SparseVectors(sv_docs)->Ids(base_ids.data())->Owner(false);

    auto param_str = R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 100,
        "remap_term_ids": true,
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 10
    })";

    vsag::JsonType pj = vsag::JsonType::Parse(param_str);
    auto ip = std::make_shared<vsag::SINDIParameter>();
    ip->FromJson(pj);
    auto index = std::make_unique<SINDI>(ip, common_param);
    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);

    // Query with original (large) term IDs
    uint32_t q_ids[] = {offset + 10, offset + 20, offset + 30};
    float q_vals[] = {1.0f, 1.0f, 1.0f};
    SparseVector qsv = {3, q_ids, q_vals, 0, nullptr};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(&qsv)->Owner(false);

    std::string sp = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 10,
            "proximity_boost": {
                "weight": 0.5,
                "boost_multiplicative": true
            }
        }
    })";
    auto result = index->KnnSearch(query, 2, sp, nullptr);
    REQUIRE(result->GetDim() == 2);
    // Doc 0 (clustered) should still rank before Doc 1 (scattered) even with remap
    REQUIRE(result->GetIds()[0] == 0);
    REQUIRE(result->GetDistances()[0] < result->GetDistances()[1]);
}

TEST_CASE("SINDI Proximity with Reorder", "[ut][SINDI][Proximity]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    SparseVector sv_docs[2];
    uint32_t ids0[] = {10, 20, 30};
    float vals0[] = {1.0f, 1.0f, 1.0f};
    uint32_t seq0[] = {10, 20, 30};
    sv_docs[0] = {3, ids0, vals0, 3, seq0};

    uint32_t ids1[] = {10, 20, 30};
    float vals1[] = {1.0f, 1.0f, 1.0f};
    std::vector<uint32_t> seq1_vec(101, 99);
    seq1_vec[0] = 10;
    seq1_vec[50] = 20;
    seq1_vec[100] = 30;
    sv_docs[1] = {3, ids1, vals1, 101, seq1_vec.data()};

    std::vector<int64_t> base_ids = {0, 1};
    auto base = vsag::Dataset::Make();
    base->NumElements(2)->SparseVectors(sv_docs)->Ids(base_ids.data())->Owner(false);

    auto param_str = R"({
        "use_reorder": true,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 200,
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 10
    })";

    vsag::JsonType pj = vsag::JsonType::Parse(param_str);
    auto ip = std::make_shared<vsag::SINDIParameter>();
    ip->FromJson(pj);
    auto index = std::make_unique<SINDI>(ip, common_param);
    index->Build(base);

    uint32_t q_ids[] = {10, 20, 30};
    float q_vals[] = {1.0f, 1.0f, 1.0f};
    SparseVector qsv = {3, q_ids, q_vals, 0, nullptr};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(&qsv)->Owner(false);

    // With reorder + proximity, both should work together
    std::string sp = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 10,
            "proximity_boost": {
                "weight": 0.5,
                "boost_multiplicative": true
            }
        }
    })";
    auto result = index->KnnSearch(query, 2, sp, nullptr);
    REQUIRE(result->GetDim() == 2);
    // Doc 0 (clustered) should rank first
    REQUIRE(result->GetIds()[0] == 0);
}

TEST_CASE("SINDI Proximity with DocPrune", "[ut][SINDI][Proximity]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    // Doc with many terms, doc_prune will remove low-weight ones
    SparseVector sv_docs[2];
    uint32_t ids0[] = {10, 20, 30, 40, 50};
    float vals0[] = {5.0f, 4.0f, 0.1f, 0.1f, 0.1f};  // 30,40,50 will be pruned
    uint32_t seq0[] = {10, 20, 30, 40, 50};
    sv_docs[0] = {5, ids0, vals0, 5, seq0};

    uint32_t ids1[] = {10, 20, 30, 40, 50};
    float vals1[] = {5.0f, 4.0f, 0.1f, 0.1f, 0.1f};
    std::vector<uint32_t> seq1_vec(101, 99);
    seq1_vec[0] = 10;
    seq1_vec[50] = 20;
    seq1_vec[60] = 30;
    seq1_vec[70] = 40;
    seq1_vec[80] = 50;
    sv_docs[1] = {5, ids1, vals1, 101, seq1_vec.data()};

    std::vector<int64_t> base_ids = {0, 1};
    auto base = vsag::Dataset::Make();
    base->NumElements(2)->SparseVectors(sv_docs)->Ids(base_ids.data())->Owner(false);

    // doc_prune_ratio=0.5 will prune ~50% of term mass
    auto param_str = R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.5,
        "window_size": 10000,
        "term_id_limit": 200,
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 10
    })";

    vsag::JsonType pj = vsag::JsonType::Parse(param_str);
    auto ip = std::make_shared<vsag::SINDIParameter>();
    ip->FromJson(pj);
    auto index = std::make_unique<SINDI>(ip, common_param);
    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);

    // Should not crash — pruned terms just have no positions
    uint32_t q_ids[] = {10, 20};
    float q_vals[] = {1.0f, 1.0f};
    SparseVector qsv = {2, q_ids, q_vals, 0, nullptr};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(&qsv)->Owner(false);

    std::string sp = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 10,
            "proximity_boost": {
                "weight": 0.5
            }
        }
    })";
    auto result = index->KnnSearch(query, 2, sp, nullptr);
    REQUIRE(result->GetDim() == 2);
}

TEST_CASE("SINDI Proximity with Filter", "[ut][SINDI][Proximity]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    SparseVector sv_docs[3];
    uint32_t ids0[] = {10, 20, 30};
    float vals0[] = {1.0f, 1.0f, 1.0f};
    uint32_t seq0[] = {10, 20, 30};
    sv_docs[0] = {3, ids0, vals0, 3, seq0};  // id=0 even → pass filter

    uint32_t ids1[] = {10, 20, 30};
    float vals1[] = {1.0f, 1.0f, 1.0f};
    uint32_t seq1[] = {10, 20, 30};
    sv_docs[1] = {3, ids1, vals1, 3, seq1};  // id=1 odd → filtered out

    uint32_t ids2[] = {10, 20, 30};
    float vals2[] = {1.0f, 1.0f, 1.0f};
    std::vector<uint32_t> seq2_vec(101, 99);
    seq2_vec[0] = 10;
    seq2_vec[50] = 20;
    seq2_vec[100] = 30;
    sv_docs[2] = {3, ids2, vals2, 101, seq2_vec.data()};  // id=2 even → pass filter

    std::vector<int64_t> base_ids = {0, 1, 2};
    auto base = vsag::Dataset::Make();
    base->NumElements(3)->SparseVectors(sv_docs)->Ids(base_ids.data())->Owner(false);

    auto param_str = R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 200,
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 10
    })";

    vsag::JsonType pj = vsag::JsonType::Parse(param_str);
    auto ip = std::make_shared<vsag::SINDIParameter>();
    ip->FromJson(pj);
    auto index = std::make_unique<SINDI>(ip, common_param);
    index->Build(base);

    uint32_t q_ids[] = {10, 20, 30};
    float q_vals[] = {1.0f, 1.0f, 1.0f};
    SparseVector qsv = {3, q_ids, q_vals, 0, nullptr};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(&qsv)->Owner(false);

    auto mock_filter = std::make_shared<MockFilter>();  // even IDs only

    std::string sp = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 10,
            "proximity_boost": {
                "weight": 0.5
            }
        }
    })";
    auto result = index->KnnSearch(query, 3, sp, mock_filter);
    // Only doc 0 and doc 2 should be returned (even IDs)
    REQUIRE(result->GetDim() == 2);
    // Doc 0 (clustered, id=0) should rank before Doc 2 (scattered, id=2)
    REQUIRE(result->GetIds()[0] == 0);
    REQUIRE(result->GetIds()[1] == 2);
}

TEST_CASE("SINDI Proximity Multi-Window", "[ut][SINDI][Proximity]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    // Use small window_size to force multiple windows
    uint32_t num_base = 200;
    uint32_t window_size = 10000;  // min allowed

    std::vector<int64_t> ids(num_base);
    std::vector<SparseVector> sv_base(num_base);
    std::vector<std::vector<uint32_t>> all_ids(num_base);
    std::vector<std::vector<float>> all_vals(num_base);
    std::vector<std::vector<uint32_t>> all_seqs(num_base);

    std::mt19937 rng(123);
    for (uint32_t i = 0; i < num_base; ++i) {
        ids[i] = i;
        uint32_t n_terms = 5 + rng() % 10;
        all_ids[i].resize(n_terms);
        all_vals[i].resize(n_terms);
        all_seqs[i].resize(20 + rng() % 30);
        for (uint32_t j = 0; j < n_terms; ++j) {
            all_ids[i][j] = rng() % 100;
            all_vals[i][j] = 1.0f + static_cast<float>(rng() % 10);
        }
        // Sort and dedup ids
        std::sort(all_ids[i].begin(), all_ids[i].end());
        all_ids[i].erase(std::unique(all_ids[i].begin(), all_ids[i].end()), all_ids[i].end());
        all_vals[i].resize(all_ids[i].size());
        n_terms = all_ids[i].size();

        for (uint32_t j = 0; j < all_seqs[i].size(); ++j) {
            all_seqs[i][j] = rng() % 100;
        }

        sv_base[i].len_ = n_terms;
        sv_base[i].ids_ = all_ids[i].data();
        sv_base[i].vals_ = all_vals[i].data();
        sv_base[i].token_seq_len_ = all_seqs[i].size();
        sv_base[i].token_sequence_ = all_seqs[i].data();
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    auto param_str = fmt::format(R"({{
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": {},
        "term_id_limit": 200,
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 10
    }})",
                                 window_size);

    vsag::JsonType pj = vsag::JsonType::Parse(param_str);
    auto ip = std::make_shared<vsag::SINDIParameter>();
    ip->FromJson(pj);
    auto index = std::make_unique<SINDI>(ip, common_param);
    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);

    // Search with proximity — should not crash across windows
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(sv_base.data())->Owner(false);

    std::string sp = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20,
            "proximity_boost": {
                "weight": 0.3
            }
        }
    })";
    auto result = index->KnnSearch(query, 10, sp, nullptr);
    REQUIRE(result->GetDim() > 0);

    // Also verify proximity_weight=0 gives results
    std::string sp_no = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20,
            "proximity_boost": {
                "weight": 0.0
            }
        }
    })";
    auto result_no = index->KnnSearch(query, 10, sp_no, nullptr);
    REQUIRE(result_no->GetDim() > 0);
}

TEST_CASE("SINDI Proximity Ordered Integration", "[ut][SINDI][Proximity]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    // Doc 0: terms in forward order [10, 20] at positions [0, 1]
    // Doc 1: terms in reverse order [10, 20] but 20 at pos 0, 10 at pos 1
    SparseVector sv_docs[2];
    uint32_t ids0[] = {10, 20};
    float vals0[] = {1.0f, 1.0f};
    uint32_t seq0[] = {10, 20};  // forward: 10@0, 20@1
    sv_docs[0] = {2, ids0, vals0, 2, seq0};

    uint32_t ids1[] = {10, 20};
    float vals1[] = {1.0f, 1.0f};
    uint32_t seq1[] = {20, 10};  // reverse: 20@0, 10@1
    sv_docs[1] = {2, ids1, vals1, 2, seq1};

    std::vector<int64_t> base_ids = {0, 1};
    auto base = vsag::Dataset::Make();
    base->NumElements(2)->SparseVectors(sv_docs)->Ids(base_ids.data())->Owner(false);

    auto param_str = R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 200,
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 10
    })";

    vsag::JsonType pj = vsag::JsonType::Parse(param_str);
    auto ip = std::make_shared<vsag::SINDIParameter>();
    ip->FromJson(pj);
    auto index = std::make_unique<SINDI>(ip, common_param);
    index->Build(base);

    // Query [10, 20] — ordered mode: 10 should come before 20
    uint32_t q_ids[] = {10, 20};
    float q_vals[] = {1.0f, 1.0f};
    SparseVector qsv = {2, q_ids, q_vals, 0, nullptr};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(&qsv)->Owner(false);

    // Unordered: both docs have same dist=1, same boost
    std::string sp_unordered = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 10,
            "proximity_boost": {
                "weight": 0.5,
                "ordered": false
            }
        }
    })";
    auto result_unordered = index->KnnSearch(query, 2, sp_unordered, nullptr);
    REQUIRE(result_unordered->GetDistances()[0] == result_unordered->GetDistances()[1]);

    // Ordered: Doc 0 is forward (dist=1), Doc 1 is reverse (dist=1*2=2)
    std::string sp_ordered = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 10,
            "proximity_boost": {
                "weight": 0.5,
                "ordered": true
            }
        }
    })";
    auto result_ordered = index->KnnSearch(query, 2, sp_ordered, nullptr);
    // Doc 0 should rank first in ordered mode
    REQUIRE(result_ordered->GetIds()[0] == 0);
    REQUIRE(result_ordered->GetDistances()[0] < result_ordered->GetDistances()[1]);
}

TEST_CASE("SINDI Phrase Filter Basic", "[ut][SINDI][PhraseFilter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    // Doc 0: [10, 20, 30] clustered at [0, 1, 2] → phrase passes (span=2)
    // Doc 1: [10, 20, 30] scattered at [0, 50, 100] → phrase fails (span=100)
    // Doc 2: [10, 20] only → phrase fails (missing term 30)
    SparseVector sv_docs[3];
    uint32_t ids0[] = {10, 20, 30};
    float vals0[] = {0.3f, 0.3f, 0.3f};
    uint32_t seq0[] = {10, 20, 30};
    sv_docs[0] = {3, ids0, vals0, 3, seq0};

    uint32_t ids1[] = {10, 20, 30};
    float vals1[] = {0.3f, 0.3f, 0.3f};
    std::vector<uint32_t> seq1_vec(101, 99);
    seq1_vec[0] = 10;
    seq1_vec[50] = 20;
    seq1_vec[100] = 30;
    sv_docs[1] = {3, ids1, vals1, 101, seq1_vec.data()};

    uint32_t ids2[] = {10, 20};
    float vals2[] = {0.3f, 0.3f};
    uint32_t seq2[] = {10, 20};
    sv_docs[2] = {2, ids2, vals2, 2, seq2};

    std::vector<int64_t> base_ids = {0, 1, 2};
    auto base = vsag::Dataset::Make();
    base->NumElements(3)->SparseVectors(sv_docs)->Ids(base_ids.data())->Owner(false);

    auto param_str = R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 200,
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 10
    })";

    vsag::JsonType pj = vsag::JsonType::Parse(param_str);
    auto ip = std::make_shared<vsag::SINDIParameter>();
    ip->FromJson(pj);
    auto index = std::make_unique<SINDI>(ip, common_param);
    index->Build(base);

    uint32_t q_ids[] = {10, 20, 30};
    float q_vals[] = {0.3f, 0.3f, 0.3f};
    SparseVector qsv = {3, q_ids, q_vals, 0, nullptr};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(&qsv)->Owner(false);

    // phrase_terms=[10,20,30], slop=5 → max_span = 5+3-1=7
    // Doc 0: span=2 ≤ 7 → pass
    // Doc 1: span=100 > 7 → filtered
    // Doc 2: missing term 30 → filtered
    std::string sp = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 10,
            "proximity_boost": {
                "weight": 0.0
            },
            "phrase_terms": [10, 20, 30],
            "phrase_slop": 5
        }
    })";
    auto result = index->KnnSearch(query, 3, sp, nullptr);
    // Only Doc 0 survives
    REQUIRE(result->GetDim() == 1);
    REQUIRE(result->GetIds()[0] == 0);
}

TEST_CASE("SINDI Phrase Filter Retained-Prefix Bound Regression", "[ut][SINDI][PhraseFilter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    // Regression for the LookupPositions binary-search bound (docs-first
    // plan verification item): a candidate doc that lives in the *pruned tail*
    // of a second phrase term must read as position-absent, exactly as the old
    // reverse-offset map did. If LookupPositions searched the full posting
    // instead of the retained prefix, doc 8 below would wrongly pass the phrase.
    //
    // Layout (base_id == insertion index, single window):
    //   docs 0..7 : term 20 only            -> term 20 posting = [0..8], doc 8 last
    //   doc 8     : terms 10 & 20 adjacent  -> in term 10's retained prefix, but
    //                                          in term 20's pruned tail at ratio 0.5
    //   doc 9     : term 10 only
    // Query phrase = [10, 20]. Only doc 8 owns both terms, so it is the sole
    // phrase candidate. With no term pruning it passes; with term_prune_ratio
    // 0.5 term 20's tail (incl. doc 8) is pruned, so doc 8 must be filtered.
    const uint32_t num_base = 10;
    std::vector<SparseVector> sv_docs(num_base);
    std::vector<std::vector<uint32_t>> ids_store(num_base);
    std::vector<std::vector<float>> vals_store(num_base);
    std::vector<std::vector<uint32_t>> seq_store(num_base);

    for (uint32_t i = 0; i < 8; ++i) {
        ids_store[i] = {20};
        vals_store[i] = {0.3f};
        seq_store[i] = {20};
        sv_docs[i] = {1, ids_store[i].data(), vals_store[i].data(), 1, seq_store[i].data()};
    }
    // doc 8: both terms adjacent at positions 0 and 1
    ids_store[8] = {10, 20};
    vals_store[8] = {0.3f, 0.3f};
    seq_store[8] = {10, 20};
    sv_docs[8] = {2, ids_store[8].data(), vals_store[8].data(), 2, seq_store[8].data()};
    // doc 9: term 10 only
    ids_store[9] = {10};
    vals_store[9] = {0.3f};
    seq_store[9] = {10};
    sv_docs[9] = {1, ids_store[9].data(), vals_store[9].data(), 1, seq_store[9].data()};

    std::vector<int64_t> base_ids(num_base);
    for (uint32_t i = 0; i < num_base; ++i) {
        base_ids[i] = i;
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_docs.data())->Ids(base_ids.data())->Owner(false);

    auto param_str = R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 200,
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 10
    })";
    vsag::JsonType pj = vsag::JsonType::Parse(param_str);
    auto ip = std::make_shared<vsag::SINDIParameter>();
    ip->FromJson(pj);
    auto index = std::make_unique<SINDI>(ip, common_param);
    index->Build(base);

    uint32_t q_ids[] = {10, 20};
    float q_vals[] = {0.3f, 0.3f};
    SparseVector qsv = {2, q_ids, q_vals, 0, nullptr};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(&qsv)->Owner(false);

    // No term pruning: full postings searched, doc 8 owns both terms adjacent
    // -> phrase passes, doc 8 is returned.
    std::string sp_no_prune = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20,
            "proximity_boost": {
                "weight": 0.0
            },
            "phrase_terms": [10, 20],
            "phrase_slop": 5
        }
    })";
    auto res_full = index->KnnSearch(query, 10, sp_no_prune, nullptr);
    bool doc8_present_full = false;
    for (int64_t i = 0; i < res_full->GetDim(); ++i) {
        if (res_full->GetIds()[i] == 8) {
            doc8_present_full = true;
        }
    }
    REQUIRE(doc8_present_full == true);

    // Term pruning 0.5: term 20's posting [0..8] retains only the head [0..3],
    // so doc 8 (last) falls in the pruned tail and LookupPositions must
    // treat term 20 as absent for doc 8 -> phrase fails -> doc 8 filtered out.
    std::string sp_pruned = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.5,
            "n_candidate": 20,
            "proximity_boost": {
                "weight": 0.0
            },
            "phrase_terms": [10, 20],
            "phrase_slop": 5
        }
    })";
    auto res_pruned = index->KnnSearch(query, 10, sp_pruned, nullptr);
    bool doc8_present_pruned = false;
    for (int64_t i = 0; i < res_pruned->GetDim(); ++i) {
        if (res_pruned->GetIds()[i] == 8) {
            doc8_present_pruned = true;
        }
    }
    REQUIRE(doc8_present_pruned == false);
}

TEST_CASE("SINDI Phrase Filter with Boost Combined", "[ut][SINDI][PhraseFilter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    // 3 docs all with terms [10, 20, 30]
    // Doc 0: clustered [0,1,2] -> phrase pass + high boost
    // Doc 1: medium [0,3,6] -> phrase pass (span=6, slop=5 -> max_span=7) +
    // medium boost
    // Doc 2: scattered [0,50,100] -> phrase fail (span=100 > 7)
    SparseVector sv_docs[3];
    uint32_t ids0[] = {10, 20, 30};
    float vals0[] = {0.3f, 0.3f, 0.3f};
    uint32_t seq0[] = {10, 20, 30};
    sv_docs[0] = {3, ids0, vals0, 3, seq0};

    uint32_t ids1[] = {10, 20, 30};
    float vals1[] = {0.3f, 0.3f, 0.3f};
    uint32_t seq1[] = {10, 99, 99, 20, 99, 99, 30};
    sv_docs[1] = {3, ids1, vals1, 7, seq1};

    uint32_t ids2[] = {10, 20, 30};
    float vals2[] = {0.3f, 0.3f, 0.3f};
    std::vector<uint32_t> seq2_vec(101, 99);
    seq2_vec[0] = 10;
    seq2_vec[50] = 20;
    seq2_vec[100] = 30;
    sv_docs[2] = {3, ids2, vals2, 101, seq2_vec.data()};

    std::vector<int64_t> base_ids = {0, 1, 2};
    auto base = vsag::Dataset::Make();
    base->NumElements(3)->SparseVectors(sv_docs)->Ids(base_ids.data())->Owner(false);

    auto param_str = R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 200,
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": 10
    })";

    vsag::JsonType pj = vsag::JsonType::Parse(param_str);
    auto ip_param = std::make_shared<vsag::SINDIParameter>();
    ip_param->FromJson(pj);
    auto index = std::make_unique<SINDI>(ip_param, common_param);
    index->Build(base);

    uint32_t q_ids[] = {10, 20, 30};
    float q_vals[] = {0.3f, 0.3f, 0.3f};
    SparseVector qsv = {3, q_ids, q_vals, 0, nullptr};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(&qsv)->Owner(false);

    // phrase filter + boost combined
    std::string sp = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 10,
            "proximity_boost": {
                "weight": 0.3
            },
            "phrase_terms": [10, 20, 30],
            "phrase_slop": 5
        }
    })";
    auto result = index->KnnSearch(query, 3, sp, nullptr);
    // Doc 2 filtered out, only Doc 0 and Doc 1 remain
    REQUIRE(result->GetDim() == 2);
    // Doc 0 (more clustered) should rank before Doc 1
    REQUIRE(result->GetIds()[0] == 0);
    REQUIRE(result->GetIds()[1] == 1);
    REQUIRE(result->GetDistances()[0] < result->GetDistances()[1]);
}

TEST_CASE("SINDI Immutable Proximity Phrase Equality", "[ut][SINDI][Proximity][PhraseFilter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    const char* sparse_value_quant_type = GENERATE("fp32", "sq8", "fp16");
    const bool remap_term_ids = GENERATE(false, true);
    const std::string use_quantization =
        std::string(sparse_value_quant_type) == "fp16"
            ? R"("fp16")"
            : (std::string(sparse_value_quant_type) == "sq8" ? "true" : "false");

    // Controlled docs so both proximity ordering and phrase filtering exercise
    // real positions. term_offset lets us optionally push term ids into the
    // remap range. Every doc shares terms {t10,t20,t30} but at different spreads.
    constexpr uint32_t term_offset_remap = 5000000;
    const uint32_t term_offset = remap_term_ids ? term_offset_remap : 0;
    const uint32_t t10 = term_offset + 10;
    const uint32_t t20 = term_offset + 20;
    const uint32_t t30 = term_offset + 30;

    constexpr uint32_t num_base = 4;
    SparseVector sv_docs[num_base];

    // Values are intentionally non-uniform: SQ8 quantization derives its
    // min/max from the base set, so all-identical values would collapse every
    // code to 0 and yield empty results. Distinct values keep the SQ8 path
    // meaningful while phrase matching (position-only) is unaffected.

    // Doc 0: clustered [0,1,2] -> phrase pass, high proximity
    uint32_t ids0[] = {t10, t20, t30};
    float vals0[] = {0.9f, 0.9f, 0.9f};
    uint32_t seq0[] = {t10, t20, t30};
    sv_docs[0] = {3, ids0, vals0, 3, seq0};

    // Doc 1: medium [0,3,6] -> phrase pass (span 6 <= 7), medium proximity
    uint32_t ids1[] = {t10, t20, t30};
    float vals1[] = {0.6f, 0.6f, 0.6f};
    uint32_t filler = term_offset + 99;
    uint32_t seq1[] = {t10, filler, filler, t20, filler, filler, t30};
    sv_docs[1] = {3, ids1, vals1, 7, seq1};

    // Doc 2: scattered [0,50,100] -> phrase fail, low proximity
    uint32_t ids2[] = {t10, t20, t30};
    float vals2[] = {0.3f, 0.3f, 0.3f};
    std::vector<uint32_t> seq2_vec(101, term_offset + 99);
    seq2_vec[0] = t10;
    seq2_vec[50] = t20;
    seq2_vec[100] = t30;
    sv_docs[2] = {3, ids2, vals2, 101, seq2_vec.data()};

    // Doc 3: missing t30 -> phrase fail, partial proximity
    uint32_t ids3[] = {t10, t20};
    float vals3[] = {0.5f, 0.5f};
    uint32_t seq3[] = {t10, t20};
    sv_docs[3] = {2, ids3, vals3, 2, seq3};

    std::vector<int64_t> base_ids = {0, 1, 2, 3};
    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_docs)->Ids(base_ids.data())->Owner(false);

    auto make_param = [&](bool immutable) {
        return fmt::format(R"({{
            "use_reorder": false,
            "use_quantization": {},
            "doc_prune_ratio": 0.0,
            "window_size": 10000,
            "term_id_limit": 200,
            "remap_term_ids": {},
            "store_positions": true,
            "max_positions_per_term": 64,
            "avg_doc_term_length": 10,
            "immutable": {}
        }})",
                           use_quantization,
                           remap_term_ids,
                           immutable);
    };

    auto source_param = std::make_shared<vsag::SINDIParameter>();
    source_param->FromJson(vsag::JsonType::Parse(make_param(false)));
    auto target_param = std::make_shared<vsag::SINDIParameter>();
    target_param->FromJson(vsag::JsonType::Parse(make_param(true)));

    auto source = std::make_unique<SINDI>(source_param, common_param);
    auto direct_immutable = std::make_unique<SINDI>(target_param, common_param);
    auto immutable_roundtrip = std::make_unique<SINDI>(target_param, common_param);

    auto build_res = source->Build(base);
    REQUIRE(build_res.empty());
    REQUIRE(direct_immutable->Build(base).empty());
    REQUIRE(direct_immutable->GetNumElements() == num_base);
    test_serializion(*direct_immutable, *immutable_roundtrip);
    REQUIRE(immutable_roundtrip->GetNumElements() == num_base);

    uint32_t q_ids[] = {t10, t20, t30};
    float q_vals[] = {0.3f, 0.3f, 0.3f};
    SparseVector qsv = {3, q_ids, q_vals, 0, nullptr};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(&qsv)->Owner(false);

    auto expect_equal = [](const DatasetPtr& a, const DatasetPtr& b) {
        REQUIRE(a->GetDim() == b->GetDim());
        for (int64_t j = 0; j < a->GetDim(); ++j) {
            REQUIRE(a->GetIds()[j] == b->GetIds()[j]);
            REQUIRE(std::abs(a->GetDistances()[j] - b->GetDistances()[j]) < 1e-3);
        }
    };

    // Proximity boost only.
    std::string sp_prox = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20,
            "proximity_boost": {
                "weight": 0.5,
                "ordered": false,
                "candidates": 10000
            }
        }
    })";
    auto source_prox_knn = source->KnnSearch(query, 4, sp_prox, nullptr);
    expect_equal(source_prox_knn, direct_immutable->KnnSearch(query, 4, sp_prox, nullptr));
    expect_equal(source_prox_knn, immutable_roundtrip->KnnSearch(query, 4, sp_prox, nullptr));
    auto source_prox_range = source->RangeSearch(query, 1.0F, sp_prox, nullptr);
    expect_equal(source_prox_range, direct_immutable->RangeSearch(query, 1.0F, sp_prox, nullptr));
    expect_equal(source_prox_range,
                 immutable_roundtrip->RangeSearch(query, 1.0F, sp_prox, nullptr));

    // Phrase filter only.
    std::string sp_phrase = fmt::format(R"({{
        "sindi": {{
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20,
            "proximity_boost": {{
                "weight": 0.0
            }},
            "phrase_terms": [{}, {}, {}],
            "phrase_slop": 5
        }}
    }})",
                                        t10,
                                        t20,
                                        t30);
    auto src_phrase = source->KnnSearch(query, 4, sp_phrase, nullptr);
    auto imm_phrase = direct_immutable->KnnSearch(query, 4, sp_phrase, nullptr);
    expect_equal(src_phrase, imm_phrase);
    expect_equal(src_phrase, immutable_roundtrip->KnnSearch(query, 4, sp_phrase, nullptr));
    // Only Doc 0 and Doc 1 pass the phrase constraint.
    REQUIRE(imm_phrase->GetDim() == 2);

    // Phrase filter + proximity boost combined.
    std::string sp_combined = fmt::format(R"({{
        "sindi": {{
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20,
            "proximity_boost": {{
                "weight": 0.3
            }},
            "phrase_terms": [{}, {}, {}],
            "phrase_slop": 5
        }}
    }})",
                                          t10,
                                          t20,
                                          t30);
    auto source_combined = source->KnnSearch(query, 4, sp_combined, nullptr);
    expect_equal(source_combined, direct_immutable->KnnSearch(query, 4, sp_combined, nullptr));
    expect_equal(source_combined, immutable_roundtrip->KnnSearch(query, 4, sp_combined, nullptr));
}

TEST_CASE("SINDI Immutable Proximity Ref Retained-Prefix Equality", "[ut][SINDI][Proximity]") {
    // Regression for the pre-resolved posting-list view (ResolveTerm /
    // LookupPositions): the resolved view must honor the same retained-prefix
    // bound as the per-candidate lookup path. A candidate that lives in the
    // pruned tail of one query term must read as position-absent, so its
    // proximity contribution for that term is empty — identical to mutable.
    // Exercised with term_prune_ratio > 0 and both remap settings.
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    const bool remap_term_ids = GENERATE(false, true);
    constexpr uint32_t term_offset_remap = 5000000;
    const uint32_t term_offset = remap_term_ids ? term_offset_remap : 0;
    const uint32_t t10 = term_offset + 10;
    const uint32_t t20 = term_offset + 20;

    // Layout (single window, base_id == insertion index):
    //   docs 0..7 : term 10 & term 20, clustered so proximity is high
    //   doc 8     : term 10 & term 20 adjacent, but it is the LAST entry in each
    //               posting, so at term_prune_ratio 0.5 it falls in the pruned
    //               tail -> LookupPositions on both the mutable and immutable
    //               backends must skip it.
    constexpr uint32_t num_base = 9;
    std::vector<SparseVector> sv_docs(num_base);
    std::vector<std::vector<uint32_t>> ids_store(num_base);
    std::vector<std::vector<float>> vals_store(num_base);
    std::vector<std::vector<uint32_t>> seq_store(num_base);

    for (uint32_t i = 0; i < num_base; ++i) {
        ids_store[i] = {t10, t20};
        // Descending values so the pruned tail (low IP) is the high doc ids.
        float v = 0.9f - static_cast<float>(i) * 0.08f;
        vals_store[i] = {v, v};
        seq_store[i] = {t10, t20};  // adjacent -> non-zero proximity when present
        sv_docs[i] = {2, ids_store[i].data(), vals_store[i].data(), 2, seq_store[i].data()};
    }

    std::vector<int64_t> base_ids(num_base);
    for (uint32_t i = 0; i < num_base; ++i) {
        base_ids[i] = i;
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_docs.data())->Ids(base_ids.data())->Owner(false);

    auto make_param = [&](bool immutable) {
        return fmt::format(R"({{
            "use_reorder": false,
            "use_quantization": false,
            "doc_prune_ratio": 0.0,
            "window_size": 10000,
            "term_id_limit": 200,
            "remap_term_ids": {},
            "store_positions": true,
            "max_positions_per_term": 64,
            "avg_doc_term_length": 10,
            "immutable": {}
        }})",
                           remap_term_ids,
                           immutable);
    };

    auto source_param = std::make_shared<vsag::SINDIParameter>();
    source_param->FromJson(vsag::JsonType::Parse(make_param(false)));
    auto target_param = std::make_shared<vsag::SINDIParameter>();
    target_param->FromJson(vsag::JsonType::Parse(make_param(true)));

    auto source = std::make_unique<SINDI>(source_param, common_param);
    auto immutable = std::make_unique<SINDI>(target_param, common_param);
    auto immutable_roundtrip = std::make_unique<SINDI>(target_param, common_param);

    REQUIRE(source->Build(base).empty());
    REQUIRE(immutable->Build(base).empty());
    REQUIRE(immutable->GetNumElements() == num_base);
    test_serializion(*immutable, *immutable_roundtrip);
    REQUIRE(immutable_roundtrip->GetNumElements() == num_base);

    uint32_t q_ids[] = {t10, t20};
    float q_vals[] = {0.5f, 0.5f};
    SparseVector qsv = {2, q_ids, q_vals, 0, nullptr};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(&qsv)->Owner(false);

    auto expect_equal = [](const DatasetPtr& a, const DatasetPtr& b) {
        REQUIRE(a->GetDim() == b->GetDim());
        for (int64_t j = 0; j < a->GetDim(); ++j) {
            REQUIRE(a->GetIds()[j] == b->GetIds()[j]);
            REQUIRE(std::abs(a->GetDistances()[j] - b->GetDistances()[j]) < 1e-3);
        }
    };

    // term_prune_ratio 0.5 activates the retained-prefix bound in both backends.
    std::string sp = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.5,
            "n_candidate": 20,
            "proximity_boost": {
                "weight": 0.5,
                "all_pairs": true,
                "ordered": false,
                "candidates": 10000
            }
        }
    })";
    expect_equal(source->KnnSearch(query, num_base, sp, nullptr),
                 immutable->KnnSearch(query, num_base, sp, nullptr));
    expect_equal(source->KnnSearch(query, num_base, sp, nullptr),
                 immutable_roundtrip->KnnSearch(query, num_base, sp, nullptr));
    expect_equal(source->RangeSearch(query, 1.0F, sp, nullptr),
                 immutable->RangeSearch(query, 1.0F, sp, nullptr));
    expect_equal(source->RangeSearch(query, 1.0F, sp, nullptr),
                 immutable_roundtrip->RangeSearch(query, 1.0F, sp, nullptr));

    // Same retained-prefix bound must hold for the phrase-filter path. Like
    // proximity, the immutable backend reaches doc positions via ResolveTerm +
    // LookupPositions, whose binary search must also stop at the retained
    // prefix. doc 8 lives in the pruned tail, so the phrase must fail
    // identically on both backends. bench cannot drive phrase_terms, so this UT
    // is the only guard for immutable phrase pruning correctness.
    std::string sp_phrase = fmt::format(R"({{
        "sindi": {{
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.5,
            "n_candidate": 20,
            "proximity_boost": {{
                "weight": 0.0
            }},
            "phrase_terms": [{}, {}],
            "phrase_slop": 5
        }}
    }})",
                                        t10,
                                        t20);
    expect_equal(source->KnnSearch(query, num_base, sp_phrase, nullptr),
                 immutable->KnnSearch(query, num_base, sp_phrase, nullptr));
    expect_equal(source->KnnSearch(query, num_base, sp_phrase, nullptr),
                 immutable_roundtrip->KnnSearch(query, num_base, sp_phrase, nullptr));
    expect_equal(source->RangeSearch(query, 1.0F, sp_phrase, nullptr),
                 immutable->RangeSearch(query, 1.0F, sp_phrase, nullptr));
    expect_equal(source->RangeSearch(query, 1.0F, sp_phrase, nullptr),
                 immutable_roundtrip->RangeSearch(query, 1.0F, sp_phrase, nullptr));
}

TEST_CASE("SINDI Proximity Benchmark", "[ut][SINDI][Proximity][ProximityBenchmark]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    // Generate test data
    constexpr uint32_t num_base = 2000;
    constexpr uint32_t num_query = 50;
    constexpr uint32_t vocab_size = 500;
    constexpr uint32_t max_doc_terms = 50;
    constexpr uint32_t doc_len = 100;
    constexpr int64_t k = 10;
    constexpr float beta = 0.3f;

    std::mt19937 rng(42);

    // Generate documents: sparse vectors + token sequences
    std::vector<std::vector<uint32_t>> all_ids(num_base);
    std::vector<std::vector<float>> all_vals(num_base);
    std::vector<std::vector<uint32_t>> all_seqs(num_base);
    std::vector<SparseVector> sv_base(num_base);
    std::vector<int64_t> base_ids_bench(num_base);

    for (uint32_t i = 0; i < num_base; ++i) {
        base_ids_bench[i] = i;

        // Generate token sequence
        all_seqs[i].resize(doc_len);
        for (uint32_t j = 0; j < doc_len; ++j) {
            all_seqs[i][j] = rng() % vocab_size;
        }

        // Extract unique terms with random weights
        std::set<uint32_t> unique;
        for (auto t : all_seqs[i]) {
            unique.insert(t);
        }
        uint32_t n_terms = std::min(static_cast<uint32_t>(unique.size()), max_doc_terms);
        all_ids[i].assign(unique.begin(), unique.end());
        all_ids[i].resize(n_terms);
        all_vals[i].resize(n_terms);
        for (uint32_t j = 0; j < n_terms; ++j) {
            all_vals[i][j] = 0.1f + static_cast<float>(rng() % 90) / 100.0f;
        }

        sv_base[i].len_ = n_terms;
        sv_base[i].ids_ = all_ids[i].data();
        sv_base[i].vals_ = all_vals[i].data();
        sv_base[i].token_seq_len_ = doc_len;
        sv_base[i].token_sequence_ = all_seqs[i].data();
    }

    // Generate queries: pick subset of terms from random docs
    std::vector<std::vector<uint32_t>> q_ids_vec(num_query);
    std::vector<std::vector<float>> q_vals_vec(num_query);
    std::vector<SparseVector> sv_queries(num_query);

    for (uint32_t qi = 0; qi < num_query; ++qi) {
        uint32_t src_doc = rng() % num_base;
        uint32_t n_q = std::min(static_cast<uint32_t>(all_ids[src_doc].size()), 8u);
        q_ids_vec[qi].resize(n_q);
        q_vals_vec[qi].resize(n_q);
        for (uint32_t j = 0; j < n_q; ++j) {
            q_ids_vec[qi][j] = all_ids[src_doc][j];
            q_vals_vec[qi][j] = 0.1f + static_cast<float>(rng() % 90) / 100.0f;
        }
        sv_queries[qi].len_ = n_q;
        sv_queries[qi].ids_ = q_ids_vec[qi].data();
        sv_queries[qi].vals_ = q_vals_vec[qi].data();
    }

    // Build index with positions
    auto bench_param_str = fmt::format(R"({{
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "store_positions": true,
        "max_positions_per_term": 64,
        "avg_doc_term_length": {}
    }})",
                                       vocab_size + 1,
                                       max_doc_terms);

    vsag::JsonType bench_pj = vsag::JsonType::Parse(bench_param_str);
    auto bench_ip = std::make_shared<vsag::SINDIParameter>();
    bench_ip->FromJson(bench_pj);

    auto bench_index = std::make_unique<SINDI>(bench_ip, common_param);
    auto bench_base = vsag::Dataset::Make();
    bench_base->NumElements(num_base)
        ->SparseVectors(sv_base.data())
        ->Ids(base_ids_bench.data())
        ->Owner(false);
    auto build_res = bench_index->Build(bench_base);
    REQUIRE(build_res.size() == 0);

    // Brute-force ground truth with proximity boost
    std::vector<std::vector<int64_t>> gt_ids(num_query);
    for (uint32_t qi = 0; qi < num_query; ++qi) {
        std::vector<std::pair<float, int64_t>> dists;
        for (uint32_t di = 0; di < num_base; ++di) {
            // Compute IP
            float ip = 0.0f;
            uint32_t qp = 0, dp = 0;
            while (qp < q_ids_vec[qi].size() && dp < all_ids[di].size()) {
                if (q_ids_vec[qi][qp] == all_ids[di][dp]) {
                    ip += q_vals_vec[qi][qp] * all_vals[di][dp];
                    qp++;
                    dp++;
                } else if (q_ids_vec[qi][qp] < all_ids[di][dp]) {
                    qp++;
                } else {
                    dp++;
                }
            }
            if (ip == 0.0f) {
                continue;
            }

            // Compute proximity boost
            std::unordered_map<uint32_t, std::vector<uint16_t>> pos_map;
            for (uint32_t p = 0; p < all_seqs[di].size(); ++p) {
                pos_map[all_seqs[di][p]].push_back(static_cast<uint16_t>(p));
            }

            std::vector<std::vector<uint16_t>> position_lists;
            for (uint32_t qj = 0; qj < q_ids_vec[qi].size(); ++qj) {
                auto it = pos_map.find(q_ids_vec[qi][qj]);
                if (it != pos_map.end()) {
                    position_lists.push_back(it->second);
                } else {
                    position_lists.emplace_back();
                }
            }

            std::vector<PosSpan> position_spans;
            position_spans.reserve(position_lists.size());
            for (const auto& list : position_lists) {
                position_spans.push_back(PosSpan{list.data(), static_cast<uint32_t>(list.size())});
            }
            float raw_boost = all_pairwise_proximity(position_spans, false);
            float pair_count = static_cast<float>(q_ids_vec[qi].size()) *
                               static_cast<float>(q_ids_vec[qi].size() - 1) / 2.0f;
            float norm_boost = (pair_count > 0) ? raw_boost / pair_count : 0.0f;
            float boosted_ip = ip * (1.0f + beta * norm_boost);
            float dist = 1.0f - boosted_ip;
            dists.push_back({dist, static_cast<int64_t>(di)});
        }
        std::sort(dists.begin(), dists.end());
        gt_ids[qi].resize(std::min(static_cast<int64_t>(dists.size()), k));
        for (uint64_t j = 0; j < gt_ids[qi].size(); ++j) {
            gt_ids[qi][j] = dists[j].second;
        }
    }

    // Search params
    std::string sp_prox = fmt::format(R"({{
        "sindi": {{
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 50,
            "proximity_boost": {{
                "weight": {},
                "boost_multiplicative": true,
                "all_pairs": true,
                "candidates": 10000
            }}
        }}
    }})",
                                      beta);

    std::string sp_baseline = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 50,
            "proximity_boost": {
                "weight": 0.0
            }
        }
    })";

    // Measure proximity search: recall + QPS
    auto query_ds = vsag::Dataset::Make();
    auto t_start_prox = std::chrono::high_resolution_clock::now();
    float total_recall = 0.0f;

    for (uint32_t qi = 0; qi < num_query; ++qi) {
        query_ds->NumElements(1)->SparseVectors(sv_queries.data() + qi)->Owner(false);
        auto result = bench_index->KnnSearch(query_ds, k, sp_prox, nullptr);

        std::set<int64_t> gt_set(gt_ids[qi].begin(), gt_ids[qi].end());
        uint32_t hits = 0;
        for (int64_t j = 0; j < result->GetDim(); ++j) {
            if (gt_set.count(result->GetIds()[j]) > 0) {
                hits++;
            }
        }
        float recall = gt_set.empty() ? 1.0f : static_cast<float>(hits) / gt_set.size();
        total_recall += recall;
    }
    auto t_end_prox = std::chrono::high_resolution_clock::now();
    double prox_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end_prox - t_start_prox).count() /
        1000.0;

    // Measure baseline search + count rank changes
    auto t_start_base = std::chrono::high_resolution_clock::now();
    uint32_t rank_changes = 0;
    for (uint32_t qi = 0; qi < num_query; ++qi) {
        query_ds->NumElements(1)->SparseVectors(sv_queries.data() + qi)->Owner(false);
        auto result_base = bench_index->KnnSearch(query_ds, k, sp_baseline, nullptr);
        auto result_prox = bench_index->KnnSearch(query_ds, k, sp_prox, nullptr);

        if (result_base->GetDim() > 0 && result_prox->GetDim() > 0 &&
            result_base->GetIds()[0] != result_prox->GetIds()[0]) {
            rank_changes++;
        }
    }
    auto t_end_base = std::chrono::high_resolution_clock::now();
    double base_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end_base - t_start_base).count() /
        1000.0;

    float avg_recall = total_recall / num_query;
    double prox_qps = num_query / (prox_ms / 1000.0);

    WARN("=== SINDI Proximity Benchmark ===");
    WARN("Data: " << num_base << " docs, " << num_query << " queries, vocab=" << vocab_size
                  << ", doc_len=" << doc_len);
    WARN("Proximity recall@" << k << " vs brute-force GT: " << avg_recall);
    WARN("Proximity search: " << prox_ms << " ms total, QPS=" << static_cast<int>(prox_qps));
    WARN("Baseline+proximity search: " << base_ms << " ms total (" << num_query << " x 2 queries)");
    WARN("Top-1 rank changes (baseline vs proximity): " << rank_changes << "/" << num_query);
    WARN("Memory estimate: " << bench_index->EstimateMemory(num_base) << " bytes");

    REQUIRE(avg_recall > 0.5f);
}
