
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
#include <set>

#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "storage/serialization_template_test.h"
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
};

}  // namespace vsag

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
        REQUIRE(result->GetNumElements() == bf_result->GetNumElements());
        REQUIRE(result->GetDim() == bf_result->GetDim());
        for (int j = 0; j < k; j++) {
            REQUIRE(result->GetIds()[j] == bf_result->GetIds()[j]);
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

TEST_CASE("SINDI Immutable Deserialize KNN Test", "[ut][SINDI]") {
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
        "immutable": false
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
    REQUIRE_THROWS_AS(index.Serialize(writer), vsag::VsagException);
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
