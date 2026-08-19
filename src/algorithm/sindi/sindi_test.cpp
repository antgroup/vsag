
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

#include "sindi.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <map>
#include <set>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "storage/serialization_template_test.h"

using namespace vsag;

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

TEST_CASE("SINDI Basic Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    // Prepare Base and Query Dataset
    uint32_t num_base = 1000;
    uint32_t num_query = 100;
    int64_t max_dim = 128;
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

    constexpr static auto param_str = R"({{
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
    SparseIndexParameterPtr bf_param = std::make_shared<SparseIndexParameters>();
    bf_param->need_sort = true;
    auto bf_index = std::make_unique<SparseIndex>(bf_param, common_param);

    // test build
    bf_index->Build(base);
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
            "n_candidate": 20,
            "use_term_lists_heap_insert": false
        }
    }
    )";

    auto query = vsag::Dataset::Make();
    auto mock_filter = std::make_shared<MockFilter>();
    auto mock_valid_filter = std::make_shared<MockValidIdFilter>();
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
        auto bf_result = bf_index->KnnSearch(query, k, search_param_str, nullptr);

        // test basic performance
        auto result = index->KnnSearch(query, k, search_param_str, nullptr);
        REQUIRE(result->GetNumElements() == bf_result->GetNumElements());
        REQUIRE(result->GetDim() == bf_result->GetDim());
        for (int j = 0; j < k; j++) {
            REQUIRE(result->GetIds()[j] == bf_result->GetIds()[j]);
            REQUIRE(std::abs(result->GetDistances()[j] - bf_result->GetDistances()[j]) < 1e-3);
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

TEST_CASE("SINDI range search ignores repeated zero-distance candidates", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    const bool immutable = GENERATE(false, true);
    const bool use_reorder = GENERATE(false, true);
    auto index_param = std::make_shared<SINDIParameter>();
    index_param->FromJson(JsonType::Parse(fmt::format(R"({{
        "use_reorder": {},
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 60000,
        "term_id_limit": 4000
    }})",
                                                      use_reorder)));
    SINDI index(index_param, common_param);

    std::vector<std::vector<uint32_t>> indices = {{5, 100, 2000}, {5, 50, 2000}, {10, 100, 3000}};
    std::vector<std::vector<float>> values = {
        {0.5F, 0.8F, 0.6F}, {0.3F, 0.9F, 0.1F}, {0.7F, 0.4F, 0.5F}};
    std::vector<SparseVector> sparse_vectors(indices.size());
    for (uint64_t i = 0; i < sparse_vectors.size(); ++i) {
        sparse_vectors[i].len_ = static_cast<uint32_t>(indices[i].size());
        sparse_vectors[i].ids_ = indices[i].data();
        sparse_vectors[i].vals_ = values[i].data();
    }
    std::vector<int64_t> ids = {1, 2, 3};
    auto base = Dataset::Make();
    base->NumElements(static_cast<int64_t>(ids.size()))
        ->Ids(ids.data())
        ->SparseVectors(sparse_vectors.data())
        ->Owner(false);
    REQUIRE(index.Build(base).empty());
    if (immutable) {
        index.SetImmutable();
    }

    auto query = Dataset::Make();
    query->NumElements(1)->SparseVectors(sparse_vectors.data())->Owner(false);
    // Force term-list heap insertion while retaining terms shared by multiple documents.
    const std::string search_params =
        R"({"sindi": {"query_prune_ratio": 0.2, "n_candidate": 100}})";
    auto result = index.RangeSearch(query, 1.0F, search_params, nullptr);

    REQUIRE(result->GetDim() == 3);
    const std::map<int64_t, float> expected_results =
        use_reorder ? std::map<int64_t, float>{{1, -0.25F}, {2, 0.79F}, {3, 0.68F}}
                    : std::map<int64_t, float>{{1, 0.0F}, {2, 0.94F}, {3, 0.68F}};
    std::set<int64_t> result_ids;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        const auto id = result->GetIds()[i];
        const auto expected = expected_results.find(id);
        REQUIRE(expected != expected_results.end());
        REQUIRE(std::abs(result->GetDistances()[i] - expected->second) < 1e-5F);
        result_ids.insert(id);
    }
    REQUIRE(result_ids == std::set<int64_t>{1, 2, 3});
}

TEST_CASE("SINDI Quantization Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    // Prepare Base and Query Dataset
    uint32_t num_base = 1000;
    uint32_t num_query = 100;
    int64_t max_dim = 128;
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

    constexpr static auto param_str = R"({{
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
    SparseIndexParameterPtr bf_param = std::make_shared<SparseIndexParameters>();
    bf_param->need_sort = true;
    auto bf_index = std::make_unique<SparseIndex>(bf_param, common_param);

    // test build
    bf_index->Build(base);
    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);
    REQUIRE(index->GetNumElements() == num_base);

    // test search process
    std::string search_param_str = R"(
    {
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 20,
            "use_term_lists_heap_insert": false
        }
    }
    )";

    auto query = vsag::Dataset::Make();
    int64_t correct_count = 0;

    for (int i = 0; i < num_query; ++i) {
        query->NumElements(1)->SparseVectors(sv_base.data() + i)->Owner(false);

        // gt
        auto bf_result = bf_index->KnnSearch(query, k, search_param_str, nullptr);

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
