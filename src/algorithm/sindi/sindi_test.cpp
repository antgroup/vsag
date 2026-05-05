
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

#include <set>

#include "impl/allocator/safe_allocator.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"
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

TEST_CASE("SINDI Remap Basic Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    // Sparse term IDs in [0, 5000000] range, but only ~1000 unique terms
    uint32_t num_base = 500;
    uint32_t num_query = 50;
    int64_t max_dim = 64;
    int64_t max_id = 5000000;  // large sparse range
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
    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    // Count unique terms to set term_id_limit
    std::set<uint32_t> unique_terms;
    for (uint32_t i = 0; i < num_base; ++i) {
        for (uint32_t j = 0; j < sv_base[i].len_; ++j) {
            unique_terms.insert(sv_base[i].ids_[j]);
        }
    }
    uint32_t term_id_limit = static_cast<uint32_t>(unique_terms.size()) + 100;  // some headroom

    auto param_str = fmt::format(R"({{
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": {},
        "remap_term_ids": true,
        "avg_doc_term_length": 64
    }})",
                                 term_id_limit);

    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);
    auto another_index = std::make_unique<SINDI>(index_param, common_param);

    // Build a brute-force index for ground truth
    SparseIndexParameterPtr bf_param = std::make_shared<SparseIndexParameters>();
    bf_param->need_sort = true;
    auto bf_index = std::make_unique<SparseIndex>(bf_param, common_param);

    // test build
    bf_index->Build(base);
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
            "n_candidate": 20,
            "use_term_lists_heap_insert": false
        }
    }
    )";

    auto query = vsag::Dataset::Make();
    for (int i = 0; i < num_query; ++i) {
        query->NumElements(1)->SparseVectors(sv_base.data() + i)->Owner(false);

        auto bf_result = bf_index->KnnSearch(query, k, search_param_str, nullptr);
        auto result = index->KnnSearch(query, k, search_param_str, nullptr);

        REQUIRE(result->GetDim() == bf_result->GetDim());
        for (int j = 0; j < result->GetDim(); j++) {
            REQUIRE(result->GetIds()[j] == bf_result->GetIds()[j]);
            REQUIRE(std::abs(result->GetDistances()[j] - bf_result->GetDistances()[j]) < 1e-3);
        }

        // test serialized index gives same results
        auto another_result = another_index->KnnSearch(query, k, search_param_str, nullptr);
        for (int j = 0; j < another_result->GetDim(); j++) {
            REQUIRE(result->GetIds()[j] == another_result->GetIds()[j]);
        }
    }

    // test unknown query terms (terms not in the index)
    {
        SparseVector unknown_query;
        uint32_t unknown_ids[] = {max_id + 100, max_id + 200};
        float unknown_vals[] = {1.0f, 2.0f};
        unknown_query.len_ = 2;
        unknown_query.ids_ = unknown_ids;
        unknown_query.vals_ = unknown_vals;
        query->NumElements(1)->SparseVectors(&unknown_query)->Owner(false);
        auto result = index->KnnSearch(query, k, search_param_str, nullptr);
        REQUIRE(result->GetDim() == 0);  // no matches since all terms are unknown
    }

    // test incremental add with new terms
    {
        uint32_t num_add = 100;
        std::vector<int64_t> add_ids(num_add);
        for (uint32_t i = 0; i < num_add; ++i) {
            add_ids[i] = num_base + i;
        }
        auto sv_add =
            fixtures::GenerateSparseVectors(num_add, max_dim, max_id, min_val, max_val, 99);
        auto add_data = vsag::Dataset::Make();
        add_data->NumElements(num_add)
            ->SparseVectors(sv_add.data())
            ->Ids(add_ids.data())
            ->Owner(false);
        auto add_res = index->Add(add_data);
        REQUIRE(index->GetNumElements() == num_base + num_add);

        // search still works after incremental add
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

    uint32_t num_base = 300;
    uint32_t num_query = 30;
    int64_t max_dim = 64;
    int64_t max_id = 1000000;
    float min_val = 0;
    float max_val = 10;
    int seed_base = 77;
    int64_t k = 10;

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base =
        fixtures::GenerateSparseVectors(num_base, max_dim, max_id, min_val, max_val, seed_base);
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
        "avg_doc_term_length": 64
    }})",
                                 term_id_limit);

    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);

    SparseIndexParameterPtr bf_param = std::make_shared<SparseIndexParameters>();
    bf_param->need_sort = true;
    auto bf_index = std::make_unique<SparseIndex>(bf_param, common_param);

    bf_index->Build(base);
    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);

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
    for (int i = 0; i < num_query; ++i) {
        query->NumElements(1)->SparseVectors(sv_base.data() + i)->Owner(false);

        auto bf_result = bf_index->KnnSearch(query, k, search_param_str, nullptr);
        auto result = index->KnnSearch(query, k, search_param_str, nullptr);

        REQUIRE(result->GetDim() == bf_result->GetDim());
        for (int j = 0; j < result->GetDim(); j++) {
            REQUIRE(result->GetIds()[j] == bf_result->GetIds()[j]);
            REQUIRE(std::abs(result->GetDistances()[j] - bf_result->GetDistances()[j]) < 1e-3);
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

    // Create index with very small term_id_limit
    auto param_str = R"({
        "use_reorder": false,
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 5,
        "remap_term_ids": true,
        "avg_doc_term_length": 64
    })";

    vsag::JsonType param_json = vsag::JsonType::Parse(param_str);
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);

    // Generate data with more than 5 unique terms
    uint32_t num_base = 10;
    int64_t max_dim = 10;
    int64_t max_id = 1000;
    auto sv_base = fixtures::GenerateSparseVectors(num_base, max_dim, max_id, 0, 10, 123);

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    // Some docs should fail because term_id_limit=5 is too small
    auto failed = index->Build(base);
    REQUIRE(failed.size() > 0);  // at least some should fail
    // But some should succeed (the first few docs with <= 5 unique terms total)
    REQUIRE(index->GetNumElements() > 0);

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}
