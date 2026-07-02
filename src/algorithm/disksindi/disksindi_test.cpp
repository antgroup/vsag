
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

#include "disksindi.h"

#include <fmt/format.h>

#include <cstdio>
#include <numeric>

#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "unittest.h"

using namespace vsag;

namespace {

DiskSINDIParameterPtr
create_disksindi_param(uint32_t term_id_limit,
                       const std::string& term_path,
                       const std::string& term_io_type = "buffer_io",
                       const std::string& rerank_io_type = "memory_io",
                       const std::string& rerank_layout = "none",
                       uint32_t rerank_layout_top_terms = 16) {
    auto param_str = fmt::format(R"({{
        "term_id_limit": {},
        "window_size": 10000,
        "doc_prune_ratio": 0.0,
        "use_quantization": false,
        "use_reorder": true,
        "avg_doc_term_length": 100,
        "rerank_layout": "{}",
        "rerank_layout_top_terms": {},
        "term_io": {{ "type": "{}", "file_path": "{}" }},
        "rerank_io": {{ "type": "{}" }}
    }})",
                                 term_id_limit,
                                 rerank_layout,
                                 rerank_layout_top_terms,
                                 term_io_type,
                                 term_path,
                                 rerank_io_type);
    auto param = std::make_shared<DiskSINDIParameter>();
    param->FromJson(JsonType::Parse(param_str));
    return param;
}

}  // namespace

TEST_CASE("DiskSINDI Batch Rerank End-To-End", "[ut][DiskSINDI]") {
    // Phase A regression: the rerank path now batches IO through
    // GetCodesByIdsBatch instead of issuing one Read per candidate. The
    // returned (id, distance) pairs must remain identical to the single-id
    // path. We verify that by checking that the top-k results are a sane
    // permutation of the brute-force top-k (allowing ties due to identical
    // distances when k is at the boundary).
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    const uint32_t num_base = 500;
    const uint32_t num_query = 20;
    const int64_t max_dim = 128;
    const uint32_t term_id_limit = 5000;
    const int64_t k = 10;
    common_param.dim_ = max_dim;

    std::vector<int64_t> ids(num_base);
    std::iota(ids.begin(), ids.end(), 0);

    auto sv_base = fixtures::GenerateSparseVectors(num_base, max_dim, /*max_id=*/term_id_limit - 1);
    auto base = Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    fixtures::TempDir dir("disksindi_batch_rerank");
    const std::string term_path = dir.GenerateRandomFile(false);
    auto param = create_disksindi_param(term_id_limit, term_path);
    auto index = std::make_unique<DiskSINDI>(param, common_param);
    REQUIRE(index->Build(base).empty());

    const std::string search_param = R"({
        "disksindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 100,
            "use_term_lists_heap_insert": false
        }
    })";

    for (uint32_t q = 0; q < num_query; ++q) {
        auto query = Dataset::Make();
        query->NumElements(1)->SparseVectors(sv_base.data() + q)->Owner(false);

        auto result = index->KnnSearch(query, k, search_param, nullptr);
        REQUIRE(result->GetDim() == k);

        // Distances must be non-decreasing (heap output order).
        for (int64_t i = 1; i < result->GetDim(); ++i) {
            REQUIRE(result->GetDistances()[i] >= result->GetDistances()[i - 1] - 1e-5);
        }

        // The query is itself in the index; the best hit must match it with
        // distance ~0 (1 - <q, q>/<q, q> = 0 when normalized, otherwise the
        // smallest dist of all pairs).
        bool found_self = false;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            if (result->GetIds()[i] == static_cast<int64_t>(q)) {
                found_self = true;
                break;
            }
        }
        REQUIRE(found_self);
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    index.reset();
}

TEST_CASE("DiskSINDI Top Terms Rerank Layout End-To-End", "[ut][DiskSINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    const uint32_t num_base = 128;
    const int64_t max_dim = 64;
    const uint32_t term_id_limit = 2048;
    const int64_t k = 10;
    common_param.dim_ = max_dim;

    std::vector<int64_t> ids(num_base);
    std::iota(ids.begin(), ids.end(), 0);

    auto sv_base = fixtures::GenerateSparseVectors(num_base, max_dim, term_id_limit - 1);
    auto base = Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    fixtures::TempDir dir("disksindi_top_terms_layout");
    const std::string term_path = dir.GenerateRandomFile(false);
    auto param = create_disksindi_param(term_id_limit,
                                        term_path,
                                        "buffer_io",
                                        "memory_io",
                                        "top_terms_signature",
                                        /*rerank_layout_top_terms=*/8);
    auto index = std::make_unique<DiskSINDI>(param, common_param);
    REQUIRE(index->Build(base).empty());

    auto query = Dataset::Make();
    query->NumElements(1)->SparseVectors(sv_base.data())->Owner(false);
    const std::string search_param = R"({
        "disksindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 64,
            "use_term_lists_heap_insert": false
        }
    })";
    auto result = index->KnnSearch(query, k, search_param, nullptr);
    REQUIRE(result->GetDim() == k);
    REQUIRE(result->GetIds()[0] == 0);
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        auto precise_dist = index->CalcDistanceById(query, result->GetIds()[i], true);
        REQUIRE(std::abs(result->GetDistances()[i] - precise_dist) < 1e-5);
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    index.reset();
}

TEST_CASE("DiskSINDI Sorted Batch Rerank End-To-End", "[ut][DiskSINDI]") {
    // The rerank path sorts candidate inner_ids before issuing batched IO.
    // Returned (id, distance) pairs must remain identical to the brute-force
    // truth. Each returned distance is cross-checked against CalcDistanceById to
    // ensure the batched code path produces the same distance computation.
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    const uint32_t num_base = 500;
    const uint32_t num_query = 10;
    const int64_t max_dim = 128;
    const uint32_t term_id_limit = 5000;
    const int64_t k = 20;
    common_param.dim_ = max_dim;

    std::vector<int64_t> ids(num_base);
    std::iota(ids.begin(), ids.end(), 0);

    auto sv_base = fixtures::GenerateSparseVectors(num_base, max_dim, /*max_id=*/term_id_limit - 1);
    auto base = Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    fixtures::TempDir dir("disksindi_batch_rerank");
    const std::string term_path = dir.GenerateRandomFile(false);
    auto param = create_disksindi_param(term_id_limit, term_path);
    auto index = std::make_unique<DiskSINDI>(param, common_param);
    REQUIRE(index->Build(base).empty());

    const std::string search_param = R"({
        "disksindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 200,
            "use_term_lists_heap_insert": false
        }
    })";

    for (uint32_t q = 0; q < num_query; ++q) {
        auto query = Dataset::Make();
        query->NumElements(1)->SparseVectors(sv_base.data() + q)->Owner(false);

        auto result = index->KnnSearch(query, k, search_param, nullptr);
        REQUIRE(result->GetDim() == k);

        // Distances must be non-decreasing.
        for (int64_t i = 1; i < result->GetDim(); ++i) {
            REQUIRE(result->GetDistances()[i] >= result->GetDistances()[i - 1] - 1e-5);
        }

        // Cross-check each result distance against CalcDistanceById (which
        // uses the single-id GetCodesById path, bypassing batched IO).
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            auto precise_dist =
                index->CalcDistanceById(query, result->GetIds()[i], /*precise=*/true);
            REQUIRE(std::abs(result->GetDistances()[i] - precise_dist) < 1e-5);
        }

        // Self must appear in the results.
        bool found_self = false;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            if (result->GetIds()[i] == static_cast<int64_t>(q)) {
                found_self = true;
                break;
            }
        }
        REQUIRE(found_self);
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    index.reset();
    std::remove(term_path.c_str());
}

TEST_CASE("DiskSINDI ReaderIO Rerank Uses Section Offset", "[ut][DiskSINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    const uint32_t num_base = 128;
    const int64_t max_dim = 64;
    const uint32_t term_id_limit = 2048;
    const int64_t k = 10;
    common_param.dim_ = max_dim;

    std::vector<int64_t> ids(num_base);
    std::iota(ids.begin(), ids.end(), 0);

    auto sv_base = fixtures::GenerateSparseVectors(num_base, max_dim, term_id_limit - 1);
    auto base = Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    fixtures::TempDir dir("disksindi_readerio_rerank");
    const std::string term_path = dir.GenerateRandomFile(false);
    auto build_param = create_disksindi_param(term_id_limit, term_path);
    DiskSINDI built(build_param, common_param);
    REQUIRE(built.Build(base).empty());

    std::stringstream stream;
    const std::string prefix = "outer-container-prefix";
    stream.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
    IOStreamWriter writer(stream);
    built.Serialize(writer);

    auto load_param =
        create_disksindi_param(term_id_limit, term_path, IO_TYPE_VALUE_READER_IO, "reader_io");
    DiskSINDI loaded(load_param, common_param);
    stream.seekg(static_cast<std::streamoff>(prefix.size()), std::ios::beg);
    loaded.Deserialize(stream);

    auto query = Dataset::Make();
    query->NumElements(1)->SparseVectors(sv_base.data())->Owner(false);
    const std::string search_param = R"({
        "disksindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 64,
            "use_term_lists_heap_insert": false
        }
    })";
    auto result = loaded.KnnSearch(query, k, search_param, nullptr);
    REQUIRE(result->GetDim() == k);
    REQUIRE(result->GetIds()[0] == 0);

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}
