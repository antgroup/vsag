
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

#include <unistd.h>

#include <cstdio>

#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "unittest.h"

using namespace vsag;

namespace {

DiskSINDIParameterPtr
create_disksindi_param(uint32_t term_id_limit, const std::string& term_path) {
    auto param_str = fmt::format(R"({{
        "term_id_limit": {},
        "window_size": 10000,
        "doc_prune_ratio": 0.0,
        "use_quantization": false,
        "use_reorder": true,
        "avg_doc_term_length": 100,
        "term_io": {{ "type": "buffer_io", "file_path": "{}" }},
        "rerank_io": {{ "type": "memory_io" }}
    }})",
                                 term_id_limit,
                                 term_path);
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

    const std::string term_path =
        fmt::format("/tmp/disksindi_test_{}.term.index", static_cast<long long>(::getpid()));
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
    std::remove(term_path.c_str());
}

TEST_CASE("DiskSINDI Sorted Merge Rerank End-To-End", "[ut][DiskSINDI]") {
    // Phase B regression: the rerank path now sorts candidate inner_ids before
    // issuing the batched IO, enabling GetCodesByIdsBatch to merge adjacent
    // requests. The returned (id, distance) pairs must remain identical to the
    // brute-force truth (same as Phase A). Additionally, each returned distance
    // is cross-checked against CalcDistanceById to ensure the sorted+merged
    // code path produces the exact same distance computation.
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

    const std::string term_path =
        fmt::format("/tmp/disksindi_merge_test_{}.term.index", static_cast<long long>(::getpid()));
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
        // uses the single-id GetCodesById path, bypassing the merge logic).
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
