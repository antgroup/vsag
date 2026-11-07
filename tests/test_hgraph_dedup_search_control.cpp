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

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "vsag/vsag.h"

namespace {

constexpr int64_t DIM = 32;
constexpr int64_t BASE_COUNT = 200;
constexpr int64_t DUP_COUNT = 100;

std::string
make_build_param(bool support_duplicate, float dup_threshold = 0.0F) {
    return fmt::format(R"({{
        "dtype": "float32",
        "metric_type": "l2",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "sq8",
            "graph_type": "nsw",
            "max_degree": 24,
            "ef_construction": 100,
            "support_duplicate": {},
            "duplicate_distance_threshold": {}
        }}
    }})",
                       DIM,
                       support_duplicate ? "true" : "false",
                       dup_threshold);
}

std::string
make_search_param(int64_t ef_search = 100,
                  int64_t max_duplicates_per_group = -1,
                  int64_t parallelism = 1) {
    return fmt::format(R"({{
        "parallelism": {},
        "hgraph": {{
            "ef_search": {},
            "max_duplicates_per_group": {}
        }}
    }})",
                       parallelism,
                       ef_search,
                       max_duplicates_per_group);
}

struct test_vectors {
    std::vector<float> base;
    std::vector<int64_t> base_ids;
    std::vector<float> duplicates;
    std::vector<int64_t> dup_ids;
    std::vector<float> queries;
};

test_vectors
generate_test_data(int64_t dim, int64_t base_count, int64_t dup_count, uint32_t seed = 42) {
    test_vectors tv;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    tv.base.resize(base_count * dim);
    tv.base_ids.resize(base_count);
    for (int64_t i = 0; i < base_count; ++i) {
        for (int64_t d = 0; d < dim; ++d) {
            tv.base[i * dim + d] = dist(rng);
        }
        tv.base_ids[i] = i;
    }

    tv.duplicates.resize(dup_count * dim);
    tv.dup_ids.resize(dup_count);
    for (int64_t i = 0; i < dup_count; ++i) {
        int64_t src = i % base_count;
        std::memcpy(
            tv.duplicates.data() + i * dim, tv.base.data() + src * dim, dim * sizeof(float));
        tv.dup_ids[i] = base_count + i;
    }

    tv.queries.resize(10 * dim);
    for (int64_t i = 0; i < 10; ++i) {
        int64_t src = i % base_count;
        std::memcpy(tv.queries.data() + i * dim, tv.base.data() + src * dim, dim * sizeof(float));
    }

    return tv;
}

vsag::IndexPtr
build_index_with_duplicates(const test_vectors& tv, const std::string& build_param) {
    auto index = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index.has_value());

    auto base_count = static_cast<int64_t>(tv.base_ids.size());
    auto base_ds = vsag::Dataset::Make();
    base_ds->NumElements(base_count)
        ->Dim(DIM)
        ->Float32Vectors(tv.base.data())
        ->Ids(tv.base_ids.data())
        ->Owner(false);
    REQUIRE(index.value()->Build(base_ds).has_value());

    auto dup_count = static_cast<int64_t>(tv.dup_ids.size());
    if (dup_count > 0) {
        auto dup_ds = vsag::Dataset::Make();
        dup_ds->NumElements(dup_count)
            ->Dim(DIM)
            ->Float32Vectors(tv.duplicates.data())
            ->Ids(tv.dup_ids.data())
            ->Owner(false);
        REQUIRE(index.value()->Add(dup_ds).has_value());
    }

    return index.value();
}

std::map<int64_t, int64_t>
count_duplicate_ids_by_group(const vsag::DatasetPtr& result, int64_t base_count) {
    std::map<int64_t, int64_t> group_dup_count;
    const auto* ids = result->GetIds();
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        if (ids[i] >= base_count) {
            int64_t src = (ids[i] - base_count) % base_count;
            group_dup_count[src]++;
        }
    }
    return group_dup_count;
}

int64_t
count_duplicate_ids(const vsag::DatasetPtr& result, int64_t base_count) {
    int64_t duplicate_count = 0;
    const auto* ids = result->GetIds();
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        if (ids[i] >= base_count) {
            ++duplicate_count;
        }
    }
    return duplicate_count;
}

struct iterator_item {
    int64_t id;
    float distance;
    int64_t page;
};

struct iterator_context_guard {
    vsag::IteratorContext*& context;

    ~iterator_context_guard() {
        delete context;
    }
};

std::vector<iterator_item>
collect_iterator_results(const vsag::IndexPtr& index,
                         const vsag::DatasetPtr& query,
                         const std::string& params,
                         int64_t page_size = 1) {
    vsag::IteratorContext* iter_ctx = nullptr;
    iterator_context_guard guard{iter_ctx};

    std::vector<iterator_item> items;
    std::set<int64_t> seen_ids;
    bool exhausted = false;
    constexpr int64_t max_pages = 32;
    vsag::FilterPtr filter = nullptr;

    for (int64_t page = 0; page < max_pages; ++page) {
        auto result = index->KnnSearch(query, page_size, params, filter, iter_ctx, false);
        REQUIRE(result.has_value());

        const auto count = result.value()->GetDim();
        if (count == 0) {
            exhausted = true;
            break;
        }
        REQUIRE(count <= page_size);
        for (int64_t i = 0; i < count; ++i) {
            const auto id = result.value()->GetIds()[i];
            REQUIRE(seen_ids.insert(id).second);
            items.push_back({id, result.value()->GetDistances()[i], page});
        }
    }

    REQUIRE(exhausted);
    return items;
}

}  // namespace

TEST_CASE("HGraph dedup search expands the entry-point group",
          "[ft][hgraph][duplicate][search_control][entry_point]") {
    constexpr int64_t entry_base_count = 1;
    constexpr int64_t entry_duplicate_count = 3;
    constexpr int64_t duplicate_limit = 2;
    auto tv = generate_test_data(DIM, entry_base_count, entry_duplicate_count);
    auto index = build_index_with_duplicates(tv, make_build_param(true, 0.001F));
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    for (const auto parallelism : {1, 2}) {
        DYNAMIC_SECTION("parallelism=" << parallelism) {
            auto result = index->KnnSearch(
                query, 1 + duplicate_limit, make_search_param(4, duplicate_limit, parallelism));
            REQUIRE(result.has_value());
            REQUIRE(result.value()->GetDim() == 1 + duplicate_limit);
            REQUIRE(count_duplicate_ids(result.value(), entry_base_count) == duplicate_limit);

            std::set<int64_t> ids(result.value()->GetIds(),
                                  result.value()->GetIds() + result.value()->GetDim());
            REQUIRE(ids.count(0) == 1);
        }
    }
}

TEST_CASE("HGraph dedup iterator expands the entry-point group across pages",
          "[ft][hgraph][duplicate][search_control][iterator][entry_point]") {
    constexpr int64_t entry_base_count = 1;
    constexpr int64_t entry_duplicate_count = 3;
    constexpr int64_t duplicate_limit = 2;
    auto tv = generate_test_data(DIM, entry_base_count, entry_duplicate_count);
    auto index = build_index_with_duplicates(tv, make_build_param(true, 0.001F));
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    const auto items =
        collect_iterator_results(index, query, make_search_param(4, duplicate_limit), 1);
    REQUIRE(items.size() == static_cast<uint64_t>(1 + duplicate_limit));

    int64_t root_count = 0;
    int64_t duplicate_count = 0;
    std::set<int64_t> duplicate_pages;
    for (const auto& item : items) {
        if (item.id == 0) {
            ++root_count;
        } else {
            ++duplicate_count;
            duplicate_pages.insert(item.page);
        }
    }
    REQUIRE(root_count == 1);
    REQUIRE(duplicate_count == duplicate_limit);
    REQUIRE(duplicate_pages.size() == static_cast<uint64_t>(duplicate_limit));
}

TEST_CASE("HGraph dedup search: max_duplicates_per_group=1 limits expansion",
          "[ft][hgraph][duplicate][search_control]") {
    constexpr int64_t multi_base_count = 20;
    constexpr int64_t multi_dup_count = 60;
    auto tv = generate_test_data(DIM, multi_base_count, multi_dup_count);
    auto index = build_index_with_duplicates(tv, make_build_param(true, 0.001F));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    auto param = make_search_param(200, 1);
    auto result = index->KnnSearch(query_ds, 20, param);
    REQUIRE(result.has_value());
    auto group_dup_count = count_duplicate_ids_by_group(result.value(), multi_base_count);
    REQUIRE(not group_dup_count.empty());
    for (const auto& [group, cnt] : group_dup_count) {
        REQUIRE(cnt <= 1);
    }
}

TEST_CASE("HGraph dedup search: max_duplicates_per_group=0 vs unlimited",
          "[ft][hgraph][duplicate][search_control]") {
    auto tv = generate_test_data(DIM, BASE_COUNT, DUP_COUNT);
    auto index = build_index_with_duplicates(tv, make_build_param(true, 0.001F));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    auto param_limit = make_search_param(200, 0);
    auto result_limit = index->KnnSearch(query_ds, 20, param_limit);
    REQUIRE(result_limit.has_value());
    const auto dup_limit = count_duplicate_ids(result_limit.value(), BASE_COUNT);

    auto param_unlimit = make_search_param(200, -1);
    auto result_unlimit = index->KnnSearch(query_ds, 20, param_unlimit);
    REQUIRE(result_unlimit.has_value());
    const auto dup_unlimit = count_duplicate_ids(result_unlimit.value(), BASE_COUNT);

    REQUIRE(dup_limit < dup_unlimit);
}

TEST_CASE("HGraph dedup search: no duplicate support ignores max_duplicates_per_group",
          "[ft][hgraph][duplicate][search_control]") {
    auto tv = generate_test_data(DIM, BASE_COUNT, 0);
    auto index = build_index_with_duplicates(tv, make_build_param(false));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    auto param = make_search_param(200, 0);
    auto result = index->KnnSearch(query_ds, 10, param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetDistances()[0] < 0.1F);
}

TEST_CASE("HGraph dedup iterator keeps limited duplicates across pages",
          "[ft][hgraph][duplicate][search_control][iterator]") {
    constexpr int64_t iterator_base_count = 2;
    constexpr int64_t iterator_duplicate_count = 6;
    auto tv = generate_test_data(DIM, iterator_base_count, iterator_duplicate_count);
    auto index = build_index_with_duplicates(tv, make_build_param(true, 0.001F));
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    const std::vector<std::pair<int64_t, int64_t>> cases{{0, 0}, {1, 1}, {2, 2}, {-1, 3}};
    for (const auto& [limit, expected_duplicate_count] : cases) {
        DYNAMIC_SECTION("max_duplicates_per_group=" << limit) {
            const auto items = collect_iterator_results(index, query, make_search_param(2, limit));
            std::map<int64_t, int64_t> duplicate_counts;
            std::map<int64_t, std::set<int64_t>> duplicate_pages;
            for (const auto& item : items) {
                REQUIRE(item.distance >= 0.0F);
                if (item.id < iterator_base_count) {
                    continue;
                }
                const auto group = (item.id - iterator_base_count) % iterator_base_count;
                ++duplicate_counts[group];
                duplicate_pages[group].insert(item.page);
            }

            if (expected_duplicate_count == 0) {
                REQUIRE(duplicate_counts.empty());
                continue;
            }

            bool found_full_group = false;
            for (const auto& [group, count] : duplicate_counts) {
                REQUIRE(count <= expected_duplicate_count);
                if (count == expected_duplicate_count) {
                    found_full_group = true;
                    REQUIRE(duplicate_pages[group].size() ==
                            static_cast<uint64_t>(expected_duplicate_count));
                }
            }
            REQUIRE(found_full_group);
        }
    }
}

TEST_CASE("HGraph dedup iterator drains pending duplicates in last-search mode",
          "[ft][hgraph][duplicate][search_control][iterator]") {
    constexpr int64_t iterator_base_count = 2;
    auto tv = generate_test_data(DIM, iterator_base_count, 6);
    auto index = build_index_with_duplicates(tv, make_build_param(true, 0.001F));
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    vsag::IteratorContext* iter_ctx = nullptr;
    iterator_context_guard guard{iter_ctx};
    vsag::FilterPtr filter = nullptr;
    const auto params = make_search_param(2, 2);
    auto first = index->KnnSearch(query, 1, params, filter, iter_ctx, false);
    REQUIRE(first.has_value());
    REQUIRE(first.value()->GetDim() == 1);
    auto last = index->KnnSearch(query, 8, params, filter, iter_ctx, true);
    REQUIRE(last.has_value());
    REQUIRE(last.value()->GetDim() > 0);

    std::set<int64_t> result_ids;
    std::map<int64_t, int64_t> duplicate_counts;
    auto collect_result = [&](const vsag::DatasetPtr& result) {
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            const auto id = result->GetIds()[i];
            REQUIRE(result_ids.insert(id).second);
            if (id < iterator_base_count) {
                continue;
            }
            const auto group = (id - iterator_base_count) % iterator_base_count;
            ++duplicate_counts[group];
        }
    };
    collect_result(first.value());
    collect_result(last.value());

    bool found_full_group = false;
    for (const auto& [group, count] : duplicate_counts) {
        (void)group;
        REQUIRE(count <= 2);
        found_full_group = found_full_group or count == 2;
    }
    REQUIRE(found_full_group);

    auto exhausted = index->KnnSearch(query, 8, params, filter, iter_ctx, true);
    REQUIRE(exhausted.has_value());
    REQUIRE(exhausted.value()->GetDim() == 0);
}

TEST_CASE("HGraph iterator ignores duplicate limit without duplicate tracking",
          "[ft][hgraph][duplicate][search_control][iterator]") {
    constexpr int64_t iterator_base_count = 2;
    auto tv = generate_test_data(DIM, iterator_base_count, 6);
    auto index = build_index_with_duplicates(tv, make_build_param(false));
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    const auto limited = collect_iterator_results(index, query, make_search_param(16, 0), 2);
    const auto unlimited = collect_iterator_results(index, query, make_search_param(16, -1), 2);
    std::set<int64_t> limited_ids;
    std::set<int64_t> unlimited_ids;
    for (const auto& item : limited) {
        limited_ids.insert(item.id);
    }
    for (const auto& item : unlimited) {
        unlimited_ids.insert(item.id);
    }

    REQUIRE(limited_ids == unlimited_ids);
    REQUIRE(limited_ids.lower_bound(iterator_base_count) != limited_ids.end());
}
