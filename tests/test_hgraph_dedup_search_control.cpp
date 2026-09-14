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
#include <memory>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "vsag/vsag.h"

namespace {

constexpr int64_t DIM = 32;

std::string
MakeBuildParam(bool support_duplicate, float duplicate_threshold = 0.0F) {
    return fmt::format(R"({{
        "dtype": "float32",
        "metric_type": "l2",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "fp32",
            "graph_type": "nsw",
            "max_degree": 24,
            "ef_construction": 100,
            "build_thread_count": 4,
            "support_duplicate": {},
            "duplicate_distance_threshold": {}
        }}
    }})",
                       DIM,
                       support_duplicate ? "true" : "false",
                       duplicate_threshold);
}

std::string
MakeSearchParam(int64_t ef_search = 100,
                int64_t max_duplicates_per_group = -1,
                int64_t parallelism = 1) {
    return fmt::format(R"({{
        "hgraph": {{
            "ef_search": {},
            "parallelism": {},
            "max_duplicates_per_group": {}
        }}
    }})",
                       ef_search,
                       parallelism,
                       max_duplicates_per_group);
}

struct TestVectors {
    std::vector<float> base;
    std::vector<int64_t> base_ids;
    std::vector<float> duplicates;
    std::vector<int64_t> duplicate_ids;
    std::vector<float> queries;
};

TestVectors
GenerateTestData(int64_t dim, int64_t base_count, int64_t duplicate_count, uint32_t seed = 42) {
    TestVectors vectors;
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);

    vectors.base.resize(base_count * dim);
    vectors.base_ids.resize(base_count);
    for (int64_t i = 0; i < base_count; ++i) {
        for (int64_t d = 0; d < dim; ++d) {
            vectors.base[i * dim + d] = distribution(random);
        }
        vectors.base_ids[i] = i;
    }

    vectors.duplicates.resize(duplicate_count * dim);
    vectors.duplicate_ids.resize(duplicate_count);
    for (int64_t i = 0; i < duplicate_count; ++i) {
        const int64_t source = i % base_count;
        std::memcpy(vectors.duplicates.data() + i * dim,
                    vectors.base.data() + source * dim,
                    dim * sizeof(float));
        vectors.duplicate_ids[i] = base_count + i;
    }

    vectors.queries.resize(10 * dim);
    for (int64_t i = 0; i < 10; ++i) {
        const int64_t source = i % base_count;
        std::memcpy(vectors.queries.data() + i * dim,
                    vectors.base.data() + source * dim,
                    dim * sizeof(float));
    }

    return vectors;
}

vsag::IndexPtr
BuildIndexWithDuplicates(const TestVectors& vectors, const std::string& build_param) {
    auto index = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index.has_value());

    const auto base_count = static_cast<int64_t>(vectors.base_ids.size());
    auto base_dataset = vsag::Dataset::Make();
    base_dataset->NumElements(base_count)
        ->Dim(DIM)
        ->Float32Vectors(vectors.base.data())
        ->Ids(vectors.base_ids.data())
        ->Owner(false);
    auto build_result = index.value()->Build(base_dataset);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    const auto duplicate_count = static_cast<int64_t>(vectors.duplicate_ids.size());
    if (duplicate_count > 0) {
        auto duplicate_dataset = vsag::Dataset::Make();
        duplicate_dataset->NumElements(duplicate_count)
            ->Dim(DIM)
            ->Float32Vectors(vectors.duplicates.data())
            ->Ids(vectors.duplicate_ids.data())
            ->Owner(false);
        auto add_result = index.value()->Add(duplicate_dataset);
        REQUIRE(add_result.has_value());
        REQUIRE(add_result.value().empty());
    }

    return index.value();
}

std::map<int64_t, int64_t>
CountDuplicateIdsByGroup(const vsag::DatasetPtr& result, int64_t base_count) {
    std::map<int64_t, int64_t> group_duplicate_count;
    const auto* ids = result->GetIds();
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        if (ids[i] >= base_count) {
            const int64_t source = (ids[i] - base_count) % base_count;
            ++group_duplicate_count[source];
        }
    }
    return group_duplicate_count;
}

int64_t
CountDuplicateIds(const vsag::DatasetPtr& result, int64_t base_count) {
    int64_t duplicate_count = 0;
    const auto* ids = result->GetIds();
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        if (ids[i] >= base_count) {
            ++duplicate_count;
        }
    }
    return duplicate_count;
}

struct IteratorItem {
    int64_t id;
    float distance;
    int64_t page;
};

struct IteratorContextGuard {
    vsag::IteratorContext*& context;

    ~IteratorContextGuard() {
        delete context;
    }
};

class AllowedIdFilter : public vsag::Filter {
public:
    explicit AllowedIdFilter(std::set<int64_t> allowed_ids) : allowed_ids_(std::move(allowed_ids)) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return allowed_ids_.count(id) != 0;
    }

private:
    std::set<int64_t> allowed_ids_;
};

std::vector<IteratorItem>
CollectIteratorResults(const vsag::IndexPtr& index,
                       const vsag::DatasetPtr& query,
                       const std::string& params,
                       int64_t page_size = 1,
                       vsag::FilterPtr filter = nullptr) {
    vsag::IteratorContext* iterator_context = nullptr;
    IteratorContextGuard guard{iterator_context};

    std::vector<IteratorItem> items;
    std::set<int64_t> seen_ids;
    bool exhausted = false;
    constexpr int64_t max_pages = 32;
    for (int64_t page = 0; page < max_pages; ++page) {
        auto result = index->KnnSearch(query, page_size, params, filter, iterator_context, false);
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
    auto vectors = GenerateTestData(DIM, entry_base_count, entry_duplicate_count);
    auto index = BuildIndexWithDuplicates(vectors, MakeBuildParam(true, 0.001F));
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(vectors.queries.data())->Owner(false);

    for (const auto parallelism : {1, 2}) {
        DYNAMIC_SECTION("parallelism=" << parallelism) {
            auto result = index->KnnSearch(
                query, 1 + duplicate_limit, MakeSearchParam(4, duplicate_limit, parallelism));
            REQUIRE(result.has_value());
            REQUIRE(result.value()->GetDim() == 1 + duplicate_limit);
            REQUIRE(CountDuplicateIds(result.value(), entry_base_count) == duplicate_limit);

            std::set<int64_t> ids(result.value()->GetIds(),
                                  result.value()->GetIds() + result.value()->GetDim());
            REQUIRE(ids.count(0) == 1);
        }
    }

    auto default_limit =
        index->KnnSearch(query, 4, R"({"hgraph":{"ef_search":4,"parallelism":2}})");
    REQUIRE(default_limit.has_value());
    REQUIRE(default_limit.value()->GetDim() == 4);
    REQUIRE(CountDuplicateIds(default_limit.value(), entry_base_count) == entry_duplicate_count);
}

TEST_CASE("HGraph dedup iterator expands the entry-point group across pages",
          "[ft][hgraph][duplicate][search_control][iterator][entry_point]") {
    constexpr int64_t entry_base_count = 1;
    constexpr int64_t entry_duplicate_count = 3;
    constexpr int64_t duplicate_limit = 2;
    auto vectors = GenerateTestData(DIM, entry_base_count, entry_duplicate_count);
    auto index = BuildIndexWithDuplicates(vectors, MakeBuildParam(true, 0.001F));
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(vectors.queries.data())->Owner(false);

    const auto items = CollectIteratorResults(index, query, MakeSearchParam(4, duplicate_limit), 1);
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

TEST_CASE("HGraph dedup search limits duplicate expansion per group",
          "[ft][hgraph][duplicate][search_control]") {
    constexpr int64_t multi_base_count = 2;
    constexpr int64_t multi_duplicate_count = 6;
    auto vectors = GenerateTestData(DIM, multi_base_count, multi_duplicate_count);
    auto index = BuildIndexWithDuplicates(vectors, MakeBuildParam(true, 0.001F));

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(vectors.queries.data())->Owner(false);

    auto result = index->KnnSearch(query, 4, MakeSearchParam(8, 1));
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 4);
    const auto group_duplicate_count = CountDuplicateIdsByGroup(result.value(), multi_base_count);
    REQUIRE(group_duplicate_count.size() == 2);
    for (int64_t group = 0; group < multi_base_count; ++group) {
        REQUIRE(group_duplicate_count.at(group) == 1);
    }
}

TEST_CASE("HGraph dedup search supports zero and unlimited duplicate expansion",
          "[ft][hgraph][duplicate][search_control]") {
    constexpr int64_t base_count = 2;
    constexpr int64_t duplicate_count = 6;
    auto vectors = GenerateTestData(DIM, base_count, duplicate_count);
    auto index = BuildIndexWithDuplicates(vectors, MakeBuildParam(true, 0.001F));

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(vectors.queries.data())->Owner(false);

    auto limited = index->KnnSearch(query, 8, MakeSearchParam(8, 0));
    REQUIRE(limited.has_value());
    REQUIRE(limited.value()->GetDim() == base_count);
    REQUIRE(CountDuplicateIds(limited.value(), base_count) == 0);

    auto unlimited = index->KnnSearch(query, 8, MakeSearchParam(8, -1));
    REQUIRE(unlimited.has_value());
    REQUIRE(unlimited.value()->GetDim() == base_count + duplicate_count);
    REQUIRE(CountDuplicateIds(unlimited.value(), base_count) == duplicate_count);
}

TEST_CASE("HGraph dedup search applies min_distance consistently at the entry point",
          "[ft][hgraph][duplicate][search_control][entry_point][min_distance]") {
    constexpr int64_t base_count = 1;
    constexpr int64_t duplicate_count = 3;
    auto vectors = GenerateTestData(DIM, base_count, duplicate_count);
    auto index = BuildIndexWithDuplicates(vectors, MakeBuildParam(true, 0.001F));
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(vectors.queries.data())->Owner(false);

    for (const auto parallelism : {1, 2}) {
        DYNAMIC_SECTION("parallelism=" << parallelism) {
            const auto params = fmt::format(
                R"({{"hgraph":{{"ef_search":4,"parallelism":{},"min_distance":0}}}})", parallelism);
            auto result = index->KnnSearch(query, 4, params);
            REQUIRE(result.has_value());
            REQUIRE(result.value()->GetDim() == 0);
        }
    }
}

TEST_CASE("HGraph dedup search ignores duplicate limit when tracking is disabled",
          "[ft][hgraph][duplicate][search_control]") {
    constexpr int64_t base_count = 2;
    constexpr int64_t duplicate_count = 6;
    auto vectors = GenerateTestData(DIM, base_count, duplicate_count);
    auto index = BuildIndexWithDuplicates(vectors, MakeBuildParam(false));

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(vectors.queries.data())->Owner(false);

    auto limited = index->KnnSearch(query, 8, MakeSearchParam(16, 0));
    auto unlimited = index->KnnSearch(query, 8, MakeSearchParam(16, -1));
    REQUIRE(limited.has_value());
    REQUIRE(unlimited.has_value());
    REQUIRE(limited.value()->GetDim() == base_count + duplicate_count);
    REQUIRE(unlimited.value()->GetDim() == base_count + duplicate_count);
    REQUIRE(std::set<int64_t>(limited.value()->GetIds(),
                              limited.value()->GetIds() + limited.value()->GetDim()) ==
            std::set<int64_t>(unlimited.value()->GetIds(),
                              unlimited.value()->GetIds() + unlimited.value()->GetDim()));
}

TEST_CASE("HGraph dedup search applies filters before counting duplicate limits",
          "[ft][hgraph][duplicate][search_control][filter]") {
    constexpr int64_t base_count = 2;
    auto vectors = GenerateTestData(DIM, base_count, 3 * base_count);
    auto index = BuildIndexWithDuplicates(vectors, MakeBuildParam(true, 0.001F));

    for (const auto parallelism : {1, 2}) {
        DYNAMIC_SECTION("parallelism=" << parallelism) {
            for (int64_t group = 0; group < base_count; ++group) {
                for (int64_t member = 0; member < 3; ++member) {
                    auto query = vsag::Dataset::Make();
                    query->NumElements(1)
                        ->Dim(DIM)
                        ->Float32Vectors(vectors.base.data() + group * DIM)
                        ->Owner(false);
                    const auto allowed_duplicate = base_count + group + member * base_count;
                    auto filter =
                        std::make_shared<AllowedIdFilter>(std::set<int64_t>{allowed_duplicate});

                    auto result =
                        index->KnnSearch(query, 1, MakeSearchParam(8, 1, parallelism), filter);
                    REQUIRE(result.has_value());
                    REQUIRE(result.value()->GetDim() == 1);
                    REQUIRE(result.value()->GetIds()[0] == allowed_duplicate);
                }
            }
        }
    }
}

TEST_CASE("HGraph duplicate limit does not change range-search expansion",
          "[ft][hgraph][duplicate][search_control][range]") {
    constexpr int64_t base_count = 1;
    constexpr int64_t duplicate_count = 3;
    auto vectors = GenerateTestData(DIM, base_count, duplicate_count);
    auto index = BuildIndexWithDuplicates(vectors, MakeBuildParam(true, 0.001F));
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(vectors.queries.data())->Owner(false);

    vsag::FilterPtr filter = nullptr;
    for (const auto parallelism : {1, 2}) {
        DYNAMIC_SECTION("parallelism=" << parallelism) {
            auto result =
                index->RangeSearch(query, 0.1F, MakeSearchParam(4, 0, parallelism), filter, -1);
            REQUIRE(result.has_value());
            REQUIRE(result.value()->GetDim() == base_count + duplicate_count);
        }
    }
}

TEST_CASE("HGraph dedup iterator keeps limited duplicates across pages",
          "[ft][hgraph][duplicate][search_control][iterator]") {
    constexpr int64_t iterator_base_count = 2;
    constexpr int64_t iterator_duplicate_count = 6;
    auto vectors = GenerateTestData(DIM, iterator_base_count, iterator_duplicate_count);
    auto index = BuildIndexWithDuplicates(vectors, MakeBuildParam(true, 0.001F));
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(vectors.queries.data())->Owner(false);

    const std::vector<std::pair<int64_t, int64_t>> cases{{0, 0}, {1, 1}, {2, 2}, {-1, 3}};
    for (const auto& [limit, expected_duplicate_count] : cases) {
        DYNAMIC_SECTION("max_duplicates_per_group=" << limit) {
            const auto items = CollectIteratorResults(index, query, MakeSearchParam(2, limit));
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

TEST_CASE("HGraph dedup iterator filters duplicate groups before applying limits",
          "[ft][hgraph][duplicate][search_control][iterator][filter]") {
    constexpr int64_t base_count = 1;
    constexpr int64_t duplicate_count = 3;
    auto vectors = GenerateTestData(DIM, base_count, duplicate_count);
    auto index = BuildIndexWithDuplicates(vectors, MakeBuildParam(true, 0.001F));
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(vectors.queries.data())->Owner(false);

    for (const auto allowed_duplicate : vectors.duplicate_ids) {
        DYNAMIC_SECTION("allowed_duplicate=" << allowed_duplicate) {
            auto filter = std::make_shared<AllowedIdFilter>(std::set<int64_t>{allowed_duplicate});
            const auto items =
                CollectIteratorResults(index, query, MakeSearchParam(4, 1), 1, filter);
            REQUIRE(items.size() == 1);
            REQUIRE(items[0].id == allowed_duplicate);
        }
    }
}

TEST_CASE("HGraph dedup iterator drains pending duplicates in last-search mode",
          "[ft][hgraph][duplicate][search_control][iterator]") {
    constexpr int64_t iterator_base_count = 2;
    auto vectors = GenerateTestData(DIM, iterator_base_count, 6);
    auto index = BuildIndexWithDuplicates(vectors, MakeBuildParam(true, 0.001F));
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(vectors.queries.data())->Owner(false);

    vsag::IteratorContext* iterator_context = nullptr;
    IteratorContextGuard guard{iterator_context};
    vsag::FilterPtr filter = nullptr;
    const auto params = MakeSearchParam(2, 2);
    auto first = index->KnnSearch(query, 1, params, filter, iterator_context, false);
    REQUIRE(first.has_value());
    REQUIRE(first.value()->GetDim() == 1);
    auto last = index->KnnSearch(query, 8, params, filter, iterator_context, true);
    REQUIRE(last.has_value());
    REQUIRE(last.value()->GetDim() > 0);

    std::set<int64_t> result_ids;
    std::map<int64_t, int64_t> duplicate_counts;
    const auto collect_result = [&](const vsag::DatasetPtr& result) {
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

    auto exhausted = index->KnnSearch(query, 8, params, filter, iterator_context, true);
    REQUIRE(exhausted.has_value());
    REQUIRE(exhausted.value()->GetDim() == 0);
}

TEST_CASE("HGraph iterator ignores duplicate limit without duplicate tracking",
          "[ft][hgraph][duplicate][search_control][iterator]") {
    constexpr int64_t iterator_base_count = 2;
    auto vectors = GenerateTestData(DIM, iterator_base_count, 6);
    auto index = BuildIndexWithDuplicates(vectors, MakeBuildParam(false));
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->Float32Vectors(vectors.queries.data())->Owner(false);

    const auto limited = CollectIteratorResults(index, query, MakeSearchParam(16, 0), 2);
    const auto unlimited = CollectIteratorResults(index, query, MakeSearchParam(16, -1), 2);
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
