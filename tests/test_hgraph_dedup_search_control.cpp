
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
#include <vector>

#include "vsag/vsag.h"

namespace {

constexpr int64_t DIM = 32;
constexpr int64_t BASE_COUNT = 200;
constexpr int64_t DUP_COUNT = 100;

std::string
MakeBuildParam(bool support_duplicate, float dup_threshold = 0.0F) {
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
MakeSearchParam(int64_t ef_search = 100,
                bool consider_duplicate = true,
                int64_t max_duplicates_per_group = -1,
                float brute_force_threshold = 0.0F) {
    return fmt::format(R"({{
        "hgraph": {{
            "ef_search": {},
            "consider_duplicate": {},
            "max_duplicates_per_group": {},
            "brute_force_threshold": {}
        }}
    }})",
                       ef_search,
                       consider_duplicate ? "true" : "false",
                       max_duplicates_per_group,
                       brute_force_threshold);
}

struct TestVectors {
    std::vector<float> base;
    std::vector<int64_t> base_ids;
    std::vector<float> duplicates;
    std::vector<int64_t> dup_ids;
    std::vector<float> queries;
};

TestVectors
GenerateTestData(int64_t dim, int64_t base_count, int64_t dup_count, uint32_t seed = 42) {
    TestVectors tv;
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
BuildIndexWithDuplicates(const TestVectors& tv, const std::string& build_param) {
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
CountDuplicateIdsByGroup(const vsag::DatasetPtr& result, int64_t base_count) {
    std::map<int64_t, int64_t> group_dup_count;
    auto* ids = result->GetIds();
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        if (ids[i] >= base_count) {
            int64_t src = (ids[i] - base_count) % base_count;
            group_dup_count[src]++;
        }
    }
    return group_dup_count;
}

int64_t
CountDuplicateIds(const vsag::DatasetPtr& result, int64_t base_count) {
    int64_t duplicate_count = 0;
    auto* ids = result->GetIds();
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        if (ids[i] >= base_count) {
            ++duplicate_count;
        }
    }
    return duplicate_count;
}

}  // namespace

TEST_CASE("HGraph dedup search: consider_duplicate=true returns dup ids",
          "[ft][hgraph][duplicate][search_control]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeBuildParam(true, 0.001F));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    auto param = MakeSearchParam(200, true, -1);
    auto result = index->KnnSearch(query_ds, 20, param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);

    bool found_dup = false;
    auto* ids = result.value()->GetIds();
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        if (ids[i] >= BASE_COUNT) {
            found_dup = true;
            break;
        }
    }
    REQUIRE(found_dup);
}

TEST_CASE("HGraph dedup search: consider_duplicate=false reduces dup ids",
          "[ft][hgraph][duplicate][search_control]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeBuildParam(true, 0.001F));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    auto param_on = MakeSearchParam(200, true, -1);
    auto result_on = index->KnnSearch(query_ds, 20, param_on);
    REQUIRE(result_on.has_value());
    int64_t dup_count_on = 0;
    for (int64_t i = 0; i < result_on.value()->GetDim(); ++i) {
        if (result_on.value()->GetIds()[i] >= BASE_COUNT) {
            ++dup_count_on;
        }
    }

    auto param_off = MakeSearchParam(200, false, -1);
    auto result_off = index->KnnSearch(query_ds, 20, param_off);
    REQUIRE(result_off.has_value());
    int64_t dup_count_off = 0;
    for (int64_t i = 0; i < result_off.value()->GetDim(); ++i) {
        if (result_off.value()->GetIds()[i] >= BASE_COUNT) {
            ++dup_count_off;
        }
    }

    REQUIRE(dup_count_off < dup_count_on);
}

TEST_CASE("HGraph dedup search: max_duplicates_per_group=1 limits expansion",
          "[ft][hgraph][duplicate][search_control]") {
    constexpr int64_t multi_base_count = 20;
    constexpr int64_t multi_dup_count = 60;
    auto tv = GenerateTestData(DIM, multi_base_count, multi_dup_count);
    auto index = BuildIndexWithDuplicates(tv, MakeBuildParam(true, 0.001F));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    auto param = MakeSearchParam(200, true, 1);
    auto result = index->KnnSearch(query_ds, 20, param);
    REQUIRE(result.has_value());
    auto group_dup_count = CountDuplicateIdsByGroup(result.value(), multi_base_count);
    for (const auto& [group, cnt] : group_dup_count) {
        REQUIRE(cnt <= 1);
    }
}

TEST_CASE("HGraph dedup search: max_duplicates_per_group=0 vs unlimited",
          "[ft][hgraph][duplicate][search_control]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeBuildParam(true, 0.001F));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    auto param_limit = MakeSearchParam(200, true, 0);
    auto result_limit = index->KnnSearch(query_ds, 20, param_limit);
    REQUIRE(result_limit.has_value());
    int64_t dup_limit = 0;
    for (int64_t i = 0; i < result_limit.value()->GetDim(); ++i) {
        if (result_limit.value()->GetIds()[i] >= BASE_COUNT) {
            ++dup_limit;
        }
    }

    auto param_unlimit = MakeSearchParam(200, true, -1);
    auto result_unlimit = index->KnnSearch(query_ds, 20, param_unlimit);
    REQUIRE(result_unlimit.has_value());
    int64_t dup_unlimit = 0;
    for (int64_t i = 0; i < result_unlimit.value()->GetDim(); ++i) {
        if (result_unlimit.value()->GetIds()[i] >= BASE_COUNT) {
            ++dup_unlimit;
        }
    }

    REQUIRE(dup_limit < dup_unlimit);
}

TEST_CASE("HGraph dedup search: no dup support ignores consider_duplicate",
          "[ft][hgraph][duplicate][search_control]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, 0);
    auto index = BuildIndexWithDuplicates(tv, MakeBuildParam(false));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    auto param = MakeSearchParam(200, true, -1);
    auto result = index->KnnSearch(query_ds, 10, param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetDistances()[0] < 0.1F);
}

TEST_CASE("HGraph dedup search: range search respects consider_duplicate",
          "[ft][hgraph][duplicate][search_control]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeBuildParam(true, 0.001F));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    float radius = 1.0F;

    auto param_on = MakeSearchParam(200, true, -1);
    auto result_on = index->RangeSearch(query_ds, radius, param_on, -1);
    REQUIRE(result_on.has_value());

    auto param_off = MakeSearchParam(200, false, -1);
    auto result_off = index->RangeSearch(query_ds, radius, param_off, -1);
    REQUIRE(result_off.has_value());

    REQUIRE(result_off.value()->GetDim() <= result_on.value()->GetDim());
}

TEST_CASE("HGraph dedup search: range search respects max_duplicates_per_group",
          "[ft][hgraph][duplicate][search_control]") {
    constexpr int64_t multi_base_count = 20;
    constexpr int64_t multi_dup_count = 60;
    auto tv = GenerateTestData(DIM, multi_base_count, multi_dup_count);
    auto index = BuildIndexWithDuplicates(tv, MakeBuildParam(true, 0.001F));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    float radius = 1.0F;

    auto param = MakeSearchParam(200, true, 1);
    auto result = index->RangeSearch(query_ds, radius, param, -1);
    REQUIRE(result.has_value());

    auto group_dup_count = CountDuplicateIdsByGroup(result.value(), multi_base_count);
    for (const auto& [group, cnt] : group_dup_count) {
        REQUIRE(cnt <= 1);
    }
}

TEST_CASE("HGraph dedup search: rejects invalid max_duplicates_per_group",
          "[ft][hgraph][duplicate][search_control]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, 0);
    auto index = BuildIndexWithDuplicates(tv, MakeBuildParam(false));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    auto param = MakeSearchParam(200, true, -2);
    auto result = index->KnnSearch(query_ds, 10, param);
    REQUIRE(not result.has_value());
}

TEST_CASE("HGraph dedup search: brute force respects max_duplicates_per_group",
          "[ft][hgraph][duplicate][search_control]") {
    constexpr int64_t multi_base_count = 20;
    constexpr int64_t multi_dup_count = 60;
    auto tv = GenerateTestData(DIM, multi_base_count, multi_dup_count);
    auto index = BuildIndexWithDuplicates(tv, MakeBuildParam(true, 0.001F));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    auto param_one = MakeSearchParam(200, true, 1, 1.0F);
    auto result_one = index->KnnSearch(query_ds, 10, param_one);
    REQUIRE(result_one.has_value());
    auto group_dup_count = CountDuplicateIdsByGroup(result_one.value(), multi_base_count);
    for (const auto& [group, cnt] : group_dup_count) {
        REQUIRE(cnt <= 1);
    }

    auto param_unlimited = MakeSearchParam(200, true, -1, 1.0F);
    auto result_unlimited = index->KnnSearch(query_ds, 10, param_unlimited);
    REQUIRE(result_unlimited.has_value());
    REQUIRE(CountDuplicateIds(result_one.value(), multi_base_count) <
            CountDuplicateIds(result_unlimited.value(), multi_base_count));
}

TEST_CASE("HGraph dedup search: brute force respects consider_duplicate in range search",
          "[ft][hgraph][duplicate][search_control]") {
    constexpr int64_t multi_base_count = 20;
    constexpr int64_t multi_dup_count = 60;
    auto tv = GenerateTestData(DIM, multi_base_count, multi_dup_count);
    auto index = BuildIndexWithDuplicates(tv, MakeBuildParam(true, 0.001F));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    float radius = 0.001F;

    auto param_on = MakeSearchParam(200, true, -1, 1.0F);
    auto result_on = index->RangeSearch(query_ds, radius, param_on, -1);
    REQUIRE(result_on.has_value());
    REQUIRE(CountDuplicateIds(result_on.value(), multi_base_count) > 0);

    auto param_off = MakeSearchParam(200, false, -1, 1.0F);
    auto result_off = index->RangeSearch(query_ds, radius, param_off, -1);
    REQUIRE(result_off.has_value());
    REQUIRE(CountDuplicateIds(result_off.value(), multi_base_count) == 0);
    REQUIRE(result_off.value()->GetDim() < result_on.value()->GetDim());
}
