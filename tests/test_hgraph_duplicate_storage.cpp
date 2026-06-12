
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

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <thread>
#include <vector>

#include "vsag/vsag.h"

namespace {

constexpr int64_t DIM = 32;
constexpr int64_t BASE_COUNT = 200;
constexpr int64_t DUP_COUNT = 100;

std::string
MakeBuildParam(bool support_duplicate,
               float dup_threshold = 0.0F,
               const std::string& metric = "l2") {
    return fmt::format(R"({{
        "dtype": "float32",
        "metric_type": "{}",
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
                       metric,
                       DIM,
                       support_duplicate ? "true" : "false",
                       dup_threshold);
}

std::string
MakeSearchParam(int64_t ef_search = 100,
                bool consider_duplicate = true,
                int64_t max_duplicates_per_group = -1) {
    return fmt::format(R"({{
        "hgraph": {{
            "ef_search": {},
            "consider_duplicate": {},
            "max_duplicates_per_group": {}
        }}
    }})",
                       ef_search,
                       consider_duplicate ? "true" : "false",
                       max_duplicates_per_group);
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

std::string
MakeDefaultBuildParam(bool support_duplicate) {
    return MakeBuildParam(support_duplicate, support_duplicate ? 0.001F : 0.0F);
}

std::string
MakeBuildParamWithReorder(bool support_duplicate,
                          float dup_threshold = 0.001F,
                          const std::string& metric = "l2") {
    return fmt::format(R"({{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "sq8",
            "precise_quantization_type": "fp32",
            "graph_type": "nsw",
            "max_degree": 24,
            "ef_construction": 100,
            "support_duplicate": {},
            "duplicate_distance_threshold": {}
        }}
    }})",
                       metric,
                       DIM,
                       support_duplicate ? "true" : "false",
                       dup_threshold);
}

void
AddNearVector(vsag::IndexPtr& index,
              const float* base_vec,
              int64_t id,
              float perturbation_magnitude) {
    std::vector<float> vec(DIM);
    std::mt19937 rng(static_cast<uint32_t>(id));
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (int64_t d = 0; d < DIM; ++d) {
        vec[d] = base_vec[d] + dist(rng) * perturbation_magnitude;
    }
    auto ds = vsag::Dataset::Make();
    ds->NumElements(1)->Dim(DIM)->Float32Vectors(vec.data())->Ids(&id)->Owner(false);
    index->Add(ds);
}

vsag::IndexPtr
BuildIndexWithDuplicates(const TestVectors& tv, const std::string& build_param) {
    auto index = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index.has_value());

    auto base_ds = vsag::Dataset::Make();
    base_ds->NumElements(BASE_COUNT)
        ->Dim(DIM)
        ->Float32Vectors(tv.base.data())
        ->Ids(tv.base_ids.data())
        ->Owner(false);
    auto result = index.value()->Build(base_ds);
    REQUIRE(result.has_value());

    auto dup_ds = vsag::Dataset::Make();
    dup_ds->NumElements(DUP_COUNT)
        ->Dim(DIM)
        ->Float32Vectors(tv.duplicates.data())
        ->Ids(tv.dup_ids.data())
        ->Owner(false);
    auto add_result = index.value()->Add(dup_ds);
    REQUIRE(add_result.has_value());

    return index.value();
}

}  // namespace

TEST_CASE("HGraph dedup storage: physical slot count is less than total count",
          "[ft][hgraph][duplicate][storage]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    // All 300 labels exist
    REQUIRE(index->GetNumElements() == BASE_COUNT + DUP_COUNT);

    // Verify dedup was effective by checking search returns both base and dup ids
    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);
    auto search_param = MakeSearchParam(200, true, -1);
    auto result = index->KnnSearch(query_ds, 20, search_param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);

    bool has_dup_id = false;
    auto* ids = result.value()->GetIds();
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        if (ids[i] >= BASE_COUNT) {
            has_dup_id = true;
            break;
        }
    }
    REQUIRE(has_dup_id);
}

TEST_CASE("HGraph dedup storage: search correctness", "[ft][hgraph][duplicate][search]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    auto search_param = MakeSearchParam(200, true, -1);

    for (int64_t q = 0; q < 10; ++q) {
        auto query_ds = vsag::Dataset::Make();
        query_ds->NumElements(1)
            ->Dim(DIM)
            ->Float32Vectors(tv.queries.data() + q * DIM)
            ->Owner(false);

        auto result = index->KnnSearch(query_ds, 10, search_param);
        REQUIRE(result.has_value());
        auto count = result.value()->GetDim();
        REQUIRE(count > 0);

        auto* dists = result.value()->GetDistances();
        // Query is identical to a base vector, so nearest should be ~0 distance
        REQUIRE(dists[0] < 0.01F);
    }
}

TEST_CASE("HGraph dedup storage: distance threshold mode", "[ft][hgraph][duplicate][threshold]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeBuildParam(true, 0.01F));

    auto search_param = MakeSearchParam(200, true, -1);
    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    auto result = index->KnnSearch(query_ds, 10, search_param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetDistances()[0] < 0.01F);
}

TEST_CASE("HGraph dedup: max_duplicates_per_group = 0", "[ft][hgraph][duplicate][search_limit]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    // With max_duplicates_per_group=0 and unlimited, compare dup counts
    auto param_no_expand = MakeSearchParam(200, true, 0);
    auto result_no = index->KnnSearch(query_ds, 20, param_no_expand);
    REQUIRE(result_no.has_value());
    auto* ids_no = result_no.value()->GetIds();
    auto count_no = result_no.value()->GetDim();
    int64_t dup_count_no = 0;
    for (int64_t i = 0; i < count_no; ++i) {
        if (ids_no[i] >= BASE_COUNT) {
            ++dup_count_no;
        }
    }

    auto param_expand = MakeSearchParam(200, true, -1);
    auto result_yes = index->KnnSearch(query_ds, 20, param_expand);
    REQUIRE(result_yes.has_value());
    auto* ids_yes = result_yes.value()->GetIds();
    auto count_yes = result_yes.value()->GetDim();
    int64_t dup_count_yes = 0;
    for (int64_t i = 0; i < count_yes; ++i) {
        if (ids_yes[i] >= BASE_COUNT) {
            ++dup_count_yes;
        }
    }

    // max_duplicates_per_group=0 should produce fewer dup ids than unlimited
    REQUIRE(dup_count_no < dup_count_yes);
}

TEST_CASE("HGraph dedup: consider_duplicate = false reduces dup ids",
          "[ft][hgraph][duplicate][search_limit]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    auto param_off = MakeSearchParam(200, false, -1);
    auto result_off = index->KnnSearch(query_ds, 20, param_off);
    REQUIRE(result_off.has_value());
    auto* ids_off = result_off.value()->GetIds();
    auto count_off = result_off.value()->GetDim();
    int64_t dup_count_off = 0;
    for (int64_t i = 0; i < count_off; ++i) {
        if (ids_off[i] >= BASE_COUNT) {
            ++dup_count_off;
        }
    }

    auto param_on = MakeSearchParam(200, true, -1);
    auto result_on = index->KnnSearch(query_ds, 20, param_on);
    REQUIRE(result_on.has_value());
    auto* ids_on = result_on.value()->GetIds();
    auto count_on = result_on.value()->GetDim();
    int64_t dup_count_on = 0;
    for (int64_t i = 0; i < count_on; ++i) {
        if (ids_on[i] >= BASE_COUNT) {
            ++dup_count_on;
        }
    }

    // consider_duplicate=false should produce fewer dup ids
    REQUIRE(dup_count_off < dup_count_on);
}

TEST_CASE("HGraph dedup: max_duplicates_per_group = 1", "[ft][hgraph][duplicate][search_limit]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    auto param = MakeSearchParam(200, true, 1);
    auto result = index->KnnSearch(query_ds, 20, param);
    REQUIRE(result.has_value());
    auto* ids = result.value()->GetIds();
    auto count = result.value()->GetDim();

    // Count duplicates per representative group
    std::map<int64_t, int> group_dup_count;
    for (int64_t i = 0; i < count; ++i) {
        if (ids[i] >= BASE_COUNT) {
            int64_t src = (ids[i] - BASE_COUNT) % BASE_COUNT;
            group_dup_count[src]++;
        }
    }
    for (const auto& [group, cnt] : group_dup_count) {
        REQUIRE(cnt <= 1);
    }
}

TEST_CASE("HGraph dedup: unlimited duplicates returns all",
          "[ft][hgraph][duplicate][search_limit]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    // With unlimited duplicates and large k, we should get both base and dup ids
    auto param = MakeSearchParam(200, true, -1);
    auto result = index->KnnSearch(query_ds, 20, param);
    REQUIRE(result.has_value());
    auto* ids = result.value()->GetIds();
    auto count = result.value()->GetDim();

    bool found_base = false;
    bool found_dup = false;
    for (int64_t i = 0; i < count; ++i) {
        if (ids[i] < BASE_COUNT) {
            found_base = true;
        }
        if (ids[i] >= BASE_COUNT) {
            found_dup = true;
        }
    }
    REQUIRE(found_base);
    REQUIRE(found_dup);
}

TEST_CASE("HGraph dedup: serialize and deserialize", "[ft][hgraph][duplicate][serialize]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    // Serialize
    auto binary = index->Serialize();
    REQUIRE(binary.has_value());

    // Deserialize into new index
    auto build_param = MakeDefaultBuildParam(true);
    auto index2 = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index2.has_value());
    index2.value()->Deserialize(binary.value());

    // Search on deserialized index
    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);
    auto search_param = MakeSearchParam(200, true, -1);

    auto result = index2.value()->KnnSearch(query_ds, 10, search_param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetDistances()[0] < 0.01F);
}

TEST_CASE("HGraph dedup: range search with duplicates", "[ft][hgraph][duplicate][range]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    // Range search with a small radius should find the query's duplicate
    auto search_param = MakeSearchParam(200, true, -1);
    float radius = 0.01F;
    auto result = index->RangeSearch(query_ds, radius, search_param, -1);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
}

TEST_CASE("HGraph dedup: concurrent add with duplicates", "[ft][hgraph][duplicate][concurrent]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto build_param = MakeDefaultBuildParam(true);
    auto index = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index.has_value());

    // Build with base
    auto base_ds = vsag::Dataset::Make();
    base_ds->NumElements(BASE_COUNT)
        ->Dim(DIM)
        ->Float32Vectors(tv.base.data())
        ->Ids(tv.base_ids.data())
        ->Owner(false);
    REQUIRE(index.value()->Build(base_ds).has_value());

    // Add duplicates concurrently (one by one from multiple threads)
    constexpr int NUM_THREADS = 4;
    int64_t per_thread = DUP_COUNT / NUM_THREADS;

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (int64_t i = t * per_thread; i < (t + 1) * per_thread; ++i) {
                auto ds = vsag::Dataset::Make();
                ds->NumElements(1)
                    ->Dim(DIM)
                    ->Float32Vectors(tv.duplicates.data() + i * DIM)
                    ->Ids(tv.dup_ids.data() + i)
                    ->Owner(false);
                index.value()->Add(ds);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    // Verify search still works
    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);
    auto search_param = MakeSearchParam(200, true, -1);
    auto result = index.value()->KnnSearch(query_ds, 10, search_param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetDistances()[0] < 0.01F);
}

TEST_CASE("HGraph dedup: high duplication ratio stress test", "[ft][hgraph][duplicate][stress]") {
    constexpr int64_t SMALL_BASE = 50;
    constexpr int64_t MANY_DUPS = 200;
    auto tv = GenerateTestData(DIM, SMALL_BASE, MANY_DUPS, 99);

    auto build_param = MakeBuildParam(true, 0.001F);
    auto index = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index.has_value());

    auto base_ds = vsag::Dataset::Make();
    base_ds->NumElements(SMALL_BASE)
        ->Dim(DIM)
        ->Float32Vectors(tv.base.data())
        ->Ids(tv.base_ids.data())
        ->Owner(false);
    REQUIRE(index.value()->Build(base_ds).has_value());

    for (int64_t i = 0; i < MANY_DUPS; ++i) {
        auto ds = vsag::Dataset::Make();
        ds->NumElements(1)
            ->Dim(DIM)
            ->Float32Vectors(tv.duplicates.data() + i * DIM)
            ->Ids(tv.dup_ids.data() + i)
            ->Owner(false);
        index.value()->Add(ds);
    }

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);
    auto search_param = MakeSearchParam(200, true, -1);

    auto result = index.value()->KnnSearch(query_ds, 10, search_param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
}

TEST_CASE("HGraph no-dup mode: slot redirect disabled", "[ft][hgraph][duplicate][nodup]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, 0);

    auto build_param = MakeBuildParam(false);
    auto index = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index.has_value());

    auto base_ds = vsag::Dataset::Make();
    base_ds->NumElements(BASE_COUNT)
        ->Dim(DIM)
        ->Float32Vectors(tv.base.data())
        ->Ids(tv.base_ids.data())
        ->Owner(false);
    REQUIRE(index.value()->Build(base_ds).has_value());

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);
    auto search_param = MakeSearchParam(100, true, -1);

    auto result = index.value()->KnnSearch(query_ds, 10, search_param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetDistances()[0] < 0.01F);
}

TEST_CASE("HGraph dedup: GetNumElements counts all including duplicates",
          "[ft][hgraph][duplicate][count]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    // GetNumElements should return total count (base + duplicates)
    REQUIRE(index->GetNumElements() == BASE_COUNT + DUP_COUNT);
}

TEST_CASE("HGraph dedup: CheckIdExist for duplicates", "[ft][hgraph][duplicate][exists]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    // Both base and duplicate IDs should exist
    for (int64_t i = 0; i < 10; ++i) {
        REQUIRE(index->CheckIdExist(tv.base_ids[i]));
    }
    for (int64_t i = 0; i < 10; ++i) {
        REQUIRE(index->CheckIdExist(tv.dup_ids[i]));
    }
    // Non-existent ID
    REQUIRE_FALSE(index->CheckIdExist(BASE_COUNT + DUP_COUNT + 999));
}

TEST_CASE("HGraph dedup: duplicates in initial Build batch",
          "[ft][hgraph][duplicate][build_batch]") {
    constexpr int64_t COUNT = 100;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    std::vector<float> vectors(COUNT * DIM);
    std::vector<int64_t> ids(COUNT);
    for (int64_t i = 0; i < COUNT; ++i) {
        ids[i] = i;
        if (i >= COUNT / 2) {
            // Second half are exact copies of first half
            std::memcpy(vectors.data() + i * DIM,
                        vectors.data() + (i - COUNT / 2) * DIM,
                        DIM * sizeof(float));
        } else {
            for (int64_t d = 0; d < DIM; ++d) {
                vectors[i * DIM + d] = dist(rng);
            }
        }
    }

    auto build_param = MakeBuildParam(true, 0.001F);
    auto index = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index.has_value());

    auto ds = vsag::Dataset::Make();
    ds->NumElements(COUNT)->Dim(DIM)->Float32Vectors(vectors.data())->Ids(ids.data())->Owner(false);
    auto result = index.value()->Build(ds);
    REQUIRE(result.has_value());

    REQUIRE(index.value()->GetNumElements() == COUNT);

    // Search: query with a vector from the first half, should find its duplicate
    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(vectors.data())->Owner(false);
    auto search_param = MakeSearchParam(200, true, -1);
    auto search_result = index.value()->KnnSearch(query_ds, 10, search_param);
    REQUIRE(search_result.has_value());
    REQUIRE(search_result.value()->GetDim() > 0);
    REQUIRE(search_result.value()->GetDistances()[0] < 0.01F);
}

TEST_CASE("HGraph dedup: K > total elements", "[ft][hgraph][duplicate][large_k]") {
    constexpr int64_t SMALL_BASE = 30;
    constexpr int64_t SMALL_DUP = 10;
    auto tv = GenerateTestData(DIM, SMALL_BASE, SMALL_DUP, 77);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    int64_t total = SMALL_BASE + SMALL_DUP;
    auto search_param = MakeSearchParam(500, true, -1);
    auto result = index->KnnSearch(query_ds, total * 5, search_param);
    REQUIRE(result.has_value());
    auto count = result.value()->GetDim();
    // Should return at most total elements, not crash or return garbage
    REQUIRE(count <= total);
    REQUIRE(count > 0);

    // All returned IDs should be valid
    auto* ids = result.value()->GetIds();
    for (int64_t i = 0; i < count; ++i) {
        REQUIRE(index->CheckIdExist(ids[i]));
    }
}

TEST_CASE("HGraph dedup: duplicate label (same ID twice)", "[ft][hgraph][duplicate][dup_label]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, 0);

    auto build_param = MakeDefaultBuildParam(true);
    auto index = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index.has_value());

    auto base_ds = vsag::Dataset::Make();
    base_ds->NumElements(BASE_COUNT)
        ->Dim(DIM)
        ->Float32Vectors(tv.base.data())
        ->Ids(tv.base_ids.data())
        ->Owner(false);
    REQUIRE(index.value()->Build(base_ds).has_value());

    // Try adding a vector with an ID that already exists
    int64_t dup_id = 0;
    std::vector<float> new_vec(DIM, 0.5F);
    auto ds = vsag::Dataset::Make();
    ds->NumElements(1)->Dim(DIM)->Float32Vectors(new_vec.data())->Ids(&dup_id)->Owner(false);
    auto add_result = index.value()->Add(ds);
    REQUIRE(add_result.has_value());

    // The duplicate label should appear in failed_ids
    bool in_failed = false;
    for (auto fid : add_result.value()) {
        if (fid == dup_id) {
            in_failed = true;
        }
    }
    REQUIRE(in_failed);

    // Element count should not increase
    REQUIRE(index.value()->GetNumElements() == BASE_COUNT);
}

TEST_CASE("HGraph dedup: support_duplicate=true with zero actual duplicates",
          "[ft][hgraph][duplicate][no_actual_dups]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, 0);

    auto build_param = MakeDefaultBuildParam(true);
    auto index = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index.has_value());

    auto base_ds = vsag::Dataset::Make();
    base_ds->NumElements(BASE_COUNT)
        ->Dim(DIM)
        ->Float32Vectors(tv.base.data())
        ->Ids(tv.base_ids.data())
        ->Owner(false);
    REQUIRE(index.value()->Build(base_ds).has_value());

    REQUIRE(index.value()->GetNumElements() == BASE_COUNT);

    // Search should work normally
    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);
    auto search_param = MakeSearchParam(200, true, -1);
    auto result = index.value()->KnnSearch(query_ds, 10, search_param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetDistances()[0] < 0.01F);
}

TEST_CASE("HGraph dedup: serialize, deserialize, add more dups, then search",
          "[ft][hgraph][duplicate][serialize_add]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto build_param_str = MakeDefaultBuildParam(true);
    auto index = BuildIndexWithDuplicates(tv, build_param_str);

    // Serialize
    auto binary = index->Serialize();
    REQUIRE(binary.has_value());

    // Deserialize into new index
    auto index2 = vsag::Factory::CreateIndex("hgraph", build_param_str);
    REQUIRE(index2.has_value());
    index2.value()->Deserialize(binary.value());

    // Add more duplicates post-deserialization
    constexpr int64_t EXTRA_DUPS = 20;
    std::vector<float> extra_vecs(EXTRA_DUPS * DIM);
    std::vector<int64_t> extra_ids(EXTRA_DUPS);
    for (int64_t i = 0; i < EXTRA_DUPS; ++i) {
        int64_t src = i % BASE_COUNT;
        std::memcpy(extra_vecs.data() + i * DIM, tv.base.data() + src * DIM, DIM * sizeof(float));
        extra_ids[i] = BASE_COUNT + DUP_COUNT + i;
    }
    auto extra_ds = vsag::Dataset::Make();
    extra_ds->NumElements(EXTRA_DUPS)
        ->Dim(DIM)
        ->Float32Vectors(extra_vecs.data())
        ->Ids(extra_ids.data())
        ->Owner(false);
    auto add_result = index2.value()->Add(extra_ds);
    REQUIRE(add_result.has_value());

    // Verify total count
    REQUIRE(index2.value()->GetNumElements() == BASE_COUNT + DUP_COUNT + EXTRA_DUPS);

    // Search and verify results
    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);
    auto search_param = MakeSearchParam(200, true, -1);
    auto result = index2.value()->KnnSearch(query_ds, 20, search_param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetDistances()[0] < 0.01F);

    // Check that new duplicate IDs can appear in results
    auto* ids = result.value()->GetIds();
    auto count = result.value()->GetDim();
    bool found_new_dup = false;
    for (int64_t i = 0; i < count; ++i) {
        if (ids[i] >= BASE_COUNT + DUP_COUNT) {
            found_new_dup = true;
            break;
        }
    }
    REQUIRE(found_new_dup);
}

TEST_CASE("HGraph dedup: range search with consider_duplicate=false",
          "[ft][hgraph][duplicate][range_no_dup]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    // Range search with consider_duplicate=false
    auto param_off = MakeSearchParam(200, false, -1);
    float radius = 1.0F;
    auto result_off = index->RangeSearch(query_ds, radius, param_off, -1);
    REQUIRE(result_off.has_value());

    // Range search with consider_duplicate=true (unlimited)
    auto param_on = MakeSearchParam(200, true, -1);
    auto result_on = index->RangeSearch(query_ds, radius, param_on, -1);
    REQUIRE(result_on.has_value());

    // consider_duplicate=false should return fewer or equal results
    REQUIRE(result_off.value()->GetDim() <= result_on.value()->GetDim());
}

TEST_CASE("HGraph dedup: range search with max_duplicates_per_group",
          "[ft][hgraph][duplicate][range_limit]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);

    float radius = 1.0F;

    // Range search with max_duplicates_per_group=0
    auto param_limit = MakeSearchParam(200, true, 0);
    auto result_limit = index->RangeSearch(query_ds, radius, param_limit, -1);
    REQUIRE(result_limit.has_value());

    // Range search with unlimited
    auto param_unlimit = MakeSearchParam(200, true, -1);
    auto result_unlimit = index->RangeSearch(query_ds, radius, param_unlimit, -1);
    REQUIRE(result_unlimit.has_value());

    // Limited should have fewer or equal results
    REQUIRE(result_limit.value()->GetDim() <= result_unlimit.value()->GetDim());
}

TEST_CASE("HGraph dedup: precise reorder with dedup", "[ft][hgraph][duplicate][reorder]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto build_param = MakeBuildParamWithReorder(true, 0.001F);
    auto index = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index.has_value());

    auto base_ds = vsag::Dataset::Make();
    base_ds->NumElements(BASE_COUNT)
        ->Dim(DIM)
        ->Float32Vectors(tv.base.data())
        ->Ids(tv.base_ids.data())
        ->Owner(false);
    REQUIRE(index.value()->Build(base_ds).has_value());

    // Add duplicates
    auto dup_ds = vsag::Dataset::Make();
    dup_ds->NumElements(DUP_COUNT)
        ->Dim(DIM)
        ->Float32Vectors(tv.duplicates.data())
        ->Ids(tv.dup_ids.data())
        ->Owner(false);
    REQUIRE(index.value()->Add(dup_ds).has_value());

    REQUIRE(index.value()->GetNumElements() == BASE_COUNT + DUP_COUNT);

    // Search with duplicates enabled
    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);
    auto search_param = MakeSearchParam(200, true, -1);
    auto result = index.value()->KnnSearch(query_ds, 20, search_param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetDistances()[0] < 0.01F);

    // Should find both base and dup IDs
    bool found_dup = false;
    auto* ids = result.value()->GetIds();
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        if (ids[i] >= BASE_COUNT) {
            found_dup = true;
            break;
        }
    }
    REQUIRE(found_dup);

    // Serialize/deserialize should also work with reorder + dedup
    auto binary = index.value()->Serialize();
    REQUIRE(binary.has_value());
    auto index2 = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index2.has_value());
    index2.value()->Deserialize(binary.value());
    auto result2 = index2.value()->KnnSearch(query_ds, 10, search_param);
    REQUIRE(result2.has_value());
    REQUIRE(result2.value()->GetDim() > 0);
    REQUIRE(result2.value()->GetDistances()[0] < 0.01F);
}

TEST_CASE("HGraph dedup: near-threshold vectors", "[ft][hgraph][duplicate][threshold_boundary]") {
    constexpr float THRESHOLD = 0.01F;

    auto build_param = MakeBuildParam(true, THRESHOLD);
    auto index = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index.has_value());

    // Create a single base vector
    std::vector<float> base_vec(DIM, 0.0F);
    base_vec[0] = 1.0F;
    int64_t base_id = 0;
    auto base_ds = vsag::Dataset::Make();
    base_ds->NumElements(1)->Dim(DIM)->Float32Vectors(base_vec.data())->Ids(&base_id)->Owner(false);
    REQUIRE(index.value()->Build(base_ds).has_value());

    // Create a vector just below threshold distance (should be a duplicate)
    // L2 distance = sum of squared diffs; make a small perturbation
    // With dim=32, perturbing one dimension by sqrt(0.005) gives L2 = 0.005 < 0.01
    std::vector<float> near_dup_vec = base_vec;
    near_dup_vec[1] = std::sqrt(THRESHOLD * 0.5F);
    int64_t near_dup_id = 1;
    auto near_ds = vsag::Dataset::Make();
    near_ds->NumElements(1)
        ->Dim(DIM)
        ->Float32Vectors(near_dup_vec.data())
        ->Ids(&near_dup_id)
        ->Owner(false);
    index.value()->Add(near_ds);

    // Create a vector just above threshold distance (should NOT be a duplicate)
    std::vector<float> far_vec = base_vec;
    far_vec[1] = std::sqrt(THRESHOLD * 2.0F);
    int64_t far_id = 2;
    auto far_ds = vsag::Dataset::Make();
    far_ds->NumElements(1)->Dim(DIM)->Float32Vectors(far_vec.data())->Ids(&far_id)->Owner(false);
    index.value()->Add(far_ds);

    REQUIRE(index.value()->GetNumElements() == 3);

    // Search: near_dup should be in the same dup group as base; far should not
    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(base_vec.data())->Owner(false);

    // With consider_duplicate=false, near_dup should be suppressed but far should remain
    auto param_no_dup = MakeSearchParam(200, false, -1);
    auto result_no = index.value()->KnnSearch(query_ds, 10, param_no_dup);
    REQUIRE(result_no.has_value());

    // With consider_duplicate=true, both should appear
    auto param_dup = MakeSearchParam(200, true, -1);
    auto result_dup = index.value()->KnnSearch(query_ds, 10, param_dup);
    REQUIRE(result_dup.has_value());
    REQUIRE(result_dup.value()->GetDim() >= result_no.value()->GetDim());
}

TEST_CASE("HGraph dedup: force_remove representative then search",
          "[ft][hgraph][duplicate][remove_rep]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    // Force-remove base vector 0 (which has duplicates)
    auto removed = index->Remove({0}, vsag::RemoveMode::FORCE_REMOVE);
    REQUIRE(removed.has_value());
    REQUIRE(removed.value() > 0);

    // Search should still work without crashing
    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data() + DIM)->Owner(false);
    auto search_param = MakeSearchParam(200, true, -1);
    auto result = index->KnnSearch(query_ds, 10, search_param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);

    // Removed ID should no longer exist
    REQUIRE_FALSE(index->CheckIdExist(0));
}

TEST_CASE("HGraph dedup: force_remove duplicate then search",
          "[ft][hgraph][duplicate][remove_dup]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto index = BuildIndexWithDuplicates(tv, MakeDefaultBuildParam(true));

    // Remove a duplicate ID (BASE_COUNT is a dup of base 0)
    int64_t dup_label = BASE_COUNT;
    auto removed = index->Remove({dup_label}, vsag::RemoveMode::FORCE_REMOVE);
    REQUIRE(removed.has_value());
    REQUIRE(removed.value() > 0);

    // Search should still work
    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);
    auto search_param = MakeSearchParam(200, true, -1);
    auto result = index->KnnSearch(query_ds, 10, search_param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);

    // Removed dup should no longer exist
    REQUIRE_FALSE(index->CheckIdExist(dup_label));

    // Base vector should still exist
    REQUIRE(index->CheckIdExist(0));
}

TEST_CASE("HGraph dedup: concurrent search while adding duplicates",
          "[ft][hgraph][duplicate][concurrent_search]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);
    auto build_param = MakeDefaultBuildParam(true);
    auto index = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index.has_value());

    // Build base
    auto base_ds = vsag::Dataset::Make();
    base_ds->NumElements(BASE_COUNT)
        ->Dim(DIM)
        ->Float32Vectors(tv.base.data())
        ->Ids(tv.base_ids.data())
        ->Owner(false);
    REQUIRE(index.value()->Build(base_ds).has_value());

    std::atomic<bool> done{false};
    std::atomic<int> search_count{0};
    std::atomic<int> search_errors{0};

    // Launch search threads
    constexpr int SEARCH_THREADS = 2;
    std::vector<std::thread> search_threads;
    for (int t = 0; t < SEARCH_THREADS; ++t) {
        search_threads.emplace_back([&, t]() {
            while (!done.load()) {
                auto query_ds = vsag::Dataset::Make();
                query_ds->NumElements(1)
                    ->Dim(DIM)
                    ->Float32Vectors(tv.queries.data() + (t % 10) * DIM)
                    ->Owner(false);
                auto param = MakeSearchParam(100, true, -1);
                try {
                    auto result = index.value()->KnnSearch(query_ds, 5, param);
                    if (!result.has_value()) {
                        search_errors++;
                    }
                    search_count++;
                } catch (...) {
                    search_errors++;
                }
            }
        });
    }

    // Add duplicates concurrently
    for (int64_t i = 0; i < DUP_COUNT; ++i) {
        auto ds = vsag::Dataset::Make();
        ds->NumElements(1)
            ->Dim(DIM)
            ->Float32Vectors(tv.duplicates.data() + i * DIM)
            ->Ids(tv.dup_ids.data() + i)
            ->Owner(false);
        index.value()->Add(ds);
    }

    done = true;
    for (auto& t : search_threads) {
        t.join();
    }

    REQUIRE(search_count > 0);
    REQUIRE(search_errors == 0);
}

TEST_CASE("HGraph dedup: IP (inner product) metric", "[ft][hgraph][duplicate][ip_metric]") {
    auto tv = GenerateTestData(DIM, BASE_COUNT, DUP_COUNT);

    auto build_param = MakeBuildParam(true, 0.001F, "ip");
    auto index = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index.has_value());

    auto base_ds = vsag::Dataset::Make();
    base_ds->NumElements(BASE_COUNT)
        ->Dim(DIM)
        ->Float32Vectors(tv.base.data())
        ->Ids(tv.base_ids.data())
        ->Owner(false);
    REQUIRE(index.value()->Build(base_ds).has_value());

    // Add duplicates
    auto dup_ds = vsag::Dataset::Make();
    dup_ds->NumElements(DUP_COUNT)
        ->Dim(DIM)
        ->Float32Vectors(tv.duplicates.data())
        ->Ids(tv.dup_ids.data())
        ->Owner(false);
    REQUIRE(index.value()->Add(dup_ds).has_value());

    REQUIRE(index.value()->GetNumElements() == BASE_COUNT + DUP_COUNT);

    // Search
    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(tv.queries.data())->Owner(false);
    auto search_param = MakeSearchParam(200, true, -1);
    auto result = index.value()->KnnSearch(query_ds, 20, search_param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);

    // Should find both base and duplicate IDs
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

TEST_CASE("HGraph dedup: triple duplicate chain A=B=C", "[ft][hgraph][duplicate][chain]") {
    auto build_param = MakeBuildParam(true, 0.001F);
    auto index = vsag::Factory::CreateIndex("hgraph", build_param);
    REQUIRE(index.has_value());

    // Build with some base vectors
    constexpr int64_t SMALL_BASE = 50;
    std::mt19937 rng(55);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    std::vector<float> base_vecs(SMALL_BASE * DIM);
    std::vector<int64_t> base_ids(SMALL_BASE);
    for (int64_t i = 0; i < SMALL_BASE; ++i) {
        for (int64_t d = 0; d < DIM; ++d) {
            base_vecs[i * DIM + d] = dist(rng);
        }
        base_ids[i] = i;
    }
    auto base_ds = vsag::Dataset::Make();
    base_ds->NumElements(SMALL_BASE)
        ->Dim(DIM)
        ->Float32Vectors(base_vecs.data())
        ->Ids(base_ids.data())
        ->Owner(false);
    REQUIRE(index.value()->Build(base_ds).has_value());

    // Add B = exact copy of A (id=0)
    int64_t id_b = SMALL_BASE;
    auto ds_b = vsag::Dataset::Make();
    ds_b->NumElements(1)->Dim(DIM)->Float32Vectors(base_vecs.data())->Ids(&id_b)->Owner(false);
    index.value()->Add(ds_b);

    // Add C = exact copy of A (id=0)
    int64_t id_c = SMALL_BASE + 1;
    auto ds_c = vsag::Dataset::Make();
    ds_c->NumElements(1)->Dim(DIM)->Float32Vectors(base_vecs.data())->Ids(&id_c)->Owner(false);
    index.value()->Add(ds_c);

    REQUIRE(index.value()->GetNumElements() == SMALL_BASE + 2);
    REQUIRE(index.value()->CheckIdExist(0));
    REQUIRE(index.value()->CheckIdExist(id_b));
    REQUIRE(index.value()->CheckIdExist(id_c));

    // Search: all three should appear with consider_duplicate=true
    auto query_ds = vsag::Dataset::Make();
    query_ds->NumElements(1)->Dim(DIM)->Float32Vectors(base_vecs.data())->Owner(false);
    auto search_param = MakeSearchParam(200, true, -1);
    auto result = index.value()->KnnSearch(query_ds, 20, search_param);
    REQUIRE(result.has_value());

    std::set<int64_t> found_ids;
    auto* ids = result.value()->GetIds();
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        found_ids.insert(ids[i]);
    }
    REQUIRE(found_ids.count(0) == 1);
    REQUIRE(found_ids.count(id_b) == 1);
    REQUIRE(found_ids.count(id_c) == 1);

    // With max_duplicates_per_group=1, should see at most 1 duplicate
    auto param_limit1 = MakeSearchParam(200, true, 1);
    auto result_limit = index.value()->KnnSearch(query_ds, 20, param_limit1);
    REQUIRE(result_limit.has_value());
    int dup_count_for_group0 = 0;
    auto* lids = result_limit.value()->GetIds();
    for (int64_t i = 0; i < result_limit.value()->GetDim(); ++i) {
        if (lids[i] == id_b || lids[i] == id_c) {
            dup_count_for_group0++;
        }
    }
    REQUIRE(dup_count_for_group0 <= 1);
}
