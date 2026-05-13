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

#include "vsag/factory.h"

#include <cmath>
#include <sstream>

#include "unittest.h"

using namespace vsag;

namespace {

DatasetPtr
make_feature_dataset() {
    constexpr int64_t kDim = 4;
    constexpr int64_t kCount = 4;

    auto* ids = new int64_t[kCount]{100, 101, 102, 103};
    auto* vectors = new float[kCount * kDim]{0.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             1.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             1.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             1.0F,
                                             0.0F};
    auto* feature_ids = new std::string[kCount]{"fid-100", "fid-101", "fid-102", "fid-103"};

    return Dataset::Make()
        ->Owner(true)
        ->NumElements(kCount)
        ->Dim(kDim)
        ->Ids(ids)
        ->Float32Vectors(vectors)
        ->FeatureIds(feature_ids);
}

DatasetPtr
make_feature_only_dataset_with_modified_labels() {
    constexpr int64_t kCount = 4;

    auto* ids = new int64_t[kCount]{100, 101, 102, 999};
    auto* feature_ids = new std::string[kCount]{"fid-100", "fid-101", "fid-102", "fid-103"};

    return Dataset::Make()->Owner(true)->NumElements(kCount)->Ids(ids)->FeatureIds(feature_ids);
}

DatasetPtr
make_partial_hit_dataset() {
    constexpr int64_t kDim = 4;
    constexpr int64_t kCount = 4;

    auto* ids = new int64_t[kCount]{100, 101, 102, 200};
    auto* vectors = new float[kCount * kDim]{0.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             1.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             1.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             0.0F,
                                             1.0F};
    auto* feature_ids = new std::string[kCount]{"fid-100", "fid-101", "fid-102", "fid-200"};

    return Dataset::Make()
        ->Owner(true)
        ->NumElements(kCount)
        ->Dim(kDim)
        ->Ids(ids)
        ->Float32Vectors(vectors)
        ->FeatureIds(feature_ids);
}

std::string
make_hgraph_parameters() {
    auto parameters = JsonType::Parse(R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 4,
        "hgraph": {
            "max_degree": 4,
            "ef_construction": 16
        }
    }
    )");
    return parameters.Dump();
}

}  // namespace

TEST_CASE("HGraph Build Cache Round Trip", "[ut][HGraph][BuildCache]") {
    auto dataset = make_feature_dataset();
    auto params = make_hgraph_parameters();

    auto index = Factory::CreateIndex("hgraph", params);
    REQUIRE(index.has_value());
    REQUIRE(index.value()->SupportsBuildCache());

    auto build_result = index.value()->Build(dataset);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    std::stringstream cache_stream(std::ios::in | std::ios::out | std::ios::binary);
    auto export_result = index.value()->ExportBuildCache(cache_stream);
    REQUIRE(export_result.has_value());
    REQUIRE_FALSE(cache_stream.str().empty());

    auto cached_index = Factory::CreateIndex("hgraph", params);
    REQUIRE(cached_index.has_value());
    BuildCacheOptions options;
    cache_stream.seekg(0);
    auto cached_build = cached_index.value()->BuildWithCache(dataset, cache_stream, options);
    REQUIRE(cached_build.has_value());
    REQUIRE(cached_build.value().empty());

    auto stats = cached_index.value()->GetBuildCacheStats();
    REQUIRE(stats.has_value());
    REQUIRE(stats.value().total_nodes == 4);
    REQUIRE(stats.value().cached_nodes == 4);
    REQUIRE(stats.value().hit_nodes == 4);
    REQUIRE(stats.value().missed_nodes == 0);
    REQUIRE(std::fabs(stats.value().cache_hit_rate - 1.0F) < 1e-6F);
}

TEST_CASE("HGraph Build Cache Survives Serialize Round Trip", "[ut][HGraph][BuildCache]") {
    auto dataset = make_feature_dataset();
    auto params = make_hgraph_parameters();

    auto index = Factory::CreateIndex("hgraph", params);
    REQUIRE(index.has_value());

    auto build_result = index.value()->Build(dataset);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    std::stringstream index_stream(std::ios::in | std::ios::out | std::ios::binary);
    auto serialize_result = index.value()->Serialize(index_stream);
    REQUIRE(serialize_result.has_value());
    REQUIRE_FALSE(index_stream.str().empty());

    auto restored_index = Factory::CreateIndex("hgraph", params);
    REQUIRE(restored_index.has_value());
    index_stream.seekg(0);
    auto deserialize_result = restored_index.value()->Deserialize(index_stream);
    REQUIRE(deserialize_result.has_value());

    std::stringstream cache_stream(std::ios::in | std::ios::out | std::ios::binary);
    auto export_result = restored_index.value()->ExportBuildCache(cache_stream);
    REQUIRE(export_result.has_value());
    REQUIRE_FALSE(cache_stream.str().empty());
}

TEST_CASE("HGraph Build Cache Can Skip Route Graph Build", "[ut][HGraph][BuildCache]") {
    auto dataset = make_feature_dataset();
    auto params = make_hgraph_parameters();

    auto index = Factory::CreateIndex("hgraph", params);
    REQUIRE(index.has_value());

    auto build_result = index.value()->Build(dataset);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    std::stringstream cache_stream(std::ios::in | std::ios::out | std::ios::binary);
    auto export_result = index.value()->ExportBuildCache(cache_stream);
    REQUIRE(export_result.has_value());

    auto cached_index = Factory::CreateIndex("hgraph", params);
    REQUIRE(cached_index.has_value());
    BuildCacheOptions options;
    options.hit_refine_rounds = 0;
    options.missed_refine_rounds = 0;
    options.build_route_graph = false;

    cache_stream.seekg(0);
    auto cached_build = cached_index.value()->BuildWithCache(dataset, cache_stream, options);
    REQUIRE(cached_build.has_value());
    REQUIRE(cached_build.value().empty());

    auto stats = cached_index.value()->GetBuildCacheStats();
    REQUIRE(stats.has_value());
    REQUIRE(stats.value().route_graph_levels == 0);
    REQUIRE(stats.value().route_graph_build_us == 0);

    std::stringstream index_stream(std::ios::in | std::ios::out | std::ios::binary);
    auto serialize_result = cached_index.value()->Serialize(index_stream);
    REQUIRE(serialize_result.has_value());
    REQUIRE_FALSE(index_stream.str().empty());

    auto restored_index = Factory::CreateIndex("hgraph", params);
    REQUIRE(restored_index.has_value());
    index_stream.seekg(0);
    auto deserialize_result = restored_index.value()->Deserialize(index_stream);
    REQUIRE(deserialize_result.has_value());
}

TEST_CASE("HGraph Build Cache Refine Tracks Partial Hits", "[ut][HGraph][BuildCache]") {
    auto day1 = make_feature_dataset();
    auto day2 = make_partial_hit_dataset();
    auto params = make_hgraph_parameters();

    auto index = Factory::CreateIndex("hgraph", params);
    REQUIRE(index.has_value());

    auto build_result = index.value()->Build(day1);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    std::stringstream cache_stream(std::ios::in | std::ios::out | std::ios::binary);
    auto export_result = index.value()->ExportBuildCache(cache_stream);
    REQUIRE(export_result.has_value());

    auto cached_index = Factory::CreateIndex("hgraph", params);
    REQUIRE(cached_index.has_value());

    BuildCacheOptions options;
    options.hit_refine_rounds = 1;
    options.missed_refine_rounds = 4;
    options.build_route_graph = false;

    cache_stream.seekg(0);
    auto cached_build = cached_index.value()->BuildWithCache(day2, cache_stream, options);
    REQUIRE(cached_build.has_value());
    REQUIRE(cached_build.value().empty());

    auto stats = cached_index.value()->GetBuildCacheStats();
    REQUIRE(stats.has_value());
    REQUIRE(stats.value().total_nodes == 4);
    REQUIRE(stats.value().cached_nodes == 4);
    REQUIRE(stats.value().hit_nodes == 3);
    REQUIRE(stats.value().missed_nodes == 1);
    REQUIRE(stats.value().hit_refine_rounds == 1);
    REQUIRE(stats.value().missed_refine_rounds == 4);
    REQUIRE(stats.value().missed_empty_seed_nodes == 1);
    REQUIRE(stats.value().missed_seed_neighbor_total == 0);
    REQUIRE(stats.value().hit_seed_neighbor_total > 0);
    REQUIRE(stats.value().hit_refine_us > 0);
    REQUIRE(stats.value().missed_refine_us > 0);
}

TEST_CASE("HGraph Build Cache Supports Parallel Refine", "[ut][HGraph][BuildCache]") {
    auto day1 = make_feature_dataset();
    auto day2 = make_partial_hit_dataset();
    auto params = make_hgraph_parameters();

    auto index = Factory::CreateIndex("hgraph", params);
    REQUIRE(index.has_value());

    auto build_result = index.value()->Build(day1);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    std::stringstream cache_stream(std::ios::in | std::ios::out | std::ios::binary);
    auto export_result = index.value()->ExportBuildCache(cache_stream);
    REQUIRE(export_result.has_value());

    auto cached_index = Factory::CreateIndex("hgraph", params);
    REQUIRE(cached_index.has_value());

    BuildCacheOptions options;
    options.hit_refine_rounds = 1;
    options.missed_refine_rounds = 2;
    options.enable_parallel_refine = true;
    options.refine_parallelism = 2;
    options.build_route_graph = false;

    cache_stream.seekg(0);
    auto cached_build = cached_index.value()->BuildWithCache(day2, cache_stream, options);
    REQUIRE(cached_build.has_value());
    REQUIRE(cached_build.value().empty());

    auto stats = cached_index.value()->GetBuildCacheStats();
    REQUIRE(stats.has_value());
    REQUIRE(stats.value().hit_refine_rounds == 1);
    REQUIRE(stats.value().missed_refine_rounds == 2);
    REQUIRE(stats.value().hit_refine_parallelism == 2);
    REQUIRE(stats.value().missed_refine_parallelism == 1);
}

TEST_CASE("HGraph PrepareFeatureIdsForBuildCache Falls Back To Row Order", "[ut][HGraph][BuildCache]") {
    auto dataset = make_feature_dataset();
    auto params = make_hgraph_parameters();

    auto index = Factory::CreateIndex("hgraph", params);
    REQUIRE(index.has_value());

    auto build_result = index.value()->Build(dataset);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    std::stringstream index_stream(std::ios::in | std::ios::out | std::ios::binary);
    auto serialize_result = index.value()->Serialize(index_stream);
    REQUIRE(serialize_result.has_value());

    auto restored_index = Factory::CreateIndex("hgraph", params);
    REQUIRE(restored_index.has_value());
    index_stream.seekg(0);
    auto deserialize_result = restored_index.value()->Deserialize(index_stream);
    REQUIRE(deserialize_result.has_value());

    auto feature_only_dataset = make_feature_only_dataset_with_modified_labels();
    auto prepare_result = restored_index.value()->PrepareFeatureIdsForBuildCache(feature_only_dataset);
    REQUIRE(prepare_result.has_value());

    std::stringstream cache_stream(std::ios::in | std::ios::out | std::ios::binary);
    auto export_result = restored_index.value()->ExportBuildCache(cache_stream);
    REQUIRE(export_result.has_value());
    REQUIRE_FALSE(cache_stream.str().empty());
}