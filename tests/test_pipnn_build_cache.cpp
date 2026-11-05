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

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "algorithm/build_cache.h"
#include "algorithm/pyramid/pyramid_build_cache.h"
#include "impl/allocator/safe_allocator.h"
#include "json_wrapper.h"
#include "vsag/vsag.h"

TEST_CASE("PiPNN public cache round trip", "[ft][pipnn_cache]") {
    const std::string index_type = GENERATE("hgraph", "pyramid");
    const std::string metric = GENERATE("l2", "ip", "cosine");
    const auto threads = GENERATE(1, 4);
    CAPTURE(index_type, metric, threads);
    constexpr uint64_t count = 384;
    constexpr uint64_t dim = 16;
    auto config = vsag::JsonType::Parse(R"({
        "dtype":"float32", "metric_type":"l2", "dim":16,
        "index_param":{"graph_type":"pipnn", "max_degree":8,
            "ef_construction":32,"build_thread_count":1,
            "base_quantization_type":"fp32","persist_source_id":true}
    })");
    if (index_type == "pyramid") {
        config["index_param"]["index_min_size"].SetInt(24);
        config["index_param"]["root_graph_type"].SetString(threads == 1 ? "single_layer"
                                                                        : "multi_layer");
    }
    config["index_param"]["pipnn_max_leaf_size"].SetInt(128);
    config["index_param"]["pipnn_min_leaf_size"].SetInt(16);
    config["index_param"]["pipnn_leaf_neighbor_count"].SetInt(16);
    config["metric_type"].SetString(metric);
    config["index_param"]["build_thread_count"].SetInt(threads);
    std::mt19937 generator(42);
    std::normal_distribution<float> distribution;
    std::vector<float> vectors(count * dim);
    std::vector<int64_t> labels(count);
    std::vector<std::string> sources(count);
    std::vector<std::vector<std::string>> paths(count);
    for (uint64_t i = 0; i < count; ++i) {
        labels[i] = static_cast<int64_t>(i);
        sources[i] = "source-" + std::to_string(i);
        paths[i] = {i % 2 == 0 ? "a/leaf" : "b/leaf"};
        for (uint64_t j = 0; j < dim; ++j) {
            vectors[i * dim + j] = distribution(generator);
        }
    }
    auto dataset = [&]() {
        auto result = vsag::Dataset::Make();
        result->NumElements(count)
            ->Dim(dim)
            ->Ids(labels.data())
            ->Float32Vectors(vectors.data())
            ->SourceID(sources.data())
            ->Owner(false);
        if (index_type == "pyramid") {
            result->Paths("", paths);
        }
        return result;
    };
    auto create = [&]() {
        auto created = vsag::Factory::CreateIndex(index_type, config.Dump());
        REQUIRE(created.has_value());
        return created.value();
    };
    auto original = create();
    REQUIRE(original->Build(dataset()).has_value());
    std::stringstream buffer;
    REQUIRE(original->ExportCache(buffer).has_value());
    const auto bytes = buffer.str();
    REQUIRE_FALSE(bytes.empty());
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    std::stringstream inspect(bytes);
    vsag::IOStreamReader reader(inspect);
    if (index_type == "hgraph") {
        vsag::BuildCache cache(allocator.get());
        cache.Deserialize(reader);
        REQUIRE(cache.pipnn != nullptr);
        REQUIRE(cache.pipnn->source_ids.size() == count);
    } else {
        vsag::PyramidBuildCache cache(allocator.get());
        cache.Deserialize(reader);
        const auto* root = cache.GetGraphCache("", "");
        REQUIRE(root != nullptr);
        REQUIRE(root->pipnn != nullptr);
        REQUIRE(cache.GetGraphCache("", "a/leaf")->pipnn != nullptr);
    }
    SECTION("unchanged data reuses all rows") {
        auto warmed = create();
        std::stringstream input(bytes);
        REQUIRE(warmed->ImportCache(input).has_value());
        REQUIRE(warmed->Build(dataset()).has_value());
        const auto stats = vsag::JsonType::Parse(warmed->GetStats());
        INFO(warmed->GetStats());
        REQUIRE(stats["build_cache_hit_rate"].GetFloat() == 1.0F);
        REQUIRE(stats["pipnn_build"]["reused_graph_rows"].GetUint64() >= count);
        std::stringstream next;
        REQUIRE(warmed->ExportCache(next).has_value());
        auto next_index = create();
        REQUIRE(next_index->ImportCache(next).has_value());
        REQUIRE(next_index->Build(dataset()).has_value());
    }
    SECTION("reordered replaced and modified sources remain searchable") {
        std::reverse(labels.begin(), labels.end());
        std::reverse(sources.begin(), sources.end());
        std::reverse(paths.begin(), paths.end());
        for (uint64_t i = 0; i < count / 2; ++i) {
            std::swap_ranges(vectors.data() + i * dim,
                             vectors.data() + (i + 1) * dim,
                             vectors.data() + (count - 1 - i) * dim);
        }
        for (uint64_t i = 0; i < count / 20; ++i) {
            sources[i] = "new-" + std::to_string(i);
            labels[i] += count;
            vectors[i * dim] += 0.25F;
        }
        // Same source identity with different content must miss too.
        vectors[(count - 1) * dim] += 0.5F;
        auto warmed = create();
        std::stringstream input(bytes);
        REQUIRE(warmed->ImportCache(input).has_value());
        REQUIRE(warmed->Build(dataset()).has_value());
        const auto stats = vsag::JsonType::Parse(warmed->GetStats());
        REQUIRE(stats["build_cache_hit_rate"].GetFloat() > 0.0F);
        REQUIRE(stats["build_cache_hit_rate"].GetFloat() < 1.0F);
        for (uint64_t i : {uint64_t{0}, count - 1}) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data() + i * dim)->Owner(false);
            if (index_type == "pyramid") {
                query->Paths("", std::vector<std::vector<std::string>>{paths[i]});
            }
            const auto* search = index_type == "hgraph" ? R"({"hgraph":{"ef_search":384}})"
                                                        : R"({"pyramid":{"ef_search":384}})";
            auto result = warmed->KnnSearch(query, 10, search);
            REQUIRE(result.has_value());
            const auto* found = result.value()->GetIds();
            REQUIRE(result.value()->GetDim() > 0);
            if (metric != "ip") {
                REQUIRE(std::find(found, found + result.value()->GetDim(), labels[i]) !=
                        found + result.value()->GetDim());
            } else {
                REQUIRE(std::find(labels.begin(), labels.end(), found[0]) != labels.end());
            }
        }
    }
    SECTION("incompatible policies and missing identities fall back to cold build") {
        config["index_param"]["max_degree"].SetInt(24);
        auto warmed = create();
        std::stringstream input(bytes);
        REQUIRE(warmed->ImportCache(input).has_value());
        REQUIRE(warmed->Build(dataset()).has_value());
        REQUIRE(vsag::JsonType::Parse(warmed->GetStats())["build_cache_hit_rate"].GetFloat() ==
                0.0F);
        auto anonymous = create();
        std::stringstream second(bytes);
        REQUIRE(anonymous->ImportCache(second).has_value());
        auto base = dataset();
        base->SourceID(nullptr);
        REQUIRE(anonymous->Build(base).has_value());
        REQUIRE(vsag::JsonType::Parse(anonymous->GetStats())["build_cache_hit_rate"].GetFloat() ==
                0.0F);
    }
    SECTION("SQ8 storage reuses the FP32 construction snapshot") {
        config["index_param"]["base_quantization_type"].SetString("sq8");
        auto warmed = create();
        std::stringstream input(bytes);
        REQUIRE(warmed->ImportCache(input).has_value());
        REQUIRE(warmed->Build(dataset()).has_value());
        REQUIRE(vsag::JsonType::Parse(warmed->GetStats())["build_cache_hit_rate"].GetFloat() ==
                1.0F);
    }
    SECTION("low eligible overlap uses the ordinary cold kernel") {
        for (uint64_t i = count * 3 / 4; i < count; ++i) {
            sources[i] = "replacement-" + std::to_string(i);
        }
        auto warmed = create();
        std::stringstream input(bytes);
        REQUIRE(warmed->ImportCache(input).has_value());
        REQUIRE(warmed->Build(dataset()).has_value());
        REQUIRE(vsag::JsonType::Parse(warmed->GetStats())["build_cache_hit_rate"].GetFloat() ==
                0.0F);
    }
    SECTION("alpha changes invalidate the policy") {
        config["index_param"]["alpha"].SetFloat(1.4F);
        auto warmed = create();
        std::stringstream input(bytes);
        REQUIRE(warmed->ImportCache(input).has_value());
        REQUIRE(warmed->Build(dataset()).has_value());
        REQUIRE(vsag::JsonType::Parse(warmed->GetStats())["build_cache_hit_rate"].GetFloat() ==
                0.0F);
    }
    SECTION("Pyramid isolates named hierarchies and moved memberships") {
        if (index_type == "pyramid") {
            config["index_param"]["hierarchies"].SetJson(vsag::JsonType::Parse(R"([
                {"name":"single", "no_build_levels":[], "root_graph_type":"single_layer"},
                {"name":"multi", "no_build_levels":[], "root_graph_type":"multi_layer"}
            ])"));
            auto make_named = [&]() {
                auto input = dataset();
                input->Paths("single", paths)->Paths("multi", paths);
                return input;
            };
            auto first = create();
            REQUIRE(first->Build(make_named()).has_value());
            std::stringstream named;
            REQUIRE(first->ExportCache(named).has_value());
            auto warmed = create();
            REQUIRE(warmed->ImportCache(named).has_value());
            auto moved = make_named();
            moved->Paths("multi", std::vector<std::vector<std::string>>(count, {"fresh/leaf"}));
            REQUIRE(warmed->Build(moved).has_value());
            const auto stats = vsag::JsonType::Parse(warmed->GetStats());
            REQUIRE(stats["build_cache_hit_nodes"].GetUint64() == count);
            REQUIRE(stats["build_cache_hit_memberships"].GetUint64() > count);
            REQUIRE(stats["build_cache_missed_memberships"].GetUint64() > 0);
        }
    }
    SECTION("duplicate groups fall back without exporting an ordinary reservoir") {
        config["index_param"]["support_duplicate"].SetBool(true);
        auto warmed = create();
        std::stringstream input(bytes);
        REQUIRE(warmed->ImportCache(input).has_value());
        REQUIRE(warmed->Build(dataset()).has_value());
        REQUIRE(vsag::JsonType::Parse(warmed->GetStats())["build_cache_hit_rate"].GetFloat() ==
                0.0F);
    }
    SECTION("adaptive pruning matches only an identical policy") {
        if (metric == "l2") {
            config["index_param"]["pipnn_adaptive_pruning"].SetBool(true);
            auto first = create();
            REQUIRE(first->Build(dataset()).has_value());
            std::stringstream adaptive;
            REQUIRE(first->ExportCache(adaptive).has_value());
            auto warmed = create();
            REQUIRE(warmed->ImportCache(adaptive).has_value());
            REQUIRE(warmed->Build(dataset()).has_value());
            REQUIRE(vsag::JsonType::Parse(warmed->GetStats())["build_cache_hit_rate"].GetFloat() ==
                    1.0F);
        }
    }
    SECTION("truncated import preserves an already imported cache") {
        auto warmed = create();
        std::stringstream input(bytes);
        REQUIRE(warmed->ImportCache(input).has_value());
        std::stringstream corrupt(bytes.substr(0, bytes.size() - 1));
        REQUIRE_FALSE(warmed->ImportCache(corrupt).has_value());
        REQUIRE(warmed->Build(dataset()).has_value());
        REQUIRE(vsag::JsonType::Parse(warmed->GetStats())["build_cache_hit_rate"].GetFloat() ==
                1.0F);
    }
}

TEST_CASE("PiPNN cache version and structural validation", "[ft][pipnn_cache]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::BuildCache original(allocator.get());
    original.source_ids_.emplace_back("a");
    original.pipnn = std::make_shared<vsag::PiPNNBuildCache>();
    original.pipnn->policy = "policy";
    original.pipnn->source_ids = {"a", "b"};
    original.pipnn->fingerprints = {123, 456};
    original.pipnn->offsets = {0, 1, 2};
    original.pipnn->targets = {1, 0};
    original.pipnn->distances = {42, 42};
    original.pipnn->hashes = {0, 0};
    original.pipnn->inner_ids = {0, 1};
    original.pipnn->graph_offsets = {0, 1, 2};
    original.pipnn->graph_targets = {1, 0};
    original.pipnn->graph_distances = {42, 42};
    std::stringstream encoded;
    vsag::IOStreamWriter writer(encoded);
    original.Serialize(writer);
    vsag::BuildCache restored(allocator.get());
    vsag::IOStreamReader reader(encoded);
    restored.Deserialize(reader);
    REQUIRE(restored.pipnn->targets == original.pipnn->targets);
    REQUIRE(restored.pipnn->fingerprints == original.pipnn->fingerprints);
    SECTION("unknown version is rejected without changing state") {
        auto bytes = encoded.str();
        const uint64_t version = 99;
        std::memcpy(bytes.data() + sizeof(uint64_t), &version, sizeof(version));
        std::stringstream input(bytes);
        vsag::IOStreamReader bad(input);
        REQUIRE_THROWS(restored.Deserialize(bad));
        REQUIRE(restored.pipnn->policy == "policy");
    }
    SECTION("nonmonotonic offsets are rejected") {
        original.pipnn->offsets = {0, 3, 2};
        std::stringstream input;
        vsag::IOStreamWriter output(input);
        original.Serialize(output);
        vsag::IOStreamReader bad(input);
        REQUIRE_THROWS(restored.Deserialize(bad));
    }
    SECTION("out of range target is rejected") {
        original.pipnn->targets[0] = 2;
        std::stringstream input;
        vsag::IOStreamWriter output(input);
        original.Serialize(output);
        vsag::IOStreamReader bad(input);
        REQUIRE_THROWS(restored.Deserialize(bad));
    }
    SECTION("legacy payload retains its format") {
        original.pipnn.reset();
        std::stringstream input;
        vsag::IOStreamWriter output(input);
        original.Serialize(output);
        uint64_t count = 0;
        const auto bytes = input.str();
        std::memcpy(&count, bytes.data(), sizeof(count));
        REQUIRE(count == 1);
        vsag::IOStreamReader legacy(input);
        restored.Deserialize(legacy);
        REQUIRE(restored.pipnn == nullptr);
        REQUIRE(restored.source_ids_[0] == "a");
    }
}
