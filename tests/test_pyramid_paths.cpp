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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <future>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "vsag/vsag.h"

namespace {

std::string
MakePyramidParameters(const std::string& graph_type, bool store_paths) {
    nlohmann::json index_param = {
        {"max_degree", 4},
        {"alpha", 1.2},
        {"ef_construction", 8},
        {"graph_type", graph_type},
        {"graph_iter_turn", 1},
        {"neighbor_sample_rate", 0.2},
        {"base_quantization_type", "fp32"},
        {"use_reorder", false},
        {"index_min_size", 100},
        {"no_build_levels", {0, 1, 2, 3}},
    };
    if (store_paths) {
        index_param["store_paths"] = true;
    }
    return nlohmann::json({{"dtype", "float32"},
                           {"metric_type", "l2"},
                           {"dim", 2},
                           {"index_param", std::move(index_param)}})
        .dump();
}

vsag::IndexPtr
MakePyramidIndex(const std::string& graph_type, bool store_paths) {
    auto result =
        vsag::Factory::CreateIndex("pyramid", MakePyramidParameters(graph_type, store_paths));
    REQUIRE(result.has_value());
    return result.value();
}

vsag::DatasetPtr
MakeDataset(const std::vector<int64_t>& ids, const std::vector<std::string>& paths) {
    REQUIRE(ids.size() == paths.size());
    auto* vectors = new float[ids.size() * 2];
    auto* raw_ids = new int64_t[ids.size()];
    auto* raw_paths = new std::string[paths.size()];
    for (uint64_t offset = 0; offset < ids.size(); ++offset) {
        vectors[offset * 2] = static_cast<float>(offset);
        vectors[offset * 2 + 1] = static_cast<float>(offset + 1);
        raw_ids[offset] = ids[offset];
        raw_paths[offset] = paths[offset];
    }

    return vsag::Dataset::Make()
        ->NumElements(static_cast<int64_t>(ids.size()))
        ->Dim(2)
        ->Float32Vectors(vectors)
        ->Ids(raw_ids)
        ->Paths(raw_paths)
        ->Owner(true);
}

vsag::DatasetPtr
GetData(const vsag::IndexPtr& index, const std::vector<int64_t>& ids) {
    auto result = index->GetDataByIds(ids.data(), static_cast<int64_t>(ids.size()));
    REQUIRE(result.has_value());
    return result.value();
}

vsag::DatasetPtr
GetDataWithFlag(const vsag::IndexPtr& index,
                const std::vector<int64_t>& ids,
                uint64_t selected_data_flag) {
    auto result = index->GetDataByIdsWithFlag(
        ids.data(), static_cast<int64_t>(ids.size()), selected_data_flag);
    REQUIRE(result.has_value());
    return result.value();
}

void
RequirePaths(const std::string* actual, const std::vector<std::string>& expected) {
    REQUIRE(actual != nullptr);
    for (uint64_t offset = 0; offset < expected.size(); ++offset) {
        REQUIRE(actual[offset] == expected[offset]);
    }
}

}  // namespace

TEST_CASE("Pyramid advertises GetDataByIds only when paths are retained",
          "[ft][pyramid][paths][feature]") {
    const auto graph_type = GENERATE(std::string("nsw"), std::string("odescent"));
    REQUIRE_FALSE(MakePyramidIndex(graph_type, false)->CheckFeature(vsag::SUPPORT_GET_DATA_BY_IDS));
    REQUIRE(MakePyramidIndex(graph_type, true)->CheckFeature(vsag::SUPPORT_GET_DATA_BY_IDS));
}

TEST_CASE("Pyramid NSW preserves legacy GetDataByIds when path storage is disabled",
          "[ft][pyramid][paths]") {
    auto index = MakePyramidIndex("nsw", false);
    REQUIRE(index->Build(MakeDataset({10, 20}, {"a", "b"})).has_value());

    // Feature discovery intentionally stays disabled without the optional sidecar; it is not an
    // invocation gate. NSW has always populated reverse label lookup, so legacy direct calls still
    // return IDs without retained paths.
    REQUIRE(GetData(index, {20, 10})->GetPaths() == nullptr);

    auto binary_set = index->Serialize();
    REQUIRE(binary_set.has_value());
    auto restored = MakePyramidIndex("nsw", false);
    REQUIRE(restored->Deserialize(binary_set.value()).has_value());
    REQUIRE(GetData(restored, {10, 20})->GetPaths() == nullptr);
}

TEST_CASE("Pyramid retained paths support empty-index serialization",
          "[ft][pyramid][paths][serialization]") {
    auto index = MakePyramidIndex("nsw", true);

    auto binary_set = index->Serialize();
    REQUIRE(binary_set.has_value());
    auto restored = MakePyramidIndex("nsw", true);
    REQUIRE(restored->Deserialize(binary_set.value()).has_value());
    REQUIRE(restored->Serialize().has_value());
}

TEST_CASE("Pyramid serialized path storage configuration must match",
          "[ft][pyramid][paths][serialization]") {
    SECTION("stored paths require an enabled reader") {
        auto index = MakePyramidIndex("nsw", true);
        REQUIRE(index->Build(MakeDataset({4}, {"path"})).has_value());
        auto binary_set = index->Serialize();
        REQUIRE(binary_set.has_value());

        auto restored = MakePyramidIndex("nsw", false);
        REQUIRE_FALSE(restored->Deserialize(binary_set.value()).has_value());
    }

    SECTION("a reader requiring paths rejects an index without them") {
        auto index = MakePyramidIndex("nsw", false);
        REQUIRE(index->Build(MakeDataset({4}, {"path"})).has_value());
        auto binary_set = index->Serialize();
        REQUIRE(binary_set.has_value());

        auto restored = MakePyramidIndex("nsw", true);
        REQUIRE_FALSE(restored->Deserialize(binary_set.value()).has_value());
    }
}

TEST_CASE("Pyramid retains paths for NSW and ODescent Build", "[ft][pyramid][paths]") {
    const auto graph_type = GENERATE(std::string("nsw"), std::string("odescent"));
    auto index = MakePyramidIndex(graph_type, true);
    REQUIRE(index->Build(MakeDataset({101, 42, 900}, {"alpha", "", "beta/gamma"})).has_value());

    auto data = GetDataWithFlag(index, {900, 101, 42}, DATA_FLAG_PATH);
    RequirePaths(data->GetPaths(), {"beta/gamma", "alpha", ""});
    REQUIRE(data->GetIds() == nullptr);
}

TEST_CASE("Pyramid Add stores paths for allocated inner IDs", "[ft][pyramid][paths]") {
    auto index = MakePyramidIndex("nsw", true);
    REQUIRE(index->Build(MakeDataset({10}, {"original"})).has_value());

    auto add_result =
        index->Add(MakeDataset({20, 10, 30}, {"path-20", "duplicate-path", "path-30"}));
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value() == std::vector<int64_t>{10});

    RequirePaths(GetDataWithFlag(index, {20, 30, 10}, DATA_FLAG_PATH)->GetPaths(),
                 {"path-20", "path-30", "original"});
}

TEST_CASE("Pyramid concurrent Add keeps paths aligned with IDs", "[ft][pyramid][paths]") {
    auto index = MakePyramidIndex("nsw", true);
    REQUIRE(index->Build(MakeDataset({10}, {"original"})).has_value());
    auto first = MakeDataset({20, 21}, {"first/20", "first/21"});
    auto second = MakeDataset({30, 31}, {"second/30", "second/31"});

    auto first_result =
        std::async(std::launch::async, [index, first]() { return index->Add(first); });
    auto second_result =
        std::async(std::launch::async, [index, second]() { return index->Add(second); });
    REQUIRE(first_result.get().has_value());
    REQUIRE(second_result.get().has_value());

    auto data = GetDataWithFlag(index, {31, 20, 30, 21}, DATA_FLAG_PATH);
    RequirePaths(data->GetPaths(), {"second/31", "first/20", "second/30", "first/21"});
}

TEST_CASE("Pyramid serialization and Clone preserve retained paths", "[ft][pyramid][paths]") {
    auto index = MakePyramidIndex("nsw", true);
    REQUIRE(index->Build(MakeDataset({4, 8}, {"left", "right"})).has_value());

    auto clone = index->Clone();
    REQUIRE(clone.has_value());
    RequirePaths(GetDataWithFlag(clone.value(), {8, 4}, DATA_FLAG_PATH)->GetPaths(),
                 {"right", "left"});

    auto binary_set = index->Serialize();
    REQUIRE(binary_set.has_value());
    auto restored = MakePyramidIndex("nsw", true);
    REQUIRE(restored->Deserialize(binary_set.value()).has_value());
    auto restored_data = GetDataWithFlag(restored, {4, 8}, DATA_FLAG_ID | DATA_FLAG_PATH);
    RequirePaths(restored_data->GetPaths(), {"left", "right"});
    REQUIRE(restored_data->GetIds() != nullptr);
    REQUIRE(restored_data->GetIds()[0] == 4);
    REQUIRE(restored_data->GetIds()[1] == 8);
}

TEST_CASE("Pyramid returns paths only when selected", "[ft][pyramid][paths]") {
    auto index = MakePyramidIndex("nsw", true);
    REQUIRE(index->Build(MakeDataset({1, 2}, {"a", "b"})).has_value());

    auto all_standard_data = GetData(index, {2, 1});
    REQUIRE(all_standard_data->GetIds() != nullptr);
    REQUIRE(all_standard_data->GetPaths() == nullptr);

    auto ids_only = GetDataWithFlag(index, {2, 1}, DATA_FLAG_ID);
    REQUIRE(ids_only->GetIds() != nullptr);
    REQUIRE(ids_only->GetPaths() == nullptr);

    auto paths_only = GetDataWithFlag(index, {2, 1}, DATA_FLAG_PATH);
    REQUIRE(paths_only->GetIds() == nullptr);
    RequirePaths(paths_only->GetPaths(), {"b", "a"});
}

TEST_CASE("Pyramid rejects selected paths when storage is disabled", "[ft][pyramid][paths]") {
    auto index = MakePyramidIndex("nsw", false);
    REQUIRE(index->Build(MakeDataset({1}, {"a"})).has_value());

    const int64_t id = 1;
    auto ids_only = index->GetDataByIdsWithFlag(&id, 1, DATA_FLAG_ID);
    REQUIRE(ids_only.has_value());
    REQUIRE(ids_only.value()->GetIds() != nullptr);
    REQUIRE(ids_only.value()->GetIds()[0] == id);
    REQUIRE(ids_only.value()->GetPaths() == nullptr);

    auto result = index->GetDataByIdsWithFlag(&id, 1, DATA_FLAG_PATH);
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}
