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
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "storage/serialization_tags.h"
#include "storage/streaming_serialization_test_utils.h"
#include "vsag/vsag.h"

namespace {

constexpr int64_t PYRAMID_PATH_TEST_DIM = 2;

struct PyramidPathRow {
    int64_t id;
    std::array<float, PYRAMID_PATH_TEST_DIM> vector;
    std::string site_path;
    std::string category_path;
};

std::string
MakePyramidPathParameters(bool store_paths) {
    nlohmann::json index_param = {
        {"base_quantization_type", "fp32"},
        {"max_degree", 4},
        {"ef_construction", 8},
        {"alpha", 1.2},
        {"graph_type", "nsw"},
        {"no_build_levels", {0, 1, 2, 3}},
        {"use_reorder", false},
        {"index_min_size", 100},
        {"hierarchies", {"site", "category"}},
    };
    if (store_paths) {
        index_param["store_paths"] = true;
    }
    return nlohmann::json({{"dtype", "float32"},
                           {"metric_type", "l2"},
                           {"dim", PYRAMID_PATH_TEST_DIM},
                           {"index_param", std::move(index_param)}})
        .dump();
}

vsag::IndexPtr
MakePyramidPathIndex(bool store_paths) {
    auto result = vsag::Factory::CreateIndex("pyramid", MakePyramidPathParameters(store_paths));
    REQUIRE(result.has_value());
    return result.value();
}

vsag::DatasetPtr
MakePyramidPathDataset(const std::vector<PyramidPathRow>& rows) {
    auto* vectors = new float[rows.size() * PYRAMID_PATH_TEST_DIM];
    auto* ids = new int64_t[rows.size()];
    auto* site_paths = new std::string[rows.size()];
    auto* category_paths = new std::string[rows.size()];
    for (uint64_t offset = 0; offset < rows.size(); ++offset) {
        ids[offset] = rows[offset].id;
        std::copy(rows[offset].vector.begin(),
                  rows[offset].vector.end(),
                  vectors + offset * PYRAMID_PATH_TEST_DIM);
        site_paths[offset] = rows[offset].site_path;
        category_paths[offset] = rows[offset].category_path;
    }
    return vsag::Dataset::Make()
        ->NumElements(static_cast<int64_t>(rows.size()))
        ->Dim(PYRAMID_PATH_TEST_DIM)
        ->Ids(ids)
        ->Float32Vectors(vectors)
        ->Paths("site", site_paths)
        ->Paths("category", category_paths)
        ->Owner(true);
}

void
RequirePyramidPaths(const vsag::IndexPtr& index,
                    const std::vector<int64_t>& ids,
                    const std::vector<std::string>& expected_site_paths,
                    const std::vector<std::string>& expected_category_paths) {
    REQUIRE(ids.size() == expected_site_paths.size());
    REQUIRE(ids.size() == expected_category_paths.size());
    auto result = index->GetDataByIdsWithFlag(
        ids.data(), static_cast<int64_t>(ids.size()), DATA_FLAG_ID | DATA_FLAG_PATH);
    REQUIRE(result.has_value());
    const auto data = result.value();
    REQUIRE(data->GetPaths() == nullptr);
    const auto* site_paths = data->GetPaths("site");
    const auto* category_paths = data->GetPaths("category");
    REQUIRE(site_paths != nullptr);
    REQUIRE(category_paths != nullptr);
    for (uint64_t offset = 0; offset < ids.size(); ++offset) {
        REQUIRE(data->GetIds()[offset] == ids[offset]);
        REQUIRE(site_paths[offset] == expected_site_paths[offset]);
        REQUIRE(category_paths[offset] == expected_category_paths[offset]);
    }
}

bool
HasPyramidPathsBlock(const std::string& bytes) {
    const auto blocks = vsag::test::ParseStreamingBlocks(bytes);
    return std::any_of(blocks.begin(), blocks.end(), [](const auto& block) {
        return block.tag == static_cast<uint32_t>(vsag::StreamSerializationTag::PYRAMID_PATHS);
    });
}

const std::vector<PyramidPathRow> INITIAL_ROWS = {
    {101, {1.0F, 0.0F}, "us/news", "media/news"},
    {205, {0.0F, 1.0F}, "uk/sports", "media/sports"},
};

}  // namespace

TEST_CASE("Pyramid BinarySet serialization preserves paths and supports Add",
          "[ft][pyramid][paths][serialization]") {
    auto index = MakePyramidPathIndex(true);
    REQUIRE(index->Build(MakePyramidPathDataset(INITIAL_ROWS)).has_value());

    auto binary_set = index->Serialize();
    REQUIRE(binary_set.has_value());
    auto restored = MakePyramidPathIndex(true);
    REQUIRE(restored->Deserialize(binary_set.value()).has_value());
    RequirePyramidPaths(
        restored, {205, 101}, {"uk/sports", "us/news"}, {"media/sports", "media/news"});

    const std::vector<PyramidPathRow> added_rows = {
        {309, {0.5F, 0.5F}, "jp/technology", "media/technology"},
    };
    auto add_result = restored->Add(MakePyramidPathDataset(added_rows));
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value().empty());
    RequirePyramidPaths(restored,
                        {309, 101, 205},
                        {"jp/technology", "us/news", "uk/sports"},
                        {"media/technology", "media/news", "media/sports"});
}

TEST_CASE("Pyramid streaming serialization preserves required paths block",
          "[ft][pyramid][paths][serialization][streaming]") {
    auto index = MakePyramidPathIndex(true);
    REQUIRE(index->Build(MakePyramidPathDataset(INITIAL_ROWS)).has_value());

    std::stringstream stream;
    REQUIRE(index->SerializeStreaming(stream).has_value());
    const auto bytes = stream.str();
    REQUIRE(HasPyramidPathsBlock(bytes));

    auto restored = MakePyramidPathIndex(true);
    std::stringstream deserialize_stream(bytes);
    REQUIRE(restored->DeserializeStreaming(deserialize_stream).has_value());
    RequirePyramidPaths(
        restored, {205, 101}, {"uk/sports", "us/news"}, {"media/sports", "media/news"});

    std::stringstream load_stream(bytes);
    auto loaded = vsag::Index::Load(load_stream, "{}");
    REQUIRE(loaded.has_value());
    RequirePyramidPaths(
        loaded.value(), {101, 205}, {"us/news", "uk/sports"}, {"media/news", "media/sports"});

    const auto missing_paths =
        vsag::test::EraseStreamingBlock(bytes, vsag::StreamSerializationTag::PYRAMID_PATHS);
    auto missing_paths_target = MakePyramidPathIndex(true);
    std::stringstream missing_deserialize_stream(missing_paths);
    REQUIRE_FALSE(
        missing_paths_target->DeserializeStreaming(missing_deserialize_stream).has_value());

    std::stringstream missing_load_stream(missing_paths);
    REQUIRE_FALSE(vsag::Index::Load(missing_load_stream, "{}").has_value());
}

TEST_CASE("Pyramid default streaming serialization omits paths block",
          "[ft][pyramid][paths][serialization][streaming]") {
    auto index = MakePyramidPathIndex(false);
    REQUIRE(index->Build(MakePyramidPathDataset(INITIAL_ROWS)).has_value());

    std::stringstream stream;
    REQUIRE(index->SerializeStreaming(stream).has_value());
    REQUIRE_FALSE(HasPyramidPathsBlock(stream.str()));
}

TEST_CASE("Pyramid default BinarySet roundtrip does not return paths",
          "[ft][pyramid][paths][serialization]") {
    auto index = MakePyramidPathIndex(false);
    REQUIRE(index->Build(MakePyramidPathDataset(INITIAL_ROWS)).has_value());

    auto binary_set = index->Serialize();
    REQUIRE(binary_set.has_value());
    auto restored = MakePyramidPathIndex(false);
    REQUIRE(restored->Deserialize(binary_set.value()).has_value());

    const std::array<int64_t, 2> ids = {205, 101};
    auto data = restored->GetDataByIds(ids.data(), static_cast<int64_t>(ids.size()));
    REQUIRE(data.has_value());
    REQUIRE(data.value()->GetPaths("site") == nullptr);
    REQUIRE(data.value()->GetPaths("category") == nullptr);
}

TEST_CASE("Pyramid BinarySet path storage configuration must match",
          "[ft][pyramid][paths][serialization]") {
    SECTION("stored paths require an enabled reader") {
        auto index = MakePyramidPathIndex(true);
        REQUIRE(index->Build(MakePyramidPathDataset(INITIAL_ROWS)).has_value());
        auto binary_set = index->Serialize();
        REQUIRE(binary_set.has_value());

        auto restored = MakePyramidPathIndex(false);
        REQUIRE_FALSE(restored->Deserialize(binary_set.value()).has_value());
    }

    SECTION("a reader requiring paths rejects an index without them") {
        auto index = MakePyramidPathIndex(false);
        REQUIRE(index->Build(MakePyramidPathDataset(INITIAL_ROWS)).has_value());
        auto binary_set = index->Serialize();
        REQUIRE(binary_set.has_value());

        auto restored = MakePyramidPathIndex(true);
        REQUIRE_FALSE(restored->Deserialize(binary_set.value()).has_value());
    }
}

TEST_CASE("Pyramid path storage supports empty serialization roundtrips",
          "[ft][pyramid][paths][serialization][streaming]") {
    SECTION("Clone") {
        auto index = MakePyramidPathIndex(true);
        auto clone = index->Clone();
        REQUIRE(clone.has_value());
        REQUIRE(clone.value()->GetNumElements() == 0);
    }

    SECTION("BinarySet") {
        auto index = MakePyramidPathIndex(true);
        auto binary_set = index->Serialize();
        REQUIRE(binary_set.has_value());

        auto restored = MakePyramidPathIndex(true);
        REQUIRE(restored->Deserialize(binary_set.value()).has_value());
    }

    SECTION("streaming") {
        auto index = MakePyramidPathIndex(true);
        std::stringstream stream;
        REQUIRE(index->SerializeStreaming(stream).has_value());

        auto restored = MakePyramidPathIndex(true);
        std::stringstream deserialize_stream(stream.str());
        REQUIRE(restored->DeserializeStreaming(deserialize_stream).has_value());
    }
}
