
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

#include "pyramid.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "unittest.h"

namespace {

constexpr int64_t PYRAMID_TEST_DIM = 4;

struct PyramidTestIndex {
    std::shared_ptr<vsag::Allocator> allocator;
    std::shared_ptr<vsag::Pyramid> index;
};

PyramidTestIndex
MakePyramidIndex(uint32_t index_min_size,
                 uint64_t build_thread_count = 1,
                 bool use_rabitq_with_sq8 = false,
                 bool split_rabitq = false,
                 bool use_mrle_split = false,
                 bool use_mrle_fp32 = false,
                 bool use_reorder = false,
                 bool store_raw_vector = false) {
    PyramidTestIndex result;
    vsag::IndexCommonParam common_param;
    common_param.dim_ = PYRAMID_TEST_DIM;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    result.allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    common_param.allocator_ = result.allocator;

    auto external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "base_io_type": "memory_io",
        "max_degree": 8,
        "ef_construction": 8,
        "alpha": 1.2,
        "graph_type": "nsw",
        "no_build_levels": [0],
        "index_min_size": 3
    })");
    if (split_rabitq) {
        external_param[vsag::PYRAMID_BASE_QUANTIZATION_TYPE].SetString("rabitq");
        external_param[vsag::PYRAMID_PRECISE_QUANTIZATION_TYPE].SetString("rabitq");
        external_param[vsag::PYRAMID_RABITQ_BITS_PER_DIM_BASE].SetInt(3);
        external_param[vsag::PYRAMID_RABITQ_BITS_PER_DIM_PRECISE].SetInt(5);
        external_param[vsag::PYRAMID_USE_REORDER].SetBool(true);
    }
    if (use_mrle_split) {
        external_param[vsag::PYRAMID_BASE_QUANTIZATION_TYPE].SetString(
            vsag::QUANTIZATION_TYPE_VALUE_TQ);
        external_param[vsag::PYRAMID_PRECISE_QUANTIZATION_TYPE].SetString(
            vsag::QUANTIZATION_TYPE_VALUE_RABITQ);
        external_param[vsag::PYRAMID_USE_REORDER].SetBool(true);
        external_param[vsag::INDEX_TQ_CHAIN].SetString("mrle, rabitq");
        external_param[vsag::INDEX_MRLE_DIM].SetInt(2);
        external_param[vsag::PYRAMID_RABITQ_BITS_PER_DIM_BASE].SetInt(3);
        external_param[vsag::PYRAMID_RABITQ_BITS_PER_DIM_PRECISE].SetInt(5);
    }
    if (use_mrle_fp32) {
        external_param[vsag::PYRAMID_BASE_QUANTIZATION_TYPE].SetString(
            vsag::QUANTIZATION_TYPE_VALUE_TQ);
        external_param[vsag::PYRAMID_USE_REORDER].SetBool(false);
        external_param[vsag::INDEX_TQ_CHAIN].SetString("mrle, fp32");
        external_param[vsag::INDEX_MRLE_DIM].SetInt(2);
    }
    external_param[vsag::PYRAMID_INDEX_MIN_SIZE].SetInt(index_min_size);
    external_param[vsag::PYRAMID_BUILD_THREAD_COUNT].SetUint64(build_thread_count);
    external_param[vsag::STORE_RAW_VECTOR].SetBool(store_raw_vector);
    external_param[vsag::PYRAMID_USE_REORDER].SetBool(use_rabitq_with_sq8 or split_rabitq or
                                                      use_mrle_split or use_reorder);
    if (use_rabitq_with_sq8) {
        external_param[vsag::PYRAMID_BASE_QUANTIZATION_TYPE].SetString("rabitq");
        external_param[vsag::PYRAMID_PRECISE_QUANTIZATION_TYPE].SetString("sq8");
        external_param[vsag::PYRAMID_BASE_IO_TYPE].SetString("block_memory_io");
        external_param[vsag::PYRAMID_PRECISE_IO_TYPE].SetString("block_memory_io");
        external_param[vsag::PYRAMID_RABITQ_BITS_PER_DIM_BASE].SetUint64(1);
    } else if (use_reorder) {
        external_param[vsag::PYRAMID_PRECISE_QUANTIZATION_TYPE].SetString(
            vsag::QUANTIZATION_TYPE_VALUE_FP32);
    }
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    result.index = std::make_shared<vsag::Pyramid>(param, common_param);
    return result;
}

vsag::DatasetPtr
MakePyramidDataset(float* vectors, int64_t* ids, std::string* paths, int64_t count) {
    return vsag::Dataset::Make()
        ->NumElements(count)
        ->Dim(PYRAMID_TEST_DIM)
        ->Ids(ids)
        ->Float32Vectors(vectors)
        ->Paths(paths)
        ->Owner(false);
}

int64_t
GetPyramidSubindexCount(const std::shared_ptr<vsag::Pyramid>& index, const char* status) {
    auto stats = vsag::JsonType::Parse(index->GetStats());
    return stats["subindex_quality"][status].GetInt();
}

float
GetPyramidDuplicateRatio(const std::shared_ptr<vsag::Pyramid>& index) {
    auto stats = vsag::JsonType::Parse(index->GetStats());
    return stats["duplicate_ratio"].GetFloat();
}

void
RequirePyramidSearchStatistics(const vsag::DatasetPtr& result, uint64_t approximate) {
    auto statistics = vsag::JsonType::Parse(result->GetStatistics());
    REQUIRE(statistics["distance_evaluations_by_phase"]["approximate"].GetUint64() == approximate);
    REQUIRE(statistics["distance_evaluations_by_phase"]["rerank"].GetUint64() > 0);
    REQUIRE(statistics["distance_evaluations"].GetUint64() ==
            statistics["distance_evaluations_by_phase"]["routing"].GetUint64() +
                statistics["distance_evaluations_by_phase"]["approximate"].GetUint64() +
                statistics["distance_evaluations_by_phase"]["rerank"].GetUint64());
    REQUIRE(statistics["distance_evaluations_by_backend"]["fp32"].GetUint64() ==
            statistics["distance_evaluations"].GetUint64());
    REQUIRE(statistics["complete"].GetBool());
}

}  // namespace

TEST_CASE("Split function tests", "[ut][pyramid]") {
    SECTION("Empty input string") {
        auto result = vsag::split("", ',');
        REQUIRE(result.empty());
    }

    SECTION("No delimiters in string") {
        auto result = vsag::split("hello", ',');
        REQUIRE(result == std::vector<std::string>{"hello"});
    }

    SECTION("Delimiter at start") {
        auto result = vsag::split(",hello,world", ',');
        REQUIRE(result == std::vector<std::string>{"hello", "world"});
    }

    SECTION("Delimiter at end") {
        auto result = vsag::split("hello,world,", ',');
        REQUIRE(result == std::vector<std::string>{"hello", "world"});
    }

    SECTION("Multiple consecutive delimiters") {
        auto result = vsag::split("a,,b,,,c", ',');
        REQUIRE(result == std::vector<std::string>{"a", "b", "c"});
    }

    SECTION("Normal split with multiple tokens") {
        auto result = vsag::split("one,two,three", ',');
        REQUIRE(result == std::vector<std::string>{"one", "two", "three"});
    }

    SECTION("All delimiters") {
        auto result = vsag::split(",,,", ',');
        REQUIRE(result.empty());
    }

    SECTION("Mixed delimiters and spaces") {
        auto result = vsag::split("  , hello,  world  ", ',');
        REQUIRE(result == std::vector<std::string>{"  ", " hello", "  world  "});
    }
}

TEST_CASE("Pyramid stats count duplicates within each leaf", "[ut][pyramid][analyzer]") {
    constexpr int64_t count = 7;
    auto test_index = MakePyramidIndex(100, 1, true);
    const auto& index = test_index.index;
    std::vector<float> vectors = {
        1.0F, 2.0F, 3.0F, 4.0F,  // leaf a: representative
        1.0F, 2.0F, 3.0F, 4.0F,  // leaf a: duplicate
        2.0F, 3.0F, 4.0F, 5.0F,  // leaf a: unique
        1.0F, 2.0F, 3.0F, 4.0F,  // leaf b: not a cross-leaf duplicate
        6.0F, 7.0F, 8.0F, 9.0F,  // leaf c: representative
        6.0F, 7.0F, 8.0F, 9.0F,  // leaf c: duplicate
        9.0F, 8.0F, 7.0F, 6.0F,  // leaf d: singleton
    };
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<std::string> paths = {"a", "a", "a", "b", "c", "c", "d"};

    REQUIRE(
        index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count)).empty());
    REQUIRE(std::abs(GetPyramidDuplicateRatio(index) - 2.0F / static_cast<float>(count)) < 1e-6F);
}

TEST_CASE("Pyramid promotes flat node at index minimum size", "[ut][pyramid]") {
    const bool split_rabitq = GENERATE(false, true);
    const bool build_all_at_once = GENERATE(false, true);
    CAPTURE(split_rabitq, build_all_at_once);
    auto test_index = MakePyramidIndex(3, 1, false, split_rabitq);
    const auto& index = test_index.index;
    std::vector<float> vectors = {
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
        1.0F,
        1.0F,
        1.0F,
        2.0F,
        2.0F,
        2.0F,
        2.0F,
    };
    std::vector<int64_t> ids = {100, 101, 102};
    std::vector<std::string> paths(3, "tenant");

    if (build_all_at_once) {
        REQUIRE(
            index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), 3)).empty());
    } else {
        REQUIRE(
            index->Add(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), 2)).empty());
        REQUIRE(GetPyramidSubindexCount(index, "flat_subindexes") == 1);
        REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 0);

        REQUIRE(index
                    ->Add(MakePyramidDataset(
                        vectors.data() + 2 * PYRAMID_TEST_DIM, ids.data() + 2, paths.data() + 2, 1))
                    .empty());
    }
    REQUIRE(GetPyramidSubindexCount(index, "flat_subindexes") == 0);
    REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 1);
    REQUIRE(GetPyramidSubindexCount(index, "total_vectors_in_graph") == 3);

    for (int64_t i = 0; i < 3; ++i) {
        auto query =
            MakePyramidDataset(vectors.data() + i * PYRAMID_TEST_DIM, nullptr, paths.data() + i, 1);
        auto result =
            index->KnnSearch(query, 1, R"({"pyramid":{"ef_search":10}})", vsag::FilterPtr{});
        REQUIRE(result->GetDim() == 1);
        REQUIRE(result->GetIds()[0] == ids[i]);
        if (split_rabitq) {
            auto stats = result->GetStatistics({"reorder_lower_bound_probe_count"});
            REQUIRE(stats.size() == 1);
            REQUIRE(std::stoul(stats[0]) > 0);
        }
    }
}

TEST_CASE("Pyramid MRLE split promotes flat nodes without raw vectors", "[ut][pyramid][MRLE]") {
    auto test_index = MakePyramidIndex(3, 4, false, false, true);
    const auto& index = test_index.index;
    std::vector<float> vectors = {
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
        0.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {100, 101, 102};
    std::vector<std::string> paths(3, "tenant");

    REQUIRE(index->Add(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), 2)).empty());
    REQUIRE(index
                ->Add(MakePyramidDataset(
                    vectors.data() + 2 * PYRAMID_TEST_DIM, ids.data() + 2, paths.data() + 2, 1))
                .empty());

    REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 1);
    auto stats = vsag::JsonType::Parse(index->GetStats());
    REQUIRE_FALSE(stats["sample_metrics_available"].GetBool());
    REQUIRE(stats.Contains("sample_metrics_unavailable_reason"));
    for (int64_t i = 0; i < 3; ++i) {
        auto query =
            MakePyramidDataset(vectors.data() + i * PYRAMID_TEST_DIM, nullptr, paths.data() + i, 1);
        auto result =
            index->KnnSearch(query, 1, R"({"pyramid":{"ef_search":10}})", vsag::FilterPtr{});
        REQUIRE(result->GetDim() == 1);
        REQUIRE(result->GetIds()[0] == ids[i]);
    }
}

TEST_CASE("Pyramid MRLE split stores raw vectors when enabled", "[ut][pyramid][MRLE]") {
    auto test_index = MakePyramidIndex(3, 1, false, false, true, false, false, true);
    const auto& index = test_index.index;
    std::vector<float> vectors = {
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
        0.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {100, 101, 102};
    std::vector<std::string> paths(3, "tenant");

    REQUIRE(index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), 3)).empty());
    auto stats = vsag::JsonType::Parse(index->GetStats());
    REQUIRE(stats["sample_metrics_available"].GetBool());
    for (int64_t i = 0; i < 3; ++i) {
        std::array<float, PYRAMID_TEST_DIM> decoded{};
        index->GetVectorByInnerId(i, decoded.data());
        REQUIRE(std::equal(decoded.begin(), decoded.end(), vectors.begin() + i * PYRAMID_TEST_DIM));
    }
}

TEST_CASE("Pyramid TQ retains vectors without precise decode source", "[ut][pyramid][TQ]") {
    auto test_index = MakePyramidIndex(3, 1, false, false, false, true);
    const auto& index = test_index.index;
    std::vector<float> vectors = {
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
        0.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {100, 101, 102};
    std::vector<std::string> paths(3, "tenant");

    REQUIRE(index->Add(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), 2)).empty());
    REQUIRE(index
                ->Add(MakePyramidDataset(
                    vectors.data() + 2 * PYRAMID_TEST_DIM, ids.data() + 2, paths.data() + 2, 1))
                .empty());

    REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 1);
    for (int64_t i = 0; i < 3; ++i) {
        std::array<float, PYRAMID_TEST_DIM> decoded{};
        index->GetVectorByInnerId(i, decoded.data());
        REQUIRE(std::equal(decoded.begin(), decoded.end(), vectors.begin() + i * PYRAMID_TEST_DIM));
    }
}

TEST_CASE("Pyramid Build stores RaBitQ and SQ8 codes in parallel", "[ut][pyramid]") {
    constexpr int64_t count = 804;
    auto test_index = MakePyramidIndex(count + 1, 4, true);

    std::vector<float> vectors(count * PYRAMID_TEST_DIM);
    std::vector<int64_t> ids(count);
    std::vector<std::string> paths(count, "tenant");
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = i;
        for (int64_t j = 0; j < PYRAMID_TEST_DIM; ++j) {
            vectors[i * PYRAMID_TEST_DIM + j] = static_cast<float>((i + j) % 101) / 100.0F;
        }
    }

    auto failed_ids = test_index.index->Build(
        MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count));

    REQUIRE(failed_ids.empty());
    REQUIRE(test_index.index->GetNumElements() == count);
    for (const int64_t inner_id : {0, 401, 803}) {
        std::vector<float> decoded(PYRAMID_TEST_DIM);
        test_index.index->GetVectorByInnerId(inner_id, decoded.data());
        for (int64_t j = 0; j < PYRAMID_TEST_DIM; ++j) {
            REQUIRE(std::abs(decoded[j] - vectors[inner_id * PYRAMID_TEST_DIM + j]) < 0.02F);
        }
    }
}

TEST_CASE("Pyramid reports statistics for flat and graph leaves", "[ut][pyramid][statistics]") {
    auto test_index = MakePyramidIndex(3, 1, false, false, false, false, true);
    const auto& index = test_index.index;
    std::vector<float> vectors = {
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
        1.0F,
        1.0F,
        1.0F,
        2.0F,
        2.0F,
        2.0F,
        2.0F,
    };
    std::vector<int64_t> ids = {100, 101, 102};
    std::vector<std::string> paths(3, "tenant");
    auto query = MakePyramidDataset(vectors.data(), nullptr, paths.data(), 1);
    const auto parameters = R"({"pyramid":{"ef_search":10}})";

    REQUIRE(index->Add(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), 2)).empty());
    auto flat_result = index->KnnSearch(query, 1, parameters, vsag::FilterPtr{});
    RequirePyramidSearchStatistics(flat_result, 2);

    REQUIRE(index
                ->Add(MakePyramidDataset(
                    vectors.data() + 2 * PYRAMID_TEST_DIM, ids.data() + 2, paths.data() + 2, 1))
                .empty());
    auto graph_result = index->KnnSearch(query, 1, parameters, vsag::FilterPtr{});
    auto graph_statistics = vsag::JsonType::Parse(graph_result->GetStatistics());
    REQUIRE(graph_statistics["distance_evaluations_by_phase"]["approximate"].GetUint64() > 0);
    RequirePyramidSearchStatistics(
        graph_result, graph_statistics["distance_evaluations_by_phase"]["approximate"].GetUint64());
}
