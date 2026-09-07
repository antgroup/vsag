
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
#include <atomic>
#include <cmath>
#include <future>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "index/index_impl.h"
#include "index_common_param.h"
#include "unittest.h"
#include "vsag/options.h"

namespace {

constexpr int64_t PYRAMID_TEST_DIM = 4;

struct PyramidTestIndex {
    std::shared_ptr<vsag::Allocator> allocator;
    std::shared_ptr<vsag::Pyramid> index;
};

class BlockSizeLimitGuard {
public:
    explicit BlockSizeLimitGuard(uint64_t block_size_limit)
        : previous_limit_(vsag::Options::Instance().block_size_limit()) {
        vsag::Options::Instance().set_block_size_limit(block_size_limit);
    }

    ~BlockSizeLimitGuard() {
        vsag::Options::Instance().set_block_size_limit(previous_limit_);
    }

private:
    uint64_t previous_limit_;
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
        external_param[vsag::PYRAMID_RABITQ_BITS_PER_DIM_BASE].SetUint64(3);
    } else if (use_reorder) {
        external_param[vsag::PYRAMID_PRECISE_QUANTIZATION_TYPE].SetString(
            vsag::QUANTIZATION_TYPE_VALUE_FP32);
    }
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    result.index = std::make_shared<vsag::Pyramid>(param, common_param);
    return result;
}

PyramidTestIndex
MakeRootPyramidIndex(const std::string& root_graph_type,
                     bool use_reorder = false,
                     const std::string& graph_type = vsag::GRAPH_TYPE_VALUE_NSW,
                     bool support_duplicate = false,
                     uint64_t build_thread_count = 1,
                     bool use_rabitq_with_sq8 = false,
                     const std::string& graph_storage_type = vsag::GRAPH_STORAGE_TYPE_VALUE_FLAT) {
    PyramidTestIndex result;
    vsag::IndexCommonParam common_param;
    common_param.dim_ = PYRAMID_TEST_DIM;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    result.allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    common_param.allocator_ = result.allocator;
    auto external = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "precise_quantization_type": "fp32",
        "base_io_type": "memory_io",
        "precise_io_type": "memory_io",
        "max_degree": 8,
        "ef_construction": 32,
        "alpha": 1.2,
        "no_build_levels": [],
        "index_min_size": 1,
        "build_thread_count": 1
    })");
    external[vsag::PYRAMID_ROOT_GRAPH_TYPE].SetString(root_graph_type);
    external[vsag::PYRAMID_GRAPH_TYPE].SetString(graph_type);
    external[vsag::PYRAMID_GRAPH_STORAGE_TYPE].SetString(graph_storage_type);
    external[vsag::PYRAMID_USE_REORDER].SetBool(use_reorder || use_rabitq_with_sq8);
    external[vsag::PYRAMID_SUPPORT_DUPLICATE].SetBool(support_duplicate);
    external[vsag::PYRAMID_BUILD_THREAD_COUNT].SetUint64(build_thread_count);
    if (use_rabitq_with_sq8) {
        external[vsag::PYRAMID_BASE_QUANTIZATION_TYPE].SetString("rabitq");
        external[vsag::PYRAMID_PRECISE_QUANTIZATION_TYPE].SetString("sq8");
        external[vsag::PYRAMID_BASE_IO_TYPE].SetString("block_memory_io");
        external[vsag::PYRAMID_PRECISE_IO_TYPE].SetString("block_memory_io");
        external[vsag::PYRAMID_RABITQ_BITS_PER_DIM_BASE].SetUint64(3);
    }
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external, common_param);
    result.index = std::make_shared<vsag::Pyramid>(param, common_param);
    return result;
}

void
FillRootVectors(std::vector<float>& vectors, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        for (int64_t d = 0; d < PYRAMID_TEST_DIM; ++d) {
            vectors[i * PYRAMID_TEST_DIM + d] =
                static_cast<float>((i * (d + 3) + d * 17) % 997) / 997.0F;
        }
    }
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

TEST_CASE("Pyramid query analyzer honors paths and removals", "[ut][pyramid][analyzer]") {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = PYRAMID_TEST_DIM;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "max_degree": 4,
        "ef_construction": 8,
        "graph_type": "nsw",
        "no_build_levels": [0, 1, 2],
        "index_min_size": 28
    })");
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    auto index = std::make_shared<vsag::Pyramid>(param, common_param);

    std::array<float, PYRAMID_TEST_DIM* 4> vectors = {10.0F,
                                                      0.0F,
                                                      0.0F,
                                                      0.0F,
                                                      1.0F,
                                                      0.0F,
                                                      0.0F,
                                                      0.0F,
                                                      0.0F,
                                                      0.0F,
                                                      0.0F,
                                                      0.0F,
                                                      100.0F,
                                                      0.0F,
                                                      0.0F,
                                                      0.0F};
    std::array<int64_t, 4> ids = {101, 102, 201, 202};
    std::array<std::string, 4> paths = {"root/a/leaf", "root/a/leaf", "root/b/leaf", "root/b/leaf"};
    auto base = vsag::Dataset::Make()
                    ->NumElements(4)
                    ->Dim(PYRAMID_TEST_DIM)
                    ->Float32Vectors(vectors.data())
                    ->Ids(ids.data())
                    ->Paths(paths.data())
                    ->Owner(false);
    REQUIRE(index->Build(base).empty());

    std::array<float, PYRAMID_TEST_DIM* 2> query_vectors = {
        0.0F, 0.0F, 0.0F, 0.0F, 100.0F, 0.0F, 0.0F, 0.0F};
    std::array<std::string, 2> query_paths = {"root/a/leaf", "root/b/leaf"};
    auto query = vsag::Dataset::Make()
                     ->NumElements(2)
                     ->Dim(PYRAMID_TEST_DIM)
                     ->Float32Vectors(query_vectors.data())
                     ->Paths(query_paths.data())
                     ->Owner(false);
    vsag::SearchRequest request;
    request.query_ = query;
    request.topk_ = 1;
    request.params_str_ = R"({"pyramid":{"ef_search":20}})";

    auto stats = vsag::JsonType::Parse(index->AnalyzeIndexBySearch(request));
    REQUIRE(std::abs(stats["recall_query"].GetFloat() - 1.0F) < 1e-6F);
    REQUIRE(std::abs(stats["avg_distance_query"].GetFloat() - 0.5F) < 1e-6F);

    REQUIRE(index->Remove(std::vector<int64_t>{102}, vsag::RemoveMode::MARK_REMOVE) == 1);
    stats = vsag::JsonType::Parse(index->AnalyzeIndexBySearch(request));
    REQUIRE(std::abs(stats["recall_query"].GetFloat() - 1.0F) < 1e-6F);
    REQUIRE(std::abs(stats["avg_distance_query"].GetFloat() - 50.0F) < 1e-6F);

    request.mode_ = vsag::SearchMode::RANGE_SEARCH;
    REQUIRE_THROWS(index->AnalyzeIndexBySearch(request));
}

TEST_CASE("Pyramid query analyzer includes graph duplicates in path ground truth",
          "[ut][pyramid][analyzer]") {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = PYRAMID_TEST_DIM;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "max_degree": 4,
        "ef_construction": 8,
        "graph_type": "nsw",
        "no_build_levels": [0],
        "index_min_size": 1,
        "support_duplicate": true
    })");
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    auto index = std::make_shared<vsag::Pyramid>(param, common_param);

    std::array<float, PYRAMID_TEST_DIM* 4> vectors = {1.0F,
                                                      2.0F,
                                                      3.0F,
                                                      4.0F,
                                                      1.0F,
                                                      2.0F,
                                                      3.0F,
                                                      4.0F,
                                                      1.0F,
                                                      2.0F,
                                                      3.0F,
                                                      4.0F,
                                                      9.0F,
                                                      8.0F,
                                                      7.0F,
                                                      6.0F};
    std::array<int64_t, 4> ids = {100, 101, 102, 103};
    std::array<std::string, 4> paths = {"tenant", "tenant", "tenant", "tenant"};
    auto base = vsag::Dataset::Make()
                    ->NumElements(4)
                    ->Dim(PYRAMID_TEST_DIM)
                    ->Float32Vectors(vectors.data())
                    ->Ids(ids.data())
                    ->Paths(paths.data())
                    ->Owner(false);
    REQUIRE(index->Build(base).empty());
    REQUIRE(index->Remove(std::vector<int64_t>{100}, vsag::RemoveMode::MARK_REMOVE) == 1);

    auto query = MakePyramidDataset(vectors.data(), nullptr, paths.data(), 1);
    vsag::SearchRequest request;
    request.query_ = query;
    request.topk_ = 1;
    request.params_str_ = R"({"pyramid":{"ef_search":10}})";

    auto stats = vsag::JsonType::Parse(index->AnalyzeIndexBySearch(request));
    REQUIRE(std::abs(stats["recall_query"].GetFloat() - 1.0F) < 1e-6F);
    REQUIRE(std::abs(stats["avg_distance_query"].GetFloat()) < 1e-6F);
}

TEST_CASE("Pyramid query analyzer supports a built root without query paths",
          "[ut][pyramid][analyzer]") {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = PYRAMID_TEST_DIM;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "max_degree": 4,
        "ef_construction": 8,
        "graph_type": "odescent",
        "no_build_levels": [],
        "index_min_size": 100
    })");
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    auto index = std::make_shared<vsag::Pyramid>(param, common_param);

    std::array<float, PYRAMID_TEST_DIM* 2> vectors = {
        0.0F, 0.0F, 0.0F, 0.0F, 10.0F, 0.0F, 0.0F, 0.0F};
    std::array<int64_t, 2> ids = {100, 101};
    std::array<std::string, 2> paths = {"", ""};
    auto base = vsag::Dataset::Make()
                    ->NumElements(2)
                    ->Dim(PYRAMID_TEST_DIM)
                    ->Float32Vectors(vectors.data())
                    ->Ids(ids.data())
                    ->Paths(paths.data())
                    ->Owner(false);
    REQUIRE(index->Build(base).empty());

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(PYRAMID_TEST_DIM)
                     ->Float32Vectors(vectors.data())
                     ->Owner(false);
    vsag::SearchRequest request;
    request.query_ = query;
    request.topk_ = 1;
    request.params_str_ = R"({"pyramid":{"ef_search":10}})";

    auto stats = vsag::JsonType::Parse(index->AnalyzeIndexBySearch(request));
    REQUIRE(std::abs(stats["recall_query"].GetFloat() - 1.0F) < 1e-6F);
    REQUIRE(std::abs(stats["avg_distance_query"].GetFloat()) < 1e-6F);
}

TEST_CASE("Pyramid query analyzer selects an available ground truth code source",
          "[ut][pyramid][raw_vector][analyzer]") {
    const bool store_raw_vector = GENERATE(false, true);
    CAPTURE(store_raw_vector);
    constexpr int64_t dim = 4;
    std::array<float, dim* 3> vectors = {
        0.0F, 0.0F, 0.0F, 0.0F, 0.123456F, 0.234567F, 0.345678F, 0.456789F, 1.0F, 1.0F, 1.0F, 1.0F};
    std::array<int64_t, 3> ids = {10, 11, 12};
    std::array<std::string, 3> paths = {"leaf", "leaf", "leaf"};

    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "tq",
        "precise_quantization_type": "rabitq",
        "use_reorder": true,
        "tq_chain": "mrle, rabitq",
        "mrle_dim": 2,
        "rabitq_bits_per_dim_base": 3,
        "rabitq_bits_per_dim_precise": 5,
        "max_degree": 4,
        "ef_construction": 8,
        "index_min_size": 4,
        "no_build_levels": [0]
    })");
    external_param[vsag::STORE_RAW_VECTOR].SetBool(store_raw_vector);
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    auto pyramid_param = std::dynamic_pointer_cast<vsag::PyramidParameters>(param);
    REQUIRE(pyramid_param != nullptr);
    REQUIRE(pyramid_param->store_raw_vector == store_raw_vector);
    REQUIRE((pyramid_param->raw_vector_param != nullptr) == store_raw_vector);
    REQUIRE(pyramid_param->precise_codes_param == nullptr);
    REQUIRE(pyramid_param->base_codes_param != nullptr);
    REQUIRE(pyramid_param->base_codes_param->name == vsag::RABITQ_SPLIT_DATA_CELL);
    auto index = std::make_shared<vsag::Pyramid>(param, common_param);
    auto base = vsag::Dataset::Make()
                    ->NumElements(3)
                    ->Dim(dim)
                    ->Float32Vectors(vectors.data())
                    ->Ids(ids.data())
                    ->Paths(paths.data())
                    ->Owner(false);
    REQUIRE(index->Build(base).empty());

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->Float32Vectors(vectors.data() + dim)
                     ->Paths(paths.data() + 1)
                     ->Owner(false);
    vsag::SearchRequest request;
    request.query_ = query;
    request.topk_ = 1;
    request.params_str_ = R"({"pyramid":{"ef_search":10}})";

    auto stats = vsag::JsonType::Parse(index->AnalyzeIndexBySearch(request));
    REQUIRE(std::abs(stats["recall_query"].GetFloat() - 1.0F) < 1e-6F);
    const auto avg_distance = stats["avg_distance_query"].GetFloat();
    REQUIRE(std::isfinite(avg_distance));
    if (store_raw_vector) {
        REQUIRE(std::abs(avg_distance) < 1e-12F);
    }
}

TEST_CASE("Pyramid promotes flat node at index minimum size", "[ut][pyramid]") {
    const bool split_rabitq = GENERATE(false, true);
    const bool build_all_at_once = GENERATE(false, true);
    CAPTURE(split_rabitq, build_all_at_once);
    auto test_index = MakePyramidIndex(3, 4, false, split_rabitq);
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

TEST_CASE("Pyramid SearchWithRequest reports reasoning for expected labels",
          "[ut][pyramid][reasoning]") {
    auto test_index = MakePyramidIndex(100);
    const auto& index = test_index.index;
    std::array<float, PYRAMID_TEST_DIM* 2> vectors = {
        0.0F, 0.0F, 0.0F, 0.0F, 10.0F, 0.0F, 0.0F, 0.0F};
    std::array<int64_t, 2> ids = {100, 101};
    std::array<std::string, 2> paths = {"tenant", "tenant"};
    REQUIRE(index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), 2)).empty());

    auto query = MakePyramidDataset(vectors.data(), nullptr, paths.data(), 1);
    vsag::SearchRequest request;
    request.query_ = query;
    request.topk_ = 1;
    request.params_str_ = R"({"pyramid":{"ef_search":10}})";
    request.expected_labels_ = {ids[0]};

    auto result = index->SearchWithRequest(request);
    REQUIRE(result != nullptr);
    REQUIRE(result->GetIds()[0] == ids[0]);
    REQUIRE(result->GetReasoning().find("1/1 expected labels found") != std::string::npos);
    REQUIRE(result->GetReasoning().find("0 missed") != std::string::npos);

    request.topk_ = 2;
    request.expected_labels_ = {ids[0], ids[1]};
    auto multi_result = index->SearchWithRequest(request);
    REQUIRE(multi_result != nullptr);
    REQUIRE(multi_result->GetDim() == 2);
    REQUIRE(multi_result->GetReasoning().find("2/2 expected labels found") != std::string::npos);

    request.threshold_ = 1.0F;
    request.expected_labels_ = {ids[0]};
    auto threshold_result = index->SearchWithRequest(request);
    REQUIRE(threshold_result != nullptr);
    REQUIRE(threshold_result->GetDim() == 1);
    REQUIRE(threshold_result->GetIds()[0] == ids[0]);

    request.expected_labels_.clear();
    request.threshold_ = std::nullopt;
    auto result_without_reasoning = index->SearchWithRequest(request);
    REQUIRE(result_without_reasoning != nullptr);
    REQUIRE(result_without_reasoning->GetIds()[0] == ids[0]);
    REQUIRE(result_without_reasoning->GetReasoning().find("expected_analysis") ==
            std::string::npos);
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

TEST_CASE("Pyramid exposes stored raw vectors", "[ut][pyramid][raw_vector]") {
    constexpr int64_t count = 3;
    const auto graph_type = GENERATE("nsw", "odescent");
    std::array<float, count* PYRAMID_TEST_DIM> vectors = {
        0.123456F,
        0.234567F,
        0.345678F,
        0.456789F,
        1.0F,
        2.0F,
        3.0F,
        4.0F,
        5.0F,
        6.0F,
        7.0F,
        8.0F,
    };
    std::array<int64_t, count> ids = {10, 42, 1001};
    std::array<std::string, count> paths = {"a", "b", "c"};

    vsag::IndexCommonParam common_param;
    common_param.dim_ = PYRAMID_TEST_DIM;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();

    auto external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "sq8",
        "store_raw_vector": true,
        "max_degree": 4,
        "ef_construction": 8,
        "no_build_levels": [0, 1]
    })");
    external_param[vsag::PYRAMID_GRAPH_TYPE].SetString(graph_type);
    auto index = std::make_shared<vsag::IndexImpl<vsag::Pyramid>>(external_param, common_param);
    auto dataset = MakePyramidDataset(
        vectors.data(), ids.data(), paths.data(), static_cast<int64_t>(ids.size()));

    REQUIRE(index->Build(dataset).has_value());
    auto restored = index->GetRawVectorByIds(ids.data(), count, nullptr);
    REQUIRE(restored.has_value());
    REQUIRE(std::equal(vectors.begin(), vectors.end(), restored.value()->GetFloat32Vectors()));

    auto distance = index->CalcDistanceById(vectors.data(), ids[0], true);
    REQUIRE(distance.has_value());
    REQUIRE(distance.value() == 0.0F);

    auto distances = index->CalcDistancesById(vectors.data(), ids.data(), count, true);
    REQUIRE(distances.has_value());
    REQUIRE(distances.value()->GetDistances()[0] == 0.0F);
}

TEST_CASE("Pyramid rejects TQ-only fields for non-TQ quantizers", "[ut][pyramid]") {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;

    auto tq_chain = vsag::JsonType::Parse(R"({"base_quantization_type":"fp32",
                                               "tq_chain":"mrle,fp32"})");
    REQUIRE_THROWS(vsag::Pyramid::CheckAndMappingExternalParam(tq_chain, common_param));

    auto mrle_dim = vsag::JsonType::Parse(R"({"base_quantization_type":"int8",
                                               "mrle_dim":64})");
    REQUIRE_THROWS(vsag::Pyramid::CheckAndMappingExternalParam(mrle_dim, common_param));
}

TEST_CASE("Pyramid multi-layer root builds routes and survives serialization",
          "[ut][pyramid][root_graph][pipnn]") {
    const auto graph_type = GENERATE(std::string(vsag::GRAPH_TYPE_VALUE_NSW),
                                     std::string(vsag::GRAPH_TYPE_VALUE_PIPNN));
    const auto graph_storage_type =
        GENERATE(std::string(vsag::GRAPH_STORAGE_TYPE_VALUE_FLAT),
                 std::string(vsag::GRAPH_STORAGE_TYPE_VALUE_COMPRESSED));
    CAPTURE(graph_type, graph_storage_type);
    constexpr int64_t count = 512;
    std::vector<float> vectors(count * PYRAMID_TEST_DIM);
    FillRootVectors(vectors, count);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<std::string> paths(count, "");
    auto source = MakeRootPyramidIndex(vsag::PYRAMID_ROOT_GRAPH_TYPE_MULTI_LAYER,
                                       false,
                                       graph_type,
                                       false,
                                       1,
                                       false,
                                       graph_storage_type);
    REQUIRE(source.index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count))
                .empty());

    auto stats = vsag::JsonType::Parse(source.index->GetStats());
    auto root_stats = stats["root_graphs"]["default"];
    REQUIRE(root_stats[vsag::PYRAMID_ROOT_GRAPH_TYPE].GetString() ==
            vsag::PYRAMID_ROOT_GRAPH_TYPE_MULTI_LAYER);
    REQUIRE(root_stats["bottom_graph_storage_type"].GetString() == graph_storage_type);
    REQUIRE(root_stats["bottom_graph_node_count"].GetUint64() == count);
    REQUIRE(root_stats["bottom_graph_size"].GetUint64() > 0);
    REQUIRE(root_stats["route_graph_count"].GetUint64() > 0);
    REQUIRE(root_stats["route_node_counts"].GetVector().front() < count);
    REQUIRE(source.index->GetMemoryUsageDetail().at("root_route_graphs") > 0);

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(PYRAMID_TEST_DIM)
                     ->Float32Vectors(vectors.data() + 137 * PYRAMID_TEST_DIM)
                     ->Owner(false);
    const auto search_params = R"({"pyramid":{"ef_search":128,"hops_limit":256}})";
    auto result = source.index->KnnSearch(query, 10, search_params, nullptr);
    REQUIRE(result->GetDim() == 10);
    auto search_stats = vsag::JsonType::Parse(result->GetStatistics());
    REQUIRE(search_stats["distance_evaluations_by_phase"]["routing"].GetUint64() > 0);

    constexpr int64_t added_count = 64;
    std::vector<float> added_vectors(added_count * PYRAMID_TEST_DIM);
    FillRootVectors(added_vectors, added_count);
    std::vector<int64_t> added_ids(added_count);
    std::iota(added_ids.begin(), added_ids.end(), count);
    std::vector<std::string> added_paths(added_count, "");
    REQUIRE(source.index
                ->Add(MakePyramidDataset(
                    added_vectors.data(), added_ids.data(), added_paths.data(), added_count))
                .empty());
    auto added_query = vsag::Dataset::Make()
                           ->NumElements(1)
                           ->Dim(PYRAMID_TEST_DIM)
                           ->Float32Vectors(added_vectors.data())
                           ->Owner(false);
    REQUIRE(source.index->KnnSearch(added_query, 10, search_params, nullptr)->GetDim() == 10);
    const auto source_result = source.index->KnnSearch(query, 10, search_params, nullptr);

    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    source.index->Serialize(writer);
    auto restored = MakeRootPyramidIndex(vsag::PYRAMID_ROOT_GRAPH_TYPE_MULTI_LAYER,
                                         false,
                                         graph_type,
                                         false,
                                         1,
                                         false,
                                         graph_storage_type);
    vsag::IOStreamReader reader(stream);
    restored.index->Deserialize(reader);
    auto restored_result = restored.index->KnnSearch(query, 10, search_params, nullptr);
    const auto source_stats = vsag::JsonType::Parse(source.index->GetStats());
    const auto restored_stats = vsag::JsonType::Parse(restored.index->GetStats());
    const auto source_root_stats = source_stats["root_graphs"]["default"];
    const auto restored_root_stats = restored_stats["root_graphs"]["default"];
    REQUIRE(restored_root_stats["bottom_graph_node_count"].GetUint64() == count + added_count);
    REQUIRE(restored_root_stats["route_graph_count"].GetUint64() ==
            source_root_stats["route_graph_count"].GetUint64());
    REQUIRE(restored_root_stats["route_node_counts"].GetVector() ==
            source_root_stats["route_node_counts"].GetVector());
    REQUIRE(restored_result->GetDim() == source_result->GetDim());
    REQUIRE(std::equal(source_result->GetIds(),
                       source_result->GetIds() + source_result->GetDim(),
                       restored_result->GetIds()));
    for (uint64_t i = 0; i < source_result->GetDim(); ++i) {
        REQUIRE(std::abs(restored_result->GetDistances()[i] - source_result->GetDistances()[i]) <
                1e-6F);
    }

    constexpr int64_t post_restore_count = 512;
    std::vector<float> post_restore_vectors(post_restore_count * PYRAMID_TEST_DIM);
    FillRootVectors(post_restore_vectors, post_restore_count);
    for (auto& value : post_restore_vectors) {
        value += 2.0F;
    }
    std::vector<int64_t> post_restore_ids(post_restore_count);
    std::iota(post_restore_ids.begin(), post_restore_ids.end(), count + added_count);
    std::vector<std::string> post_restore_paths(post_restore_count, "");
    REQUIRE(restored.index
                ->Add(MakePyramidDataset(post_restore_vectors.data(),
                                         post_restore_ids.data(),
                                         post_restore_paths.data(),
                                         post_restore_count))
                .empty());
    REQUIRE(restored.index->GetNumElements() == count + added_count + post_restore_count);
    for (const int64_t offset : {int64_t{0}, post_restore_count - 1}) {
        auto post_restore_query =
            vsag::Dataset::Make()
                ->NumElements(1)
                ->Dim(PYRAMID_TEST_DIM)
                ->Float32Vectors(post_restore_vectors.data() + offset * PYRAMID_TEST_DIM)
                ->Owner(false);
        const auto post_restore_result =
            restored.index->KnnSearch(post_restore_query, 1, search_params, nullptr);
        REQUIRE(post_restore_result->GetDim() == 1);
        REQUIRE(post_restore_result->GetIds()[0] == post_restore_ids[offset]);
        REQUIRE(std::abs(post_restore_result->GetDistances()[0]) < 1e-6F);
    }
}

TEST_CASE("Pyramid NSW and PiPNN builds retain descendant path search",
          "[ut][pyramid][pipnn][path]") {
    const auto graph_type = GENERATE(std::string(vsag::GRAPH_TYPE_VALUE_NSW),
                                     std::string(vsag::GRAPH_TYPE_VALUE_PIPNN));
    CAPTURE(graph_type);
    constexpr int64_t count = 128;
    std::vector<float> vectors(count * PYRAMID_TEST_DIM);
    FillRootVectors(vectors, count);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<std::string> paths(count, "tenant/leaf");
    auto test_index = MakeRootPyramidIndex(
        vsag::PYRAMID_ROOT_GRAPH_TYPE_SINGLE_LAYER, false, graph_type, false, 4);
    REQUIRE(
        test_index.index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count))
            .empty());

    const auto stats = vsag::JsonType::Parse(test_index.index->GetStats());
    REQUIRE(stats["root_graphs"]["default"]["bottom_graph_node_count"].GetUint64() == count);
    std::string query_path = "tenant/leaf";
    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(PYRAMID_TEST_DIM)
                     ->Float32Vectors(vectors.data() + 73 * PYRAMID_TEST_DIM)
                     ->Paths(&query_path)
                     ->Owner(false);
    auto result = test_index.index->KnnSearch(
        query, 1, R"({"pyramid":{"ef_search":128,"subindex_ef_search":128}})", nullptr);
    REQUIRE(result->GetDim() == 1);
    REQUIRE(result->GetIds()[0] == ids[73]);
}

TEST_CASE("Pyramid NSW Build and empty Add share routed construction",
          "[ut][pyramid][root_graph][build]") {
    constexpr int64_t count = 512;
    std::vector<float> vectors(count * PYRAMID_TEST_DIM);
    FillRootVectors(vectors, count);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<std::string> paths(count, "");

    auto built = MakeRootPyramidIndex(vsag::PYRAMID_ROOT_GRAPH_TYPE_MULTI_LAYER,
                                      true,
                                      vsag::GRAPH_TYPE_VALUE_NSW,
                                      false,
                                      1,
                                      true);
    auto added = MakeRootPyramidIndex(vsag::PYRAMID_ROOT_GRAPH_TYPE_MULTI_LAYER,
                                      true,
                                      vsag::GRAPH_TYPE_VALUE_NSW,
                                      false,
                                      1,
                                      true);
    auto dataset = MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count);
    REQUIRE(built.index->Build(dataset).empty());
    REQUIRE(added.index->Add(dataset).empty());

    const auto built_stats = vsag::JsonType::Parse(built.index->GetStats());
    const auto added_stats = vsag::JsonType::Parse(added.index->GetStats());
    const auto built_root = built_stats["root_graphs"]["default"];
    const auto added_root = added_stats["root_graphs"]["default"];
    REQUIRE(built_root["bottom_graph_node_count"].GetUint64() == count);
    REQUIRE(added_root["bottom_graph_node_count"].GetUint64() == count);
    REQUIRE(built_root["route_node_counts"].GetVector() ==
            added_root["route_node_counts"].GetVector());

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(PYRAMID_TEST_DIM)
                     ->Float32Vectors(vectors.data() + 137 * PYRAMID_TEST_DIM)
                     ->Owner(false);
    const auto search_params = R"({"pyramid":{"ef_search":128,"hops_limit":256}})";
    const auto built_result = built.index->KnnSearch(query, 10, search_params, nullptr);
    const auto added_result = added.index->KnnSearch(query, 10, search_params, nullptr);
    REQUIRE(std::equal(built_result->GetIds(),
                       built_result->GetIds() + built_result->GetDim(),
                       added_result->GetIds()));
}

TEST_CASE("Pyramid flat routed root crosses memory blocks and resizes",
          "[ut][pyramid][root_graph][parallel]") {
    BlockSizeLimitGuard block_size_guard(256 * 1024);
    constexpr int64_t initial_count = 8192;
    constexpr int64_t added_count = 1024;
    constexpr int64_t total_count = initial_count + added_count;
    std::vector<float> vectors(total_count * PYRAMID_TEST_DIM);
    FillRootVectors(vectors, total_count);
    std::vector<int64_t> ids(total_count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<std::string> paths(total_count, "");

    auto test_index = MakeRootPyramidIndex(
        vsag::PYRAMID_ROOT_GRAPH_TYPE_MULTI_LAYER, false, vsag::GRAPH_TYPE_VALUE_NSW, false, 8);
    REQUIRE(test_index.index
                ->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), initial_count))
                .empty());
    REQUIRE(test_index.index
                ->Add(MakePyramidDataset(vectors.data() + initial_count * PYRAMID_TEST_DIM,
                                         ids.data() + initial_count,
                                         paths.data() + initial_count,
                                         added_count))
                .empty());

    const auto stats = vsag::JsonType::Parse(test_index.index->GetStats());
    const auto root_stats = stats["root_graphs"]["default"];
    REQUIRE(root_stats["bottom_graph_storage_type"].GetString() == "flat");
    REQUIRE(root_stats["bottom_graph_node_count"].GetUint64() == total_count);
    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(PYRAMID_TEST_DIM)
                     ->Float32Vectors(vectors.data() + (total_count - 1) * PYRAMID_TEST_DIM)
                     ->Owner(false);
    const auto result =
        test_index.index->KnnSearch(query, 10, R"({"pyramid":{"ef_search":128}})", nullptr);
    // A directed approximate graph does not guarantee that every query reaches k nodes. Verify
    // that traversal remains usable and that the vector appended after the block resize is intact.
    REQUIRE(result->GetDim() > 0);
    REQUIRE(result->GetDim() <= 10);
    REQUIRE(result->GetDistances()[0] < 1e-6F);
    REQUIRE(
        std::abs(test_index.index->CalcDistanceById(
            vectors.data() + (total_count - 1) * PYRAMID_TEST_DIM, ids[total_count - 1], false)) <
        1e-6F);
}

TEST_CASE("Pyramid routed root supports concurrent Add and Search",
          "[ut][pyramid][root_graph][concurrent]") {
    constexpr int64_t initial_count = 512;
    constexpr int64_t added_count = 128;
    std::vector<float> initial_vectors(initial_count * PYRAMID_TEST_DIM);
    std::vector<float> added_vectors(added_count * PYRAMID_TEST_DIM);
    FillRootVectors(initial_vectors, initial_count);
    FillRootVectors(added_vectors, added_count);
    std::vector<int64_t> initial_ids(initial_count);
    std::vector<int64_t> added_ids(added_count);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    std::iota(added_ids.begin(), added_ids.end(), initial_count);
    std::vector<std::string> initial_paths(initial_count, "");
    std::vector<std::string> added_paths(added_count, "");

    auto source = MakeRootPyramidIndex(vsag::PYRAMID_ROOT_GRAPH_TYPE_MULTI_LAYER, true);
    REQUIRE(
        source.index
            ->Build(MakePyramidDataset(
                initial_vectors.data(), initial_ids.data(), initial_paths.data(), initial_count))
            .empty());

    auto add_future = std::async(std::launch::async, [&]() {
        return source.index->Add(MakePyramidDataset(
            added_vectors.data(), added_ids.data(), added_paths.data(), added_count));
    });
    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(PYRAMID_TEST_DIM)
                     ->Float32Vectors(initial_vectors.data() + 137 * PYRAMID_TEST_DIM)
                     ->Owner(false);
    const auto search_params = R"({"pyramid":{"ef_search":128,"hops_limit":256}})";
    for (uint64_t i = 0; i < 32; ++i) {
        REQUIRE(source.index->KnnSearch(query, 10, search_params, nullptr)->GetDim() == 10);
    }
    REQUIRE(add_future.get().empty());
    const auto stats = vsag::JsonType::Parse(source.index->GetStats());
    REQUIRE(stats["root_graphs"]["default"]["bottom_graph_node_count"].GetUint64() ==
            initial_count + added_count);
}

TEST_CASE("Pyramid factor controls reorder candidates without changing final topk",
          "[ut][pyramid][factor]") {
    const auto root_graph_type = GENERATE(std::string(vsag::PYRAMID_ROOT_GRAPH_TYPE_SINGLE_LAYER),
                                          std::string(vsag::PYRAMID_ROOT_GRAPH_TYPE_MULTI_LAYER));
    constexpr int64_t count = 128;
    std::vector<float> vectors(count * PYRAMID_TEST_DIM);
    FillRootVectors(vectors, count);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<std::string> paths(count, "");
    auto test_index = MakeRootPyramidIndex(root_graph_type, true);
    REQUIRE(
        test_index.index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count))
            .empty());
    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(PYRAMID_TEST_DIM)
                     ->Float32Vectors(vectors.data())
                     ->Owner(false);
    const std::array<std::pair<std::string, uint32_t>, 5> cases = {
        std::pair{R"({"pyramid":{"ef_search":20}})", 20U},
        std::pair{R"({"pyramid":{"ef_search":20,"factor":1.0}})", 20U},
        std::pair{R"({"pyramid":{"ef_search":20,"factor":2.0}})", 10U},
        std::pair{R"({"pyramid":{"ef_search":20,"factor":10.0}})", 20U},
        std::pair{R"({"pyramid":{"ef_search":20,"factor":3.0e38}})", 20U}};
    for (const auto& [params, expected_candidates] : cases) {
        auto result = test_index.index->KnnSearch(query, 5, params, nullptr);
        REQUIRE(result->GetDim() == 5);
        auto stats = vsag::JsonType::Parse(result->GetStatistics());
        REQUIRE(stats["reorder_candidate_count"].GetInt() == expected_candidates);
        REQUIRE(stats["reorder_distance_count"].GetInt() == expected_candidates);
    }

    vsag::SearchRequest request;
    request.query_ = query;
    request.topk_ = 5;
    request.params_str_ = R"({"pyramid":{"ef_search":20,"factor":2.0}})";
    const auto request_result = test_index.index->SearchWithRequest(request);
    REQUIRE(request_result->GetDim() == 5);
    REQUIRE(vsag::JsonType::Parse(request_result->GetStatistics())["reorder_candidate_count"]
                .GetUint64() == 10);

    auto large_topk = test_index.index->KnnSearch(
        query, 25, R"({"pyramid":{"ef_search":20,"factor":2.0}})", nullptr);
    REQUIRE(large_topk->GetDim() == 25);
    REQUIRE(
        vsag::JsonType::Parse(large_topk->GetStatistics())["reorder_candidate_count"].GetInt() ==
        25);

    auto no_reorder = MakeRootPyramidIndex(root_graph_type, false);
    REQUIRE(
        no_reorder.index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count))
            .empty());
    auto no_reorder_result = no_reorder.index->KnnSearch(
        query, 5, R"({"pyramid":{"ef_search":20,"factor":2.0}})", nullptr);
    REQUIRE(no_reorder_result->GetDim() == 5);
    REQUIRE(vsag::JsonType::Parse(no_reorder_result->GetStatistics())["reorder_candidate_count"]
                .GetInt() == 0);

    REQUIRE_THROWS(test_index.index->KnnSearch(
        query, 5, R"({"pyramid":{"ef_search":20,"factor":0.0}})", nullptr));
    REQUIRE_THROWS(test_index.index->KnnSearch(
        query, 5, R"({"pyramid":{"ef_search":20,"factor":-1.0}})", nullptr));
}

TEST_CASE("Pyramid factor limits merged subgraph candidates without changing subgraph search",
          "[ut][pyramid][factor]") {
    constexpr int64_t count = 512;
    std::vector<float> vectors(count * PYRAMID_TEST_DIM);
    FillRootVectors(vectors, count);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<std::string> paths(count);
    for (int64_t i = 0; i < count; ++i) {
        paths[i] = i % 2 == 0 ? "tenant/left" : "tenant/right";
    }
    auto test_index = MakePyramidIndex(1, 1, false, false, false, false, true);
    REQUIRE(
        test_index.index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count))
            .empty());

    std::string query_path = "tenant/left|tenant/right";
    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(PYRAMID_TEST_DIM)
                     ->Float32Vectors(vectors.data() + 137 * PYRAMID_TEST_DIM)
                     ->Paths(&query_path)
                     ->Owner(false);
    const std::array<std::pair<std::string, uint32_t>, 3> cases = {
        std::pair{R"({"pyramid":{"ef_search":100,"subindex_ef_search":20}})", 40U},
        std::pair{R"({"pyramid":{"ef_search":100,"subindex_ef_search":20,"factor":1.0}})", 40U},
        std::pair{R"({"pyramid":{"ef_search":100,"subindex_ef_search":20,"factor":2.0}})", 10U}};
    uint64_t approximate_distance_count = 0;
    for (uint64_t i = 0; i < cases.size(); ++i) {
        const auto& [params, expected_candidates] = cases[i];
        const auto result = test_index.index->KnnSearch(query, 5, params, nullptr);
        REQUIRE(result->GetDim() == 5);
        const auto stats = vsag::JsonType::Parse(result->GetStatistics());
        REQUIRE(stats["reorder_candidate_count"].GetUint64() == expected_candidates);
        const auto current_approximate_count =
            stats["distance_evaluations_by_phase"]["approximate"].GetUint64();
        if (i == 0) {
            approximate_distance_count = current_approximate_count;
        } else {
            REQUIRE(current_approximate_count == approximate_distance_count);
        }
    }

    auto split_index = MakePyramidIndex(1, 1, false, true);
    REQUIRE(split_index.index
                ->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count))
                .empty());
    const auto no_factor = split_index.index->KnnSearch(
        query,
        5,
        R"({"threshold":1e30,"pyramid":{"ef_search":100,"subindex_ef_search":1,"rabitq_one_bit_search":true,"rabitq_error_rate":1e-6}})",
        nullptr);
    const auto factor_one = split_index.index->KnnSearch(
        query,
        5,
        R"({"threshold":1e30,"pyramid":{"ef_search":100,"subindex_ef_search":1,"rabitq_one_bit_search":true,"rabitq_error_rate":1e-6,"factor":1.0}})",
        nullptr);
    const auto no_factor_stats = vsag::JsonType::Parse(no_factor->GetStatistics());
    const auto factor_one_stats = vsag::JsonType::Parse(factor_one->GetStatistics());
    REQUIRE(no_factor->GetDim() == 5);
    REQUIRE(factor_one->GetDim() == 5);
    REQUIRE(no_factor_stats["reorder_candidate_count"].GetUint64() ==
            factor_one_stats["reorder_candidate_count"].GetUint64());
    REQUIRE(no_factor_stats["reorder_distance_count"].GetUint64() ==
            factor_one_stats["reorder_distance_count"].GetUint64());
    REQUIRE(no_factor_stats["reorder_candidate_count"].GetUint64() >= 100);
    REQUIRE(no_factor_stats["reorder_lower_bound_probe_count"].GetUint64() > 0);
    REQUIRE(no_factor_stats["reorder_distance_count"].GetUint64() >= 100);
}

TEST_CASE("Pyramid legacy deserialization rejects incompatible root graph type",
          "[ut][pyramid][root_graph][serialization]") {
    constexpr int64_t count = 32;
    std::vector<float> vectors(count * PYRAMID_TEST_DIM);
    FillRootVectors(vectors, count);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<std::string> paths(count, "");

    const std::array<std::pair<std::string, std::string>, 2> cases = {
        std::pair{std::string(vsag::PYRAMID_ROOT_GRAPH_TYPE_SINGLE_LAYER),
                  std::string(vsag::PYRAMID_ROOT_GRAPH_TYPE_MULTI_LAYER)},
        std::pair{std::string(vsag::PYRAMID_ROOT_GRAPH_TYPE_MULTI_LAYER),
                  std::string(vsag::PYRAMID_ROOT_GRAPH_TYPE_SINGLE_LAYER)}};
    for (const auto& [serialized_type, configured_type] : cases) {
        auto source = MakeRootPyramidIndex(serialized_type);
        REQUIRE(source.index
                    ->Build(MakePyramidDataset(
                        vectors.data(), ids.data(), paths.data(), static_cast<int64_t>(ids.size())))
                    .empty());
        std::stringstream stream;
        vsag::IOStreamWriter writer(stream);
        source.index->Serialize(writer);

        auto target = MakeRootPyramidIndex(configured_type);
        vsag::IOStreamReader reader(stream);
        try {
            target.index->Deserialize(reader);
            FAIL("incompatible root graph type must be rejected");
        } catch (const vsag::VsagException& error) {
            REQUIRE(std::string(error.what()).find("Pyramid index parameter not match") !=
                    std::string::npos);
        }
    }
}

TEST_CASE("Pyramid multi-layer root routes duplicate representatives",
          "[ut][pyramid][root_graph][duplicate][pipnn]") {
    const auto graph_type = GENERATE(std::string(vsag::GRAPH_TYPE_VALUE_NSW),
                                     std::string(vsag::GRAPH_TYPE_VALUE_PIPNN));
    CAPTURE(graph_type);
    constexpr int64_t count = 512;
    std::vector<float> vectors(count * PYRAMID_TEST_DIM);
    std::vector<int64_t> ids(count);
    std::vector<std::string> paths(count, "");
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = i;
        const auto representative = i / 2;
        for (int64_t d = 0; d < PYRAMID_TEST_DIM; ++d) {
            vectors[i * PYRAMID_TEST_DIM + d] =
                static_cast<float>((representative * (d + 3) + d * 17) % 997) / 997.0F;
        }
    }
    auto test_index =
        MakeRootPyramidIndex(vsag::PYRAMID_ROOT_GRAPH_TYPE_MULTI_LAYER, false, graph_type, true);
    REQUIRE(
        test_index.index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count))
            .empty());
    const auto stats = vsag::JsonType::Parse(test_index.index->GetStats());
    const auto bottom_count =
        stats["root_graphs"]["default"]["bottom_graph_node_count"].GetUint64();
    REQUIRE(bottom_count < count);
    REQUIRE(stats["root_graphs"]["default"]["route_node_counts"].GetVector().front() <
            bottom_count);

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(PYRAMID_TEST_DIM)
                     ->Float32Vectors(vectors.data() + 200 * PYRAMID_TEST_DIM)
                     ->Owner(false);
    auto result = test_index.index->KnnSearch(
        query, 10, R"({"pyramid":{"ef_search":256,"hops_limit":512}})", nullptr);
    REQUIRE(result->GetDim() == 10);
    std::set<int64_t> result_ids(result->GetIds(), result->GetIds() + result->GetDim());
    REQUIRE(result_ids.count(200) == 1);
}

TEST_CASE("Pyramid applies hops limit to non-root graphs", "[ut][pyramid][hops_limit]") {
    constexpr int64_t count = 512;
    std::vector<float> vectors(count * PYRAMID_TEST_DIM);
    FillRootVectors(vectors, count);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<std::string> paths(count, "tenant/leaf");
    auto test_index = MakePyramidIndex(1);
    REQUIRE(
        test_index.index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count))
            .empty());
    std::string query_path = "tenant/leaf";
    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(PYRAMID_TEST_DIM)
                     ->Float32Vectors(vectors.data() + 311 * PYRAMID_TEST_DIM)
                     ->Paths(&query_path)
                     ->Owner(false);
    auto unlimited =
        test_index.index->KnnSearch(query, 1, R"({"pyramid":{"ef_search":2}})", nullptr);
    auto limited = test_index.index->KnnSearch(
        query, 1, R"({"pyramid":{"ef_search":2,"hops_limit":3}})", nullptr);
    const auto unlimited_stats = vsag::JsonType::Parse(unlimited->GetStatistics());
    const auto limited_stats = vsag::JsonType::Parse(limited->GetStatistics());
    REQUIRE(unlimited_stats["hops"].GetInt() > limited_stats["hops"].GetInt());
    REQUIRE(limited_stats["hops"].GetInt() <= 3);
}

TEST_CASE("Pyramid PiPNN multi-layer root supports RaBitQ with SQ8 reorder",
          "[ut][pipnn][pyramid]") {
    constexpr int64_t count = 48;
    std::vector<float> vectors((count + 1) * PYRAMID_TEST_DIM);
    FillRootVectors(vectors, count + 1);
    std::vector<int64_t> ids(count + 1);
    std::iota(ids.begin(), ids.end(), 1000);
    std::vector<std::string> paths(count + 1, "tenant");

    auto test_index = MakeRootPyramidIndex(vsag::PYRAMID_ROOT_GRAPH_TYPE_MULTI_LAYER,
                                           false,
                                           vsag::GRAPH_TYPE_VALUE_PIPNN,
                                           false,
                                           4,
                                           true);
    REQUIRE(
        test_index.index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count))
            .empty());
    auto stats = vsag::JsonType::Parse(test_index.index->GetStats());
    REQUIRE(stats["root_graphs"]["default"]["route_graph_count"].GetUint64() > 0);

    auto added = MakePyramidDataset(
        vectors.data() + count * PYRAMID_TEST_DIM, ids.data() + count, paths.data() + count, 1);
    REQUIRE(test_index.index->Add(added).empty());

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    vsag::IOStreamWriter writer(stream);
    test_index.index->Serialize(writer);
    auto restored = MakeRootPyramidIndex(vsag::PYRAMID_ROOT_GRAPH_TYPE_MULTI_LAYER,
                                         false,
                                         vsag::GRAPH_TYPE_VALUE_PIPNN,
                                         false,
                                         4,
                                         true);
    stream.seekg(0);
    vsag::IOStreamReader reader(stream);
    restored.index->Deserialize(reader);

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(PYRAMID_TEST_DIM)
                     ->Float32Vectors(vectors.data() + count * PYRAMID_TEST_DIM)
                     ->Owner(false);
    auto result =
        restored.index->KnnSearch(query, 5, R"({"pyramid":{"ef_search":64}})", vsag::FilterPtr{});
    REQUIRE(std::find(result->GetIds(), result->GetIds() + result->GetDim(), ids[count]) !=
            result->GetIds() + result->GetDim());
}
