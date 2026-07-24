
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

#include <array>
#include <future>
#include <sstream>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "unittest.h"
#include "vsag/index.h"

namespace {

constexpr int64_t PYRAMID_TEST_DIM = 4;

struct PyramidTestIndex {
    std::shared_ptr<vsag::Allocator> allocator;
    std::shared_ptr<vsag::Pyramid> index;
};

PyramidTestIndex
MakePyramidIndex(uint32_t index_min_size) {
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
    external_param[vsag::PYRAMID_INDEX_MIN_SIZE].SetInt(index_min_size);
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

TEST_CASE("Pyramid promotes flat node at index minimum size", "[ut][pyramid]") {
    auto test_index = MakePyramidIndex(3);
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

    REQUIRE(index->Add(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), 2)).empty());
    REQUIRE(GetPyramidSubindexCount(index, "flat_subindexes") == 1);
    REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 0);

    REQUIRE(index
                ->Add(MakePyramidDataset(
                    vectors.data() + 2 * PYRAMID_TEST_DIM, ids.data() + 2, paths.data() + 2, 1))
                .empty());
    REQUIRE(GetPyramidSubindexCount(index, "flat_subindexes") == 0);
    REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 1);
    REQUIRE(GetPyramidSubindexCount(index, "total_vectors_in_graph") == 3);

    for (int64_t i = 0; i < 3; ++i) {
        auto query =
            MakePyramidDataset(vectors.data() + i * PYRAMID_TEST_DIM, nullptr, paths.data() + i, 1);
        auto result =
            index->KnnSearch(query, 1, R"({"pyramid":{"ef_search":10}})", vsag::FilterPtr{});
        REQUIRE(result->GetIds()[0] == ids[i]);
    }
}

TEST_CASE("Pyramid stores and restores raw vectors", "[ut][pyramid][raw_vector]") {
    constexpr int64_t dim = 4;
    constexpr int64_t count = 3;
    std::array<float, 12> vectors = {
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        0.123456F,
        0.234567F,
        0.345678F,
        0.456789F,
        1.0F,
        1.0F,
        1.0F,
        1.0F,
    };
    std::array<int64_t, count> ids = {10, 11, 12};
    std::array<std::string, count> paths = {"leaf", "leaf", "leaf"};

    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
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
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    auto pyramid_param = std::dynamic_pointer_cast<vsag::PyramidParameters>(param);

    REQUIRE(pyramid_param != nullptr);
    REQUIRE(pyramid_param->store_raw_vector);
    REQUIRE(pyramid_param->raw_vector_param != nullptr);
    REQUIRE(pyramid_param->raw_vector_param->quantizer_parameter->GetTypeName() == "fp32");

    auto dataset = vsag::Dataset::Make()
                       ->NumElements(count)
                       ->Dim(dim)
                       ->Float32Vectors(vectors.data())
                       ->Ids(ids.data())
                       ->Paths(paths.data())
                       ->Owner(false);
    auto index = std::make_shared<vsag::Pyramid>(pyramid_param, common_param);
    index->InitFeatures();
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS));
    REQUIRE(index->Build(dataset).empty());

    std::array<float, dim> restored{};
    index->GetVectorByInnerId(1, restored.data());
    for (int64_t i = 0; i < dim; ++i) {
        REQUIRE(restored[i] == vectors[dim + i]);
    }
    REQUIRE(index->CalcDistanceById(vectors.data() + dim, ids[1], true) == 0.0F);

    auto binary_set = index->vsag::InnerIndexInterface::Serialize();
    auto loaded = std::make_shared<vsag::Pyramid>(pyramid_param, common_param);
    loaded->vsag::InnerIndexInterface::Deserialize(binary_set);
    restored.fill(0.0F);
    loaded->GetVectorByInnerId(1, restored.data());
    for (int64_t i = 0; i < dim; ++i) {
        REQUIRE(restored[i] == vectors[dim + i]);
    }

    std::stringstream streaming_buffer;
    index->SerializeStreaming(streaming_buffer);
    auto streaming_loaded = std::make_shared<vsag::Pyramid>(pyramid_param, common_param);
    std::stringstream streaming_reader(streaming_buffer.str());
    streaming_loaded->DeserializeStreaming(streaming_reader);
    restored.fill(0.0F);
    streaming_loaded->GetVectorByInnerId(1, restored.data());
    for (int64_t i = 0; i < dim; ++i) {
        REQUIRE(restored[i] == vectors[dim + i]);
    }
}

TEST_CASE("Pyramid stores raw vectors during ODescent build", "[ut][pyramid][raw_vector]") {
    constexpr int64_t dim = 4;
    std::array<float, 8> vectors = {
        0.123456F, 0.234567F, 0.345678F, 0.456789F, 1.0F, 1.0F, 1.0F, 1.0F};
    std::array<int64_t, 2> ids = {10, 11};
    std::array<std::string, 2> paths = {"leaf", "leaf"};

    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();

    auto external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "sq8",
        "store_raw_vector": true,
        "graph_type": "odescent",
        "max_degree": 4,
        "no_build_levels": [0, 1]
    })");
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    auto index = std::make_shared<vsag::Pyramid>(param, common_param);
    auto dataset = vsag::Dataset::Make()
                       ->NumElements(2)
                       ->Dim(dim)
                       ->Float32Vectors(vectors.data())
                       ->Ids(ids.data())
                       ->Paths(paths.data())
                       ->Owner(false);

    REQUIRE(index->Build(dataset).empty());
    std::array<float, dim> restored{};
    index->GetVectorByInnerId(0, restored.data());
    for (int64_t i = 0; i < dim; ++i) {
        REQUIRE(restored[i] == vectors[i]);
    }
}

TEST_CASE("Pyramid reuses FP32 codes as raw vectors", "[ut][pyramid][raw_vector]") {
    constexpr int64_t dim = 4;
    std::array<float, dim> vector = {0.123456F, 0.234567F, 0.345678F, 0.456789F};
    std::array<int64_t, 1> ids = {10};
    std::array<std::string, 1> paths = {"leaf"};

    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();

    auto external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "use_reorder": true,
        "precise_quantization_type": "sq8",
        "store_raw_vector": true,
        "max_degree": 4,
        "ef_construction": 8,
        "no_build_levels": [0, 1]
    })");
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    auto index = std::make_shared<vsag::Pyramid>(param, common_param);
    auto dataset = vsag::Dataset::Make()
                       ->NumElements(1)
                       ->Dim(dim)
                       ->Float32Vectors(vector.data())
                       ->Ids(ids.data())
                       ->Paths(paths.data())
                       ->Owner(false);

    REQUIRE(index->Build(dataset).empty());
    std::array<float, dim> restored{};
    index->GetVectorByInnerId(0, restored.data());
    REQUIRE(restored == vector);
}

TEST_CASE("Pyramid streaming load preserves cosine raw vectors", "[ut][pyramid][raw_vector]") {
    constexpr int64_t dim = 4;
    std::array<float, 8> vectors = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::array<int64_t, 2> ids = {10, 11};
    std::array<std::string, 2> paths = {"leaf", "leaf"};

    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_COSINE;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();

    auto external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "store_raw_vector": true,
        "max_degree": 4,
        "ef_construction": 8,
        "no_build_levels": [0, 1]
    })");
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    auto index = std::make_shared<vsag::Pyramid>(param, common_param);
    auto dataset = vsag::Dataset::Make()
                       ->NumElements(2)
                       ->Dim(dim)
                       ->Float32Vectors(vectors.data())
                       ->Ids(ids.data())
                       ->Paths(paths.data())
                       ->Owner(false);

    REQUIRE(index->Build(dataset).empty());
    std::stringstream stream;
    index->SerializeStreaming(stream);

    std::stringstream load_stream(stream.str());
    auto loaded = vsag::Index::Load(load_stream, "{}");
    REQUIRE(loaded.has_value());
    auto raw = loaded.value()->GetRawVectorByIds(ids.data(), 2);
    REQUIRE(raw.has_value());
    const auto* restored = raw.value()->GetFloat32Vectors();
    for (int64_t i = 0; i < dim * 2; ++i) {
        REQUIRE(restored[i] == vectors[i]);
    }
}

TEST_CASE("Pyramid legacy deserialize validates raw vector config before reading",
          "[ut][pyramid][raw_vector]") {
    constexpr int64_t dim = 4;
    std::array<float, dim> vector = {1.0F, 2.0F, 3.0F, 4.0F};
    std::array<int64_t, 1> ids = {10};
    std::array<std::string, 1> paths = {"leaf"};

    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();

    auto stored_external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "sq8",
        "store_raw_vector": true,
        "max_degree": 4,
        "ef_construction": 8,
        "no_build_levels": [0, 1]
    })");
    auto stored_param =
        vsag::Pyramid::CheckAndMappingExternalParam(stored_external_param, common_param);
    auto stored = std::make_shared<vsag::Pyramid>(stored_param, common_param);
    auto dataset = vsag::Dataset::Make()
                       ->NumElements(1)
                       ->Dim(dim)
                       ->Float32Vectors(vector.data())
                       ->Ids(ids.data())
                       ->Paths(paths.data())
                       ->Owner(false);
    REQUIRE(stored->Build(dataset).empty());
    auto binary_set = stored->vsag::InnerIndexInterface::Serialize();

    auto target_external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "sq8",
        "store_raw_vector": false,
        "max_degree": 4,
        "ef_construction": 8,
        "no_build_levels": [0, 1]
    })");
    auto target_param =
        vsag::Pyramid::CheckAndMappingExternalParam(target_external_param, common_param);
    auto target = std::make_shared<vsag::Pyramid>(target_param, common_param);

    REQUIRE_THROWS(target->vsag::InnerIndexInterface::Deserialize(binary_set));
    REQUIRE(target->GetNumElements() == 0);
}

TEST_CASE("Pyramid reports live raw vector memory", "[ut][pyramid][raw_vector][memory]") {
    constexpr int64_t dim = 4;
    std::array<float, 8> vectors = {
        0.123456F, 0.234567F, 0.345678F, 0.456789F, 1.0F, 2.0F, 3.0F, 4.0F};
    std::array<int64_t, 2> ids = {10, 11};
    std::array<std::string, 2> paths = {"leaf", "leaf"};

    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
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
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    auto index = std::make_shared<vsag::Pyramid>(param, common_param);
    auto dataset = vsag::Dataset::Make()
                       ->NumElements(2)
                       ->Dim(dim)
                       ->Float32Vectors(vectors.data())
                       ->Ids(ids.data())
                       ->Paths(paths.data())
                       ->Owner(false);

    REQUIRE(index->Build(dataset).empty());
    REQUIRE(index->GetMemoryUsage() > 0);
    auto detail = index->GetMemoryUsageDetail();
    REQUIRE(detail.at("base_codes") > 0);
    REQUIRE(detail.at("raw_vector") >= sizeof(float) * dim * 2);
    REQUIRE(index->GetMemoryUsage() >= detail.at("raw_vector"));
}

TEST_CASE("Pyramid raw vector serialization handles IO storage changes",
          "[ut][pyramid][raw_vector]") {
    constexpr int64_t dim = 4;
    constexpr int64_t count = 2;
    std::array<float, dim* count> vectors = {
        0.123456F, 0.234567F, 0.345678F, 0.456789F, 1.0F, 2.0F, 3.0F, 4.0F};
    std::array<int64_t, count> ids = {10, 11};
    std::array<std::string, count> paths = {"leaf", "leaf"};

    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();

    fixtures::TempDir dir("pyramid_raw_vector_io");
    auto make_param = [&](bool use_dedicated_storage) {
        auto external_param = vsag::JsonType::Parse(R"({
            "base_quantization_type": "fp32",
            "store_raw_vector": true,
            "max_degree": 4,
            "ef_construction": 8,
            "no_build_levels": [0, 1]
        })");
        auto io_type = use_dedicated_storage ? "buffer_io" : "block_memory_io";
        external_param["base_io_type"].SetString(io_type);
        external_param["base_file_path"].SetString(dir.GenerateRandomFile(false));
        external_param["raw_vector_io_type"].SetString(io_type);
        external_param["raw_vector_file_path"].SetString(dir.GenerateRandomFile(false));
        return vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    };
    auto dataset = vsag::Dataset::Make()
                       ->NumElements(count)
                       ->Dim(dim)
                       ->Float32Vectors(vectors.data())
                       ->Ids(ids.data())
                       ->Paths(paths.data())
                       ->Owner(false);

    auto round_trip = [&](bool producer_uses_dedicated_storage, bool streaming) {
        auto producer = std::make_shared<vsag::Pyramid>(make_param(producer_uses_dedicated_storage),
                                                        common_param);
        REQUIRE(producer->Build(dataset).empty());

        auto loaded = std::make_shared<vsag::Pyramid>(
            make_param(not producer_uses_dedicated_storage), common_param);
        if (streaming) {
            std::stringstream buffer;
            producer->SerializeStreaming(buffer);
            std::stringstream reader(buffer.str());
            loaded->DeserializeStreaming(reader);
        } else {
            auto binary_set = producer->vsag::InnerIndexInterface::Serialize();
            loaded->vsag::InnerIndexInterface::Deserialize(binary_set);
        }

        auto restored = loaded->GetDataByIds(ids.data(), count);
        REQUIRE(restored->GetFloat32Vectors() != nullptr);
        for (int64_t i = 0; i < dim * count; ++i) {
            REQUIRE(restored->GetFloat32Vectors()[i] == vectors[i]);
        }

        auto memory_detail = loaded->GetMemoryUsageDetail();
        if (producer_uses_dedicated_storage) {
            REQUIRE(memory_detail.count("raw_vector") == 0);
        } else {
            REQUIRE(memory_detail.at("raw_vector") >= sizeof(float) * dim * count);
        }
    };

    SECTION("legacy alias to dedicated") {
        round_trip(false, false);
    }
    SECTION("legacy dedicated to alias") {
        round_trip(true, false);
    }
    SECTION("streaming alias to dedicated") {
        round_trip(false, true);
    }
    SECTION("streaming dedicated to alias") {
        round_trip(true, true);
    }
}

TEST_CASE("Pyramid does not alias normalized FP32 codes as cosine raw vectors",
          "[ut][pyramid][raw_vector]") {
    constexpr int64_t dim = 4;
    std::array<float, dim> vector = {1.0F, 2.0F, 3.0F, 4.0F};
    std::array<int64_t, 1> ids = {10};
    std::array<std::string, 1> paths = {"leaf"};

    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_COSINE;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();

    auto external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "store_raw_vector": true,
        "max_degree": 4,
        "ef_construction": 8,
        "no_build_levels": [0, 1]
    })");
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    auto pyramid_param = std::dynamic_pointer_cast<vsag::PyramidParameters>(param);
    REQUIRE(pyramid_param != nullptr);
    vsag::JsonType no_molds;
    no_molds["hold_molds"].SetBool(false);
    pyramid_param->base_codes_param->quantizer_parameter->FromJson(no_molds);

    auto index = std::make_shared<vsag::Pyramid>(pyramid_param, common_param);
    auto dataset = vsag::Dataset::Make()
                       ->NumElements(1)
                       ->Dim(dim)
                       ->Float32Vectors(vector.data())
                       ->Ids(ids.data())
                       ->Paths(paths.data())
                       ->Owner(false);
    REQUIRE(index->Build(dataset).empty());

    auto memory_detail = index->GetMemoryUsageDetail();
    REQUIRE(memory_detail.at("raw_vector") >= sizeof(float) * dim);
    auto restored = index->GetDataByIds(ids.data(), 1);
    REQUIRE(restored->GetFloat32Vectors() != nullptr);
    for (int64_t i = 0; i < dim; ++i) {
        REQUIRE(restored->GetFloat32Vectors()[i] == vector[i]);
    }
}

TEST_CASE("Pyramid reports memory while adding vectors", "[ut][pyramid][raw_vector][memory]") {
    constexpr int64_t dim = 4;
    constexpr int64_t add_count = 256;
    std::array<float, dim> initial_vector = {0.0F, 0.0F, 0.0F, 0.0F};
    std::array<int64_t, 1> initial_id = {1};
    std::array<std::string, 1> initial_path = {"leaf"};
    std::vector<float> vectors(dim * add_count, 1.0F);
    std::vector<int64_t> ids(add_count);
    std::vector<std::string> paths(add_count, "leaf");
    for (int64_t i = 0; i < add_count; ++i) {
        ids[i] = i + 2;
    }

    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "sq8",
        "store_raw_vector": true,
        "max_degree": 4,
        "ef_construction": 8
    })");
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    auto index = std::make_shared<vsag::Pyramid>(param, common_param);
    auto initial_dataset = vsag::Dataset::Make()
                               ->NumElements(1)
                               ->Dim(dim)
                               ->Float32Vectors(initial_vector.data())
                               ->Ids(initial_id.data())
                               ->Paths(initial_path.data())
                               ->Owner(false);
    REQUIRE(index->Build(initial_dataset).empty());

    std::promise<void> add_started;
    auto started = add_started.get_future();
    auto add_result = std::async(std::launch::async, [&]() {
        add_started.set_value();
        for (int64_t i = 0; i < add_count; ++i) {
            auto add_dataset = vsag::Dataset::Make()
                                   ->NumElements(1)
                                   ->Dim(dim)
                                   ->Float32Vectors(vectors.data() + i * dim)
                                   ->Ids(ids.data() + i)
                                   ->Paths(paths.data() + i)
                                   ->Owner(false);
            if (not index->Add(add_dataset).empty()) {
                return false;
            }
        }
        return true;
    });
    started.wait();
    for (int64_t i = 0; i < 32; ++i) {
        auto detail = index->GetMemoryUsageDetail();
        REQUIRE(detail.at("label_table") > 0);
        REQUIRE(detail.at("raw_vector") > 0);
    }
    REQUIRE(add_result.get());
    REQUIRE(index->GetNumElements() == add_count + 1);
}
