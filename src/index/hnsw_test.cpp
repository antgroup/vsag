
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

#include "hnsw.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

#include "data_type.h"
#include "datacell/graph_datacell_parameter.h"
#include "fixtures.h"
#include "impl/logger/logger.h"
#include "io/memory_io_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "storage/serialization.h"
#include "vsag/bitset.h"
#include "vsag/errors.h"
#include "vsag/options.h"

using namespace vsag;

HnswParameters
parse_hnsw_params(IndexCommonParam index_common_param) {
    auto build_parameter_json = R"(
        {
            "max_degree": 12,
            "ef_construction": 100
        }
    )";
    auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
    return HnswParameters::FromJson(parsed_params, index_common_param);
}

TEST_CASE("build & add", "[ut][hnsw]") {
    logger::set_level(logger::level::debug);
    int64_t dim = 128;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = allocator;

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 12;
    hnsw_obj.ef_construction = 100;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);

    std::vector<int64_t> ids(1);
    int64_t incorrect_dim = 63;
    std::vector<float> vectors(incorrect_dim);

    auto dataset = Dataset::Make();
    dataset->Dim(incorrect_dim)
        ->NumElements(1)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    SECTION("build with incorrect dim") {
        auto result = index->Build(dataset);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == ErrorType::INVALID_ARGUMENT);
    }

    SECTION("add with incorrect dim") {
        auto result = index->Add(dataset);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("build with allocator", "[ut][hnsw]") {
    logger::set_level(logger::level::debug);

    int64_t dim = 128;
    IndexCommonParam common_param;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = allocator;

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 12;
    hnsw_obj.ef_construction = 100;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();

    const int64_t num_elements = 10;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

    auto dataset = Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(1)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());
}

TEST_CASE("knn_search", "[ut][hnsw]") {
    logger::set_level(logger::level::debug);

    int64_t dim = 128;
    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 12;
    hnsw_obj.ef_construction = 100;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();

    const int64_t num_elements = 10;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

    auto dataset = Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(1)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto build_result = index->Build(dataset);
    REQUIRE(build_result.has_value());

    auto query = Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);
    int64_t k = 10;

    JsonType params;
    params["hnsw"]["ef_search"].SetInt(100);

    SECTION("invalid parameters k is 0") {
        auto result = index->KnnSearch(query, 0, params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters k less than 0") {
        auto result = index->KnnSearch(query, -1, params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters hnsw not found") {
        JsonType invalid_params{};
        auto result = index->KnnSearch(query, k, invalid_params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters ef_search not found") {
        JsonType invalid_params;
        invalid_params["hnsw"].SetJson(JsonType());
        auto result = index->KnnSearch(query, k, invalid_params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == ErrorType::INVALID_ARGUMENT);
    }

    SECTION("query length is not 1") {
        auto query2 = Dataset::Make();
        query2->NumElements(2)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);
        auto result = index->KnnSearch(query2, k, params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == ErrorType::INVALID_ARGUMENT);
    }

    SECTION("dimension not equal") {
        auto query2 = Dataset::Make();
        query2->NumElements(1)->Dim(dim - 1)->Float32Vectors(vectors.data())->Owner(false);
        auto result = index->KnnSearch(query2, k, params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("range_search", "[ut][hnsw]") {
    logger::set_level(logger::level::debug);

    int64_t dim = 128;
    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 12;
    hnsw_obj.ef_construction = 100;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();

    const int64_t num_elements = 10;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

    auto dataset = Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(num_elements)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto build_result = index->Build(dataset);
    REQUIRE(build_result.has_value());

    auto query = Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);
    float radius = 9.9f;
    JsonType params;
    params["hnsw"]["ef_search"].SetInt(100);

    SECTION("successful case with smaller range_search_limit") {
        int64_t range_search_limit = num_elements - 1;
        auto result = index->RangeSearch(query, 100, params.Dump(), range_search_limit);
        REQUIRE(result.has_value());
        REQUIRE((*result)->GetDim() == range_search_limit);
    }

    SECTION("successful case with larger range_search_limit") {
        int64_t range_search_limit = num_elements + 1;
        auto result = index->RangeSearch(query, 100, params.Dump(), range_search_limit);
        REQUIRE(result.has_value());
        REQUIRE((*result)->GetDim() == num_elements);
    }

    SECTION("invalid parameter range_search_limit less than 0") {
        int64_t range_search_limit = -1;
        auto result = index->RangeSearch(query, 1000, params.Dump(), range_search_limit);
        REQUIRE(result.has_value());
        REQUIRE((*result)->GetDim() == num_elements);
    }

    SECTION("invalid parameter range_search_limit equals to 0") {
        int64_t range_search_limit = 0;
        auto result = index->RangeSearch(query, 1000, params.Dump(), range_search_limit);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameter radius equals to 0") {
        auto result = index->RangeSearch(query, 0, params.Dump());
        REQUIRE(result.has_value());
    }

    SECTION("invalid parameter radius less than 0") {
        auto result = index->RangeSearch(query, -1, params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters hnsw not found") {
        JsonType invalid_params{};
        auto result = index->RangeSearch(query, radius, invalid_params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters ef_search not found") {
        JsonType invalid_params;
        invalid_params["hnsw"].SetJson(JsonType());
        auto result = index->RangeSearch(query, radius, invalid_params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == ErrorType::INVALID_ARGUMENT);
    }

    SECTION("query length is not 1") {
        auto query2 = Dataset::Make();
        query2->NumElements(2)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);
        auto result = index->RangeSearch(query2, radius, params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("serialize empty index", "[ut][hnsw]") {
    logger::set_level(logger::level::debug);

    int64_t dim = 128;
    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 12;
    hnsw_obj.ef_construction = 100;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();

    SECTION("serialize to binaryset") {
        auto result = index->Serialize();
        REQUIRE(result.has_value());
        REQUIRE(result.value().Contains(SERIAL_META_KEY));
        auto metadata = std::make_shared<Metadata>(result.value().Get(SERIAL_META_KEY));
        REQUIRE(metadata->EmptyIndex());
    }

    SECTION("serialize to fstream") {
        fixtures::TempDir dir("hnsw_test_serialize_empty_index");
        std::fstream out_stream(dir.path + "empty_index.bin", std::ios::out | std::ios::binary);
        auto result = index->Serialize(out_stream);
        REQUIRE(result.has_value());
        out_stream.close();
        std::fstream in_stream(dir.path + "empty_index.bin", std::ios::in | std::ios::binary);
        IOStreamReader reader(in_stream);
        auto footer = Footer::Parse(reader);
        REQUIRE(footer->GetMetadata()->EmptyIndex());
    }
}

TEST_CASE("deserialize on not empty index", "[ut][hnsw]") {
    logger::set_level(logger::level::debug);

    int64_t dim = 128;
    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 12;
    hnsw_obj.ef_construction = 100;
    hnsw_obj.use_conjugate_graph = true;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();

    const int64_t num_elements = 10;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

    auto dataset = Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(1)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    SECTION("serialize to binaryset") {
        auto binary_set = index->Serialize();
        REQUIRE(binary_set.has_value());

        auto voidresult = index->Deserialize(binary_set.value());
        REQUIRE_FALSE(voidresult.has_value());
        REQUIRE(voidresult.error().type == ErrorType::INDEX_NOT_EMPTY);
        auto another_index = std::make_shared<HNSW>(hnsw_obj, common_param);
        another_index->InitMemorySpace();
        auto deserialize_result = another_index->Deserialize(binary_set.value());
        REQUIRE(deserialize_result.has_value());
    }

    SECTION("serialize to fstream") {
        fixtures::TempDir dir("hnsw_test_deserialize_on_not_empty_index");
        std::fstream out_stream(dir.path + "index.bin", std::ios::out | std::ios::binary);
        auto serialize_result = index->Serialize(out_stream);
        REQUIRE(serialize_result.has_value());
        out_stream.close();

        std::fstream in_stream(dir.path + "index.bin", std::ios::in | std::ios::binary);
        auto voidresult = index->Deserialize(in_stream);
        REQUIRE_FALSE(voidresult.has_value());
        REQUIRE(voidresult.error().type == ErrorType::INDEX_NOT_EMPTY);
        in_stream.close();
    }
}

TEST_CASE("static hnsw", "[ut][hnsw]") {
    logger::set_level(logger::level::debug);

    int64_t dim = 128;
    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 12;
    hnsw_obj.ef_construction = 100;
    hnsw_obj.use_static = true;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();

    const int64_t num_elements = 10;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

    auto dataset = Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(9)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    auto one_vector = Dataset::Make();
    one_vector->Dim(dim)
        ->NumElements(1)
        ->Ids(ids.data() + 9)
        ->Float32Vectors(vectors.data() + 9 * dim)
        ->Owner(false);
    result = index->Add(one_vector);
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().type == ErrorType::UNSUPPORTED_INDEX_OPERATION);

    JsonType params;
    params["hnsw"]["ef_search"].SetInt(100);

    auto knn_result = index->KnnSearch(one_vector, 1, params.Dump());
    REQUIRE(knn_result.has_value());

    auto range_result = index->RangeSearch(one_vector, 1, params.Dump());
    REQUIRE_FALSE(range_result.has_value());
    REQUIRE(range_result.error().type == ErrorType::UNSUPPORTED_INDEX_OPERATION);

    SECTION("incorrect dim") {
        IndexCommonParam incorrect_common_param;
        incorrect_common_param.dim_ = 127;
        incorrect_common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
        incorrect_common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
        HnswParameters incorrect_hnsw_obj = parse_hnsw_params(incorrect_common_param);
        incorrect_hnsw_obj.use_static = true;
        incorrect_hnsw_obj.max_degree = 12;
        incorrect_hnsw_obj.ef_construction = 100;
        REQUIRE_THROWS(std::make_shared<HNSW>(incorrect_hnsw_obj, incorrect_common_param));
    }

    auto remove_result = index->Index::Remove(ids[0]);
    REQUIRE_FALSE(remove_result.has_value());
    REQUIRE(remove_result.error().type == ErrorType::UNSUPPORTED_INDEX_OPERATION);
}

TEST_CASE("hnsw add vector with duplicated id", "[ut][hnsw]") {
    logger::set_level(logger::level::debug);
    int64_t dim = 128;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 12;
    hnsw_obj.ef_construction = 100;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();

    std::vector<int64_t> ids{1};
    std::vector<float> vectors(dim);

    auto first_time = Dataset::Make();
    first_time->Dim(dim)
        ->NumElements(1)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto result = index->Add(first_time);
    REQUIRE(result.has_value());
    // expect failed id list empty
    REQUIRE(result.value().empty());

    auto second_time = Dataset::Make();
    second_time->Dim(dim)
        ->NumElements(1)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto result2 = index->Add(second_time);
    REQUIRE(result2.has_value());
    // expected failed id list == {1}
    REQUIRE(result2.value().size() == 1);
    REQUIRE(result2.value()[0] == ids[0]);
}

TEST_CASE("build with reversed edges", "[ut][hnsw]") {
    logger::set_level(logger::level::debug);
    int64_t dim = 128;
    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 12;
    hnsw_obj.ef_construction = 100;
    hnsw_obj.use_reversed_edges = true;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();

    const int64_t num_elements = 1000;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

    auto dataset = Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(num_elements)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    REQUIRE(index->CheckGraphIntegrity());

    {
        fixtures::TempDir dir("test_index_serialize_via_stream");

        // serialize to file stream
        std::fstream out_file(dir.path + "index.bin", std::ios::out | std::ios::binary);
        REQUIRE(index->Serialize(out_file).has_value());
        out_file.close();

        // deserialize from file stream
        std::fstream in_file(dir.path + "index.bin", std::ios::in | std::ios::binary);
        in_file.seekg(0, std::ios::end);
        int64_t length = in_file.tellg();
        in_file.seekg(0, std::ios::beg);
        auto new_index = std::make_shared<HNSW>(hnsw_obj, common_param);
        new_index->InitMemorySpace();
        REQUIRE(new_index->Deserialize(in_file).has_value());
        REQUIRE(new_index->CheckGraphIntegrity());
    }

    // Serialize(multi-file)
    {
        fixtures::TempDir dir("test_index_serialize_via_stream");

        if (auto bs = index->Serialize(); bs.has_value()) {
            auto keys = bs->GetKeys();
            for (auto key : keys) {
                Binary b = bs->Get(key);
                std::ofstream file(dir.path + "hnsw.index." + key, std::ios::binary);
                file.write((const char*)b.data.get(), b.size);
                file.close();
            }
            std::ofstream metafile(dir.path + "hnsw.index._metafile", std::ios::out);
            for (auto key : keys) {
                metafile << key << std::endl;
            }
            metafile.close();
        } else if (bs.error().type == ErrorType::NO_ENOUGH_MEMORY) {
            std::cerr << "no enough memory to serialize index" << std::endl;
        }

        std::ifstream metafile(dir.path + "hnsw.index._metafile", std::ios::in);
        std::vector<std::string> keys;
        std::string line;
        while (std::getline(metafile, line)) {
            keys.push_back(line);
        }
        metafile.close();

        BinarySet bs;
        for (auto key : keys) {
            std::ifstream file(dir.path + "hnsw.index." + key, std::ios::in);
            file.seekg(0, std::ios::end);
            Binary b;
            b.size = file.tellg();
            b.data.reset(new int8_t[b.size]);
            file.seekg(0, std::ios::beg);
            file.read((char*)b.data.get(), b.size);
            bs.Set(key, b);
        }

        auto new_index = std::make_shared<HNSW>(hnsw_obj, common_param);
        new_index->InitMemorySpace();
        REQUIRE(new_index->Deserialize(bs).has_value());
        REQUIRE(new_index->CheckGraphIntegrity());
    }
}

TEST_CASE("feedback with invalid argument", "[ut][hnsw]") {
    Options::Instance().logger()->SetLevel(Logger::Level::kDEBUG);
    // parameters
    int64_t num_vectors = 1000;
    int64_t k = 10;
    int64_t dim = 128;
    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 16;
    hnsw_obj.ef_construction = 200;
    hnsw_obj.use_conjugate_graph = true;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();

    JsonType search_parameters;
    search_parameters["hnsw"]["ef_search"].SetInt(200);

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_vectors, dim);
    auto query = Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);

    SECTION("index feedback with k = 0") {
        REQUIRE(index->Feedback(query, 0, search_parameters.Dump(), -1).error().type ==
                ErrorType::INVALID_ARGUMENT);
        REQUIRE(index->Feedback(query, 0, search_parameters.Dump()).error().type ==
                ErrorType::INVALID_ARGUMENT);
    }

    SECTION("index feedback with invalid global optimum tag id") {
        auto feedback_result = index->Feedback(query, k, search_parameters.Dump(), -1000);
        REQUIRE(feedback_result.error().type == ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("redundant feedback and empty enhancement", "[ut][hnsw]") {
    Options::Instance().logger()->SetLevel(Logger::Level::kDEBUG);

    // parameters
    int64_t num_base = 10;
    int64_t num_query = 1;
    int64_t k = 10;
    int64_t dim = 128;

    IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 16;
    hnsw_obj.ef_construction = 200;
    hnsw_obj.use_conjugate_graph = true;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();

    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(num_base, dim);
    auto base = Dataset::Make();
    base->NumElements(num_base)
        ->Dim(dim)
        ->Ids(base_ids.data())
        ->Float32Vectors(base_vectors.data())
        ->Owner(false);
    // build index
    auto buildindex = index->Build(base);
    REQUIRE(buildindex.has_value());

    JsonType search_parameters;
    search_parameters["hnsw"]["ef_search"].SetInt(200);
    search_parameters["hnsw"]["use_conjugate_graph"].SetBool(true);

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_query, dim);
    auto query = Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);

    auto search_result = index->KnnSearch(query, k, search_parameters.Dump());
    REQUIRE(search_result.has_value());

    SECTION("index redundant feedback") {
        auto feedback_result =
            index->Feedback(query, k, search_parameters.Dump(), search_result.value()->GetIds()[0]);
        REQUIRE(*feedback_result == k - 1);

        auto redundant_feedback_result =
            index->Feedback(query, k, search_parameters.Dump(), search_result.value()->GetIds()[0]);
        REQUIRE(*redundant_feedback_result == 0);
    }

    SECTION("index search with empty enhancement") {
        auto enhanced_search_result = index->KnnSearch(query, k, search_parameters.Dump());
        REQUIRE(enhanced_search_result.has_value());
        for (int i = 0; i < search_result.value()->GetNumElements(); i++) {
            REQUIRE(search_result.value()->GetIds()[i] ==
                    enhanced_search_result.value()->GetIds()[i]);
        }
    }
}

TEST_CASE("feedback and pretrain without use conjugate graph", "[ut][hnsw]") {
    Options::Instance().logger()->SetLevel(Logger::Level::kDEBUG);

    // parameters
    int64_t num_base = 10;
    int64_t num_query = 1;
    int64_t k = 10;
    int64_t dim = 128;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 16;
    hnsw_obj.ef_construction = 200;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();
    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(num_base, dim);
    auto base = Dataset::Make();
    base->NumElements(num_base)
        ->Dim(dim)
        ->Ids(base_ids.data())
        ->Float32Vectors(base_vectors.data())
        ->Owner(false);
    // build index
    auto buildindex = index->Build(base);
    REQUIRE(buildindex.has_value());

    JsonType search_parameters;
    search_parameters["hnsw"]["ef_search"].SetInt(200);

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_query, dim);
    auto query = Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);

    auto feedback_result = index->Feedback(query, k, search_parameters.Dump());
    REQUIRE(feedback_result.error().type == ErrorType::UNSUPPORTED_INDEX_OPERATION);

    std::vector<int64_t> base_tag_ids;
    base_tag_ids.push_back(10000);
    auto pretrain_result = index->Pretrain(base_tag_ids, 10, search_parameters.Dump());
    REQUIRE(pretrain_result.error().type == ErrorType::UNSUPPORTED_INDEX_OPERATION);
}

TEST_CASE("feedback and pretrain on empty index", "[ut][hnsw]") {
    Options::Instance().logger()->SetLevel(Logger::Level::kDEBUG);

    // parameters
    int64_t dim = 128;
    int64_t num_base = 0;
    int64_t num_query = 1;
    int64_t k = 100;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 16;
    hnsw_obj.ef_construction = 200;
    hnsw_obj.use_conjugate_graph = true;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();

    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(num_base, dim);
    auto base = Dataset::Make();
    base->NumElements(num_base)
        ->Dim(dim)
        ->Ids(base_ids.data())
        ->Float32Vectors(base_vectors.data())
        ->Owner(false);
    // build index
    auto buildindex = index->Build(base);
    REQUIRE(buildindex.has_value());

    JsonType search_parameters;
    search_parameters["hnsw"]["ef_search"].SetInt(200);

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_query, dim);
    auto query = Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);

    auto feedback_result = index->Feedback(query, k, search_parameters.Dump());
    REQUIRE(*feedback_result == 0);

    std::vector<int64_t> base_tag_ids;
    base_tag_ids.push_back(10000);
    auto pretrain_result = index->Pretrain(base_tag_ids, k, search_parameters.Dump());
    REQUIRE(*pretrain_result == 0);
}

TEST_CASE("invalid pretrain", "[ut][hnsw]") {
    Options::Instance().logger()->SetLevel(Logger::Level::kDEBUG);

    // parameters
    int64_t num_base = 10;
    int64_t num_query = 1;
    int64_t k = 100;
    int64_t dim = 128;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 16;
    hnsw_obj.ef_construction = 200;
    hnsw_obj.use_conjugate_graph = true;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();

    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(num_base, dim);
    auto base = Dataset::Make();
    base->NumElements(num_base)
        ->Dim(dim)
        ->Ids(base_ids.data())
        ->Float32Vectors(base_vectors.data())
        ->Owner(false);
    // build index
    auto buildindex = index->Build(base);
    REQUIRE(buildindex.has_value());

    JsonType search_parameters;
    search_parameters["hnsw"]["ef_search"].SetInt(200);

    SECTION("invalid base tag id") {
        std::vector<int64_t> base_tag_ids;
        base_tag_ids.push_back(10000);
        auto pretrain_result = index->Pretrain(base_tag_ids, 10, search_parameters.Dump());
        REQUIRE(pretrain_result.error().type == ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid k") {
        std::vector<int64_t> base_tag_ids;
        base_tag_ids.push_back(0);
        auto pretrain_result = index->Pretrain(base_tag_ids, 0, search_parameters.Dump());
        REQUIRE(pretrain_result.error().type == ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid search parameter") {
        JsonType invalid_search_parameters;
        invalid_search_parameters["hnsw"]["ef_search"].SetInt(-1);

        std::vector<int64_t> base_tag_ids;
        base_tag_ids.push_back(0);
        auto pretrain_result = index->Pretrain(base_tag_ids, 10, invalid_search_parameters.Dump());
        REQUIRE(pretrain_result.error().type == ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("get distance by label", "[ut][hnsw]") {
    Options::Instance().logger()->SetLevel(Logger::Level::kDEBUG);

    // parameters
    int dim = 128;
    int64_t num_base = 1;

    // data
    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(num_base, dim);

    // hnsw index
    hnswlib::L2Space space(dim);

    SECTION("hnsw test") {
        DefaultAllocator allocator;
        auto* alg_hnsw = new hnswlib::HierarchicalNSW(&space, 100, &allocator);
        alg_hnsw->init_memory_space();
        alg_hnsw->addPoint(base_vectors.data(), 0);
        fixtures::dist_t distance = alg_hnsw->getDistanceByLabel(0, base_vectors.data());
        REQUIRE(distance == 0);
        REQUIRE_THROWS(alg_hnsw->getDistanceByLabel(-1, base_vectors.data()));
        delete alg_hnsw;
    }

    SECTION("static hnsw test") {
        DefaultAllocator allocator;
        auto* alg_hnsw_static = new hnswlib::StaticHierarchicalNSW(&space, 100, &allocator);
        alg_hnsw_static->init_memory_space();
        alg_hnsw_static->addPoint(base_vectors.data(), 0);
        fixtures::dist_t distance = alg_hnsw_static->getDistanceByLabel(0, base_vectors.data());
        REQUIRE(distance == 0);
        REQUIRE_THROWS(alg_hnsw_static->getDistanceByLabel(-1, base_vectors.data()));
        delete alg_hnsw_static;
    }
}

TEST_CASE("get min and max id", "[ut][hnsw]") {
    Options::Instance().logger()->SetLevel(Logger::Level::kDEBUG);

    // parameters
    int dim = 128;
    int64_t num_base = 1;

    // data
    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(num_base, dim);

    // hnsw index
    hnswlib::L2Space space(dim);

    SECTION("hnsw test") {
        DefaultAllocator allocator;
        auto* alg_hnsw = new hnswlib::HierarchicalNSW(&space, 100, &allocator);
        alg_hnsw->init_memory_space();
        alg_hnsw->addPoint(base_vectors.data(), 0);
        alg_hnsw->addPoint(base_vectors.data(), 5);
        auto get_min_max_res = alg_hnsw->getMinAndMaxId();
        int64_t min_id = get_min_max_res.first;
        int64_t max_id = get_min_max_res.second;

        REQUIRE(min_id == 0);
        REQUIRE(max_id == 5);
        delete alg_hnsw;
    }

    SECTION("static hnsw test") {
        DefaultAllocator allocator;
        auto* alg_hnsw_static = new hnswlib::StaticHierarchicalNSW(&space, 100, &allocator);
        alg_hnsw_static->init_memory_space();
        alg_hnsw_static->addPoint(base_vectors.data(), 0);
        alg_hnsw_static->addPoint(base_vectors.data(), 5);
        auto get_min_max_res = alg_hnsw_static->getMinAndMaxId();
        int64_t min_id = get_min_max_res.first;
        int64_t max_id = get_min_max_res.second;

        REQUIRE(min_id == 0);
        REQUIRE(max_id == 5);
        delete alg_hnsw_static;
    }
}

TEST_CASE("get data by label", "[ut][hnsw]") {
    Options::Instance().logger()->SetLevel(Logger::Level::kDEBUG);

    // parameters
    int dim = 128;
    int64_t num_base = 1;

    // data
    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(num_base, dim);

    // hnsw index
    hnswlib::L2Space space(dim);

    SECTION("hnsw test") {
        DefaultAllocator allocator;
        auto* alg_hnsw = new hnswlib::HierarchicalNSW(&space, 100, &allocator);
        std::vector<int8_t> base_data(dim * sizeof(float));
        alg_hnsw->init_memory_space();
        alg_hnsw->addPoint(base_vectors.data(), 0);
        fixtures::dist_t distance = alg_hnsw->getDistanceByLabel(0, alg_hnsw->getDataByLabel(0));

        alg_hnsw->copyDataByLabel(0, base_data.data());
        fixtures::dist_t distance_validate = alg_hnsw->getDistanceByLabel(0, base_data.data());

        REQUIRE(distance == 0);
        REQUIRE(distance == distance_validate);
        REQUIRE_THROWS(alg_hnsw->getDistanceByLabel(-1, base_vectors.data()));
        delete alg_hnsw;
    }

    SECTION("static hnsw test") {
        DefaultAllocator allocator;
        auto* alg_hnsw_static = new hnswlib::StaticHierarchicalNSW(&space, 100, &allocator);
        std::vector<int8_t> base_data(dim * sizeof(float));
        alg_hnsw_static->init_memory_space();
        alg_hnsw_static->addPoint(base_vectors.data(), 0);
        fixtures::dist_t distance =
            alg_hnsw_static->getDistanceByLabel(0, alg_hnsw_static->getDataByLabel(0));

        alg_hnsw_static->copyDataByLabel(0, base_data.data());
        fixtures::dist_t distance_validate =
            alg_hnsw_static->getDistanceByLabel(0, base_data.data());

        REQUIRE(distance == 0);
        REQUIRE(distance == distance_validate);
        REQUIRE_THROWS(alg_hnsw_static->getDistanceByLabel(-1, base_vectors.data()));
        delete alg_hnsw_static;
    }
}

TEST_CASE("extract/set data and graph", "[ut][hnsw]") {
    logger::set_level(logger::level::debug);

    int64_t dim = 128;
    IndexCommonParam common_param;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    common_param.dim_ = dim;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = allocator;

    HnswParameters hnsw_obj = parse_hnsw_params(common_param);
    hnsw_obj.max_degree = 12;
    hnsw_obj.ef_construction = 100;
    auto index = std::make_shared<HNSW>(hnsw_obj, common_param);
    index->InitMemorySpace();

    const int64_t num_elements = 2000;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

    auto dataset = Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(num_elements / 2)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    auto param = std::make_shared<FlattenDataCellParameter>();
    param->io_parameter = std::make_shared<vsag::MemoryIOParameter>();
    param->quantizer_parameter = std::make_shared<vsag::FP32QuantizerParameter>();
    vsag::GraphDataCellParamPtr graph_param_ptr = std::make_shared<vsag::GraphDataCellParameter>();
    graph_param_ptr->io_parameter_ = std::make_shared<vsag::MemoryIOParameter>();

    FlattenInterfacePtr flatten_interface = FlattenInterface::MakeInstance(param, common_param);
    GraphInterfacePtr graph_interface = GraphInterface::MakeInstance(graph_param_ptr, common_param);
    Vector<LabelType> ids_vector(allocator.get());

    IdMapFunction id_map = [](int64_t id) -> std::tuple<bool, int64_t> {
        return std::make_tuple(true, id);
    };
    REQUIRE(index->ExtractDataAndGraph(
        flatten_interface, graph_interface, ids_vector, id_map, allocator.get()));

    auto another_index = std::make_shared<HNSW>(hnsw_obj, common_param);
    another_index->InitMemorySpace();
    REQUIRE(another_index->SetDataAndGraph(flatten_interface, graph_interface, ids_vector));

    dataset->Dim(dim)
        ->NumElements(num_elements / 2)
        ->Ids(ids.data() + num_elements / 2)
        ->Float32Vectors(vectors.data() + num_elements / 2 * dim)
        ->Owner(false);
    another_index->Add(dataset, AddMode::DEFAULT);

    JsonType search_parameters;

    search_parameters["hnsw"]["ef_search"].SetInt(200);
    int correct = 0;
    for (int i = 0; i < num_elements; ++i) {
        auto query = Dataset::Make();
        query->Dim(dim)->Float32Vectors(vectors.data() + i * dim)->NumElements(1)->Owner(false);
        auto query_result = another_index->KnnSearch(query, 10, search_parameters.Dump());
        REQUIRE(query_result.has_value());
        correct += query_result.value()->GetIds()[0] == ids[i] ? 1 : 0;
    }
    float recall = correct / (float)num_elements;
    REQUIRE(recall > 0.99);
}

TEST_CASE("update mark-deleted vector", "[ut][hnsw]") {
    logger::set_level(logger::level::debug);

    // parameters
    int dim = 128;
    int base_size = 100;
    int delete_size = 50;
    int update_size = 50;

    // create hnsw
    hnswlib::L2Space space(dim);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto* alg_hnsw = new hnswlib::HierarchicalNSW(&space, 100, allocator.get());
    alg_hnsw->init_memory_space();

    // data and build index
    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(base_size, dim);
    for (auto i = 0; i < base_size; i++) {
        alg_hnsw->addPoint(base_vectors.data() + i * dim, base_ids[i]);
    }

    // start remove
    for (auto i = 0; i < delete_size; i++) {
        REQUIRE(alg_hnsw->getCurrentElementCount() == base_size);
        REQUIRE(alg_hnsw->getDeletedCount() == i);
        REQUIRE(alg_hnsw->getDeletedElements().size() == i);

        alg_hnsw->markDelete(base_ids[i]);

        REQUIRE(alg_hnsw->getDeletedElements().count(base_ids[i]) != 0);
    }

    // update
    for (auto i = 0; i < update_size; i++) {
        auto old_label = base_ids[i] + delete_size / 2;
        auto new_label = old_label + base_size;
        bool is_deleted = alg_hnsw->isMarkedDeleted(old_label);
        alg_hnsw->updateLabel(old_label, new_label);
        if (is_deleted) {
            REQUIRE(alg_hnsw->getDeletedElements().count(old_label) == 0);
            REQUIRE(alg_hnsw->getDeletedElements().count(new_label) != 0);
            REQUIRE(alg_hnsw->getDeletedElements()[new_label] == old_label);
        } else {
            REQUIRE(not alg_hnsw->isValidLabel(old_label));
            REQUIRE(alg_hnsw->isValidLabel(new_label));
        }
    }

    delete alg_hnsw;
}
