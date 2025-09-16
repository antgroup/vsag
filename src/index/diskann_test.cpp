
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

#include "diskann.h"

#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <tuple>

#include "diskann_zparameters.h"
#include "distance.h"
#include "fixtures.h"
#include "index_common_param.h"
#include "utils/timer.h"
#include "vsag/errors.h"

vsag::DiskannParameters
parse_diskann_params(vsag::IndexCommonParam index_common_param) {
    auto build_parameter_json = R"(
        {
            "max_degree": 16,
            "ef_construction": 100,
            "pq_dims": 32,
            "pq_sample_rate": 1.0
        }
    )";
    auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
    return vsag::DiskannParameters::FromJson(parsed_params, index_common_param);
}

TEST_CASE("diskann build", "[ut][diskann]") {
    vsag::logger::set_level(vsag::logger::level::debug);
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    vsag::DiskannParameters diskann_obj = parse_diskann_params(common_param);
    diskann_obj.metric = diskann::Metric::L2;
    diskann_obj.pq_sample_rate = 1.0f;
    diskann_obj.pq_dims = 16;
    diskann_obj.max_degree = 12;
    diskann_obj.ef_construction = 100;
    diskann_obj.use_bsa = false;
    diskann_obj.use_reference = false;
    diskann_obj.use_preload = false;

    auto index = std::make_shared<vsag::DiskANN>(diskann_obj, common_param);

    int64_t num_elements = 10;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, common_param.dim_);

    SECTION("build with incorrect dim") {
        int64_t incorrect_dim = common_param.dim_ - 1;
        auto dataset = vsag::Dataset::Make();
        dataset->Dim(incorrect_dim)
            ->NumElements(num_elements)
            ->Ids(ids.data())
            ->Float32Vectors(vectors.data())
            ->Owner(false);
        auto result = index->Build(dataset);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("build twice") {
        auto dataset = vsag::Dataset::Make();
        dataset->Dim(common_param.dim_)
            ->NumElements(10)
            ->Ids(ids.data())
            ->Float32Vectors(vectors.data())
            ->Owner(false);
        REQUIRE(index->Build(dataset).has_value());

        auto result = index->Build(dataset);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::BUILD_TWICE);
    }
}

TEST_CASE("build & search empty index for diskann", "[ut][diskann]") {
    vsag::logger::set_level(vsag::logger::level::debug);
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    vsag::DiskannParameters diskann_obj = parse_diskann_params(common_param);
    diskann_obj.metric = diskann::Metric::L2;
    diskann_obj.pq_sample_rate = 1.0f;
    diskann_obj.pq_dims = 16;
    diskann_obj.max_degree = 12;
    diskann_obj.ef_construction = 100;
    diskann_obj.use_bsa = false;
    diskann_obj.use_reference = false;
    diskann_obj.use_preload = false;

    auto index = std::make_shared<vsag::DiskANN>(diskann_obj, common_param);

    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(0);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(1, common_param.dim_);
    auto one_vector = vsag::Dataset::Make();
    one_vector->NumElements(1)
        ->Dim(common_param.dim_)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto search_parameters = R"(
    {
        "diskann": {
            "ef_search": 100,
            "beam_search": 4,
            "io_limit": 100,
            "use_reorder": false
        }
    }
    )";

    auto knnsearch = index->KnnSearch(one_vector, 10, search_parameters);
    REQUIRE(knnsearch.has_value());
    REQUIRE(knnsearch.value()->GetNumElements() == 1);
    REQUIRE(knnsearch.value()->GetDim() == 0);

    auto rangesearch = index->RangeSearch(one_vector, 10, search_parameters);
    REQUIRE(rangesearch.has_value());
    REQUIRE(rangesearch.value()->GetNumElements() == 1);
    REQUIRE(rangesearch.value()->GetDim() == 0);
}

TEST_CASE("build diskann index with one vector", "[ut][diskann]") {
    vsag::logger::set_level(vsag::logger::level::debug);
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    vsag::DiskannParameters diskann_obj = parse_diskann_params(common_param);
    diskann_obj.metric = diskann::Metric::L2;
    diskann_obj.pq_sample_rate = 1.0f;
    diskann_obj.pq_dims = 16;
    diskann_obj.max_degree = 12;
    diskann_obj.ef_construction = 100;
    diskann_obj.use_bsa = false;
    diskann_obj.use_reference = false;
    diskann_obj.use_preload = false;

    auto index = std::make_shared<vsag::DiskANN>(diskann_obj, common_param);
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(1, common_param.dim_);
    auto one_vector = vsag::Dataset::Make();
    one_vector->NumElements(1)
        ->Dim(common_param.dim_)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto result = index->Build(one_vector);
    REQUIRE(result.has_value());
    auto search_parameters = R"(
    {
        "diskann": {
            "ef_search": 100,
            "beam_search": 4,
            "io_limit": 100,
            "use_reorder": false
        }
    }
    )";
    auto knnsearch = index->KnnSearch(one_vector, 10, search_parameters);
    REQUIRE(knnsearch.has_value());
    REQUIRE(knnsearch.value()->GetDim() == 1);
    REQUIRE(knnsearch.value()->GetIds()[0] == ids[0]);
}

TEST_CASE("diskann knn_search", "[ut][diskann]") {
    vsag::logger::set_level(vsag::logger::level::debug);
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    vsag::DiskannParameters diskann_obj = parse_diskann_params(common_param);
    diskann_obj.metric = diskann::Metric::L2;
    diskann_obj.pq_sample_rate = 1.0f;
    diskann_obj.pq_dims = 16;
    diskann_obj.max_degree = 12;
    diskann_obj.ef_construction = 100;
    diskann_obj.use_bsa = false;
    diskann_obj.use_reference = false;
    diskann_obj.use_preload = false;

    auto index = std::make_shared<vsag::DiskANN>(diskann_obj, common_param);

    int64_t num_elements = 100;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, common_param.dim_);

    auto dataset = vsag::Dataset::Make();
    dataset->Dim(common_param.dim_)
        ->NumElements(num_elements)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto build_result = index->Build(dataset);
    REQUIRE(build_result.has_value());

    auto query = vsag::Dataset::Make();
    query->Dim(common_param.dim_)
        ->NumElements(1)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    int64_t k = 10;
    vsag::JsonType params;
    params["diskann"]["ef_search"].SetInt(100);
    params["diskann"]["beam_search"].SetInt(4);
    params["diskann"]["io_limit"].SetInt(200);

    SECTION("index empty") {
        auto empty_index = std::make_shared<vsag::DiskANN>(diskann_obj, common_param);
        auto result = empty_index->KnnSearch(query, k, params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INDEX_EMPTY);
    }

    SECTION("invalid parameters k is 0") {
        auto result = index->KnnSearch(query, 0, params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters k less than 0") {
        auto result = index->KnnSearch(query, -1, params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("dimension not equal") {
        auto query2 = vsag::Dataset::Make();
        query2->NumElements(1)
            ->Dim(common_param.dim_ - 1)
            ->Float32Vectors(vectors.data())
            ->Owner(false);
        auto result = index->KnnSearch(query2, k, params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters diskann not found") {
        vsag::JsonType invalid_params{};
        auto result = index->KnnSearch(query, k, invalid_params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters beam_search not found") {
        vsag::JsonType invalid_params;
        invalid_params["diskann"]["ef_search"].SetInt(100);
        invalid_params["diskann"]["io_limit"].SetInt(200);

        auto result = index->KnnSearch(query, k, invalid_params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters io_limit not found") {
        vsag::JsonType invalid_params;
        invalid_params["diskann"]["ef_search"].SetInt(100);
        invalid_params["diskann"]["beam_search"].SetInt(4);

        auto result = index->KnnSearch(query, k, invalid_params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters ef_search not found") {
        vsag::JsonType invalid_params;
        invalid_params["diskann"]["beam_search"].SetInt(4);
        invalid_params["diskann"]["io_limit"].SetInt(200);

        auto result = index->KnnSearch(query, k, invalid_params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("range_search", "[ut][diskann]") {
    vsag::logger::set_level(vsag::logger::level::debug);
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    vsag::DiskannParameters diskann_obj = parse_diskann_params(common_param);
    diskann_obj.metric = diskann::Metric::L2;
    diskann_obj.pq_sample_rate = 1.0f;
    diskann_obj.pq_dims = 16;
    diskann_obj.max_degree = 12;
    diskann_obj.ef_construction = 100;
    diskann_obj.use_bsa = false;
    diskann_obj.use_reference = false;
    diskann_obj.use_preload = false;

    auto index = std::make_shared<vsag::DiskANN>(diskann_obj, common_param);

    int64_t num_elements = 100;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, common_param.dim_);

    auto dataset = vsag::Dataset::Make();
    dataset->Dim(common_param.dim_)
        ->NumElements(num_elements)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto build_result = index->Build(dataset);
    REQUIRE(build_result.has_value());

    auto query = vsag::Dataset::Make();
    query->Dim(common_param.dim_)
        ->NumElements(1)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    float radius = 9.9f;
    vsag::JsonType params;
    params["diskann"]["ef_search"].SetInt(100);
    params["diskann"]["beam_search"].SetInt(4);
    params["diskann"]["io_limit"].SetInt(200);

    SECTION("successful case with smaller range_search_limit") {
        int64_t range_search_limit = num_elements - 1;
        auto result = index->RangeSearch(query, 1000, params.Dump(), range_search_limit);
        REQUIRE(result.has_value());
        REQUIRE((*result)->GetDim() == range_search_limit);
    }

    SECTION("successful case with larger range_search_limit") {
        int64_t range_search_limit = num_elements + 1;
        auto result = index->RangeSearch(query, 1000, params.Dump(), range_search_limit);
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
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("index empty") {
        auto empty_index = std::make_shared<vsag::DiskANN>(diskann_obj, common_param);
        auto result = empty_index->RangeSearch(query, radius, params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INDEX_EMPTY);
    }

    SECTION("invalid parameter radius equals to 0") {
        auto query2 = vsag::Dataset::Make();
        query2->NumElements(1)
            ->Dim(common_param.dim_)
            ->Float32Vectors(vectors.data())
            ->Owner(false);
        auto result = index->RangeSearch(query2, 0, params.Dump());
        REQUIRE(result.has_value());
    }

    SECTION("invalid parameter radius less than 0") {
        auto query2 = vsag::Dataset::Make();
        query2->NumElements(1)
            ->Dim(common_param.dim_)
            ->Float32Vectors(vectors.data())
            ->Owner(false);
        auto result = index->RangeSearch(query2, -1, params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("dimension not equal") {
        auto query2 = vsag::Dataset::Make();
        query2->NumElements(1)
            ->Dim(common_param.dim_ - 1)
            ->Float32Vectors(vectors.data())
            ->Owner(false);
        auto result = index->RangeSearch(query2, radius, params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("query length is not 1") {
        auto query2 = vsag::Dataset::Make();
        query2->NumElements(2)
            ->Dim(common_param.dim_)
            ->Float32Vectors(vectors.data())
            ->Owner(false);
        auto result = index->RangeSearch(query2, radius, params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters diskann not found") {
        vsag::JsonType invalid_params{};
        auto result = index->RangeSearch(query, radius, invalid_params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters beam_search not found") {
        vsag::JsonType invalid_params;
        invalid_params["diskann"]["ef_search"].SetInt(100);
        invalid_params["diskann"]["io_limit"].SetInt(200);

        auto result = index->RangeSearch(query, radius, invalid_params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters io_limit not found") {
        vsag::JsonType invalid_params;
        invalid_params["diskann"]["ef_search"].SetInt(100);
        invalid_params["diskann"]["beam_search"].SetInt(4);

        auto result = index->RangeSearch(query, radius, invalid_params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters ef_search not found") {
        vsag::JsonType invalid_params;
        invalid_params["diskann"]["beam_search"].SetInt(4);
        invalid_params["diskann"]["io_limit"].SetInt(200);

        auto result = index->RangeSearch(query, radius, invalid_params.Dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("serialize empty index", "[ut][diskann]") {
    vsag::logger::set_level(vsag::logger::level::debug);
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    vsag::DiskannParameters diskann_obj = parse_diskann_params(common_param);
    diskann_obj.metric = diskann::Metric::L2;
    diskann_obj.pq_sample_rate = 1.0f;
    diskann_obj.pq_dims = 16;
    diskann_obj.max_degree = 12;
    diskann_obj.ef_construction = 100;
    diskann_obj.use_bsa = false;
    diskann_obj.use_reference = false;
    diskann_obj.use_preload = false;

    auto index = std::make_shared<vsag::DiskANN>(diskann_obj, common_param);

    auto result = index->Serialize();
    REQUIRE(result.has_value());
}

TEST_CASE("deserialize on not empty index", "[ut][diskann]") {
    vsag::logger::set_level(vsag::logger::level::debug);
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    vsag::DiskannParameters diskann_obj = parse_diskann_params(common_param);
    diskann_obj.metric = diskann::Metric::L2;
    diskann_obj.pq_sample_rate = 1.0f;
    diskann_obj.pq_dims = 16;
    diskann_obj.max_degree = 12;
    diskann_obj.ef_construction = 100;
    diskann_obj.use_bsa = false;
    diskann_obj.use_reference = false;
    diskann_obj.use_preload = false;

    auto index = std::make_shared<vsag::DiskANN>(diskann_obj, common_param);

    int64_t num_elements = 100;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, common_param.dim_);

    auto dataset = vsag::Dataset::Make();
    dataset->Dim(common_param.dim_)
        ->NumElements(num_elements)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    auto binary_set = index->Serialize();
    REQUIRE(binary_set.has_value());

    auto voidresult = index->Deserialize(binary_set.value());
    REQUIRE_FALSE(voidresult.has_value());
    REQUIRE(voidresult.error().type == vsag::ErrorType::INDEX_NOT_EMPTY);
}

TEST_CASE("split building process", "[ut][diskann]") {
    vsag::logger::set_level(vsag::logger::level::debug);
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    vsag::DiskannParameters diskann_obj = parse_diskann_params(common_param);
    diskann_obj.metric = diskann::Metric::L2;
    diskann_obj.pq_sample_rate = 1.0f;
    diskann_obj.pq_dims = 16;
    diskann_obj.max_degree = 12;
    diskann_obj.ef_construction = 100;
    diskann_obj.use_bsa = false;
    diskann_obj.use_reference = false;
    diskann_obj.use_preload = false;

    int64_t num_elements = 1000;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, common_param.dim_);

    auto dataset = vsag::Dataset::Make();
    dataset->Dim(common_param.dim_)
        ->NumElements(num_elements)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    vsag::Index::Checkpoint checkpoint;
    std::shared_ptr<vsag::DiskANN> partial_index;
    double partial_time = 0;
    {
        vsag::Timer timer(partial_time);
        while (not checkpoint.finish) {
            partial_index = std::make_shared<vsag::DiskANN>(diskann_obj, common_param);
            checkpoint = partial_index->ContinueBuild(dataset, checkpoint.data).value();
        }
    }

    vsag::JsonType parameters;
    parameters["diskann"]["ef_search"].SetInt(10);
    parameters["diskann"]["beam_search"].SetInt(4);
    parameters["diskann"]["io_limit"].SetInt(20);
    float correct = 0;
    for (int i = 0; i < num_elements; i++) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(common_param.dim_)
            ->Float32Vectors(vectors.data() + i * common_param.dim_)
            ->Owner(false);
        int64_t k = 2;
        if (auto result = partial_index->KnnSearch(query, k, parameters.Dump());
            result.has_value()) {
            if (result.value()->GetNumElements() == 1) {
                REQUIRE(!std::isinf(result.value()->GetDistances()[0]));
                if (result.value()->GetIds()[0] == i) {
                    correct++;
                }
            }
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to search on index: internalError" << std::endl;
            exit(-1);
        }
    }
    float recall_partial = correct / 1000;

    double full_time = 0;
    {
        vsag::Timer timer(full_time);
        std::shared_ptr<vsag::DiskANN> full_index =
            std::make_shared<vsag::DiskANN>(diskann_obj, common_param);
        full_index->Build(dataset);
    }
    correct = 0;
    for (int i = 0; i < num_elements; i++) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(common_param.dim_)
            ->Float32Vectors(vectors.data() + i * common_param.dim_)
            ->Owner(false);
        int64_t k = 2;
        if (auto result = partial_index->KnnSearch(query, k, parameters.Dump());
            result.has_value()) {
            if (result.value()->GetNumElements() == 1) {
                REQUIRE(!std::isinf(result.value()->GetDistances()[0]));
                if (result.value()->GetIds()[0] == i) {
                    correct++;
                }
            }
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to search on index: internalError" << std::endl;
            exit(-1);
        }
    }
    float recall_full = correct / 1000;
    vsag::logger::debug("Recall: " + std::to_string(recall_full));
    REQUIRE(recall_full == recall_partial);
}

TEST_CASE("diskann serialize in streaming", "[ut][diskann]") {
    vsag::logger::set_level(vsag::logger::level::debug);
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    vsag::DiskannParameters diskann_obj = parse_diskann_params(common_param);
    diskann_obj.metric = diskann::Metric::L2;
    diskann_obj.pq_sample_rate = 1.0F;
    diskann_obj.pq_dims = 16;
    diskann_obj.max_degree = 12;
    diskann_obj.ef_construction = 100;
    diskann_obj.use_bsa = false;
    diskann_obj.use_reference = false;
    diskann_obj.use_preload = false;

    auto index = std::make_shared<vsag::DiskANN>(diskann_obj, common_param);

    int64_t num_elements = 100;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, common_param.dim_);

    auto dataset = vsag::Dataset::Make();
    dataset->Dim(common_param.dim_)
        ->NumElements(num_elements)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    fixtures::TempDir dir("diskann_serialize_in_streaming");
    auto filename = dir.GenerateRandomFile();

    std::ofstream out_file(filename);
    auto serialize_result = index->Serialize(out_file);
    REQUIRE(serialize_result.has_value());
    out_file.close();

    index = nullptr;

    index = std::make_shared<vsag::DiskANN>(diskann_obj, common_param);
    std::ifstream in_file(filename);
    auto deserialize_result = index->Deserialize(in_file);
    REQUIRE(deserialize_result.has_value());
    in_file.close();
}
