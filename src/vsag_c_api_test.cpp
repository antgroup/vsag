
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

#include <sys/stat.h>
#include <vsag/vsag_c_api.h>

#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <random>

#include "fixtures.h"
#include "simd/fp32_simd.h"

static constexpr const char* index_param = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 64,
            "ef_construction": 200,
            "alpha":1.2
        }
    }
)";

static constexpr const char* index_name = "hgraph";

static constexpr const char* hgraph_search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 200
        }
    }
)";

static constexpr int64_t num_vectors = 500;
static constexpr int64_t dim = 128;

class VsagTestCase {
public:
    std::vector<int64_t> ids{};
    std::vector<float> datas{};

    VsagTestCase() {
        std::mt19937 rng(47);
        std::uniform_real_distribution<float> distrib_real;
        ids.resize(num_vectors);
        datas.resize(num_vectors * dim);
        for (int64_t i = 0; i < num_vectors; ++i) {
            ids[i] = i;
        }
        for (int64_t i = 0; i < dim * num_vectors; ++i) {
            datas[i] = distrib_real(rng);
        }
    }

    void
    CheckIndex(vsag_index_t index) {
        int64_t topk = 10;
        std::vector<int64_t> results(topk);
        std::vector<float> scores(topk);
        SearchResult_t result;
        result.dists = scores.data();
        result.ids = results.data();
        for (int i = 0; i < num_vectors; ++i) {
            auto ret = vsag_index_knn_search(
                index, datas.data() + i * dim, dim, topk, hgraph_search_parameters, &result);
            REQUIRE(ret.code == VSAG_SUCCESS);
            bool in_results = false;
            for (int64_t j = 0; j < result.count; ++j) {
                if (result.ids[j] == i) {
                    in_results = true;
                    break;
                }
            }
            REQUIRE(in_results);
        }
    }
};

TEST_CASE("vsag_c_api basic test", "[vsag_c_api][ut]") {
    auto index = vsag_index_factory(index_name, index_param);
    REQUIRE(index != nullptr);
    VsagTestCase test_case;
    Error_t ret =
        vsag_index_build(index, test_case.datas.data(), test_case.ids.data(), dim, num_vectors);
    REQUIRE(ret.code == VSAG_SUCCESS);
    test_case.CheckIndex(index);
    // for test serialize and deserialize file
    {
        auto dir = fixtures::TempDir("vsag_c_api");
        auto path = dir.GenerateRandomFile();
        ret = vsag_serialize_file(index, path.data());
        REQUIRE(ret.code == VSAG_SUCCESS);
        vsag_index_t index2 = vsag_index_factory(index_name, index_param);
        ret = vsag_deserialize_file(index2, path.data());
        REQUIRE(ret.code == VSAG_SUCCESS);
        REQUIRE(index2 != nullptr);
        test_case.CheckIndex(index2);
        vsag_index_destroy(index2);
    }

    // for test serialize and deserialize write func and read func
    {
        auto dir = fixtures::TempDir("vsag_c_api");
        thread_local auto path = dir.GenerateRandomFile();
        ret = vsag_serialize_file(index, path.data());
        using WriteFuncType = void (*)(OffsetType offset, SizeType size, const void* data);
        WriteFuncType write_func = [](OffsetType offset, SizeType size, const void* data) {
            std::ofstream ofile(path, std::ios::binary | std::ios::app);
            ofile.seekp(offset);
            ofile.write(reinterpret_cast<const char*>(data), size);
            ofile.close();
        };
        ret = vsag_serialize_write_func(index, write_func);
        REQUIRE(ret.code == VSAG_SUCCESS);

        using SizeFuncType = SizeType (*)();
        SizeFuncType size_func = []() {
            struct stat st;
            stat(path.data(), &st);
            return static_cast<SizeType>(st.st_size);
        };

        using ReadFuncType = void (*)(OffsetType offset, SizeType size, void* data);
        ReadFuncType read_func = [](OffsetType offset, SizeType size, void* data) {
            std::ifstream ifile(path.data(), std::ios::binary);
            ifile.seekg(offset);
            ifile.read(reinterpret_cast<char*>(data), size);
            ifile.close();
        };
        vsag_index_t index2 = vsag_index_factory(index_name, index_param);
        ret = vsag_deserialize_read_func(index2, read_func, size_func);
        REQUIRE(ret.code == VSAG_SUCCESS);
        REQUIRE(index2 != nullptr);
        test_case.CheckIndex(index2);
        vsag_index_destroy(index2);
    }

    vsag_index_destroy(index);
}

TEST_CASE("vsag_c_api factory and destroy", "[vsag_c_api][ut]") {
    // Test factory with valid parameters
    auto index = vsag_index_factory(index_name, index_param);
    REQUIRE(index != nullptr);

    // Test factory with invalid parameters
    auto invalid_index = vsag_index_factory("invalid_index", index_param);
    REQUIRE(invalid_index == nullptr);

    // Test destroy
    Error_t ret = vsag_index_destroy(index);
    REQUIRE(ret.code == VSAG_SUCCESS);
}

TEST_CASE("vsag_c_api build and add", "[vsag_c_api][ut]") {
    auto index = vsag_index_factory(index_name, index_param);
    REQUIRE(index != nullptr);

    // Test build
    VsagTestCase test_case;
    int64_t add_num_vectors = 50;
    Error_t ret = vsag_index_build(
        index, test_case.datas.data(), test_case.ids.data(), dim, num_vectors - add_num_vectors);
    REQUIRE(ret.code == VSAG_SUCCESS);

    ret = vsag_index_add(index,
                         test_case.datas.data() + (num_vectors - add_num_vectors) * dim,
                         test_case.ids.data() + (num_vectors - add_num_vectors),
                         dim,
                         add_num_vectors);
    REQUIRE(ret.code == VSAG_SUCCESS);

    test_case.CheckIndex(index);

    vsag_index_destroy(index);
}

TEST_CASE("vsag_c_api train", "[vsag_c_api][ut]") {
    auto index = vsag_index_factory(index_name, index_param);
    REQUIRE(index != nullptr);

    VsagTestCase test_case;
    // Test train
    Error_t ret =
        vsag_index_train(index, test_case.datas.data(), test_case.ids.data(), dim, num_vectors);
    REQUIRE(ret.code == VSAG_SUCCESS);

    ret = vsag_index_add(index, test_case.datas.data(), test_case.ids.data(), dim, num_vectors);
    REQUIRE(ret.code == VSAG_SUCCESS);

    test_case.CheckIndex(index);

    vsag_index_destroy(index);
}

TEST_CASE("vsag_c_api knn search(with filter)", "[vsag_c_api][ut]") {
    auto index = vsag_index_factory(index_name, index_param);
    REQUIRE(index != nullptr);

    VsagTestCase test_case;

    Error_t ret =
        vsag_index_build(index, test_case.datas.data(), test_case.ids.data(), dim, num_vectors);
    REQUIRE(ret.code == VSAG_SUCCESS);

    // Test knn search

    int64_t topk = 10;
    std::vector<int64_t> results(topk);
    std::vector<float> scores(topk);
    SearchResult_t result;
    result.dists = scores.data();
    result.ids = results.data();

    // Test with filter
    auto filter_func = [](int64_t id) -> bool {
        return id % 2 == 0;  // Only accept even IDs
    };

    ret = vsag_index_knn_search_with_filter(
        index, test_case.datas.data(), dim, topk, hgraph_search_parameters, filter_func, &result);
    REQUIRE(ret.code == VSAG_SUCCESS);

    // Verify that all returned IDs are even
    for (int64_t i = 0; i < result.count; ++i) {
        REQUIRE(result.ids[i] % 2 == 0);
    }

    vsag_index_destroy(index);
}

TEST_CASE("vsag_c_api range search", "[vsag_c_api][ut]") {
    auto index = vsag_index_factory(index_name, index_param);
    REQUIRE(index != nullptr);

    VsagTestCase test_case;

    Error_t ret =
        vsag_index_build(index, test_case.datas.data(), test_case.ids.data(), dim, num_vectors);
    REQUIRE(ret.code == VSAG_SUCCESS);

    // Test range search

    float radius = 10.0F;
    std::vector<int64_t> results(500);  // Large enough buffer
    std::vector<float> scores(500);
    SearchResult_t result;
    result.dists = scores.data();
    result.ids = results.data();

    auto idx = random() % num_vectors;
    ret = vsag_index_range_search(
        index, test_case.datas.data() + idx * dim, dim, radius, hgraph_search_parameters, &result);
    REQUIRE(ret.code == VSAG_SUCCESS);

    // Test range search with filter
    auto filter_func = [](int64_t id) -> bool {
        return id < 50;  // Only accept IDs less than 50
    };

    ret = vsag_index_range_search_with_filter(index,
                                              test_case.datas.data() + idx * dim,
                                              dim,
                                              radius,
                                              hgraph_search_parameters,
                                              filter_func,
                                              &result);
    REQUIRE(ret.code == VSAG_SUCCESS);

    // Verify that all returned IDs are less than 50
    for (int64_t i = 0; i < result.count; ++i) {
        REQUIRE(result.ids[i] < 50);
    }

    vsag_index_destroy(index);
}

TEST_CASE("vsag_c_api clone and export model", "[vsag_c_api][ut]") {
    auto index = vsag_index_factory(index_name, index_param);
    REQUIRE(index != nullptr);

    VsagTestCase test_case;

    Error_t ret =
        vsag_index_build(index, test_case.datas.data(), test_case.ids.data(), dim, num_vectors);
    REQUIRE(ret.code == VSAG_SUCCESS);

    // Test clone
    vsag_index_t cloned_index = nullptr;
    ret = vsag_index_clone(index, &cloned_index);
    REQUIRE(ret.code == VSAG_SUCCESS);
    REQUIRE(cloned_index != nullptr);
    test_case.CheckIndex(cloned_index);

    // Test export model
    vsag_index_t model_index = nullptr;
    ret = vsag_index_export_model(index, &model_index);
    REQUIRE(ret.code == VSAG_SUCCESS);
    REQUIRE(model_index != nullptr);

    ret =
        vsag_index_add(model_index, test_case.datas.data(), test_case.ids.data(), dim, num_vectors);
    REQUIRE(ret.code == VSAG_SUCCESS);
    test_case.CheckIndex(model_index);

    vsag_index_destroy(index);
    vsag_index_destroy(cloned_index);
    vsag_index_destroy(model_index);
}

TEST_CASE("vsag_c_api calculate distance by ids", "[vsag_c_api][ut]") {
    auto index = vsag_index_factory(index_name, index_param);
    REQUIRE(index != nullptr);

    VsagTestCase test_case;

    Error_t ret =
        vsag_index_build(index, test_case.datas.data(), test_case.ids.data(), dim, num_vectors);
    REQUIRE(ret.code == VSAG_SUCCESS);

    // Test calculate distance by ids
    std::vector<int64_t> query_ids = {0, 1, 2};
    std::vector<float> distances(query_ids.size());

    ret = vsag_index_calculate_distance_by_ids(index,
                                               test_case.datas.data() + 5 * dim,
                                               dim,
                                               query_ids.data(),
                                               query_ids.size(),
                                               distances.data());
    REQUIRE(ret.code == VSAG_SUCCESS);

    // Verify distances are calculated (should be non-negative)
    for (size_t i = 0; i < distances.size(); ++i) {
        auto gt_dist = vsag::FP32ComputeL2Sqr(
            test_case.datas.data() + 5 * dim, test_case.datas.data() + query_ids[i] * dim, dim);
        REQUIRE(distances[i] == gt_dist);
    }

    vsag_index_destroy(index);
}

TEST_CASE("vsag_c_api update operations and get vector by ids", "[vsag_c_api][ut]") {
    auto index = vsag_index_factory(index_name, index_param);
    REQUIRE(index != nullptr);

    VsagTestCase test_case;

    Error_t ret =
        vsag_index_build(index, test_case.datas.data(), test_case.ids.data(), dim, num_vectors);
    REQUIRE(ret.code == VSAG_SUCCESS);

    // Test get vector by ids
    std::vector<int64_t> get_ids = {8, 9, 10};
    std::vector<float> vectors(get_ids.size() * dim);
    ret = vsag_index_get_vector_by_ids(index, get_ids.data(), get_ids.size(), vectors.data());
    REQUIRE(ret.code == VSAG_SUCCESS);

    for (size_t i = 0; i < get_ids.size(); ++i) {
        auto* gt_vec = test_case.datas.data() + get_ids[i] * dim;
        for (int64_t j = 0; j < dim; ++j) {
            REQUIRE(vectors[i * dim + j] == gt_vec[j]);
        }
    }

    // Test update ids
    std::vector<int64_t> old_ids = {3, 4, 5};
    std::vector<int64_t> new_ids = {3000, 4000, 5000};
    ret = vsag_index_update_ids(index, old_ids.data(), new_ids.data(), dim, old_ids.size());
    REQUIRE(ret.code == VSAG_SUCCESS);

    ret = vsag_index_get_vector_by_ids(index, new_ids.data(), new_ids.size(), vectors.data());
    REQUIRE(ret.code == VSAG_SUCCESS);

    for (size_t i = 0; i < new_ids.size(); ++i) {
        auto* gt_vec = test_case.datas.data() + old_ids[i] * dim;
        for (int64_t j = 0; j < dim; ++j) {
            REQUIRE(vectors[i * dim + j] == gt_vec[j]);
        }
    }

    // Test update vector
    std::vector<float> new_vector(dim);
    for (int64_t i = 0; i < dim; ++i) {
        new_vector[i] = test_case.datas[6 * dim + i];
    }
    ret = vsag_index_update_vector(index, 6, new_vector.data(), dim);
    REQUIRE(ret.code == VSAG_SUCCESS);

    // Test update vector force
    ret = vsag_index_update_vector_force(index, 7, new_vector.data(), dim);
    REQUIRE(ret.code == VSAG_SUCCESS);

    std::vector<int64_t> update_ids = {6, 7};
    ret = vsag_index_get_vector_by_ids(index, update_ids.data(), update_ids.size(), vectors.data());
    REQUIRE(ret.code == VSAG_SUCCESS);

    for (int64_t i = 0; i < update_ids.size(); ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            REQUIRE(vectors[i * dim + j] == new_vector[j]);
        }
    }

    vsag_index_destroy(index);
}
