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

#include <vsag/vsag_c_api.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "simd/fp32_simd.h"
#include "unittest.h"

namespace {

// Reproducible graph; parallel build behavior is covered by dedicated HGraph tests.
constexpr const char* index_param = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 64,
            "ef_construction": 200,
            "build_thread_count": 1,
            "alpha":1.2
        }
    }
)";
constexpr const char* index_name = "hgraph";
constexpr const char* search_parameters = R"({"hgraph": {"ef_search": 200}})";
constexpr int64_t num_vectors = 500;
constexpr int64_t dim = 128;

// Releases the error and returns its code, so a test can compare against VSAG_SUCCESS.
int32_t
code_of(vsag_error_t* error) {
    const auto code = vsag_error_code(error);
    vsag_error_destroy(error);
    return code;
}

#define REQUIRE_OK(expr) REQUIRE(code_of(expr) == VSAG_SUCCESS)

struct DatasetDeleter {
    void
    operator()(vsag_dataset_t* dataset) const {
        vsag_dataset_destroy(dataset);
    }
};
struct IndexDeleter {
    void
    operator()(vsag_index_t* index) const {
        vsag_index_destroy(index);
    }
};
struct EngineDeleter {
    void
    operator()(vsag_engine_t* engine) const {
        vsag_engine_destroy(engine);
    }
};
struct OptionsDeleter {
    void
    operator()(vsag_engine_options_t* options) const {
        vsag_engine_options_destroy(options);
    }
};
struct FilterDeleter {
    void
    operator()(vsag_filter_t* filter) const {
        vsag_filter_destroy(filter);
    }
};
struct ResultDeleter {
    void
    operator()(vsag_search_result_t* result) const {
        vsag_search_result_destroy(result);
    }
};
using DatasetPtr = std::unique_ptr<vsag_dataset_t, DatasetDeleter>;
using IndexPtr = std::unique_ptr<vsag_index_t, IndexDeleter>;
using EnginePtr = std::unique_ptr<vsag_engine_t, EngineDeleter>;
using OptionsPtr = std::unique_ptr<vsag_engine_options_t, OptionsDeleter>;
using FilterPtr = std::unique_ptr<vsag_filter_t, FilterDeleter>;
using ResultPtr = std::unique_ptr<vsag_search_result_t, ResultDeleter>;

DatasetPtr
make_dense(const float* vectors, int64_t count, const int64_t* ids = nullptr) {
    vsag_dataset_t* raw = nullptr;
    REQUIRE_OK(vsag_dataset_create(&raw));
    DatasetPtr dataset(raw);
    REQUIRE_OK(vsag_dataset_set_num_elements(raw, count));
    REQUIRE_OK(vsag_dataset_set_dim(raw, dim));
    REQUIRE_OK(vsag_dataset_set_float32_vectors(raw, vectors));
    if (ids != nullptr) {
        REQUIRE_OK(vsag_dataset_set_ids(raw, ids));
    }
    return dataset;
}

IndexPtr
make_index(const char* name = index_name, const char* param = index_param) {
    vsag_index_t* raw = nullptr;
    REQUIRE_OK(vsag_index_create(name, param, &raw));
    REQUIRE(raw != nullptr);
    return IndexPtr(raw);
}

struct TestData {
    std::vector<int64_t> ids;
    std::vector<float> datas;

    TestData() : ids(num_vectors), datas(num_vectors * dim) {
        std::mt19937 rng(47);
        std::uniform_real_distribution<float> distrib;
        for (int64_t i = 0; i < num_vectors; ++i) {
            ids[i] = i;
        }
        for (auto& value : datas) {
            value = distrib(rng);
        }
    }

    IndexPtr
    BuildIndex() const {
        auto index = make_index();
        auto base = make_dense(datas.data(), num_vectors, ids.data());
        REQUIRE_OK(vsag_index_build(index.get(), base.get()));
        return index;
    }

    const float*
    Row(int64_t i) const {
        return datas.data() + i * dim;
    }

    // Every vector must find itself among its own top-10 results.
    void
    CheckIndex(const vsag_index_t* index) const {
        for (int64_t i = 0; i < num_vectors; ++i) {
            auto query = make_dense(Row(i), 1);
            vsag_search_result_t* raw = nullptr;
            REQUIRE_OK(
                vsag_index_knn_search(index, query.get(), 10, search_parameters, nullptr, &raw));
            ResultPtr result(raw);
            const auto* result_ids = vsag_search_result_ids(raw);
            REQUIRE(std::find(result_ids, result_ids + vsag_search_result_count(raw), i) !=
                    result_ids + vsag_search_result_count(raw));
        }
    }
};

}  // namespace

TEST_CASE("vsag_c_api error objects", "[vsag_c_api][ut]") {
    REQUIRE(vsag_c_api_version() == VSAG_C_API_VERSION);
    REQUIRE(vsag_error_code(nullptr) == VSAG_SUCCESS);
    REQUIRE(std::string(vsag_error_message(nullptr)).empty());
    vsag_error_destroy(nullptr);

    vsag_index_t* index = nullptr;
    auto* error = vsag_index_create("no_such_index", index_param, &index);
    REQUIRE(error != nullptr);
    REQUIRE(vsag_error_code(error) == VSAG_UNSUPPORTED_INDEX);
    REQUIRE(index == nullptr);
    REQUIRE_FALSE(std::string(vsag_error_message(error)).empty());
    vsag_error_destroy(error);

    REQUIRE(code_of(vsag_index_create(nullptr, index_param, &index)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_create(index_name, index_param, nullptr)) == VSAG_INVALID_ARGUMENT);
    vsag_index_destroy(nullptr);
    vsag_dataset_destroy(nullptr);
    vsag_filter_destroy(nullptr);
    vsag_search_result_destroy(nullptr);
}

TEST_CASE("vsag_c_api build, search and serialize", "[vsag_c_api][ut]") {
    TestData data;
    auto index = data.BuildIndex();
    data.CheckIndex(index.get());

    int64_t count = 0;
    REQUIRE_OK(vsag_index_get_num_elements(index.get(), &count));
    REQUIRE(count == num_vectors);
    int64_t memory = 0;
    REQUIRE_OK(vsag_index_get_memory_usage(index.get(), &memory));
    REQUIRE(memory > 0);
    int32_t exist = 0;
    REQUIRE_OK(vsag_index_check_id_exist(index.get(), 3, &exist));
    REQUIRE(exist == 1);
    REQUIRE_OK(vsag_index_check_id_exist(index.get(), 100000, &exist));
    REQUIRE(exist == 0);
    int64_t min_id = -1;
    int64_t max_id = -1;
    REQUIRE_OK(vsag_index_get_min_max_id(index.get(), &min_id, &max_id));
    REQUIRE(min_id == 0);
    REQUIRE(max_id == num_vectors - 1);

    auto dir = fixtures::TempDir("vsag_c_api");
    auto path = dir.GenerateRandomFile();
    REQUIRE_OK(vsag_index_serialize_file(index.get(), path.data()));

    char* analysis = nullptr;
    REQUIRE_OK(vsag_analyze_index(
        path.data(), "default", "", "auto", "", "", "", "default", 100, &analysis));
    REQUIRE(analysis != nullptr);
    const std::string report(analysis);
    vsag_string_destroy(analysis);
    REQUIRE(report.find("\"schema_version\"") != std::string::npos);
    REQUIRE(report.find("\"static_analysis\"") != std::string::npos);
    analysis = nullptr;
    REQUIRE(code_of(vsag_analyze_index(
                path.data(), "default", "", "auto", "", "", "", "default", 0, &analysis)) ==
            VSAG_INVALID_ARGUMENT);
    REQUIRE(analysis == nullptr);
    REQUIRE(code_of(vsag_analyze_index(
                path.data(), "default", "", "auto", "", "", "", "default", 100, nullptr)) ==
            VSAG_INVALID_ARGUMENT);
    vsag_string_destroy(nullptr);

    auto loaded = make_index();
    REQUIRE_OK(vsag_index_deserialize_file(loaded.get(), path.data()));
    data.CheckIndex(loaded.get());

    REQUIRE(code_of(vsag_index_serialize_file(index.get(), "/nonexistent_dir/index.bin")) !=
            VSAG_SUCCESS);
    REQUIRE(code_of(vsag_index_deserialize_file(loaded.get(), "/nonexistent_dir/index.bin")) !=
            VSAG_SUCCESS);
}

namespace {
struct MemoryStorage {
    std::vector<char> bytes;
    int writes_before_failure{-1};  // -1: never fail
    int reads_before_failure{-1};
};

int32_t
write_to_memory(void* user_data, uint64_t offset, uint64_t size, const void* data) {
    auto* storage = static_cast<MemoryStorage*>(user_data);
    if (storage->writes_before_failure == 0) {
        return -1;
    }
    if (storage->writes_before_failure > 0) {
        --storage->writes_before_failure;
    }
    if (storage->bytes.size() < offset + size) {
        storage->bytes.resize(offset + size);
    }
    std::memcpy(storage->bytes.data() + offset, data, size);
    return 0;
}

int32_t
read_from_memory(void* user_data, uint64_t offset, uint64_t size, void* dest) {
    auto* storage = static_cast<MemoryStorage*>(user_data);
    if (storage->reads_before_failure == 0 or offset + size > storage->bytes.size()) {
        return -1;
    }
    if (storage->reads_before_failure > 0) {
        --storage->reads_before_failure;
    }
    std::memcpy(dest, storage->bytes.data() + offset, size);
    return 0;
}

uint64_t
size_of_memory(void* user_data) {
    return static_cast<MemoryStorage*>(user_data)->bytes.size();
}
}  // namespace

TEST_CASE("vsag_c_api serialize through host storage", "[vsag_c_api][ut]") {
    TestData data;
    auto index = data.BuildIndex();

    MemoryStorage storage;
    REQUIRE_OK(vsag_index_serialize_writer(index.get(), write_to_memory, &storage));
    REQUIRE_FALSE(storage.bytes.empty());

    auto loaded = make_index();
    REQUIRE_OK(
        vsag_index_deserialize_reader(loaded.get(), read_from_memory, size_of_memory, &storage));
    data.CheckIndex(loaded.get());

    // A failing write or read callback surfaces as an error instead of a partial success.
    MemoryStorage failing_writer;
    failing_writer.writes_before_failure = 0;
    REQUIRE(code_of(vsag_index_serialize_writer(index.get(), write_to_memory, &failing_writer)) !=
            VSAG_SUCCESS);
    MemoryStorage failing_reader{storage.bytes};
    failing_reader.reads_before_failure = 1;
    auto target = make_index();
    REQUIRE(code_of(vsag_index_deserialize_reader(
                target.get(), read_from_memory, size_of_memory, &failing_reader)) != VSAG_SUCCESS);

    REQUIRE(code_of(vsag_index_serialize_writer(nullptr, write_to_memory, &storage)) ==
            VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_serialize_writer(index.get(), nullptr, &storage)) ==
            VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_deserialize_reader(
                nullptr, read_from_memory, size_of_memory, &storage)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_deserialize_reader(
                target.get(), read_from_memory, nullptr, &storage)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_deserialize_reader(target.get(), nullptr, nullptr, &storage)) ==
            VSAG_INVALID_ARGUMENT);
}

TEST_CASE("vsag_c_api add, train, remove and update", "[vsag_c_api][ut]") {
    TestData data;
    {
        auto index = make_index();
        auto base = make_dense(data.datas.data(), num_vectors, data.ids.data());
        REQUIRE_OK(vsag_index_train(index.get(), base.get()));
        uint64_t failed = 99;
        REQUIRE_OK(vsag_index_add(index.get(), base.get(), &failed));
        REQUIRE(failed == 0);
        data.CheckIndex(index.get());
        // Adding the same ids again only reports failures; the index is unchanged.
        REQUIRE_OK(vsag_index_add(index.get(), base.get(), nullptr));
    }

    auto index = data.BuildIndex();

    uint64_t removed = 0;
    std::vector<int64_t> to_remove{1, 2, 100000};
    REQUIRE_OK(vsag_index_remove(index.get(), to_remove.data(), to_remove.size(), &removed));
    REQUIRE(removed == 2);
    REQUIRE_OK(vsag_index_remove(index.get(), nullptr, 0, nullptr));

    std::vector<int64_t> old_ids{3, 4, 5};
    std::vector<int64_t> new_ids{3000, 4000, 5000};
    REQUIRE_OK(vsag_index_update_ids(index.get(), old_ids.data(), new_ids.data(), 3));
    std::vector<float> vectors(3 * dim);
    REQUIRE_OK(vsag_index_get_vectors_by_ids(index.get(), new_ids.data(), 3, vectors.data()));
    for (int64_t i = 0; i < 3; ++i) {
        REQUIRE(std::equal(
            vectors.begin() + i * dim, vectors.begin() + (i + 1) * dim, data.Row(old_ids[i])));
    }
    REQUIRE(code_of(vsag_index_update_ids(index.get(), old_ids.data(), new_ids.data(), 1)) !=
            VSAG_SUCCESS);

    auto replacement = make_dense(data.Row(6), 1);
    REQUIRE_OK(vsag_index_update_vector(index.get(), 6, replacement.get(), 0));
    REQUIRE_OK(vsag_index_update_vector(index.get(), 7, replacement.get(), 1));
    std::vector<int64_t> updated{6, 7};
    REQUIRE_OK(vsag_index_get_vectors_by_ids(index.get(), updated.data(), 2, vectors.data()));
    for (int64_t i = 0; i < 2; ++i) {
        REQUIRE(
            std::equal(vectors.begin() + i * dim, vectors.begin() + (i + 1) * dim, data.Row(6)));
    }
}

TEST_CASE("vsag_c_api filters", "[vsag_c_api][ut]") {
    TestData data;
    auto index = data.BuildIndex();
    auto query = make_dense(data.Row(10), 1);

    auto search = [&](const vsag_filter_t* filter) {
        vsag_search_result_t* raw = nullptr;
        REQUIRE_OK(
            vsag_index_knn_search(index.get(), query.get(), 20, search_parameters, filter, &raw));
        ResultPtr result(raw);
        const auto* ids = vsag_search_result_ids(raw);
        return std::vector<int64_t>(ids, ids + vsag_search_result_count(raw));
    };
    REQUIRE(search(nullptr)[0] == 10);

    vsag_filter_t* raw = nullptr;
    std::vector<int64_t> listed{10, 11, 12};
    REQUIRE_OK(vsag_filter_create_id_list(listed.data(), listed.size(), VSAG_FILTER_ALLOW, &raw));
    FilterPtr allow(raw);
    auto allowed = search(allow.get());
    REQUIRE(allowed.size() <= 3);
    for (auto id : allowed) {
        REQUIRE(std::count(listed.begin(), listed.end(), id) == 1);
    }

    REQUIRE_OK(vsag_filter_create_id_list(listed.data(), listed.size(), VSAG_FILTER_DENY, &raw));
    FilterPtr deny(raw);
    for (auto id : search(deny.get())) {
        REQUIRE(std::count(listed.begin(), listed.end(), id) == 0);
    }

    std::vector<uint8_t> bitmap((num_vectors + 7) / 8, 0);
    for (int64_t id = 0; id < num_vectors; id += 2) {
        bitmap[id >> 3] |= static_cast<uint8_t>(1U << (id & 7));
    }
    REQUIRE_OK(vsag_filter_create_bitmap(bitmap.data(), num_vectors, &raw));
    FilterPtr even(raw);
    auto evens = search(even.get());
    REQUIRE_FALSE(evens.empty());
    for (auto id : evens) {
        REQUIRE(id % 2 == 0);
    }

    REQUIRE_OK(vsag_filter_create_id_range(100, 199, &raw));
    FilterPtr range(raw);
    auto in_range = search(range.get());
    REQUIRE_FALSE(in_range.empty());
    for (auto id : in_range) {
        REQUIRE((id >= 100 and id <= 199));
    }

    REQUIRE(code_of(vsag_filter_create_id_list(nullptr, 1, VSAG_FILTER_ALLOW, &raw)) ==
            VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_filter_create_id_list(listed.data(), 3, 7, &raw)) ==
            VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_filter_create_bitmap(nullptr, 8, &raw)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_filter_create_id_range(0, 1, nullptr)) == VSAG_INVALID_ARGUMENT);
}

TEST_CASE("vsag_c_api range search and statistics", "[vsag_c_api][ut]") {
    TestData data;
    auto index = data.BuildIndex();
    auto query = make_dense(data.Row(10), 1);

    vsag_search_result_t* raw = nullptr;
    REQUIRE_OK(vsag_index_range_search(
        index.get(), query.get(), 1e9F, 5, search_parameters, nullptr, &raw));
    ResultPtr limited(raw);
    REQUIRE(vsag_search_result_count(raw) == 5);

    REQUIRE_OK(vsag_index_range_search(
        index.get(), query.get(), 0.0001F, -1, search_parameters, nullptr, &raw));
    ResultPtr tight(raw);
    REQUIRE(vsag_search_result_count(raw) >= 1);
    REQUIRE(vsag_search_result_ids(raw)[0] == 10);
    REQUIRE(vsag_search_result_distances(raw)[0] == 0.0F);

    REQUIRE(code_of(vsag_index_range_search(
                index.get(), query.get(), 1.0F, 0, search_parameters, nullptr, &raw)) ==
            VSAG_INVALID_ARGUMENT);

    REQUIRE_OK(
        vsag_index_knn_search(index.get(), query.get(), 5, search_parameters, nullptr, &raw));
    ResultPtr searched(raw);
    REQUIRE_FALSE(std::string(vsag_search_result_statistics(raw)).empty());
    REQUIRE(vsag_search_result_count(nullptr) == 0);
    REQUIRE(vsag_search_result_ids(nullptr) == nullptr);
    REQUIRE(vsag_search_result_distances(nullptr) == nullptr);
    REQUIRE(std::string(vsag_search_result_statistics(nullptr)).empty());
}

TEST_CASE("vsag_c_api flat float32 batch search", "[vsag_c_api][ut]") {
    TestData data;
    auto index = data.BuildIndex();

    constexpr int64_t k = 10;
    constexpr uint64_t num_queries = 20;
    std::vector<int64_t> ids(num_queries * k, -1);
    std::vector<float> dists(num_queries * k, -1.0F);
    std::vector<int64_t> counts(num_queries, 0);
    REQUIRE_OK(vsag_index_knn_search_float32(index.get(),
                                             data.datas.data(),
                                             num_queries,
                                             dim,
                                             k,
                                             search_parameters,
                                             nullptr,
                                             ids.data(),
                                             dists.data(),
                                             counts.data()));
    for (uint64_t i = 0; i < num_queries; ++i) {
        REQUIRE(counts[i] == k);
        REQUIRE(ids[i * k] == static_cast<int64_t>(i));
    }

    vsag_filter_t* raw = nullptr;
    REQUIRE_OK(vsag_filter_create_id_range(100, 199, &raw));
    FilterPtr range(raw);
    REQUIRE_OK(vsag_index_knn_search_float32(index.get(),
                                             data.datas.data(),
                                             1,
                                             dim,
                                             k,
                                             search_parameters,
                                             range.get(),
                                             ids.data(),
                                             dists.data(),
                                             nullptr));
    REQUIRE((ids[0] >= 100 and ids[0] <= 199));

    REQUIRE(code_of(vsag_index_knn_search_float32(index.get(),
                                                  data.datas.data(),
                                                  1,
                                                  dim,
                                                  0,
                                                  search_parameters,
                                                  nullptr,
                                                  ids.data(),
                                                  dists.data(),
                                                  nullptr)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_knn_search_float32(index.get(),
                                                  data.datas.data(),
                                                  1,
                                                  dim + 1,
                                                  k,
                                                  search_parameters,
                                                  nullptr,
                                                  ids.data(),
                                                  dists.data(),
                                                  nullptr)) != VSAG_SUCCESS);
}

TEST_CASE("vsag_c_api distances, clone and export model", "[vsag_c_api][ut]") {
    TestData data;
    auto index = data.BuildIndex();

    std::vector<int64_t> query_ids{0, 1, 2};
    std::vector<float> distances(query_ids.size());
    auto query = make_dense(data.Row(5), 1);
    REQUIRE_OK(vsag_index_calc_distances_by_ids(
        index.get(), query.get(), query_ids.data(), query_ids.size(), distances.data()));
    for (size_t i = 0; i < query_ids.size(); ++i) {
        REQUIRE(distances[i] == vsag::FP32ComputeL2Sqr(data.Row(5), data.Row(query_ids[i]), dim));
    }

    vsag_index_t* raw = nullptr;
    REQUIRE_OK(vsag_index_clone(index.get(), &raw));
    IndexPtr cloned(raw);
    data.CheckIndex(cloned.get());

    REQUIRE_OK(vsag_index_export_model(index.get(), &raw));
    IndexPtr model(raw);
    auto base = make_dense(data.datas.data(), num_vectors, data.ids.data());
    REQUIRE_OK(vsag_index_add(model.get(), base.get(), nullptr));
    data.CheckIndex(model.get());
}

TEST_CASE("vsag_c_api float16 vectors", "[vsag_c_api][ut]") {
    const char* param = R"({
        "dtype": "float16", "metric_type": "l2", "dim": 16,
        "index_param": {"base_quantization_type": "fp16", "max_degree": 16,
                        "ef_construction": 100, "build_thread_count": 1}
    })";
    auto index = make_index("hgraph", param);

    constexpr int64_t n = 50;
    std::vector<uint16_t> vectors(n * 16);
    std::mt19937 rng(7);
    for (auto& value : vectors) {
        value = static_cast<uint16_t>(0x3C00 + rng() % 0x0400);  // finite halves in [1, 2)
    }
    std::vector<int64_t> ids(n);
    for (int64_t i = 0; i < n; ++i) {
        ids[i] = i;
    }
    vsag_dataset_t* raw = nullptr;
    REQUIRE_OK(vsag_dataset_create(&raw));
    DatasetPtr base(raw);
    REQUIRE_OK(vsag_dataset_set_num_elements(raw, n));
    REQUIRE_OK(vsag_dataset_set_dim(raw, 16));
    REQUIRE_OK(vsag_dataset_set_ids(raw, ids.data()));
    REQUIRE_OK(vsag_dataset_set_float16_vectors(raw, vectors.data()));
    REQUIRE_OK(vsag_index_build(index.get(), base.get()));

    int64_t count = 0;
    REQUIRE_OK(vsag_index_get_num_elements(index.get(), &count));
    REQUIRE(count == n);
}

TEST_CASE("vsag_c_api sparse vectors", "[vsag_c_api][ut]") {
    const char* param = R"({
        "dim": 16, "dtype": "sparse", "metric_type": "ip",
        "index_param": {"use_reorder": true, "doc_prune_ratio": 0.0,
                        "window_size": 10000, "term_id_limit": 16}
    })";
    auto index = make_index("sindi", param);

    std::vector<uint32_t> pointers{0, 2, 3};
    std::vector<uint32_t> indices{1, 3, 5};
    std::vector<float> values{1.0F, 2.0F, 3.0F};
    std::vector<int64_t> ids{10, 20};
    auto make_sparse = [&](const uint32_t* index_pointers, int64_t rows) {
        vsag_dataset_t* raw = nullptr;
        REQUIRE_OK(vsag_dataset_create(&raw));
        DatasetPtr dataset(raw);
        REQUIRE_OK(vsag_dataset_set_num_elements(raw, rows));
        REQUIRE_OK(vsag_dataset_set_ids(raw, ids.data()));
        REQUIRE_OK(
            vsag_dataset_set_sparse_vectors(raw, index_pointers, indices.data(), values.data()));
        return dataset;
    };

    std::vector<uint32_t> bad_pointers{1, 2, 3};
    REQUIRE(code_of(vsag_index_build(index.get(), make_sparse(bad_pointers.data(), 2).get())) ==
            VSAG_INVALID_ARGUMENT);
    std::vector<uint32_t> decreasing{0, 3, 2};
    REQUIRE(code_of(vsag_index_build(index.get(), make_sparse(decreasing.data(), 2).get())) ==
            VSAG_INVALID_ARGUMENT);
    REQUIRE_OK(vsag_index_build(index.get(), make_sparse(pointers.data(), 2).get()));

    // The first row doubles as the query.
    std::vector<int64_t> query_ids{0};
    vsag_dataset_t* raw = nullptr;
    REQUIRE_OK(vsag_dataset_create(&raw));
    DatasetPtr query(raw);
    REQUIRE_OK(vsag_dataset_set_num_elements(raw, 1));
    REQUIRE_OK(
        vsag_dataset_set_sparse_vectors(raw, pointers.data(), indices.data(), values.data()));
    vsag_search_result_t* result = nullptr;
    REQUIRE_OK(vsag_index_knn_search(
        index.get(), query.get(), 2, R"({"sindi": {"n_candidate": 10}})", nullptr, &result));
    ResultPtr guard(result);
    REQUIRE(vsag_search_result_count(result) >= 1);
    REQUIRE(vsag_search_result_ids(result)[0] == 10);
}

TEST_CASE("vsag_c_api invalid arguments", "[vsag_c_api][ut]") {
    TestData data;
    auto index = data.BuildIndex();
    auto query = make_dense(data.Row(0), 1);
    vsag_search_result_t* result = nullptr;
    vsag_index_t* other = nullptr;

    REQUIRE(code_of(vsag_index_build(nullptr, query.get())) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_build(index.get(), nullptr)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_train(nullptr, query.get())) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_add(nullptr, query.get(), nullptr)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_remove(nullptr, nullptr, 0, nullptr)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_remove(index.get(), nullptr, 1, nullptr)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_update_ids(nullptr, nullptr, nullptr, 0)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_update_vector(nullptr, 0, query.get(), 0)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_knn_search(
                nullptr, query.get(), 1, search_parameters, nullptr, &result)) ==
            VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_knn_search(
                index.get(), query.get(), 0, search_parameters, nullptr, &result)) ==
            VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_knn_search(
                index.get(), query.get(), 1, nullptr, nullptr, &result)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_knn_search(
                index.get(), query.get(), 1, search_parameters, nullptr, nullptr)) ==
            VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_range_search(
                nullptr, query.get(), 1.0F, -1, search_parameters, nullptr, &result)) ==
            VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_calc_distances_by_ids(nullptr, query.get(), nullptr, 0, nullptr)) ==
            VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_get_vectors_by_ids(nullptr, nullptr, 1, nullptr)) ==
            VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_clone(nullptr, &other)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_export_model(nullptr, &other)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_serialize_file(nullptr, "x")) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_deserialize_file(index.get(), nullptr)) == VSAG_INVALID_ARGUMENT);
    int64_t value = 0;
    int32_t flag = 0;
    REQUIRE(code_of(vsag_index_get_num_elements(nullptr, &value)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_get_num_elements(index.get(), nullptr)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_get_memory_usage(nullptr, &value)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_check_id_exist(nullptr, 0, &flag)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_get_min_max_id(nullptr, &value, &value)) == VSAG_INVALID_ARGUMENT);

    // Dataset validation.
    vsag_dataset_t* raw = nullptr;
    REQUIRE(code_of(vsag_dataset_create(nullptr)) == VSAG_INVALID_ARGUMENT);
    REQUIRE_OK(vsag_dataset_create(&raw));
    DatasetPtr empty(raw);
    REQUIRE(code_of(vsag_index_build(index.get(), empty.get())) == VSAG_INVALID_ARGUMENT);
    REQUIRE_OK(vsag_dataset_set_num_elements(raw, 2));
    REQUIRE(code_of(vsag_index_build(index.get(), empty.get())) == VSAG_INVALID_ARGUMENT);
    REQUIRE_OK(vsag_dataset_set_float32_vectors(raw, data.datas.data()));
    REQUIRE(code_of(vsag_index_build(index.get(), empty.get())) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_index_knn_search(
                index.get(), empty.get(), 1, search_parameters, nullptr, &result)) ==
            VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_dataset_set_dim(nullptr, 1)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_dataset_set_num_elements(nullptr, 1)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_dataset_set_ids(nullptr, nullptr)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_dataset_set_float32_vectors(nullptr, nullptr)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_dataset_set_float16_vectors(nullptr, nullptr)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_dataset_set_int8_vectors(nullptr, nullptr)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_dataset_set_sparse_vectors(nullptr, nullptr, nullptr, nullptr)) ==
            VSAG_INVALID_ARGUMENT);
}

TEST_CASE("vsag_c_api callback filter", "[vsag_c_api][ut]") {
    TestData data;
    auto index = data.BuildIndex();
    auto query = make_dense(data.Row(10), 1);

    std::atomic<int> calls{0};
    auto keep_even = [](void* user_data, int64_t id) -> int32_t {
        static_cast<std::atomic<int>*>(user_data)->fetch_add(1);
        return id % 2 == 0 ? 1 : 0;
    };
    vsag_filter_t* raw = nullptr;
    REQUIRE_OK(vsag_filter_create_callback(keep_even, &calls, &raw));
    FilterPtr filter(raw);

    vsag_search_result_t* result = nullptr;
    REQUIRE_OK(vsag_index_knn_search(
        index.get(), query.get(), 10, search_parameters, filter.get(), &result));
    ResultPtr guard(result);
    REQUIRE(vsag_search_result_count(result) > 0);
    for (int64_t i = 0; i < vsag_search_result_count(result); ++i) {
        REQUIRE(vsag_search_result_ids(result)[i] % 2 == 0);
    }
    REQUIRE(calls.load() > 0);

    REQUIRE(code_of(vsag_filter_create_callback(nullptr, nullptr, &raw)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_filter_create_callback(keep_even, nullptr, nullptr)) ==
            VSAG_INVALID_ARGUMENT);
}

namespace {
struct CountingAllocator {
    std::atomic<int64_t> allocations{0};
    std::atomic<int64_t> live{0};
};

void*
counting_allocate(void* user_data, uint64_t size) {
    auto* counters = static_cast<CountingAllocator*>(user_data);
    counters->allocations++;
    counters->live++;
    return std::malloc(size);
}

void
counting_deallocate(void* user_data, void* ptr) {
    if (ptr != nullptr) {
        static_cast<CountingAllocator*>(user_data)->live--;
    }
    std::free(ptr);
}

void*
counting_reallocate(void* user_data, void* ptr, uint64_t size) {
    auto* counters = static_cast<CountingAllocator*>(user_data);
    if (ptr == nullptr) {
        counters->allocations++;
        counters->live++;
    }
    return std::realloc(ptr, size);
}

// Runs every task on a fresh thread; the destructor joins them all.
struct ThreadPerTaskPool {
    std::mutex mutex;
    std::vector<std::thread> threads;
    std::atomic<int> submitted{0};
    std::atomic<int> rejected{0};
    std::atomic<bool> reject{false};

    ~ThreadPerTaskPool() {
        for (auto& thread : threads) {
            thread.join();
        }
    }
};

int32_t
submit_to_thread(void* user_data, vsag_task_fn task, void* task_data) {
    auto* pool = static_cast<ThreadPerTaskPool*>(user_data);
    if (pool->reject.load()) {
        pool->rejected++;
        return -1;
    }
    pool->submitted++;
    std::lock_guard lock(pool->mutex);
    pool->threads.emplace_back([task, task_data]() { task(task_data); });
    return 0;
}

constexpr const char* parallel_index_param = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 32,
            "ef_construction": 100,
            "build_thread_count": 4
        }
    }
)";
}  // namespace

namespace {
std::atomic<bool> allocator_exhausted{false};

void*
exhaustible_allocate(void* /*user_data*/, uint64_t size) {
    return allocator_exhausted.load() ? nullptr : std::malloc(size);
}

void
exhaustible_deallocate(void* /*user_data*/, void* ptr) {
    std::free(ptr);
}

void*
exhaustible_reallocate(void* /*user_data*/, void* ptr, uint64_t size) {
    return allocator_exhausted.load() ? nullptr : std::realloc(ptr, size);
}
}  // namespace

TEST_CASE("vsag_c_api host allocator failure is an error, not a crash", "[vsag_c_api][ut]") {
    TestData data;
    vsag_engine_options_t* raw_options = nullptr;
    REQUIRE_OK(vsag_engine_options_create(&raw_options));
    OptionsPtr options(raw_options);
    REQUIRE_OK(vsag_engine_options_set_allocator(raw_options,
                                                 "exhaustible",
                                                 exhaustible_allocate,
                                                 exhaustible_deallocate,
                                                 exhaustible_reallocate,
                                                 nullptr));
    vsag_engine_t* raw_engine = nullptr;
    REQUIRE_OK(vsag_engine_create(raw_options, &raw_engine));
    EnginePtr engine(raw_engine);
    vsag_index_t* raw_index = nullptr;
    REQUIRE_OK(vsag_engine_create_index(raw_engine, index_name, index_param, &raw_index));
    IndexPtr index(raw_index);

    allocator_exhausted = true;
    auto base = make_dense(data.datas.data(), num_vectors, data.ids.data());
    auto* error = vsag_index_build(index.get(), base.get());
    allocator_exhausted = false;
    REQUIRE(error != nullptr);
    REQUIRE(vsag_error_code(error) == VSAG_NO_ENOUGH_MEMORY);
    vsag_error_destroy(error);
}

TEST_CASE("vsag_c_api engine with host allocator and thread pool", "[vsag_c_api][ut]") {
    TestData data;
    CountingAllocator counters;
    ThreadPerTaskPool pool;

    vsag_engine_options_t* raw_options = nullptr;
    REQUIRE_OK(vsag_engine_options_create(&raw_options));
    OptionsPtr options(raw_options);
    REQUIRE_OK(vsag_engine_options_set_allocator(raw_options,
                                                 "counting",
                                                 counting_allocate,
                                                 counting_deallocate,
                                                 counting_reallocate,
                                                 &counters));
    REQUIRE_OK(vsag_engine_options_set_thread_pool(raw_options, submit_to_thread, &pool));

    vsag_engine_t* raw_engine = nullptr;
    REQUIRE_OK(vsag_engine_create(raw_options, &raw_engine));
    EnginePtr engine(raw_engine);
    options.reset();

    vsag_index_t* raw_index = nullptr;
    REQUIRE_OK(vsag_engine_create_index(raw_engine, index_name, parallel_index_param, &raw_index));
    IndexPtr index(raw_index);
    // The index keeps the host resources alive after the engine handle is gone.
    engine.reset();

    auto base = make_dense(data.datas.data(), num_vectors, data.ids.data());
    REQUIRE_OK(vsag_index_build(index.get(), base.get()));
    data.CheckIndex(index.get());
    REQUIRE(counters.allocations.load() > 0);
    REQUIRE(pool.submitted.load() > 0);

    // A pool that refuses work makes the library run it on the calling thread.
    pool.reject = true;
    vsag_engine_options_t* reject_options = nullptr;
    REQUIRE_OK(vsag_engine_options_create(&reject_options));
    OptionsPtr reject_options_owner(reject_options);
    REQUIRE_OK(vsag_engine_options_set_thread_pool(reject_options, submit_to_thread, &pool));
    vsag_engine_t* rejecting_engine = nullptr;
    REQUIRE_OK(vsag_engine_create(reject_options, &rejecting_engine));
    EnginePtr rejecting(rejecting_engine);
    vsag_index_t* fallback_index = nullptr;
    REQUIRE_OK(vsag_engine_create_index(
        rejecting_engine, index_name, parallel_index_param, &fallback_index));
    IndexPtr fallback_owner(fallback_index);
    const int submitted_before = pool.submitted.load();
    REQUIRE_OK(vsag_index_build(fallback_index, base.get()));
    REQUIRE(pool.rejected.load() > 0);
    REQUIRE(pool.submitted.load() == submitted_before);
    data.CheckIndex(fallback_index);

    index.reset();
    REQUIRE(counters.live.load() == 0);

    REQUIRE(code_of(vsag_engine_create_index(nullptr, index_name, index_param, &raw_index)) ==
            VSAG_INVALID_ARGUMENT);
    vsag_engine_options_t* invalid_options = nullptr;
    REQUIRE_OK(vsag_engine_options_create(&invalid_options));
    OptionsPtr invalid(invalid_options);
    REQUIRE(code_of(vsag_engine_options_set_allocator(invalid_options,
                                                      nullptr,
                                                      nullptr,
                                                      counting_deallocate,
                                                      counting_reallocate,
                                                      nullptr)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_engine_options_set_thread_pool(invalid_options, nullptr, nullptr)) ==
            VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_engine_options_set_allocator(nullptr,
                                                      nullptr,
                                                      counting_allocate,
                                                      counting_deallocate,
                                                      counting_reallocate,
                                                      nullptr)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_engine_options_create(nullptr)) == VSAG_INVALID_ARGUMENT);
}

namespace {
struct LogRecords {
    std::atomic<int> count{0};
    std::atomic<int> max_level{-1};
};

void
collect_log(void* user_data, int32_t level, const char* message) {
    auto* records = static_cast<LogRecords*>(user_data);
    if (message == nullptr) {
        return;
    }
    records->count++;
    int previous = records->max_level.load();
    while (previous < level and not records->max_level.compare_exchange_weak(previous, level)) {
    }
}
}  // namespace

TEST_CASE("vsag_c_api logger callback", "[vsag_c_api][ut]") {
    REQUIRE(code_of(vsag_set_log_level(-1)) == VSAG_INVALID_ARGUMENT);
    REQUIRE(code_of(vsag_set_log_level(VSAG_LOG_OFF + 1)) == VSAG_INVALID_ARGUMENT);

    LogRecords records;
    REQUIRE_OK(vsag_set_log_level(VSAG_LOG_INFO));
    REQUIRE_OK(vsag_set_logger(collect_log, &records));

    // Serializing logs at info level. Records may arrive from several threads at once.
    TestData data;
    auto index = data.BuildIndex();
    auto dir = fixtures::TempDir("vsag_c_api_log");
    std::vector<std::string> paths{dir.GenerateRandomFile(), dir.GenerateRandomFile()};
    std::vector<std::thread> threads;
    for (const auto& path : paths) {
        threads.emplace_back(
            [&]() { vsag_error_destroy(vsag_index_serialize_file(index.get(), path.c_str())); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    REQUIRE(records.count.load() > 0);

    // Below the configured level nothing is delivered.
    REQUIRE_OK(vsag_set_log_level(VSAG_LOG_OFF));
    const int before = records.count.load();
    REQUIRE_OK(vsag_index_serialize_file(index.get(), paths[0].c_str()));
    REQUIRE(records.count.load() == before);

    // Restoring the built-in logger stops delivery to the callback.
    REQUIRE_OK(vsag_set_log_level(VSAG_LOG_INFO));
    REQUIRE_OK(vsag_set_logger(nullptr, nullptr));
    REQUIRE_OK(vsag_index_serialize_file(index.get(), paths[0].c_str()));
    REQUIRE(records.count.load() == before);
}
