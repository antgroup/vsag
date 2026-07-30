
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

#include "eval_dataset.h"

#include <H5Cpp.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "case/eval_case.h"
#include "evaluator.h"
#include "monitor/recall_monitor.h"
#include "vsag/factory.h"

namespace {

using vsag::SparseVector;
using vsag::eval::EvalDataset;
using vsag::eval::EvalDatasetPtr;

EvalDatasetPtr
BuildSparseDataset(bool with_token_sequences, bool all_empty = false) {
    // Build a tiny sparse dataset (3 train, 2 test) and optionally attach
    // the original tokenized term-id sequences.
    auto ds = std::make_shared<EvalDataset>();

    // Sparse train vectors.
    std::vector<SparseVector> train(3);
    std::vector<SparseVector> test(2);
    if (not all_empty) {
        train[0].len_ = 2;
        train[0].ids_ = new uint32_t[2]{1, 5};
        train[0].vals_ = new float[2]{0.5f, 1.0f};
        train[1].len_ = 1;
        train[1].ids_ = new uint32_t[1]{3};
        train[1].vals_ = new float[1]{0.25f};

        test[0].len_ = 2;
        test[0].ids_ = new uint32_t[2]{2, 7};
        test[0].vals_ = new float[2]{0.4f, 0.6f};
        test[1].len_ = 1;
        test[1].ids_ = new uint32_t[1]{9};
        test[1].vals_ = new float[1]{1.0f};
    }

    if (with_token_sequences) {
        train[0].token_seq_len_ = 4;
        train[0].token_sequence_ = new uint32_t[4]{10, 20, 10, 30};
        train[1].token_seq_len_ = 2;
        train[1].token_sequence_ = new uint32_t[2]{42, 42};
        // train[2] intentionally has no original document.
        train[2].token_seq_len_ = 0;
        train[2].token_sequence_ = nullptr;

        test[0].token_seq_len_ = 1;
        test[0].token_sequence_ = new uint32_t[1]{99};
        test[1].token_seq_len_ = 3;
        test[1].token_sequence_ = new uint32_t[3]{1, 2, 3};
    }

    // Inject the sparse vectors via friend access through pointer.
    // EvalDataset's members are private; expose a tiny helper through
    // direct memory write would be invasive. Instead we build via Save's
    // public API by reaching through a shim subclass.
    struct ShimDataset : public EvalDataset {
        std::vector<SparseVector>&
        train() {
            return this->sparse_train_;
        }
        std::vector<SparseVector>&
        test() {
            return this->sparse_test_;
        }
        void
        set_metric(const std::string& m) {
            this->metric_ = m;
        }
        void
        set_type(const std::string& t) {
            this->vector_type_ = t;
        }
        void
        set_counts(int64_t base, int64_t query) {
            this->number_of_base_ = base;
            this->number_of_query_ = query;
        }
        void
        set_neighbors_shape(int64_t q, int64_t k) {
            this->neighbors_shape_ = {q, k};
        }
        void
        set_neighbors(int64_t* ptr) {
            this->neighbors_.reset(ptr);
        }
        void
        set_distances(float* ptr) {
            this->distances_.reset(ptr);
        }
    };

    auto shim = std::make_shared<ShimDataset>();
    shim->set_type(vsag::SPARSE_VECTORS);
    shim->set_metric("ip");
    shim->train() = std::move(train);
    shim->test() = std::move(test);
    shim->set_counts(static_cast<int64_t>(shim->train().size()),
                     static_cast<int64_t>(shim->test().size()));

    // Minimal ground truth (k=1 per query; values do not matter for the
    // load/save round-trip we are testing).
    constexpr int64_t kK = 1;
    int64_t* nb = new int64_t[shim->test().size() * kK];
    float* dist = new float[shim->test().size() * kK];
    for (int64_t i = 0; i < static_cast<int64_t>(shim->test().size()); ++i) {
        nb[i] = 0;
        dist[i] = 0.0f;
    }
    shim->set_neighbors_shape(static_cast<int64_t>(shim->test().size()), kK);
    shim->set_neighbors(nb);
    shim->set_distances(dist);

    return shim;
}

std::string
TempPath(const std::string& tag) {
    std::string path = "/tmp/eval_dataset_test_" + tag + ".hdf5";
    std::remove(path.c_str());
    return path;
}

}  // namespace

TEST_CASE("EvalDataset builds a dense in-memory view with original ids", "[ut][eval_dataset]") {
    constexpr int64_t dim = 2;
    std::vector<float> base_vectors{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> base_ids{101, 7, 42};
    auto base = vsag::Dataset::Make();
    base->NumElements(3)
        ->Dim(dim)
        ->Ids(base_ids.data())
        ->Float32Vectors(base_vectors.data())
        ->Owner(false);

    std::vector<float> query_vectors{1.0F, 2.0F, 5.0F, 6.0F};
    auto queries = vsag::Dataset::Make();
    queries->NumElements(2)->Dim(dim)->Float32Vectors(query_vectors.data())->Owner(false);

    std::vector<int64_t> ground_truth_ids{101, 7, 42, 7};
    auto ground_truth = vsag::Dataset::Make();
    ground_truth->NumElements(2)->Dim(2)->Ids(ground_truth_ids.data())->Owner(false);

    auto dataset = EvalDataset::FromDatasets(base, queries, ground_truth, "l2");
    REQUIRE(dataset->GetTrain() == base_vectors.data());
    REQUIRE(dataset->GetTest() == query_vectors.data());
    REQUIRE(dataset->GetTrainIds() == base_ids.data());
    REQUIRE(dataset->GetMetric() == "euclidean");
    REQUIRE(dataset->GetGroundTruthK() == 2);
    REQUIRE(dataset->GetNeighbors(1)[0] == 42);
    REQUIRE(static_cast<const float*>(dataset->GetOneTrainById(7))[0] == 3.0F);
    REQUIRE(dataset->GetOneTrainById(999) == nullptr);

    int64_t one_result = 101;
    vsag::eval::SearchRecord record{
        &one_result, dataset->GetNeighbors(0), dataset.get(), dataset->GetOneTest(0), 1, 2};
    vsag::eval::RecallMonitor recall_monitor(1);
    recall_monitor.SetMetrics("avg_recall");
    recall_monitor.Record(&record);
    REQUIRE(recall_monitor.GetResult()["recall_avg"].get<double>() == 0.5);

    auto without_ground_truth = EvalDataset::FromDatasets(base, queries, nullptr, "cosine");
    REQUIRE(without_ground_truth->GetGroundTruthK() == 0);
    REQUIRE(without_ground_truth->GetNeighbors(0) == nullptr);

    const auto save_path = TempPath("in_memory_save");
    REQUIRE_THROWS_WITH(EvalDataset::Save(dataset, save_path),
                        "saving an in-memory EvalDataset view is not supported");
    REQUIRE_FALSE(std::filesystem::exists(save_path));
}

TEST_CASE("EvalDataset rejects incomplete and incompatible HDF5 schemas", "[ut][eval_dataset]") {
    const std::vector<std::string> required_datasets{"train", "test", "neighbors", "distances"};
    for (const auto& missing : required_datasets) {
        const auto path = TempPath("missing_" + missing);
        {
            H5::H5File file(path, H5F_ACC_TRUNC);
            hsize_t dims[2] = {1, 1};
            H5::DataSpace space(2, dims);
            for (const auto& name : required_datasets) {
                if (name != missing) {
                    file.createDataSet(name, H5::PredType::NATIVE_FLOAT, space);
                }
            }
        }
        REQUIRE_THROWS_WITH(
            EvalDataset::Load(path),
            Catch::Matchers::ContainsSubstring("missing required HDF5 dataset '" + missing + "'"));
        std::remove(path.c_str());
    }

    const auto path = TempPath("dimension_mismatch");
    {
        H5::H5File file(path, H5F_ACC_TRUNC);
        hsize_t train_dims[2] = {1, 2};
        hsize_t test_dims[2] = {1, 3};
        hsize_t result_dims[2] = {1, 1};
        H5::DataSpace train_space(2, train_dims);
        H5::DataSpace test_space(2, test_dims);
        H5::DataSpace result_space(2, result_dims);
        file.createDataSet("train", H5::PredType::NATIVE_FLOAT, train_space);
        file.createDataSet("test", H5::PredType::NATIVE_FLOAT, test_space);
        file.createDataSet("neighbors", H5::PredType::NATIVE_INT64, result_space);
        file.createDataSet("distances", H5::PredType::NATIVE_FLOAT, result_space);
    }
    REQUIRE_THROWS_WITH(
        EvalDataset::Load(path),
        Catch::Matchers::ContainsSubstring("train and test vector dimensions must match"));
    std::remove(path.c_str());

    const auto result_shape_path = TempPath("result_shape_mismatch");
    {
        H5::H5File file(result_shape_path, H5F_ACC_TRUNC);
        hsize_t vector_dims[2] = {2, 2};
        hsize_t neighbor_dims[2] = {2, 1};
        hsize_t distance_dims[2] = {1, 1};
        H5::DataSpace vector_space(2, vector_dims);
        H5::DataSpace neighbor_space(2, neighbor_dims);
        H5::DataSpace distance_space(2, distance_dims);
        file.createDataSet("train", H5::PredType::NATIVE_FLOAT, vector_space);
        file.createDataSet("test", H5::PredType::NATIVE_FLOAT, vector_space);
        file.createDataSet("neighbors", H5::PredType::NATIVE_INT64, neighbor_space);
        file.createDataSet("distances", H5::PredType::NATIVE_FLOAT, distance_space);
    }
    REQUIRE_THROWS_WITH(
        EvalDataset::Load(result_shape_path),
        Catch::Matchers::ContainsSubstring("distances shape must match neighbors shape"));
    std::remove(result_shape_path.c_str());

    const auto query_count_path = TempPath("query_count_mismatch");
    {
        H5::H5File file(query_count_path, H5F_ACC_TRUNC);
        hsize_t train_dims[2] = {2, 2};
        hsize_t test_dims[2] = {2, 2};
        hsize_t result_dims[2] = {1, 1};
        H5::DataSpace train_space(2, train_dims);
        H5::DataSpace test_space(2, test_dims);
        H5::DataSpace result_space(2, result_dims);
        file.createDataSet("train", H5::PredType::NATIVE_FLOAT, train_space);
        file.createDataSet("test", H5::PredType::NATIVE_FLOAT, test_space);
        file.createDataSet("neighbors", H5::PredType::NATIVE_INT64, result_space);
        file.createDataSet("distances", H5::PredType::NATIVE_FLOAT, result_space);
    }
    REQUIRE_THROWS_WITH(
        EvalDataset::Load(query_count_path),
        Catch::Matchers::ContainsSubstring("neighbors row count must match query count"));
    std::remove(query_count_path.c_str());

    const auto unsupported_type_path = TempPath("unsupported_type");
    {
        H5::H5File file(unsupported_type_path, H5F_ACC_TRUNC);
        hsize_t dims[2] = {1, 1};
        H5::DataSpace space(2, dims);
        file.createDataSet("train", H5::PredType::NATIVE_FLOAT, space);
        file.createDataSet("test", H5::PredType::NATIVE_FLOAT, space);
        file.createDataSet("neighbors", H5::PredType::NATIVE_INT64, space);
        file.createDataSet("distances", H5::PredType::NATIVE_FLOAT, space);
        H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
        auto attribute = file.createAttribute("type", string_type, H5::DataSpace(H5S_SCALAR));
        std::string type = "unsupported";
        attribute.write(string_type, type);
    }
    REQUIRE_THROWS_WITH(
        EvalDataset::Load(unsupported_type_path),
        Catch::Matchers::ContainsSubstring("unsupported HDF5 dataset type 'unsupported'"));
    std::remove(unsupported_type_path.c_str());

    const auto unsupported_rank_path = TempPath("unsupported_rank");
    {
        H5::H5File file(unsupported_rank_path, H5F_ACC_TRUNC);
        hsize_t train_dims[3] = {1, 1, 1};
        hsize_t matrix_dims[2] = {1, 1};
        H5::DataSpace train_space(3, train_dims);
        H5::DataSpace matrix_space(2, matrix_dims);
        file.createDataSet("train", H5::PredType::NATIVE_FLOAT, train_space);
        file.createDataSet("test", H5::PredType::NATIVE_FLOAT, matrix_space);
        file.createDataSet("neighbors", H5::PredType::NATIVE_INT64, matrix_space);
        file.createDataSet("distances", H5::PredType::NATIVE_FLOAT, matrix_space);
    }
    REQUIRE_THROWS_WITH(EvalDataset::Load(unsupported_rank_path),
                        Catch::Matchers::ContainsSubstring("unsupported dataset rank: 3"));
    std::remove(unsupported_rank_path.c_str());

    const auto label_shape_path = TempPath("label_shape_mismatch");
    {
        H5::H5File file(label_shape_path, H5F_ACC_TRUNC);
        hsize_t train_dims[2] = {2, 2};
        hsize_t test_dims[2] = {1, 2};
        hsize_t result_dims[2] = {1, 1};
        hsize_t train_label_dims[1] = {1};
        hsize_t test_label_dims[1] = {1};
        H5::DataSpace train_space(2, train_dims);
        H5::DataSpace test_space(2, test_dims);
        H5::DataSpace result_space(2, result_dims);
        H5::DataSpace train_label_space(1, train_label_dims);
        H5::DataSpace test_label_space(1, test_label_dims);
        file.createDataSet("train", H5::PredType::NATIVE_FLOAT, train_space);
        file.createDataSet("test", H5::PredType::NATIVE_FLOAT, test_space);
        file.createDataSet("neighbors", H5::PredType::NATIVE_INT64, result_space);
        file.createDataSet("distances", H5::PredType::NATIVE_FLOAT, result_space);
        file.createDataSet("train_labels", H5::PredType::NATIVE_INT64, train_label_space);
        file.createDataSet("test_labels", H5::PredType::NATIVE_INT64, test_label_space);
    }
    REQUIRE_THROWS_WITH(EvalDataset::Load(label_shape_path),
                        Catch::Matchers::ContainsSubstring(
                            "train_labels must be one-dimensional with one label per base vector"));
    std::remove(label_shape_path.c_str());

    const auto unsupported_distance_path = TempPath("unsupported_distance");
    {
        H5::H5File file(unsupported_distance_path, H5F_ACC_TRUNC);
        hsize_t dims[2] = {1, 1};
        H5::DataSpace space(2, dims);
        file.createDataSet("train", H5::PredType::NATIVE_FLOAT, space);
        file.createDataSet("test", H5::PredType::NATIVE_FLOAT, space);
        file.createDataSet("neighbors", H5::PredType::NATIVE_INT64, space);
        file.createDataSet("distances", H5::PredType::NATIVE_FLOAT, space);
        H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
        auto attribute = file.createAttribute("distance", string_type, H5::DataSpace(H5S_SCALAR));
        std::string distance = "unsupported";
        attribute.write(string_type, distance);
    }
    REQUIRE_THROWS_WITH(EvalDataset::Load(unsupported_distance_path),
                        Catch::Matchers::ContainsSubstring(
                            "unsupported HDF5 distance 'unsupported' for dense vectors"));
    std::remove(unsupported_distance_path.c_str());
}

TEST_CASE("EvaluateSearch validates inputs and propagates search errors", "[ut][eval_dataset]") {
    constexpr int64_t dim = 2;
    std::vector<float> base_vectors{0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
    std::vector<int64_t> base_ids{0, 1, 2, 3};
    auto base = vsag::Dataset::Make();
    base->NumElements(4)
        ->Dim(dim)
        ->Ids(base_ids.data())
        ->Float32Vectors(base_vectors.data())
        ->Owner(false);

    std::vector<float> query_vectors{0.0F, 0.0F, 1.0F, 1.0F};
    auto queries = vsag::Dataset::Make();
    queries->NumElements(2)->Dim(dim)->Float32Vectors(query_vectors.data())->Owner(false);

    std::vector<int64_t> ground_truth_ids{0, 3};
    auto ground_truth = vsag::Dataset::Make();
    ground_truth->NumElements(2)->Dim(1)->Ids(ground_truth_ids.data())->Owner(false);
    auto dataset = EvalDataset::FromDatasets(base, queries, ground_truth, "l2");

    const std::string create_params = R"(
        {
            "dim": 2,
            "dtype": "float32",
            "metric_type": "l2",
            "index_param": {
                "base_quantization_type": "fp32",
                "max_degree": 8,
                "ef_construction": 20
            }
        })";
    auto created = vsag::Factory::CreateIndex("hgraph", create_params);
    REQUIRE(created.has_value());
    auto index = created.value();

    vsag::eval::EvalConfig build_config;
    build_config.index_name = "hgraph";
    build_config.enable_tps = false;
    build_config.enable_memory = false;
    const auto build_result = vsag::eval::EvaluateBuild(index, dataset, build_config);
    REQUIRE(build_result.contains("duration(s)"));
    REQUIRE_FALSE(build_result.contains("tps"));
    REQUIRE(build_result["index_info"].is_object());
    REQUIRE(build_result["index_info"].empty());

    vsag::eval::EvalConfig config;
    config.index_name = "hgraph";
    config.search_param = R"({"hgraph":{"ef_search":8}})";
    config.top_k = 1;
    config.search_query_count = 2;
    config.num_threads_searching = 2;
    config.enable_recall = false;
    config.enable_percent_recall = false;
    config.enable_qps = true;
    config.enable_tps = false;
    config.enable_memory = false;
    config.enable_latency = false;
    config.enable_percent_latency = false;

    const auto qps_only = vsag::eval::EvaluateSearch(index, dataset, config);
    REQUIRE(qps_only.contains("qps"));
    REQUIRE(qps_only["measurement_sample_count"].get<uint64_t>() == 2);
    REQUIRE(qps_only["index_info"].is_object());
    REQUIRE(qps_only["index_info"].empty());

    config.search_param = R"({"hgraph":{"ef_search":0}})";
    REQUIRE_THROWS_WITH(
        vsag::eval::EvaluateSearch(index, dataset, config),
        Catch::Matchers::ContainsSubstring("query error: ef_search(0) must be at least 1"));

    config.search_param = R"({"hgraph":{"ef_search":8}})";
    config.top_k = 0;
    REQUIRE_THROWS_WITH(vsag::eval::EvaluateSearch(index, dataset, config),
                        "evaluation top_k must be positive");

    config.top_k = 1;
    config.search_mode = "range";
    REQUIRE_THROWS_WITH(vsag::eval::EvaluateSearch(index, dataset, config),
                        "in-memory evaluation supports only knn search mode");

    config.search_mode = "knn";
    config.num_threads_searching = 0;
    REQUIRE_THROWS_WITH(vsag::eval::EvaluateSearch(index, dataset, config),
                        "evaluation search thread count must be positive");

    config.num_threads_searching = 2;
    config.search_query_count = 0;
    REQUIRE_THROWS_WITH(vsag::eval::EvaluateSearch(index, dataset, config),
                        "evaluation search query count must be positive");

    config.search_query_count = 2;
    config.top_k = 1;
    config.enable_recall = true;
    auto without_ground_truth = EvalDataset::FromDatasets(base, queries, nullptr, "l2");
    REQUIRE_THROWS_WITH(vsag::eval::EvaluateSearch(index, without_ground_truth, config),
                        "evaluation ground truth must contain at least top_k neighbors per query");

    config.top_k = 2;
    REQUIRE_THROWS_WITH(vsag::eval::EvaluateSearch(index, dataset, config),
                        "evaluation ground truth must contain at least top_k neighbors per query");
}

TEST_CASE("EvalCase builds and searches an in-memory dataset with original ids",
          "[ut][eval_dataset]") {
    constexpr int64_t dim = 2;
    std::vector<float> base_vectors{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> base_ids{101, 7, 42};
    auto base = vsag::Dataset::Make();
    base->NumElements(3)
        ->Dim(dim)
        ->Ids(base_ids.data())
        ->Float32Vectors(base_vectors.data())
        ->Owner(false);

    std::vector<float> query_vectors{5.0F, 6.0F};
    auto queries = vsag::Dataset::Make();
    queries->NumElements(1)->Dim(dim)->Float32Vectors(query_vectors.data())->Owner(false);

    std::vector<int64_t> ground_truth_ids{42};
    auto ground_truth = vsag::Dataset::Make();
    ground_truth->NumElements(1)->Dim(1)->Ids(ground_truth_ids.data())->Owner(false);
    auto dataset = EvalDataset::FromDatasets(base, queries, ground_truth, "l2");

    vsag::eval::EvalConfig config;
    config.index_name = "brute_force";
    config.index_path = "/tmp/eval_dataset_memory.index";
    config.build_param = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 2,
        "index_param": {
            "base_quantization_type": "fp32",
            "store_raw_vector": true
        }
    })";
    config.top_k = 1;
    config.search_query_count = 1;
    config.enable_memory = false;
    std::remove(config.index_path.c_str());

    auto build = vsag::eval::EvalCase::MakeInstance(config, "build", dataset);
    REQUIRE(build->Run()["action"] == "build");
    auto search = vsag::eval::EvalCase::MakeInstance(config, "search", dataset);
    const auto result = search->Run();
    REQUIRE(result["action"] == "search");
    REQUIRE(result["recall_avg"].get<double>() == 1.0);
    REQUIRE(result["statistics_query_count"].get<uint64_t>() == 1);
    REQUIRE(std::isfinite(result["qps"].get<double>()));
    REQUIRE(std::isfinite(result["latency_avg(ms)"].get<double>()));
    REQUIRE(std::isfinite(result["latency_detail(ms)"]["p99"].get<double>()));

    std::vector<float> concurrent_query_vectors{5.0F, 6.0F, 1.0F, 2.0F};
    auto concurrent_queries = vsag::Dataset::Make();
    concurrent_queries->NumElements(2)
        ->Dim(dim)
        ->Float32Vectors(concurrent_query_vectors.data())
        ->Owner(false);
    std::vector<int64_t> concurrent_ground_truth_ids{42, 101};
    auto concurrent_ground_truth = vsag::Dataset::Make();
    concurrent_ground_truth->NumElements(2)
        ->Dim(1)
        ->Ids(concurrent_ground_truth_ids.data())
        ->Owner(false);
    auto concurrent_dataset =
        EvalDataset::FromDatasets(base, concurrent_queries, concurrent_ground_truth, "l2");
    config.search_query_count = 2;
    config.num_threads_searching = 2;
    auto concurrent_search =
        vsag::eval::EvalCase::MakeInstance(config, "search", concurrent_dataset);
    const auto concurrent_result = concurrent_search->Run();
    REQUIRE(concurrent_result["recall_avg"].get<double>() == 1.0);
    REQUIRE(concurrent_result["statistics_query_count"].get<uint64_t>() == 2);
    REQUIRE(std::isfinite(concurrent_result["qps"].get<double>()));
    REQUIRE(std::isfinite(concurrent_result["latency_avg(ms)"].get<double>()));
    REQUIRE(std::isfinite(concurrent_result["latency_detail(ms)"]["p99"].get<double>()));
    std::remove(config.index_path.c_str());
}

TEST_CASE("EvalDataset sparse round-trip without token sequences", "[ut][eval_dataset]") {
    auto path = TempPath("nosqry");
    {
        auto ds = BuildSparseDataset(/*with_token_sequences=*/false);
        EvalDataset::Save(ds, path);
    }

    // Verify the optional datasets are absent on disk.
    {
        H5::H5File file(path, H5F_ACC_RDONLY);
        H5::Group root = file.openGroup("/");
        bool has_train_token = false;
        bool has_test_token = false;
        hsize_t numObj = root.getNumObjs();
        for (unsigned i = 0; i < numObj; ++i) {
            std::string n = root.getObjnameByIdx(i);
            if (n == "train_token_sequences")
                has_train_token = true;
            if (n == "test_token_sequences")
                has_test_token = true;
        }
        REQUIRE_FALSE(has_train_token);
        REQUIRE_FALSE(has_test_token);
    }

    // Loading must succeed and token_sequence_ fields stay empty.
    auto loaded = EvalDataset::Load(path);
    REQUIRE(loaded->GetVectorType() == vsag::SPARSE_VECTORS);
    REQUIRE(loaded->GetNumberOfBase() == 3);
    REQUIRE(loaded->GetNumberOfQuery() == 2);
    const auto* train = static_cast<const SparseVector*>(loaded->GetTrain());
    REQUIRE(loaded->GetTrainIds()[0] == 0);
    REQUIRE(loaded->GetTrainIds()[2] == 2);
    REQUIRE(loaded->GetOneTrainById(1) == train + 1);
    for (int i = 0; i < 3; ++i) {
        REQUIRE(train[i].token_seq_len_ == 0);
        REQUIRE(train[i].token_sequence_ == nullptr);
    }
    std::remove(path.c_str());
}

TEST_CASE("EvalDataset sparse round-trip preserves all-empty records", "[ut][eval_dataset]") {
    auto path = TempPath("all_empty_sparse");
    {
        auto ds = BuildSparseDataset(/*with_token_sequences=*/false, /*all_empty=*/true);
        EvalDataset::Save(ds, path);
    }

    auto loaded = EvalDataset::Load(path);
    REQUIRE(loaded->GetVectorType() == vsag::SPARSE_VECTORS);
    REQUIRE(loaded->GetNumberOfBase() == 3);
    REQUIRE(loaded->GetNumberOfQuery() == 2);
    REQUIRE(loaded->GetDim() == 0);
    const auto* train = static_cast<const SparseVector*>(loaded->GetTrain());
    const auto* test = static_cast<const SparseVector*>(loaded->GetTest());
    for (uint64_t i = 0; i < 3; ++i) {
        REQUIRE(train[i].len_ == 0);
    }
    for (uint64_t i = 0; i < 2; ++i) {
        REQUIRE(test[i].len_ == 0);
    }
    std::remove(path.c_str());
}

TEST_CASE("EvalDataset sparse round-trip with token sequences", "[ut][eval_dataset]") {
    auto path = TempPath("withseq");
    {
        auto ds = BuildSparseDataset(/*with_token_sequences=*/true);
        EvalDataset::Save(ds, path);
    }

    auto loaded = EvalDataset::Load(path);
    REQUIRE(loaded->GetVectorType() == vsag::SPARSE_VECTORS);
    REQUIRE(loaded->GetNumberOfBase() == 3);
    REQUIRE(loaded->GetNumberOfQuery() == 2);

    const auto* train = static_cast<const SparseVector*>(loaded->GetTrain());
    REQUIRE(train[0].token_seq_len_ == 4);
    REQUIRE(train[0].token_sequence_[0] == 10u);
    REQUIRE(train[0].token_sequence_[2] == 10u);  // duplicates preserved
    REQUIRE(train[1].token_seq_len_ == 2);
    REQUIRE(train[1].token_sequence_[0] == 42u);
    REQUIRE(train[1].token_sequence_[1] == 42u);
    REQUIRE(train[2].token_seq_len_ == 0);
    REQUIRE(train[2].token_sequence_ == nullptr);

    const auto* test = static_cast<const SparseVector*>(loaded->GetTest());
    REQUIRE(test[0].token_seq_len_ == 1);
    REQUIRE(test[0].token_sequence_[0] == 99u);
    REQUIRE(test[1].token_seq_len_ == 3);
    REQUIRE(test[1].token_sequence_[0] == 1u);
    REQUIRE(test[1].token_sequence_[2] == 3u);
    std::remove(path.c_str());
}

TEST_CASE("EvalDataset legacy sparse file without token_sequences key still loads",
          "[ut][eval_dataset]") {
    // Synthesize a minimal legacy sparse HDF5 file by hand (no
    // train_token_sequences / test_token_sequences keys), and verify
    // EvalDataset::Load still succeeds.
    auto path = TempPath("legacy");
    {
        H5::H5File file(path, H5F_ACC_TRUNC);

        H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
        {
            auto a = file.createAttribute("type", str_type, H5::DataSpace(H5S_SCALAR));
            std::string v = "sparse";
            a.write(str_type, v);
        }
        {
            auto a = file.createAttribute("distance", str_type, H5::DataSpace(H5S_SCALAR));
            std::string v = "ip";
            a.write(str_type, v);
        }

        // Encode 1 train and 1 test sparse vector each. Layout:
        //   uint32 len | uint32 ids[len] | float vals[len]
        auto encode =
            [](uint32_t len, const std::vector<uint32_t>& ids, const std::vector<float>& vals) {
                std::vector<char> buf(sizeof(uint32_t) + len * (sizeof(uint32_t) + sizeof(float)));
                char* p = buf.data();
                std::memcpy(p, &len, sizeof(uint32_t));
                p += sizeof(uint32_t);
                std::memcpy(p, ids.data(), len * sizeof(uint32_t));
                p += len * sizeof(uint32_t);
                std::memcpy(p, vals.data(), len * sizeof(float));
                return buf;
            };
        auto train_buf = encode(2, {1, 4}, {0.5f, 1.0f});
        auto test_buf = encode(1, {2}, {0.7f});
        {
            hsize_t dims[1] = {static_cast<hsize_t>(train_buf.size())};
            H5::DataSpace sp(1, dims);
            auto ds = file.createDataSet("/train", H5::PredType::ALPHA_I8, sp);
            ds.write(train_buf.data(), H5::PredType::NATIVE_CHAR);
        }
        {
            hsize_t dims[1] = {static_cast<hsize_t>(test_buf.size())};
            H5::DataSpace sp(1, dims);
            auto ds = file.createDataSet("/test", H5::PredType::ALPHA_I8, sp);
            ds.write(test_buf.data(), H5::PredType::NATIVE_CHAR);
        }
        // Minimal ground truth: shape (1, 1).
        {
            hsize_t dims[2] = {1, 1};
            H5::DataSpace sp(2, dims);
            int64_t nb[1] = {0};
            auto ds = file.createDataSet("/neighbors", H5::PredType::NATIVE_INT64, sp);
            ds.write(nb, H5::PredType::NATIVE_INT64);
        }
        {
            hsize_t dims[2] = {1, 1};
            H5::DataSpace sp(2, dims);
            float dv[1] = {0.0f};
            auto ds = file.createDataSet("/distances", H5::PredType::NATIVE_FLOAT, sp);
            ds.write(dv, H5::PredType::NATIVE_FLOAT);
        }
    }

    auto loaded = EvalDataset::Load(path);
    REQUIRE(loaded->GetVectorType() == vsag::SPARSE_VECTORS);
    REQUIRE(loaded->GetNumberOfBase() == 1);
    REQUIRE(loaded->GetNumberOfQuery() == 1);
    const auto* train = static_cast<const SparseVector*>(loaded->GetTrain());
    REQUIRE(train[0].token_seq_len_ == 0);
    REQUIRE(train[0].token_sequence_ == nullptr);
    // The reader rebuilds offsets from the byte stream when the file does
    // not store them. For a single 2-nnz record the layout is
    //   [u32 len=2][u32 ids[2]][f32 vals[2]] = 4 + 8 + 8 = 20 bytes.
    const auto& train_off = loaded->GetSparseTrainOffsets();
    REQUIRE(train_off.size() == 2);
    REQUIRE(train_off[0] == 0u);
    REQUIRE(train_off[1] == 20u);
    std::remove(path.c_str());
}

TEST_CASE("EvalDataset sparse round-trip writes record-offset indexes", "[ut][eval_dataset]") {
    auto path = TempPath("offsets");
    {
        auto ds = BuildSparseDataset(/*with_token_sequences=*/true);
        EvalDataset::Save(ds, path);
    }

    // Verify the four offsets datasets exist on disk and have the correct
    // contents. The /train byte layout is:
    //   record 0 (len=2): 4 + 8 + 8 = 20 bytes -> starts at 0
    //   record 1 (len=1): 4 + 4 + 4 = 12 bytes -> starts at 20
    //   record 2 (len=0): 4 bytes              -> starts at 32
    //   total                                  = 36 bytes
    {
        H5::H5File file(path, H5F_ACC_RDONLY);
        std::vector<uint64_t> train_off(4);
        H5::DataSet ds = file.openDataSet("/train_offsets");
        ds.read(train_off.data(), H5::PredType::NATIVE_UINT64);
        REQUIRE(train_off == std::vector<uint64_t>{0, 20, 32, 36});

        std::vector<uint64_t> test_off(3);
        H5::DataSet tds = file.openDataSet("/test_offsets");
        tds.read(test_off.data(), H5::PredType::NATIVE_UINT64);
        // record 0 (len=2): 20, record 1 (len=1): 12 -> offsets {0, 20, 32}
        REQUIRE(test_off == std::vector<uint64_t>{0, 20, 32});

        std::vector<uint64_t> train_token_off(4);
        H5::DataSet ttds = file.openDataSet("/train_token_sequences_offsets");
        ttds.read(train_token_off.data(), H5::PredType::NATIVE_UINT64);
        // train tokens: seq_lens 4, 2, 0
        //   record 0: 4 + 16 = 20 bytes  -> 0
        //   record 1: 4 + 8  = 12 bytes  -> 20
        //   record 2: 4      = 4  bytes  -> 32
        //   total                        = 36 bytes
        REQUIRE(train_token_off == std::vector<uint64_t>{0, 20, 32, 36});

        std::vector<uint64_t> test_token_off(3);
        H5::DataSet etds = file.openDataSet("/test_token_sequences_offsets");
        etds.read(test_token_off.data(), H5::PredType::NATIVE_UINT64);
        // test tokens: seq_lens 1, 3
        //   record 0: 4 + 4  = 8  bytes -> 0
        //   record 1: 4 + 12 = 16 bytes -> 8
        //   total                       = 24 bytes
        REQUIRE(test_token_off == std::vector<uint64_t>{0, 8, 24});
    }

    // Loading must succeed and surface the offsets through the public API.
    auto loaded = EvalDataset::Load(path);
    REQUIRE(loaded->GetSparseTrainOffsets() == std::vector<uint64_t>{0, 20, 32, 36});
    REQUIRE(loaded->GetSparseTestOffsets() == std::vector<uint64_t>{0, 20, 32});
    REQUIRE(loaded->GetTrainTokenSequenceOffsets() == std::vector<uint64_t>{0, 20, 32, 36});
    REQUIRE(loaded->GetTestTokenSequenceOffsets() == std::vector<uint64_t>{0, 8, 24});
    std::remove(path.c_str());
}

TEST_CASE("EvalDataset rejects sparse files with token_sequences missing offsets",
          "[ut][eval_dataset]") {
    // Contract: whenever a *_token_sequences byte stream is present, the
    // companion *_token_sequences_offsets dataset MUST also exist. Files
    // that violate this invariant are considered malformed.
    auto path = TempPath("seq_no_off");
    {
        auto ds = BuildSparseDataset(/*with_token_sequences=*/true);
        EvalDataset::Save(ds, path);
    }
    // Delete only /train_token_sequences_offsets, keep the byte stream.
    {
        H5::H5File file(path, H5F_ACC_RDWR);
        file.unlink("/train_token_sequences_offsets");
    }
    REQUIRE_THROWS(EvalDataset::Load(path));
    std::remove(path.c_str());
}

TEST_CASE("EvalDataset rejects sparse files with orphan token_sequences_offsets",
          "[ut][eval_dataset]") {
    // The reverse direction: a *_token_sequences_offsets dataset must not
    // appear without its byte-stream counterpart.
    auto path = TempPath("orphan_off");
    {
        auto ds = BuildSparseDataset(/*with_token_sequences=*/true);
        EvalDataset::Save(ds, path);
    }
    {
        H5::H5File file(path, H5F_ACC_RDWR);
        file.unlink("/test_token_sequences");
    }
    REQUIRE_THROWS(EvalDataset::Load(path));
    std::remove(path.c_str());
}

TEST_CASE("EvalDataset rejects sparse files with corrupted offsets", "[ut][eval_dataset]") {
    auto path = TempPath("badoff");
    {
        auto ds = BuildSparseDataset(/*with_token_sequences=*/false);
        EvalDataset::Save(ds, path);
    }
    // Corrupt /train_offsets: replace the sentinel with a wrong total size.
    {
        H5::H5File file(path, H5F_ACC_RDWR);
        H5::DataSet ds = file.openDataSet("/train_offsets");
        std::vector<uint64_t> tmp(4);
        ds.read(tmp.data(), H5::PredType::NATIVE_UINT64);
        tmp.back() += 7;  // sentinel no longer matches byte stream length
        ds.write(tmp.data(), H5::PredType::NATIVE_UINT64);
    }
    REQUIRE_THROWS(EvalDataset::Load(path));
    std::remove(path.c_str());
}
