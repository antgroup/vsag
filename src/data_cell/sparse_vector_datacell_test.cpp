
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

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "data_cell/sparse_vector_datacell_parameter.h"
#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "index/index_common_param.h"
#include "quantization/sparse_quantization/sparse_quantizer_parameter.h"
#include "thread_pool.h"

namespace vsag {

TEST_CASE("SparseDataCell Basic Test", "[ut][SparseDataCell] ") {
    std::string io_type = GENERATE("memory_io", "block_memory_io");
    constexpr const char* param_temp =
        R"(
        {{
            "io_params": {{
                "type": "{}"
            }},
            "quantization_params": {{
                "type": "sparse"
            }}
        }}
        )";
    int64_t max_dim = 100;
    auto param_str = fmt::format(param_temp, io_type);
    JsonType parsed_json = JsonType::parse(param_str);
    auto param = std::make_shared<SparseVectorDataCellParameter>();
    param->FromJson(parsed_json);
    IndexCommonParam index_common_param;
    index_common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    index_common_param.metric_ = MetricType::METRIC_TYPE_IP;
    index_common_param.dim_ = max_dim;
    auto data_cell = FlattenInterface::MakeInstance(param, index_common_param);
    REQUIRE(data_cell->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_SPARSE);
    REQUIRE(data_cell->GetMetricType() == MetricType::METRIC_TYPE_IP);

    size_t base_count = 1000;
    auto sparse_vectors = fixtures::GenerateSparseVectors(base_count, max_dim);
    std::vector<InnerIdType> idx(base_count);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937(47));
    auto half_count = base_count / 2;
    data_cell->Train(sparse_vectors.data(), base_count);
    for (int i = 0; i < half_count; ++i) {
        data_cell->InsertVector(sparse_vectors.data() + i, idx[i]);
    }
    data_cell->BatchInsertVector(
        sparse_vectors.data() + half_count, half_count / 2, idx.data() + half_count);
    data_cell->BatchInsertVector(sparse_vectors.data() + half_count + half_count / 2,
                                 half_count / 2,
                                 idx.data() + half_count + half_count / 2);

    for (int i = 0; i < base_count - 1; ++i) {
        fixtures::dist_t distance = data_cell->ComputePairVectors(idx[i], idx[i + 1]);
        REQUIRE(distance == fixtures::GetSparseDistance(sparse_vectors[i], sparse_vectors[i + 1]));
    }
    auto query_sparse_vectors = fixtures::GenerateSparseVectors(1, 100);
    SECTION("accuracy") {
        auto computer = data_cell->FactoryComputer(query_sparse_vectors.data());
        std::shared_ptr<float[]> dist = std::shared_ptr<float[]>(new float[base_count]);
        data_cell->Query(dist.get(), computer, idx.data(), 1);
        data_cell->Query(dist.get() + 1, computer, idx.data() + 1, base_count - 1);
        for (int i = 0; i < base_count; ++i) {
            fixtures::dist_t distance =
                fixtures::GetSparseDistance(query_sparse_vectors[0], sparse_vectors[i]);
            REQUIRE(distance == dist[i]);
        }
    }
    SECTION("serialize and deserialize") {
        fixtures::TempDir dir("flatten");
        auto path = dir.GenerateRandomFile();
        std::ofstream outfile(path.c_str(), std::ios::binary);
        IOStreamWriter writer(outfile);
        data_cell->Serialize(writer);
        outfile.close();

        auto new_data_cell = FlattenInterface::MakeInstance(param, index_common_param);

        std::ifstream infile(path.c_str(), std::ios::binary);
        IOStreamReader reader(infile);
        new_data_cell->Deserialize(reader);
        infile.close();
        auto computer = new_data_cell->FactoryComputer(query_sparse_vectors.data());
        std::shared_ptr<float[]> dist = std::shared_ptr<float[]>(new float[base_count]);
        new_data_cell->Query(dist.get(), computer, idx.data(), 1);
        new_data_cell->Query(dist.get() + 1, computer, idx.data() + 1, base_count - 1);
        for (int i = 0; i < base_count; ++i) {
            fixtures::dist_t distance =
                fixtures::GetSparseDistance(query_sparse_vectors[0], sparse_vectors[i]);
            REQUIRE(distance == dist[i]);
        }
    }
    for (auto& item : sparse_vectors) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    for (auto& item : query_sparse_vectors) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SparseDataCell Concurrent Test", "[ut][SparseDataCell][concurrent] ") {
    std::string io_type = GENERATE("memory_io", "block_memory_io");
    constexpr const char* param_temp =
        R"(
        {{
            "io_params": {{
                "type": "{}"
            }},
            "quantization_params": {{
                "type": "sparse"
            }}
        }}
        )";
    int64_t max_dim = 100;
    auto param_str = fmt::format(param_temp, io_type);
    JsonType parsed_json = JsonType::parse(param_str);
    auto param = std::make_shared<SparseVectorDataCellParameter>();
    param->FromJson(parsed_json);
    IndexCommonParam index_common_param;
    index_common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    index_common_param.metric_ = MetricType::METRIC_TYPE_IP;
    index_common_param.dim_ = max_dim;
    auto data_cell = FlattenInterface::MakeInstance(param, index_common_param);
    REQUIRE(data_cell->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_SPARSE);
    REQUIRE(data_cell->GetMetricType() == MetricType::METRIC_TYPE_IP);

    size_t base_count = 1000;
    auto sparse_vectors = fixtures::GenerateSparseVectors(base_count, max_dim);
    std::vector<InnerIdType> idx(base_count);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937(47));
    data_cell->Train(sparse_vectors.data(), base_count);
    data_cell->Resize(base_count);
    fixtures::ThreadPool thread_pool(4);
    std::vector<std::future<void>> futures;
    futures.push_back(thread_pool.enqueue([&]() {
        data_cell->BatchInsertVector(sparse_vectors.data(), base_count / 2, idx.data());
    }));
    for (int i = base_count / 2; i < base_count; ++i) {
        futures.push_back(thread_pool.enqueue(
            [&, i]() { data_cell->InsertVector(sparse_vectors.data() + i, idx[i]); }));
    }
    for (auto& future : futures) {
        future.get();
    }

    for (int i = 0; i < base_count - 1; ++i) {
        fixtures::dist_t distance = data_cell->ComputePairVectors(idx[i], idx[i + 1]);
        REQUIRE(distance == fixtures::GetSparseDistance(sparse_vectors[i], sparse_vectors[i + 1]));
    }
    auto query_sparse_vectors = fixtures::GenerateSparseVectors(1, 100);
    SECTION("accuracy") {
        auto computer = data_cell->FactoryComputer(query_sparse_vectors.data());
        std::shared_ptr<float[]> dist = std::shared_ptr<float[]>(new float[base_count]);
        data_cell->Query(dist.get(), computer, idx.data(), 1);
        data_cell->Query(dist.get() + 1, computer, idx.data() + 1, base_count - 1);
        for (int i = 0; i < base_count; ++i) {
            fixtures::dist_t distance =
                fixtures::GetSparseDistance(query_sparse_vectors[0], sparse_vectors[i]);
            REQUIRE(distance == dist[i]);
        }
    }
    for (auto& item : sparse_vectors) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    for (auto& item : query_sparse_vectors) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

}  // namespace vsag
