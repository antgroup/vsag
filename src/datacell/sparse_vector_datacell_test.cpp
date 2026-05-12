
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

#include <unordered_map>

#include "datacell/sparse_vector_datacell_parameter.h"
#include "framework/test_thread_pool.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "quantization/sparse_quantization/sparse_quantizer_parameter.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"

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
    JsonType parsed_json = JsonType::Parse(param_str);
    auto param = std::make_shared<SparseVectorDataCellParameter>();
    param->FromJson(parsed_json);
    IndexCommonParam index_common_param;
    index_common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    index_common_param.metric_ = MetricType::METRIC_TYPE_IP;
    index_common_param.dim_ = max_dim;
    auto data_cell = FlattenInterface::MakeInstance(param, index_common_param);
    REQUIRE(data_cell->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_SPARSE);
    REQUIRE(data_cell->GetMetricType() == MetricType::METRIC_TYPE_IP);

    uint64_t base_count = 1000;
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
        std::vector<float> dist(base_count);
        data_cell->Query(dist.data(), computer, idx.data(), 1);
        data_cell->Query(dist.data() + 1, computer, idx.data() + 1, base_count - 1);
        for (int i = 0; i < base_count; ++i) {
            fixtures::dist_t distance =
                fixtures::GetSparseDistance(query_sparse_vectors[0], sparse_vectors[i]);
            REQUIRE(distance == dist[i]);
        }
    }
    SECTION("serialize and deserialize") {
        auto new_data_cell = FlattenInterface::MakeInstance(param, index_common_param);
        test_serializion(*data_cell, *new_data_cell);
        auto computer = new_data_cell->FactoryComputer(query_sparse_vectors.data());
        std::vector<float> dist(base_count);
        new_data_cell->Query(dist.data(), computer, idx.data(), 1);
        new_data_cell->Query(dist.data() + 1, computer, idx.data() + 1, base_count - 1);
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
    JsonType parsed_json = JsonType::Parse(param_str);
    auto param = std::make_shared<SparseVectorDataCellParameter>();
    param->FromJson(parsed_json);
    IndexCommonParam index_common_param;
    index_common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    index_common_param.metric_ = MetricType::METRIC_TYPE_IP;
    index_common_param.dim_ = max_dim;
    auto data_cell = FlattenInterface::MakeInstance(param, index_common_param);
    REQUIRE(data_cell->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_SPARSE);
    REQUIRE(data_cell->GetMetricType() == MetricType::METRIC_TYPE_IP);

    uint64_t base_count = 1000;
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
        std::vector<float> dist(base_count);
        data_cell->Query(dist.data(), computer, idx.data(), 1);
        data_cell->Query(dist.data() + 1, computer, idx.data() + 1, base_count - 1);
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

TEST_CASE("SparseDataCell Direct Read and Sparse Vector Retrieval", "[ut][SparseDataCell]") {
    constexpr const char* param_str = R"(
        {
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "sparse"
            }
        }
        )";
    auto param = std::make_shared<SparseVectorDataCellParameter>();
    param->FromJson(JsonType::Parse(param_str));
    IndexCommonParam index_common_param;
    index_common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    index_common_param.metric_ = MetricType::METRIC_TYPE_IP;
    index_common_param.dim_ = 16;
    auto data_cell = FlattenInterface::MakeInstance(param, index_common_param);

    auto sparse_vectors = fixtures::GenerateSparseVectors(4, index_common_param.dim_);
    data_cell->Train(sparse_vectors.data(), sparse_vectors.size());
    data_cell->BatchInsertVector(sparse_vectors.data(), sparse_vectors.size());

    bool need_release = true;
    const auto* codes = data_cell->GetCodesById(2, need_release);
    REQUIRE_FALSE(need_release);
    auto len = *reinterpret_cast<const uint32_t*>(codes);
    REQUIRE(len == sparse_vectors[2].len_);

    auto code_size = sizeof(uint32_t) * (2 * sparse_vectors[2].len_ + 1);
    std::vector<uint8_t> copied_codes(code_size);
    REQUIRE(data_cell->GetCodesById(2, copied_codes.data()));
    REQUIRE(std::memcmp(codes, copied_codes.data(), code_size) == 0);

    SparseVector retrieved;
    data_cell->GetSparseVectorByInnerId(2, &retrieved, index_common_param.allocator_.get());
    REQUIRE(retrieved.len_ == sparse_vectors[2].len_);
    std::unordered_map<uint32_t, float> expected;
    for (uint32_t i = 0; i < sparse_vectors[2].len_; ++i) {
        expected[sparse_vectors[2].ids_[i]] = sparse_vectors[2].vals_[i];
    }
    for (uint32_t i = 0; i < retrieved.len_; ++i) {
        REQUIRE(expected.count(retrieved.ids_[i]) == 1);
        REQUIRE(retrieved.vals_[i] == expected[retrieved.ids_[i]]);
    }
    index_common_param.allocator_->Deallocate(retrieved.ids_);
    index_common_param.allocator_->Deallocate(retrieved.vals_);

    auto new_data_cell = FlattenInterface::MakeInstance(param, index_common_param);
    test_serializion(*data_cell, *new_data_cell);
    SparseVector restored;
    new_data_cell->GetSparseVectorByInnerId(2, &restored, index_common_param.allocator_.get());
    REQUIRE(restored.len_ == sparse_vectors[2].len_);
    for (uint32_t i = 0; i < restored.len_; ++i) {
        REQUIRE(expected.count(restored.ids_[i]) == 1);
        REQUIRE(restored.vals_[i] == expected[restored.ids_[i]]);
    }
    index_common_param.allocator_->Deallocate(restored.ids_);
    index_common_param.allocator_->Deallocate(restored.vals_);

    for (auto& item : sparse_vectors) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SparseDataCell Rejects Old Format", "[ut][SparseDataCell]") {
    constexpr const char* param_str = R"(
        {
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "sparse"
            }
        }
        )";
    auto param = std::make_shared<SparseVectorDataCellParameter>();
    param->FromJson(JsonType::Parse(param_str));
    IndexCommonParam index_common_param;
    index_common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    index_common_param.metric_ = MetricType::METRIC_TYPE_IP;
    index_common_param.dim_ = 16;
    auto data_cell = FlattenInterface::MakeInstance(param, index_common_param);

    std::stringstream ss;
    IOStreamWriter writer(ss);
    uint32_t old_format_marker = 1;
    StreamWriter::WriteObj(writer, old_format_marker);
    ss.seekg(0, std::ios::beg);
    IOStreamReader reader(ss);
    REQUIRE_THROWS_AS(data_cell->Deserialize(reader), VsagException);
}

}  // namespace vsag
