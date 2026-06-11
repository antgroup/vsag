
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
    SECTION("get sparse vector by inner id") {
        for (int i = 0; i < 10; ++i) {
            SparseVector data;
            data_cell->GetSparseVectorByInnerId(idx[i], &data, index_common_param.allocator_.get());
            std::vector<std::pair<uint32_t, float>> expected;
            std::vector<std::pair<uint32_t, float>> actual;
            for (uint32_t j = 0; j < sparse_vectors[i].len_; ++j) {
                expected.emplace_back(sparse_vectors[i].ids_[j], sparse_vectors[i].vals_[j]);
            }
            for (uint32_t j = 0; j < data.len_; ++j) {
                actual.emplace_back(data.ids_[j], data.vals_[j]);
            }
            std::sort(expected.begin(), expected.end());
            REQUIRE(actual == expected);
            index_common_param.allocator_->Deallocate(data.ids_);
            index_common_param.allocator_->Deallocate(data.vals_);
        }
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

namespace {

// Rebuilds the byte layout produced by SparseVectorDataCell::Serialize before
// the 64-bit offset fix landed (the v1 layout):
//   [FlattenInterface header: total_count, max_capacity, code_size]
//   [uint32 current_offset_]
//   [uint64 io_size][io payload]
//   [uint64 offset_io_size][N × {uint32 offset, uint32 size}]
//   [quantizer payload]
// Used to exercise the deserialization-side compatibility path against bytes
// bit-identical to what a pre-fix VSAG would have emitted.
void
write_legacy_format(std::stringstream& legacy_stream,
                    uint32_t total_count,
                    uint32_t max_capacity,
                    uint32_t code_size,
                    const std::string& io_bytes,
                    const std::vector<std::pair<uint32_t, uint32_t>>& legacy_entries,
                    const std::string& quantizer_bytes) {
    auto write = [&](const void* data, uint64_t n) {
        legacy_stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    };
    write(&total_count, sizeof(total_count));
    write(&max_capacity, sizeof(max_capacity));
    write(&code_size, sizeof(code_size));
    uint32_t current_offset_v1 = 0;
    for (const auto& [off, sz] : legacy_entries) {
        current_offset_v1 = std::max(current_offset_v1, off + sz);
    }
    write(&current_offset_v1, sizeof(current_offset_v1));
    uint64_t io_size = io_bytes.size();
    write(&io_size, sizeof(io_size));
    write(io_bytes.data(), io_size);
    uint64_t legacy_offset_io_size = legacy_entries.size() * 2 * sizeof(uint32_t);
    write(&legacy_offset_io_size, sizeof(legacy_offset_io_size));
    for (const auto& [off, sz] : legacy_entries) {
        write(&off, sizeof(off));
        write(&sz, sizeof(sz));
    }
    write(quantizer_bytes.data(), quantizer_bytes.size());
}

}  // namespace

TEST_CASE("SparseDataCell Backward Compat Deserialize", "[ut][SparseDataCell]") {
    constexpr const char* param_str = R"({
        "io_params": {"type": "memory_io"},
        "quantization_params": {"type": "sparse"}
    })";
    JsonType parsed_json = JsonType::Parse(param_str);
    auto param = std::make_shared<SparseVectorDataCellParameter>();
    param->FromJson(parsed_json);
    IndexCommonParam index_common_param;
    index_common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    index_common_param.metric_ = MetricType::METRIC_TYPE_IP;
    index_common_param.dim_ = 100;

    auto src = FlattenInterface::MakeInstance(param, index_common_param);
    constexpr uint32_t base_count = 16;
    auto sparse_vectors = fixtures::GenerateSparseVectors(base_count, 100);
    src->Train(sparse_vectors.data(), base_count);
    for (uint32_t i = 0; i < base_count; ++i) {
        src->InsertVector(sparse_vectors.data() + i, i);
    }

    std::string io_bytes;
    std::vector<std::pair<uint32_t, uint32_t>> entries;
    for (uint32_t i = 0; i < base_count; ++i) {
        bool need_release{false};
        const auto* codes = src->GetCodesById(i, need_release);
        uint32_t len = *reinterpret_cast<const uint32_t*>(codes);
        uint32_t blob_size = (len * 2 + 1) * sizeof(uint32_t);
        entries.emplace_back(static_cast<uint32_t>(io_bytes.size()), blob_size);
        io_bytes.append(reinterpret_cast<const char*>(codes), blob_size);
        if (need_release) {
            src->Release(codes);
        }
    }

    // Grab the quantizer's serialized bytes from the tail of a full new-format dump.
    std::stringstream full_ss;
    IOStreamWriter full_writer(full_ss);
    src->Serialize(full_writer);
    std::string full = full_ss.str();
    // new-format prefix: FlattenInterface(12) + sentinel(4) + version(4) + current_offset(8)
    //                  + io_section(8 + io_bytes.size())
    //                  + offset_section(8 + N * sizeof(DocLocation) where the packed
    //                    DocLocation is 12 bytes: uint64 offset + uint32 size, no padding).
    const uint64_t doc_location_size_on_disk = 12;
    const uint64_t prefix =
        12 + 4 + 4 + 8 + (8 + io_bytes.size()) + (8 + base_count * doc_location_size_on_disk);
    REQUIRE(full.size() > prefix);
    std::string quantizer_bytes = full.substr(prefix);

    std::stringstream legacy_ss;
    write_legacy_format(legacy_ss,
                        /*total_count=*/base_count,
                        /*max_capacity=*/base_count,
                        /*code_size=*/0,
                        io_bytes,
                        entries,
                        quantizer_bytes);

    auto loaded = FlattenInterface::MakeInstance(param, index_common_param);
    IOStreamReader reader(legacy_ss);
    loaded->Deserialize(reader);

    auto query_sparse_vectors = fixtures::GenerateSparseVectors(1, 100);
    auto computer_src = src->FactoryComputer(query_sparse_vectors.data());
    auto computer_loaded = loaded->FactoryComputer(query_sparse_vectors.data());
    std::vector<InnerIdType> all_ids(base_count);
    std::iota(all_ids.begin(), all_ids.end(), 0);
    std::vector<float> dist_src(base_count);
    std::vector<float> dist_loaded(base_count);
    src->Query(dist_src.data(), computer_src, all_ids.data(), base_count);
    loaded->Query(dist_loaded.data(), computer_loaded, all_ids.data(), base_count);
    for (uint32_t i = 0; i < base_count; ++i) {
        REQUIRE(dist_src[i] == dist_loaded[i]);
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

TEST_CASE("SparseDataCell New Format Sentinel", "[ut][SparseDataCell]") {
    // Confirms the post-fix Serialize() emits the 64-bit-capable v2 layout (a
    // 0xFFFFFFFF sentinel right after the FlattenInterface header) and that the
    // resulting bytes round-trip losslessly. Combined with the backward-compat
    // case above, this fully covers the format-detection branches.
    constexpr const char* param_str = R"({
        "io_params": {"type": "block_memory_io"},
        "quantization_params": {"type": "sparse"}
    })";
    JsonType parsed_json = JsonType::Parse(param_str);
    auto param = std::make_shared<SparseVectorDataCellParameter>();
    param->FromJson(parsed_json);
    IndexCommonParam index_common_param;
    index_common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    index_common_param.metric_ = MetricType::METRIC_TYPE_IP;
    index_common_param.dim_ = 100;

    auto cell = FlattenInterface::MakeInstance(param, index_common_param);
    constexpr uint32_t base_count = 32;
    auto sparse_vectors = fixtures::GenerateSparseVectors(base_count, 100);
    cell->Train(sparse_vectors.data(), base_count);
    for (uint32_t i = 0; i < base_count; ++i) {
        cell->InsertVector(sparse_vectors.data() + i, i);
    }

    std::stringstream ss;
    IOStreamWriter writer(ss);
    cell->Serialize(writer);
    std::string bytes = ss.str();

    REQUIRE(bytes.size() > 12 + sizeof(uint32_t));
    uint32_t sentinel_on_disk = 0;
    std::memcpy(&sentinel_on_disk, bytes.data() + 12, sizeof(uint32_t));
    REQUIRE(sentinel_on_disk == std::numeric_limits<uint32_t>::max());

    auto reloaded = FlattenInterface::MakeInstance(param, index_common_param);
    std::stringstream rss(bytes);
    IOStreamReader reader(rss);
    reloaded->Deserialize(reader);

    auto query_sparse_vectors = fixtures::GenerateSparseVectors(1, 100);
    auto computer = reloaded->FactoryComputer(query_sparse_vectors.data());
    std::vector<InnerIdType> all_ids(base_count);
    std::iota(all_ids.begin(), all_ids.end(), 0);
    std::vector<float> dist(base_count);
    reloaded->Query(dist.data(), computer, all_ids.data(), base_count);
    for (uint32_t i = 0; i < base_count; ++i) {
        fixtures::dist_t expected =
            fixtures::GetSparseDistance(query_sparse_vectors[0], sparse_vectors[i]);
        REQUIRE(expected == dist[i]);
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
