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

#include <fmt/format.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>

#include "datacell/flatten_interface.h"
#include "datacell/multi_vector_datacell_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "storage/serialization.h"
#include "unittest.h"
#include "vsag/dataset.h"

namespace vsag {
namespace {

FlattenInterfacePtr
MakeMultiVectorDataCell(const std::string& io_type,
                        int64_t dim,
                        const std::shared_ptr<Allocator>& allocator) {
    constexpr const char* param_template =
        R"(
        {{
            "io_params": {{
                "type": "{}"
            }}
        }}
        )";
    const std::string param_str = fmt::format(param_template, io_type);
    JsonType parsed_json = JsonType::Parse(param_str);
    MultiVectorDataCellParamPtr param = std::make_shared<MultiVectorDataCellParameter>();
    param->FromJson(parsed_json);

    IndexCommonParam index_common_param;
    index_common_param.allocator_ = allocator;
    index_common_param.metric_ = MetricType::METRIC_TYPE_IP;
    index_common_param.dim_ = dim;
    return FlattenInterface::MakeInstance(param, index_common_param);
}

void
FillMultiVectors(const std::vector<uint32_t>& token_counts,
                 uint32_t dim,
                 std::vector<std::vector<float>>& token_storage,
                 std::vector<MultiVector>& multi_vectors) {
    const uint64_t doc_count = static_cast<uint64_t>(token_counts.size());
    token_storage.resize(doc_count);
    multi_vectors.resize(doc_count);

    for (uint64_t doc_id = 0; doc_id < doc_count; ++doc_id) {
        const uint64_t float_count = static_cast<uint64_t>(token_counts[doc_id]) * dim;
        token_storage[doc_id].resize(float_count);
        for (uint64_t pos = 0; pos < float_count; ++pos) {
            token_storage[doc_id][pos] = static_cast<float>(doc_id * 100 + pos);
        }
        multi_vectors[doc_id].len_ = token_counts[doc_id];
        multi_vectors[doc_id].vectors_ = token_storage[doc_id].data();
    }
}

float
CalMaxSimIP(const float* query,
            uint32_t query_token_count,
            const float* doc,
            uint32_t doc_token_count,
            uint32_t dim) {
    float total = 0.0F;
    for (uint32_t q = 0; q < query_token_count; ++q) {
        float min_dist = std::numeric_limits<float>::max();
        for (uint32_t d = 0; d < doc_token_count; ++d) {
            float ip = 0.0F;
            for (uint32_t k = 0; k < dim; ++k) {
                ip += query[q * dim + k] * doc[d * dim + k];
            }
            float dist = 1.0F - ip;
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        total += min_dist;
    }
    return total;
}

}  // namespace

TEST_CASE("MultiVectorDataCell inserts variable-length documents", "[ut][MultiVectorDataCell]") {
    const std::string io_type = GENERATE("memory_io", "block_memory_io");
    constexpr uint32_t dim = 4;
    const std::vector<uint32_t> token_counts = {1, 3, 2, 5, 4};
    std::vector<std::vector<float>> token_storage;
    std::vector<MultiVector> multi_vectors;
    FillMultiVectors(token_counts, dim, token_storage, multi_vectors);

    std::shared_ptr<Allocator> allocator = SafeAllocator::FactoryDefaultAllocator();
    FlattenInterfacePtr data_cell = MakeMultiVectorDataCell(io_type, dim, allocator);
    data_cell->Resize(static_cast<InnerIdType>(multi_vectors.size()));

    data_cell->InsertVector(multi_vectors.data(), 0);
    data_cell->InsertVector(multi_vectors.data() + 1, 1);
    std::vector<InnerIdType> idx = {2, 3, 4};
    data_cell->BatchInsertVector(
        multi_vectors.data() + 2, static_cast<InnerIdType>(idx.size()), idx.data());

    REQUIRE(data_cell->TotalCount() == token_counts.size());
}

TEST_CASE("MultiVectorDataCell batch insert reserves ids when idx is null",
          "[ut][MultiVectorDataCell]") {
    const std::string io_type = GENERATE("memory_io", "block_memory_io");
    constexpr uint32_t dim = 4;
    const std::vector<uint32_t> token_counts = {1, 3, 2, 5, 4};
    std::vector<std::vector<float>> token_storage;
    std::vector<MultiVector> multi_vectors;
    FillMultiVectors(token_counts, dim, token_storage, multi_vectors);

    std::shared_ptr<Allocator> allocator = SafeAllocator::FactoryDefaultAllocator();
    FlattenInterfacePtr data_cell = MakeMultiVectorDataCell(io_type, dim, allocator);
    data_cell->BatchInsertVector(
        multi_vectors.data(), static_cast<InnerIdType>(multi_vectors.size()), nullptr);

    REQUIRE(data_cell->TotalCount() == token_counts.size());
}

TEST_CASE("MultiVectorDataCell GetCodesById reads back inserted data",
          "[ut][MultiVectorDataCell]") {
    const std::string io_type = GENERATE("memory_io", "block_memory_io");
    constexpr uint32_t dim = 4;
    const std::vector<uint32_t> token_counts = {1, 3, 2, 5, 4};
    std::vector<std::vector<float>> token_storage;
    std::vector<MultiVector> multi_vectors;
    FillMultiVectors(token_counts, dim, token_storage, multi_vectors);

    std::shared_ptr<Allocator> allocator = SafeAllocator::FactoryDefaultAllocator();
    FlattenInterfacePtr data_cell = MakeMultiVectorDataCell(io_type, dim, allocator);
    data_cell->Resize(static_cast<InnerIdType>(multi_vectors.size()));

    for (uint64_t i = 0; i < multi_vectors.size(); ++i) {
        data_cell->InsertVector(multi_vectors.data() + i, static_cast<InnerIdType>(i));
    }

    for (uint64_t i = 0; i < multi_vectors.size(); ++i) {
        bool need_release = false;
        const uint8_t* codes = data_cell->GetCodesById(static_cast<InnerIdType>(i), need_release);
        REQUIRE(need_release == true);

        uint32_t len = 0;
        std::memcpy(&len, codes, sizeof(uint32_t));
        REQUIRE(len == token_counts[i]);

        const auto* floats = reinterpret_cast<const float*>(codes + sizeof(uint32_t));
        for (uint64_t j = 0; j < static_cast<uint64_t>(len) * dim; ++j) {
            REQUIRE(floats[j] == token_storage[i][j]);
        }

        data_cell->Release(codes);
    }
}

TEST_CASE("MultiVectorDataCell Query computes MaxSim distances", "[ut][MultiVectorDataCell]") {
    const std::string io_type = GENERATE("memory_io", "block_memory_io");
    constexpr uint32_t dim = 4;

    // doc0: 2 tokens
    std::vector<float> doc0_data = {0.1F, 0.3F, 0.5F, 0.7F, 0.2F, 0.4F, 0.6F, 0.8F};
    // doc1: 3 tokens
    std::vector<float> doc1_data = {
        0.9F, 0.1F, 0.2F, 0.3F, 0.4F, 0.8F, 0.1F, 0.5F, 0.3F, 0.6F, 0.7F, 0.2F};
    // doc2: 1 token
    std::vector<float> doc2_data = {0.5F, 0.3F, 0.4F, 0.6F};

    MultiVector doc0{2, doc0_data.data()};
    MultiVector doc1{3, doc1_data.data()};
    MultiVector doc2{1, doc2_data.data()};
    std::vector<MultiVector> docs = {doc0, doc1, doc2};

    std::shared_ptr<Allocator> allocator = SafeAllocator::FactoryDefaultAllocator();
    FlattenInterfacePtr data_cell = MakeMultiVectorDataCell(io_type, dim, allocator);
    data_cell->Resize(static_cast<InnerIdType>(docs.size()));
    for (uint64_t i = 0; i < docs.size(); ++i) {
        data_cell->InsertVector(docs.data() + i, static_cast<InnerIdType>(i));
    }

    // query: 2 tokens
    std::vector<float> query_data = {0.7F, 0.2F, 0.3F, 0.1F, 0.1F, 0.5F, 0.8F, 0.4F};
    MultiVector query_mv{2, query_data.data()};
    ComputerInterfacePtr computer = data_cell->FactoryComputer(&query_mv);
    REQUIRE(computer != nullptr);

    std::vector<InnerIdType> idx = {0, 1, 2};
    std::vector<float> dists(3, 0.0F);
    data_cell->Query(
        dists.data(), computer, idx.data(), static_cast<InnerIdType>(idx.size()), nullptr);

    constexpr float kEpsilon = 1e-5F;
    const std::vector<std::pair<const float*, uint32_t>> doc_list = {
        {doc0_data.data(), 2}, {doc1_data.data(), 3}, {doc2_data.data(), 1}};
    for (uint64_t i = 0; i < doc_list.size(); ++i) {
        float expected =
            CalMaxSimIP(query_data.data(), 2, doc_list[i].first, doc_list[i].second, dim);
        REQUIRE(std::abs(dists[i] - expected) < kEpsilon);
    }
}

TEST_CASE("MultiVectorDataCell Serialize/Deserialize round-trip", "[ut][MultiVectorDataCell]") {
    const std::string io_type = GENERATE("memory_io", "block_memory_io");
    constexpr uint32_t dim = 4;

    std::vector<float> doc0_data = {0.1F, 0.3F, 0.5F, 0.7F, 0.2F, 0.4F, 0.6F, 0.8F};
    std::vector<float> doc1_data = {
        0.9F, 0.1F, 0.2F, 0.3F, 0.4F, 0.8F, 0.1F, 0.5F, 0.3F, 0.6F, 0.7F, 0.2F};
    std::vector<float> doc2_data = {0.5F, 0.3F, 0.4F, 0.6F};

    MultiVector doc0{2, doc0_data.data()};
    MultiVector doc1{3, doc1_data.data()};
    MultiVector doc2{1, doc2_data.data()};
    std::vector<MultiVector> docs = {doc0, doc1, doc2};

    std::shared_ptr<Allocator> allocator = SafeAllocator::FactoryDefaultAllocator();
    FlattenInterfacePtr original = MakeMultiVectorDataCell(io_type, dim, allocator);
    original->Resize(static_cast<InnerIdType>(docs.size()));
    for (uint64_t i = 0; i < docs.size(); ++i) {
        original->InsertVector(docs.data() + i, static_cast<InnerIdType>(i));
    }

    std::vector<float> query_data = {0.7F, 0.2F, 0.3F, 0.1F, 0.1F, 0.5F, 0.8F, 0.4F};
    MultiVector query_mv{2, query_data.data()};

    std::vector<InnerIdType> idx = {0, 1, 2};
    std::vector<float> dists_before(3, 0.0F);
    {
        ComputerInterfacePtr computer = original->FactoryComputer(&query_mv);
        original->Query(dists_before.data(),
                        computer,
                        idx.data(),
                        static_cast<InnerIdType>(idx.size()),
                        nullptr);
    }

    std::stringstream ss;
    IOStreamWriter writer(ss);
    original->Serialize(writer);

    FlattenInterfacePtr restored = MakeMultiVectorDataCell(io_type, dim, allocator);
    IOStreamReader reader(ss);
    restored->Deserialize(reader);

    REQUIRE(restored->TotalCount() == original->TotalCount());

    std::vector<float> dists_after(3, 0.0F);
    {
        ComputerInterfacePtr computer = restored->FactoryComputer(&query_mv);
        restored->Query(dists_after.data(),
                        computer,
                        idx.data(),
                        static_cast<InnerIdType>(idx.size()),
                        nullptr);
    }

    for (uint64_t i = 0; i < 3; ++i) {
        REQUIRE(dists_before[i] == dists_after[i]);
    }
}

TEST_CASE("MultiVectorDataCell rejects invalid InsertVector inputs", "[ut][MultiVectorDataCell]") {
    constexpr uint32_t dim = 4;
    std::shared_ptr<Allocator> allocator = SafeAllocator::FactoryDefaultAllocator();
    FlattenInterfacePtr data_cell = MakeMultiVectorDataCell("memory_io", dim, allocator);
    data_cell->Resize(4);

    SECTION("nullptr vector") {
        REQUIRE_THROWS(data_cell->InsertVector(nullptr, 0));
    }

    SECTION("zero token count") {
        std::vector<float> dummy(dim, 1.0F);
        MultiVector mv{0, dummy.data()};
        REQUIRE_THROWS(data_cell->InsertVector(&mv, 0));
    }

    SECTION("nullptr tokens") {
        MultiVector mv{2, nullptr};
        REQUIRE_THROWS(data_cell->InsertVector(&mv, 0));
    }
}

TEST_CASE("MultiVectorDataCell rejects invalid BatchInsertVector inputs",
          "[ut][MultiVectorDataCell]") {
    constexpr uint32_t dim = 4;
    std::shared_ptr<Allocator> allocator = SafeAllocator::FactoryDefaultAllocator();
    FlattenInterfacePtr data_cell = MakeMultiVectorDataCell("memory_io", dim, allocator);

    SECTION("nullptr vectors array") {
        REQUIRE_THROWS(data_cell->BatchInsertVector(nullptr, 3, nullptr));
    }
}

TEST_CASE("MultiVectorDataCell rejects invalid FactoryComputer inputs",
          "[ut][MultiVectorDataCell]") {
    constexpr uint32_t dim = 4;
    std::shared_ptr<Allocator> allocator = SafeAllocator::FactoryDefaultAllocator();
    FlattenInterfacePtr data_cell = MakeMultiVectorDataCell("memory_io", dim, allocator);

    SECTION("nullptr query") {
        REQUIRE_THROWS(data_cell->FactoryComputer(nullptr));
    }

    SECTION("zero query token count") {
        std::vector<float> dummy(dim, 1.0F);
        MultiVector mv{0, dummy.data()};
        REQUIRE_THROWS(data_cell->FactoryComputer(&mv));
    }

    SECTION("nullptr query vectors") {
        MultiVector mv{2, nullptr};
        REQUIRE_THROWS(data_cell->FactoryComputer(&mv));
    }
}

}  // namespace vsag
