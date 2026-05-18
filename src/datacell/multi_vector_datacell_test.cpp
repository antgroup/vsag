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

#include <cstdint>
#include <vector>

#include "datacell/flatten_interface.h"
#include "datacell/multi_vector_datacell_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
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

}  // namespace vsag
