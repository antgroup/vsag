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

#include "hgraph_rabitq_fused_datacell.h"

#include <atomic>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>

#include "impl/allocator/safe_allocator.h"
#include "io/memory_io/memory_io_parameter.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"

namespace vsag {

TEST_CASE("HGraph RaBitQ fused node layout and serialization", "[ut][HGraphRaBitQFusedDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = 960;

    auto graph_param = std::make_shared<GraphDataCellParameter>();
    graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
    graph_param->max_degree_ = 32;
    graph_param->init_max_capacity_ = 8;
    graph_param->support_remove_ = true;
    graph_param->remove_flag_bit_ = 8;

    constexpr uint64_t one_bit_size = 136;
    constexpr uint64_t supplement_size = 848;
    auto graph = std::make_shared<HGraphRaBitQFusedDataCell>(
        graph_param, one_bit_size, supplement_size, common_param);

    REQUIRE(reinterpret_cast<uintptr_t>(graph->GetNodeRecord(0)) % 64 == 0);
    REQUIRE(graph->RecordSize() % 64 == 0);
    REQUIRE(graph->OneBitOffset() == 152);

    Vector<uint8_t> one_bit(one_bit_size, 0x5A, allocator.get());
    Vector<uint8_t> supplement(supplement_size, 0xA5, allocator.get());
    graph->SetNodeCodes(0, 42, 7, one_bit.data(), supplement.data());
    Vector<InnerIdType> neighbors({1, 2, 3}, allocator.get());
    Vector<InnerIdType> empty_neighbors(allocator.get());
    graph->InsertNeighborsById(0, neighbors);
    graph->InsertNeighborsById(1, empty_neighbors);
    graph->InsertNeighborsById(2, empty_neighbors);
    graph->InsertNeighborsById(3, empty_neighbors);

    const auto* record = graph->GetNodeRecord(0);
    REQUIRE(graph->GetLabel(record) == 42);
    REQUIRE(graph->GetClusterId(record) == 7);
    REQUIRE(std::memcmp(graph->GetOneBitCode(record), one_bit.data(), one_bit_size) == 0);
    REQUIRE(std::memcmp(graph->GetSupplementCode(record), supplement.data(), supplement_size) == 0);

    auto restored = std::make_shared<HGraphRaBitQFusedDataCell>(
        graph_param, one_bit_size, supplement_size, common_param);
    test_serializion(*graph, *restored);
    const auto* restored_record = restored->GetNodeRecord(0);
    REQUIRE(restored->GetLabel(restored_record) == 42);
    REQUIRE(restored->GetClusterId(restored_record) == 7);
    REQUIRE(std::memcmp(restored->GetOneBitCode(restored_record), one_bit.data(), one_bit_size) ==
            0);
    REQUIRE(std::memcmp(restored->GetSupplementCode(restored_record),
                        supplement.data(),
                        supplement_size) == 0);

    graph->Move(0, 4);
    const auto* moved_record = graph->GetNodeRecord(4);
    REQUIRE(graph->GetLabel(moved_record) == 42);
    REQUIRE(graph->GetClusterId(moved_record) == 7);
}

TEST_CASE("HGraph RaBitQ fused delete version invalidates old edges",
          "[ut][HGraphRaBitQFusedDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    auto graph_param = std::make_shared<GraphDataCellParameter>();
    graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
    graph_param->max_degree_ = 4;
    graph_param->init_max_capacity_ = 4;
    graph_param->support_remove_ = true;
    graph_param->remove_flag_bit_ = 8;

    auto graph = std::make_shared<HGraphRaBitQFusedDataCell>(graph_param, 16, 16, common_param);
    Vector<InnerIdType> one_neighbor({1}, allocator.get());
    Vector<InnerIdType> empty_neighbors(allocator.get());
    graph->InsertNeighborsById(0, one_neighbor);
    graph->InsertNeighborsById(1, empty_neighbors);

    Vector<InnerIdType> neighbors(allocator.get());
    graph->GetNeighbors(0, neighbors);
    REQUIRE(neighbors == Vector<InnerIdType>({1}, allocator.get()));
    graph->DeleteNeighborsById(1);
    graph->GetNeighbors(0, neighbors);
    REQUIRE(neighbors.empty());
    graph->RecoverDeleteNeighborsById(1);
    graph->GetNeighbors(0, neighbors);
    REQUIRE(neighbors == Vector<InnerIdType>({1}, allocator.get()));
}

TEST_CASE("HGraph RaBitQ fused deserialize validates its wire layout",
          "[ut][HGraphRaBitQFusedDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    auto graph_param = std::make_shared<GraphDataCellParameter>();
    graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
    graph_param->max_degree_ = 32;
    graph_param->init_max_capacity_ = 8;
    graph_param->support_remove_ = true;
    graph_param->remove_flag_bit_ = 8;

    constexpr uint64_t one_bit_size = 16;
    constexpr uint64_t supplement_size = 16;
    auto graph = std::make_shared<HGraphRaBitQFusedDataCell>(
        graph_param, one_bit_size, supplement_size, common_param);
    Vector<InnerIdType> empty_neighbors(allocator.get());
    graph->InsertNeighborsById(0, empty_neighbors);
    std::stringstream stream;
    IOStreamWriter writer(stream);
    graph->Serialize(writer);
    const auto payload = stream.str();

    uint64_t cursor = sizeof(std::atomic<InnerIdType>);
    const uint64_t capacity_offset = cursor;
    cursor += sizeof(InnerIdType);
    cursor += sizeof(uint32_t);  // maximum degree
    const uint64_t version_offset = cursor;
    cursor += sizeof(uint32_t);
    const uint64_t record_size_offset = cursor;
    cursor += sizeof(uint64_t);
    cursor += sizeof(uint64_t);  // neighbors offset
    cursor += sizeof(uint64_t);  // cluster id offset
    cursor += sizeof(uint64_t);  // label offset
    cursor += sizeof(uint64_t);  // one-bit offset
    const uint64_t supplement_offset_offset = cursor;
    cursor += sizeof(uint64_t);
    const uint64_t one_bit_size_offset = cursor;
    cursor += sizeof(uint64_t);
    cursor += sizeof(uint64_t);  // supplement code size
    const uint64_t support_remove_offset = cursor;
    cursor += sizeof(bool);
    const uint64_t remove_flag_bit_offset = cursor;
    cursor += sizeof(uint32_t);
    const uint64_t codec_model_size_offset = cursor;

    uint64_t codec_model_size = 0;
    std::memcpy(
        &codec_model_size, payload.data() + codec_model_size_offset, sizeof(codec_model_size));
    const uint64_t payload_size_offset =
        codec_model_size_offset + sizeof(codec_model_size) + codec_model_size;

    auto overwrite = [](std::string value, uint64_t offset, const auto& replacement) {
        std::memcpy(value.data() + offset, &replacement, sizeof(replacement));
        return value;
    };
    auto require_rejected = [&](const std::string& value) {
        auto restored = std::make_shared<HGraphRaBitQFusedDataCell>(
            graph_param, one_bit_size, supplement_size, common_param);
        std::stringstream malformed_stream(value);
        IOStreamReader reader(malformed_stream);
        REQUIRE_THROWS(restored->Deserialize(reader));
    };

    SECTION("serialization version") {
        require_rejected(overwrite(payload, version_offset, uint32_t{2}));
    }

    SECTION("remove flag representation") {
        require_rejected(overwrite(payload, support_remove_offset, uint8_t{2}));
    }

    SECTION("remove flag bit count") {
        require_rejected(overwrite(
            payload, remove_flag_bit_offset, static_cast<uint32_t>(sizeof(InnerIdType) * 8)));
    }

    SECTION("non-monotonic code offsets") {
        require_rejected(overwrite(payload, supplement_offset_offset, uint64_t{0}));
    }

    SECTION("cache-line record stride") {
        require_rejected(overwrite(payload, record_size_offset, graph->RecordSize() + 1));
    }

    SECTION("code size") {
        require_rejected(overwrite(payload, one_bit_size_offset, one_bit_size + 1));
    }

    SECTION("capacity times stride overflow") {
        constexpr uint64_t largest_aligned_stride = std::numeric_limits<uint64_t>::max() - 63;
        require_rejected(overwrite(payload, record_size_offset, largest_aligned_stride));
    }

    SECTION("payload byte count") {
        require_rejected(overwrite(payload, payload_size_offset, uint64_t{0}));
    }

    SECTION("count exceeds capacity") {
        require_rejected(overwrite(payload, capacity_offset, InnerIdType{0}));
    }

    SECTION("constructor maximum degree") {
        graph_param->max_degree_ = 31;
        require_rejected(payload);
    }
}

}  // namespace vsag
