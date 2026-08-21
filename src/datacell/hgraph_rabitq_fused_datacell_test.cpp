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

    constexpr uint64_t one_bit_size = 132;
    constexpr uint64_t supplement_size = 860;
    auto graph = std::make_shared<HGraphRaBitQFusedDataCell>(
        graph_param, one_bit_size, supplement_size, common_param);

    REQUIRE(reinterpret_cast<uintptr_t>(graph->GetNodeRecord(0)) % 64 == 0);
    REQUIRE(graph->NeighborsOffset() == 4);
    REQUIRE(graph->ClusterIdOffset() == 132);
    REQUIRE(graph->LabelOffset() == 136);
    REQUIRE(graph->OneBitOffset() == 144);
    REQUIRE(graph->SupplementOffset() == 276);
    REQUIRE(graph->RecordSize() == 1152);

    Vector<uint8_t> one_bit(one_bit_size, 0x5A, allocator.get());
    Vector<uint8_t> supplement(supplement_size, 0xA5, allocator.get());
    graph->SetNodeCodes(0, 42, 7, one_bit.data(), supplement.data());
    Vector<InnerIdType> neighbors({1, 2, 3}, allocator.get());
    Vector<InnerIdType> empty_neighbors(allocator.get());
    graph->InsertNeighborsById(0, neighbors);
    graph->InsertNeighborsById(1, empty_neighbors);
    graph->InsertNeighborsById(2, empty_neighbors);
    graph->InsertNeighborsById(3, empty_neighbors);
    REQUIRE(graph->GetNeighborData(graph->GetNodeRecord(0))[0] == 1);
    REQUIRE(graph->GetNeighborData(graph->GetNodeRecord(0))[1] == 2);
    REQUIRE(graph->GetNeighborData(graph->GetNodeRecord(0))[2] == 3);

    const auto* record = graph->GetNodeRecord(0);
    REQUIRE(graph->GetLabel(record) == 42);
    REQUIRE(graph->GetClusterId(record) == 7);
    REQUIRE(std::memcmp(graph->GetOneBitCode(record), one_bit.data(), one_bit_size) == 0);
    REQUIRE(std::memcmp(graph->GetSupplementCode(record), supplement.data(), supplement_size) == 0);

    auto restored = std::make_shared<HGraphRaBitQFusedDataCell>(
        graph_param, one_bit_size, supplement_size, common_param);
    test_serializion(*graph, *restored);
    REQUIRE(restored->CodecModel().empty());
    const auto* restored_record = restored->GetNodeRecord(0);
    REQUIRE(restored->GetLabel(restored_record) == 42);
    REQUIRE(restored->GetClusterId(restored_record) == 7);
    REQUIRE(std::memcmp(restored->GetOneBitCode(restored_record), one_bit.data(), one_bit_size) ==
            0);
    REQUIRE(std::memcmp(restored->GetSupplementCode(restored_record),
                        supplement.data(),
                        supplement_size) == 0);

    graph->Resize(16);
    REQUIRE(graph->MaxCapacity() == 16);
    REQUIRE(graph->GetLabel(graph->GetNodeRecord(0)) == 42);
    graph->Resize(2);
    REQUIRE(graph->MaxCapacity() == 16);
    graph->Seal();
    REQUIRE(graph->FusedStorageSealed());
    REQUIRE_THROWS(graph->Resize(32));
    REQUIRE_THROWS(graph->InsertNeighborsById(0, neighbors));
    REQUIRE_THROWS(graph->SetFusedCodes(0, 7, one_bit.data(), supplement.data()));
}

TEST_CASE("HGraph RaBitQ fused deserialize validates its wire layout",
          "[ut][HGraphRaBitQFusedDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = 8;

    auto graph_param = std::make_shared<GraphDataCellParameter>();
    graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
    graph_param->max_degree_ = 32;
    graph_param->init_max_capacity_ = 8;

    constexpr uint64_t one_bit_size = 16;
    constexpr uint64_t supplement_size = 16;
    auto graph = std::make_shared<HGraphRaBitQFusedDataCell>(
        graph_param, one_bit_size, supplement_size, common_param);
    const uint64_t expected_codec_model_size =
        16 + 16 * static_cast<uint64_t>(common_param.dim_) * sizeof(float);
    graph->SetCodecModel(std::string(expected_codec_model_size, '\0'));
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
        require_rejected(overwrite(payload, version_offset, uint32_t{1}));
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

    SECTION("codec model length is bounded before allocation") {
        require_rejected(
            overwrite(payload, codec_model_size_offset, std::numeric_limits<uint64_t>::max()));
    }

    SECTION("node payload is bounded before allocation") {
        constexpr InnerIdType large_capacity = std::numeric_limits<InnerIdType>::max();
        auto malformed = overwrite(payload, capacity_offset, large_capacity);
        const uint64_t declared_bytes = static_cast<uint64_t>(large_capacity) * graph->RecordSize();
        malformed = overwrite(malformed, payload_size_offset, declared_bytes);
        require_rejected(malformed);
    }

    SECTION("count exceeds capacity") {
        require_rejected(overwrite(payload, capacity_offset, InnerIdType{0}));
    }

    SECTION("constructor maximum degree") {
        graph_param->max_degree_ = 31;
        require_rejected(payload);
    }

    SECTION("constructor rejects removal") {
        graph_param->support_remove_ = true;
        REQUIRE_THROWS(std::make_shared<HGraphRaBitQFusedDataCell>(
            graph_param, one_bit_size, supplement_size, common_param));
    }

    SECTION("constructor rejects reverse edges") {
        graph_param->use_reverse_edges_ = true;
        REQUIRE_THROWS(std::make_shared<HGraphRaBitQFusedDataCell>(
            graph_param, one_bit_size, supplement_size, common_param));
    }
}

}  // namespace vsag
