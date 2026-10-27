// Copyright 2026-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <catch2/generators/catch_generators_range.hpp>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

#include "storage/serialization.h"
#include "storage/streaming_serialization_test_utils.h"
#include "unittest.h"
#include "vsag/vsag.h"

namespace vsag {

TEST_CASE("HGraph forward block entry points", "[ut][hgraph][streaming_serialization]") {
    const bool reorder = GENERATE(false, true);
    const bool deduplicate = GENERATE(false, true);
    const auto param = JsonType::Parse(R"({
        "dtype":"float32", "metric_type":"l2", "dim":16, "extra_info_size":8,
        "index_param": {
            "base_quantization_type":"sq8", "precise_quantization_type":"fp32",
            "use_attribute_filter":true, "store_raw_vector":true,
            "persist_source_id":true, "use_conjugate_graph":true,
            "build_thread_count":1, "max_degree":8, "ef_construction":32
        }
    })");
    auto parameters = param;
    parameters["index_param"]["use_reorder"].SetBool(reorder);
    parameters["index_param"]["support_duplicate"].SetBool(deduplicate);
    parameters["index_param"]["deduplicate_storage"].SetBool(deduplicate);
    const auto parameter_string = parameters.Dump();
    constexpr int64_t count = 16;
    std::vector<int64_t> ids(count);
    std::vector<float> vectors(count * 16);
    std::vector<char> extra(count * 8, 'x');
    std::vector<std::string> sources(count);
    std::vector<AttributeSet> attributes(count);
    std::vector<AttributeValue<std::string>> values(count);
    for (uint64_t i = 0; i < count; ++i) {
        ids[i] = static_cast<int64_t>(i);
        sources[i] = std::to_string(i);
        values[i].name_ = "group";
        values[i].GetValue() = {"allowed"};
        attributes[i].attrs_.push_back(&values[i]);
        for (uint64_t d = 0; d < 16; ++d) {
            vectors[i * 16 + d] = static_cast<float>(i + d);
        }
    }
    auto base = Dataset::Make();
    base->NumElements(count)
        ->Dim(16)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->ExtraInfos(extra.data())
        ->SourceID(sources.data())
        ->AttributeSets(attributes.data())
        ->Owner(false);
    auto created = Factory::CreateIndex("hgraph", parameter_string);
    REQUIRE(created.has_value());
    REQUIRE(created.value()->Build(base).has_value());
    std::stringstream output;
    REQUIRE(created.value()->SerializeStreaming(output).has_value());
    const auto bytes = output.str();
    auto restored = Factory::CreateIndex("hgraph", parameter_string);
    REQUIRE(restored.has_value());
    std::istringstream input(bytes);
    REQUIRE(restored.value()->DeserializeStreaming(input).has_value());
    REQUIRE(restored.value()->GetNumElements() == count);

    LoadParameters load_parameters;
    if (reorder) {
        const auto block =
            test::FindStreamingBlock(bytes, StreamSerializationTag::HIGH_PRECISION_CODES);
        auto payload =
            std::make_shared<std::string>(bytes.substr(block.payload_offset, block.payload_size));
        auto external = Factory::CreateReadFuncReader(
            [payload](uint64_t offset, uint64_t length, void* dest) {
                std::memcpy(dest, payload->data() + offset, length);
            },
            payload->size());
        load_parameters.Set("precise_io_type", "reader_io").SetReader("precise_reader", external);
    }
    std::istringstream load_input(bytes);
    auto loaded = Index::Load(load_input, load_parameters);
    REQUIRE(loaded.has_value());
    auto query = Dataset::Make();
    query->NumElements(1)->Dim(16)->Float32Vectors(vectors.data())->Owner(false);
    const auto search_parameters = R"({"hgraph":{"ef_search":32}})";
    auto expected = created.value()->KnnSearch(query, 5, search_parameters);
    REQUIRE(expected.has_value());
    for (const auto& candidate : {restored.value(), loaded.value()}) {
        auto result = candidate->KnnSearch(query, 5, search_parameters);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() == expected.value()->GetDim());
        for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
            REQUIRE(result.value()->GetIds()[i] == expected.value()->GetIds()[i]);
        }
    }
}

}  // namespace vsag

namespace vsag {

TEST_CASE("HGraph streaming rejects corrupted counts before large allocation",
          "[ut][hgraph][streaming_serialization][count_safety]") {
    const bool duplicate = GENERATE(false, true);
    std::vector<std::string> corruptions{
        "labels", "source_count", "source_length", "truncated_count", "truncated_labels"};
    if (!duplicate) {
        corruptions.emplace_back("remap");
    }
    const auto corruption = GENERATE_COPY(Catch::Generators::from_range(corruptions));
    auto param = JsonType::Parse(R"({
        "dtype":"float32", "metric_type":"l2", "dim":16,
        "index_param": {"base_quantization_type":"sq8", "persist_source_id":true,
                        "build_thread_count":1, "max_degree":8, "ef_construction":32}
    })");
    param["index_param"]["support_duplicate"].SetBool(duplicate);
    const auto parameters = param.Dump();
    constexpr uint64_t count = 32;
    std::vector<int64_t> ids(count);
    std::vector<float> vectors(count * 16);
    std::vector<std::string> sources(count, "source");
    for (uint64_t i = 0; i < count; ++i) {
        ids[i] = static_cast<int64_t>(i);
        for (uint64_t j = 0; j < 16; ++j) {
            vectors[i * 16 + j] = static_cast<float>(i + j);
        }
    }
    auto data = Dataset::Make();
    data->NumElements(count)
        ->Dim(16)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->SourceID(sources.data())
        ->Owner(false);
    auto index = Factory::CreateIndex("hgraph", parameters);
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(data).has_value());
    std::stringstream output;
    REQUIRE(index.value()->SerializeStreaming(output).has_value());
    auto bytes = output.str();
    const auto block = test::FindStreamingBlock(bytes, StreamSerializationTag::LABEL_TABLE);
    const auto label_count = test::ReadStreamingObj<uint64_t>(bytes, block.payload_offset);
    auto source_offset = block.payload_offset + sizeof(uint64_t) + label_count * sizeof(int64_t);
    const auto remap_offset = source_offset;
    if (!duplicate) {
        const auto remap_count = test::ReadStreamingObj<uint64_t>(bytes, remap_offset);
        source_offset += sizeof(uint64_t) + remap_count * (sizeof(LabelType) + sizeof(InnerIdType));
    }
    const uint64_t oversized = count ^ (uint64_t{1} << 29);
    if (corruption == "labels") {
        test::WriteStreamingObj(bytes, block.payload_offset, oversized);
    } else if (corruption == "remap") {
        test::WriteStreamingObj(bytes, remap_offset, oversized);
    } else if (corruption == "source_count") {
        test::WriteStreamingObj(bytes, source_offset + sizeof(uint64_t), oversized);
    } else if (corruption == "source_length") {
        test::WriteStreamingObj(bytes, source_offset + 2 * sizeof(uint64_t), oversized);
    } else {
        const uint64_t keep = corruption == "truncated_count" ? 4 : sizeof(uint64_t) + 8;
        bytes.erase(block.payload_offset + keep, block.payload_size - keep);
        test::WriteStreamingObj(bytes, block.header_offset + 16, keep);
    }
    // Exercise both a stale CRC (bit corruption) and a valid CRC over malformed data.
    const bool repair_crc = GENERATE(false, true);
    if (repair_crc) {
        const auto size = test::ReadStreamingObj<uint64_t>(bytes, block.header_offset + 16);
        const auto checksum = StreamHeader::CalculateChecksum(
            std::string_view(bytes.data() + block.payload_offset, size));
        test::WriteStreamingObj(bytes, block.header_offset + 24, checksum);
    }
    test::GuardedAllocator allocator;
    allocator.limit = std::numeric_limits<uint64_t>::max();
    auto restored = Factory::CreateIndex("hgraph", parameters, &allocator);
    REQUIRE(restored.has_value());
    allocator.limit = 1024 * 1024;
    allocator.largest_request = 0;
    std::istringstream input(bytes);
    auto result = restored.value()->DeserializeStreaming(input);
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().type != ErrorType::NO_ENOUGH_MEMORY);
    REQUIRE(allocator.largest_request <= allocator.limit);
}

}  // namespace vsag
