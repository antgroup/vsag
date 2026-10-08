// Copyright 2026-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <sstream>
#include <vector>

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
