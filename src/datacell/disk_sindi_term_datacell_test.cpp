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

#include "disk_sindi_term_datacell.h"

#include <fmt/format.h>

#include <cstring>
#include <sstream>

#include "datacell/mutable_sindi_term_datacell.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "io/io_parameter.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("DiskSindiTermDataCell restores payload io", "[ut][DiskSindiTermDataCell]") {
    fixtures::TempDir dir("disk_sindi_term_datacell");
    auto io_type = GENERATE(IO_TYPE_VALUE_MMAP_IO, IO_TYPE_VALUE_BUFFER_IO, IO_TYPE_VALUE_ASYNC_IO);
    auto io_path = dir.GenerateRandomFile(true);
    constexpr uint32_t term_id_limit = 10;
    constexpr uint32_t window_size = 10000;
    constexpr uint32_t target_inner_id = 7;
    constexpr uint32_t target_term_id = 3;
    constexpr float target_value = 2.0F;

    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    auto io_param = IOParameter::GetIOParameterByJson(
        JsonType::Parse(fmt::format(R"({{"type":"{}","file_path":"{}"}})", io_type, io_path)));

    SparseVector vector;
    uint32_t ids[] = {target_term_id};
    float vals[] = {target_value};
    vector.len_ = 1;
    vector.ids_ = ids;
    vector.vals_ = vals;
    auto source =
        std::make_shared<MutableSindiTermDataCell>(1.0F,
                                                   term_id_limit,
                                                   window_size,
                                                   common_param.allocator_.get(),
                                                   SparseValueQuantizationType::FP32,
                                                   std::make_shared<QuantizationParams>());
    source->InsertVector(vector, target_inner_id);
    source->Finalize();

    const auto term_dict_count = source->GetTermDictCount();
    REQUIRE(term_dict_count == target_term_id + 1);
    std::stringstream stream;
    IOStreamWriter writer(stream);
    source->SerializeTermLayout(writer, term_dict_count);
    const auto bytes = stream.str();
    uint64_t serialized_term_dict_count = 0;
    std::memcpy(&serialized_term_dict_count, bytes.data(), sizeof(serialized_term_dict_count));
    REQUIRE(serialized_term_dict_count == term_dict_count);
    const auto payload_size_offset =
        sizeof(uint64_t) + static_cast<uint64_t>(term_dict_count) * sizeof(DiskTermEntry);
    uint64_t payload_size = 0;
    std::memcpy(&payload_size, bytes.data() + payload_size_offset, sizeof(payload_size));
    REQUIRE(writer.GetCursor() == payload_size_offset + sizeof(uint64_t) + payload_size);
    stream.seekg(0, std::ios::beg);

    auto restored = DiskSindiTermDataCellInterface::MakeInstance(1.0F,
                                                                 term_id_limit,
                                                                 common_param.allocator_.get(),
                                                                 false,
                                                                 nullptr,
                                                                 window_size,
                                                                 io_param,
                                                                 common_param);
    IOStreamReader reader(stream);
    restored->DeserializeTermLayout(reader, 1, target_inner_id + 1);
    REQUIRE(reader.GetCursor() == writer.GetCursor());

    Vector<uint32_t> query_terms(common_param.allocator_.get());
    query_terms.push_back(target_term_id);
    auto query_term_buffers = restored->LoadQueryTermBuffers(query_terms);

    REQUIRE(query_term_buffers.size() == 1);
    REQUIRE(query_term_buffers.count(target_term_id) == 1);
    const auto& term_buffer = query_term_buffers[target_term_id];
    REQUIRE(term_buffer.window_offsets.size() == 2);
    REQUIRE(term_buffer.window_offsets[0] == 0);
    REQUIRE(term_buffer.window_offsets[1] == 1);
    REQUIRE(term_buffer.window_offsets.back() == 1);
    REQUIRE(term_buffer.IdsData()[0] == target_inner_id);
    REQUIRE(term_buffer.ValuesSize() == sizeof(float));
    float restored_value = 0.0F;
    std::memcpy(&restored_value, term_buffer.ValuesData(), sizeof(float));
    REQUIRE(restored_value == target_value);
}

TEST_CASE("DiskSindiTermDataCell expands sparse window metadata", "[ut][DiskSindiTermDataCell]") {
    fixtures::TempDir dir("disk_sindi_sparse_window_metadata");
    constexpr uint32_t term_id_limit = 32;
    constexpr uint32_t window_size = 4;
    constexpr uint32_t term_id = 7;

    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    auto io_param = IOParameter::GetIOParameterByJson(JsonType::Parse(
        fmt::format(R"({{"type":"mmap_io","file_path":"{}"}})", dir.GenerateRandomFile(true))));
    auto source =
        std::make_shared<MutableSindiTermDataCell>(1.0F,
                                                   term_id_limit,
                                                   window_size,
                                                   common_param.allocator_.get(),
                                                   SparseValueQuantizationType::FP32,
                                                   std::make_shared<QuantizationParams>());
    uint32_t ids[] = {term_id};
    float values[] = {1.0F};
    SparseVector vector;
    vector.len_ = 1;
    vector.ids_ = ids;
    vector.vals_ = values;
    source->InsertVector(vector, 1);
    source->InsertVector(vector, 9);
    source->Finalize();

    const auto term_dict_count = source->GetTermDictCount();
    std::stringstream stream;
    IOStreamWriter writer(stream);
    source->SerializeTermLayout(writer, term_dict_count);

    stream.seekg(0, std::ios::beg);
    IOStreamReader reader(stream);
    uint64_t serialized_term_dict_count = 0;
    StreamReader::ReadObj(reader, serialized_term_dict_count);
    REQUIRE(serialized_term_dict_count == term_dict_count);
    const auto term_dict_size =
        static_cast<uint64_t>(serialized_term_dict_count) * sizeof(DiskTermEntry);
    std::vector<DiskTermEntry> term_dict(term_dict_count);
    reader.Read(reinterpret_cast<char*>(term_dict.data()), term_dict_size);
    uint64_t payload_size = 0;
    StreamReader::ReadObj(reader, payload_size);
    const auto payload_start = reader.GetCursor();
    REQUIRE(writer.GetCursor() == payload_start + payload_size);
    const auto& entry = term_dict[term_id];
    REQUIRE(entry.posting_count == 2);
    reader.PushSeek(payload_start + entry.posting_payload_offset);
    uint32_t non_empty_window_count = 0;
    StreamReader::ReadObj(reader, non_empty_window_count);
    REQUIRE(non_empty_window_count == 2);
    TermWindowMeta first;
    TermWindowMeta second;
    StreamReader::ReadObj(reader, first);
    StreamReader::ReadObj(reader, second);
    reader.PopSeek();
    REQUIRE(first.window_id == 0);
    REQUIRE(first.posting_count == 1);
    REQUIRE(second.window_id == 2);
    REQUIRE(second.posting_count == 1);

    auto restored = DiskSindiTermDataCellInterface::MakeInstance(1.0F,
                                                                 term_id_limit,
                                                                 common_param.allocator_.get(),
                                                                 false,
                                                                 nullptr,
                                                                 window_size,
                                                                 io_param,
                                                                 common_param);
    reader.Seek(0);
    restored->DeserializeTermLayout(reader, 3, 10);

    Vector<uint32_t> query_terms(common_param.allocator_.get());
    query_terms.push_back(term_id);
    const auto buffers = restored->LoadQueryTermBuffers(query_terms);
    const auto& buffer = buffers.at(term_id);
    REQUIRE(buffer.window_offsets.size() == 4);
    REQUIRE(buffer.window_offsets[0] == 0);
    REQUIRE(buffer.window_offsets[1] == 1);
    REQUIRE(buffer.window_offsets[2] == 1);
    REQUIRE(buffer.window_offsets[3] == 2);
    REQUIRE(buffer.window_offsets.back() == 2);
    REQUIRE(buffer.IdsData()[0] == 1);
    REQUIRE(buffer.IdsData()[1] == 1);
}

TEST_CASE("DiskSindiTermDataCell rejects memory io", "[ut][DiskSindiTermDataCell]") {
    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    auto io_param = IOParameter::GetIOParameterByJson(JsonType::Parse(R"({"type":"memory_io"})"));

    REQUIRE_THROWS_WITH(
        DiskSindiTermDataCellInterface::MakeInstance(
            1.0F, 10, common_param.allocator_.get(), false, nullptr, 10000, io_param, common_param),
        Catch::Matchers::ContainsSubstring("unsupported SINDIV2 term io type"));
}
