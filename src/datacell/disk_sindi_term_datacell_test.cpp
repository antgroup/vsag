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

#include <array>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstring>
#include <limits>
#include <sstream>

#include "datacell/mutable_sindi_term_datacell.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "io/common/io_parameter.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"

using namespace vsag;

namespace {

class StringReader : public Reader {
public:
    explicit StringReader(std::string bytes) : bytes_(std::move(bytes)) {
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        REQUIRE(offset <= bytes_.size());
        REQUIRE(len <= bytes_.size() - offset);
        std::memcpy(dest, bytes_.data() + offset, len);
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {
        Read(offset, len, dest);
        callback(IOErrorCode::IO_SUCCESS, "success");
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return bytes_.size();
    }

private:
    std::string bytes_;
};

}  // namespace

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
        std::make_shared<MutableSindiTermDataCell>(term_id_limit,
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

    auto restored = DiskSindiTermDataCellInterface::MakeInstance(term_id_limit,
                                                                 common_param.allocator_.get(),
                                                                 SparseValueQuantizationType::FP32,
                                                                 nullptr,
                                                                 window_size,
                                                                 io_param,
                                                                 common_param);
    IOStreamReader reader(stream);
    restored->DeserializeTermLayout(reader, 1, target_inner_id + 1);
    REQUIRE(reader.GetCursor() == writer.GetCursor());

    Vector<uint32_t> query_terms(common_param.allocator_.get());
    query_terms.push_back(target_term_id);
    const auto memory_usage_before_load = restored->GetMemoryUsage();
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
    if (io_type == IO_TYPE_VALUE_MMAP_IO) {
        REQUIRE(term_buffer.external_ids != nullptr);
        REQUIRE(term_buffer.external_values != nullptr);
        REQUIRE(restored->GetMemoryUsage() == memory_usage_before_load);
        const auto reloaded_buffers = restored->LoadQueryTermBuffers(query_terms);
        REQUIRE(reloaded_buffers.at(target_term_id).IdsData() == term_buffer.IdsData());
        REQUIRE(reloaded_buffers.at(target_term_id).ValuesData() == term_buffer.ValuesData());
        REQUIRE(restored->GetMemoryUsage() == memory_usage_before_load);
    } else {
        REQUIRE(term_buffer.external_ids == nullptr);
        REQUIRE(term_buffer.external_values == nullptr);
    }
    float restored_value = 0.0F;
    std::memcpy(&restored_value, term_buffer.ValuesData(), sizeof(float));
    REQUIRE(restored_value == target_value);
}

TEST_CASE("DiskSindiTermDataCell validates payloads before query", "[ut][DiskSindiTermDataCell]") {
    fixtures::TempDir dir("disk_sindi_payload_validation");
    constexpr uint32_t term_id_limit = 8;
    constexpr uint32_t window_size = 4;
    constexpr uint32_t term_id = 3;

    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    auto source =
        std::make_shared<MutableSindiTermDataCell>(term_id_limit,
                                                   window_size,
                                                   common_param.allocator_.get(),
                                                   SparseValueQuantizationType::FP32,
                                                   std::make_shared<QuantizationParams>());
    uint32_t term = term_id;
    std::array<float, 2> values = {1.0F, 2.0F};
    for (uint32_t document = 0; document < values.size(); ++document) {
        SparseVector vector{1, &term, values.data() + document};
        source->InsertVector(vector, document);
    }
    source->Finalize();

    std::stringstream stream;
    IOStreamWriter writer(stream);
    source->SerializeTermLayout(writer, source->GetTermDictCount());
    const auto serialized = stream.str();
    uint64_t term_dict_count = 0;
    std::memcpy(&term_dict_count, serialized.data(), sizeof(term_dict_count));
    REQUIRE(term_dict_count == term_id + 1);
    std::vector<DiskTermEntry> term_dict(term_dict_count);
    std::memcpy(term_dict.data(),
                serialized.data() + sizeof(term_dict_count),
                term_dict.size() * sizeof(DiskTermEntry));
    const auto payload_size_offset =
        sizeof(term_dict_count) + term_dict.size() * sizeof(DiskTermEntry);
    uint64_t payload_size = 0;
    std::memcpy(&payload_size, serialized.data() + payload_size_offset, sizeof(payload_size));
    const auto payload_start = payload_size_offset + sizeof(payload_size);
    const auto& entry = term_dict[term_id];
    REQUIRE(entry.posting_count == 2);
    REQUIRE(payload_size == entry.posting_payload_size);
    const auto term_payload_start = payload_start + entry.posting_payload_offset;
    const auto ids_offset = sizeof(uint32_t) + sizeof(TermWindowMeta);
    const auto values_offset = ids_offset + entry.posting_count * sizeof(uint16_t) +
                               sindi_datacell_utils::GetIdsPadding(entry.posting_count);

    auto duplicate_ids = serialized;
    uint16_t first_id = 0;
    std::memcpy(
        &first_id, duplicate_ids.data() + term_payload_start + ids_offset, sizeof(first_id));
    std::memcpy(duplicate_ids.data() + term_payload_start + ids_offset + sizeof(first_id),
                &first_id,
                sizeof(first_id));
    const auto* duplicate_payload =
        reinterpret_cast<const uint8_t*>(duplicate_ids.data() + term_payload_start);
    REQUIRE_THROWS_WITH(sindi_datacell_utils::ViewTermPayload(duplicate_payload,
                                                              entry.posting_payload_size,
                                                              entry,
                                                              1,
                                                              window_size,
                                                              2,
                                                              sizeof(float),
                                                              common_param.allocator_.get()),
                        Catch::Matchers::ContainsSubstring("duplicate ids"));
    REQUIRE_NOTHROW(sindi_datacell_utils::ViewTrustedTermPayload(duplicate_payload,
                                                                 entry.posting_payload_size,
                                                                 entry,
                                                                 1,
                                                                 window_size,
                                                                 2,
                                                                 sizeof(float),
                                                                 common_param.allocator_.get()));

    auto non_finite_value = serialized;
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(non_finite_value.data() + term_payload_start + values_offset, &nan, sizeof(nan));
    const auto* non_finite_payload =
        reinterpret_cast<const uint8_t*>(non_finite_value.data() + term_payload_start);
    REQUIRE_THROWS_WITH(sindi_datacell_utils::ViewTermPayload(non_finite_payload,
                                                              entry.posting_payload_size,
                                                              entry,
                                                              1,
                                                              window_size,
                                                              2,
                                                              sizeof(float),
                                                              common_param.allocator_.get()),
                        Catch::Matchers::ContainsSubstring("non-finite"));
    REQUIRE_NOTHROW(sindi_datacell_utils::ViewTrustedTermPayload(non_finite_payload,
                                                                 entry.posting_payload_size,
                                                                 entry,
                                                                 1,
                                                                 window_size,
                                                                 2,
                                                                 sizeof(float),
                                                                 common_param.allocator_.get()));

    const auto io_type = GENERATE(IO_TYPE_VALUE_BUFFER_IO, IO_TYPE_VALUE_MMAP_IO);
    DYNAMIC_SECTION("deserialize validates io_type=" << io_type) {
        auto io_param = IOParameter::GetIOParameterByJson(JsonType::Parse(fmt::format(
            R"({{"type":"{}","file_path":"{}"}})", io_type, dir.GenerateRandomFile(true))));
        auto restored =
            DiskSindiTermDataCellInterface::MakeInstance(term_id_limit,
                                                         common_param.allocator_.get(),
                                                         SparseValueQuantizationType::FP32,
                                                         nullptr,
                                                         window_size,
                                                         io_param,
                                                         common_param);
        std::stringstream corrupted_stream(duplicate_ids);
        IOStreamReader reader(corrupted_stream);
        REQUIRE_THROWS_WITH(restored->DeserializeTermLayout(reader, 1, 2),
                            Catch::Matchers::ContainsSubstring("duplicate ids"));
    }

    auto reader_io_param =
        IOParameter::GetIOParameterByJson(JsonType::Parse(R"({"type":"reader_io"})"));
    auto reader_restored =
        DiskSindiTermDataCellInterface::MakeInstance(term_id_limit,
                                                     common_param.allocator_.get(),
                                                     SparseValueQuantizationType::FP32,
                                                     nullptr,
                                                     window_size,
                                                     reader_io_param,
                                                     common_param);
    std::stringstream corrupted_stream(duplicate_ids);
    IOStreamReader layout_reader(corrupted_stream);
    REQUIRE_NOTHROW(reader_restored->DeserializeTermLayout(layout_reader, 1, 2));
    REQUIRE_THROWS_WITH(reader_restored->SetIO(std::make_shared<StringReader>(duplicate_ids)),
                        Catch::Matchers::ContainsSubstring("duplicate ids"));
}

TEST_CASE("DiskSindiTermDataCell applies term prune without term-list heap insertion",
          "[ut][DiskSindiTermDataCell]") {
    fixtures::TempDir dir("disk_sindi_term_prune");
    constexpr uint32_t term_id_limit = 8;
    constexpr uint32_t window_size = 4;
    uint32_t term = 3;

    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    auto io_param = IOParameter::GetIOParameterByJson(JsonType::Parse(
        fmt::format(R"({{"type":"buffer_io","file_path":"{}"}})", dir.GenerateRandomFile(true))));
    auto source =
        std::make_shared<MutableSindiTermDataCell>(term_id_limit,
                                                   window_size,
                                                   common_param.allocator_.get(),
                                                   SparseValueQuantizationType::FP32,
                                                   std::make_shared<QuantizationParams>());
    std::array<float, 6> values = {1.0F, 3.0F, 2.0F, 1.0F, 4.0F, 2.0F};
    for (uint32_t document = 0; document < values.size(); ++document) {
        SparseVector vector{1, &term, values.data() + document};
        source->InsertVector(vector, document);
    }
    source->SortByValue(0);
    source->SortByValue(1);
    source->Finalize();

    std::stringstream stream;
    IOStreamWriter writer(stream);
    source->SerializeTermLayout(writer, source->GetTermDictCount());
    stream.seekg(0, std::ios::beg);

    auto restored = DiskSindiTermDataCellInterface::MakeInstance(term_id_limit,
                                                                 common_param.allocator_.get(),
                                                                 SparseValueQuantizationType::FP32,
                                                                 nullptr,
                                                                 window_size,
                                                                 io_param,
                                                                 common_param);
    IOStreamReader reader(stream);
    restored->DeserializeTermLayout(reader, 2, values.size());

    Vector<uint32_t> query_terms(common_param.allocator_.get());
    query_terms.push_back(term);
    SindiQueryContext query_context(common_param.allocator_.get());
    query_context.query_term_buffers = restored->LoadQueryTermBuffers(query_terms);

    float query_value = 1.0F;
    SparseVector query{1, &term, &query_value};
    SINDIV2SearchParameter search_parameter;
    search_parameter.term_retain_threshold = 2;
    auto computer = std::make_shared<SparseTermComputer>(
        query, search_parameter, common_param.allocator_.get(), 2);

    std::array<float, window_size> distances{};
    restored->QueryWindow(distances.data(), 0, computer, false, query_context);
    REQUIRE(distances[0] == 0.0F);
    REQUIRE(distances[1] == -3.0F);
    REQUIRE(distances[2] == 0.0F);
    REQUIRE(distances[3] == 0.0F);

    distances.fill(0.0F);
    restored->QueryWindow(distances.data(), 1, computer, false, query_context);
    REQUIRE(distances[0] == -4.0F);
    REQUIRE(distances[1] == 0.0F);
}

#if HAVE_LIBAIO
TEST_CASE("DiskSindiTermDataCell batch loads async term payloads", "[ut][DiskSindiTermDataCell]") {
    fixtures::TempDir dir("disk_sindi_async_term_multiread");
    constexpr uint32_t term_id_limit = 32;
    constexpr uint32_t window_size = 4;

    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    auto io_param = IOParameter::GetIOParameterByJson(JsonType::Parse(
        fmt::format(R"({{"type":"async_io","file_path":"{}"}})", dir.GenerateRandomFile(true))));

    auto source =
        std::make_shared<MutableSindiTermDataCell>(term_id_limit,
                                                   window_size,
                                                   common_param.allocator_.get(),
                                                   SparseValueQuantizationType::FP32,
                                                   std::make_shared<QuantizationParams>());
    uint32_t first_ids[] = {2, 5};
    float first_values[] = {1.25F, 0.5F};
    SparseVector first{2, first_ids, first_values};
    source->InsertVector(first, 1);

    uint32_t second_ids[] = {2, 9};
    float second_values[] = {2.5F, 3.0F};
    SparseVector second{2, second_ids, second_values};
    source->InsertVector(second, 6);

    uint32_t third_ids[] = {5, 9};
    float third_values[] = {4.0F, 5.0F};
    SparseVector third{2, third_ids, third_values};
    source->InsertVector(third, 10);
    source->Finalize();

    std::stringstream stream;
    IOStreamWriter writer(stream);
    source->SerializeTermLayout(writer, source->GetTermDictCount());
    stream.seekg(0, std::ios::beg);

    auto restored = DiskSindiTermDataCellInterface::MakeInstance(term_id_limit,
                                                                 common_param.allocator_.get(),
                                                                 SparseValueQuantizationType::FP32,
                                                                 nullptr,
                                                                 window_size,
                                                                 io_param,
                                                                 common_param);
    IOStreamReader reader(stream);
    restored->DeserializeTermLayout(reader, 3, 11);

    Vector<uint32_t> query_terms(common_param.allocator_.get());
    query_terms = {2, 5, 2, 7, 9, term_id_limit + 1};
    const auto buffers = restored->LoadQueryTermBuffers(query_terms);

    REQUIRE(buffers.size() == 3);
    REQUIRE(buffers.count(7) == 0);
    REQUIRE(buffers.count(term_id_limit + 1) == 0);

    const auto& term_two = buffers.at(2);
    REQUIRE(term_two.window_offsets.size() == 4);
    REQUIRE(term_two.window_offsets[0] == 0);
    REQUIRE(term_two.window_offsets[1] == 1);
    REQUIRE(term_two.window_offsets[2] == 2);
    REQUIRE(term_two.window_offsets[3] == 2);
    REQUIRE(term_two.IdsData()[0] == 1);
    REQUIRE(term_two.IdsData()[1] == 2);

    const auto& term_five = buffers.at(5);
    REQUIRE(term_five.window_offsets.size() == 4);
    REQUIRE(term_five.window_offsets[0] == 0);
    REQUIRE(term_five.window_offsets[1] == 1);
    REQUIRE(term_five.window_offsets[2] == 1);
    REQUIRE(term_five.window_offsets[3] == 2);
    REQUIRE(term_five.IdsData()[0] == 1);
    REQUIRE(term_five.IdsData()[1] == 2);

    const auto& term_nine = buffers.at(9);
    REQUIRE(term_nine.window_offsets.size() == 4);
    REQUIRE(term_nine.window_offsets[0] == 0);
    REQUIRE(term_nine.window_offsets[1] == 0);
    REQUIRE(term_nine.window_offsets[2] == 1);
    REQUIRE(term_nine.window_offsets[3] == 2);
    REQUIRE(term_nine.IdsData()[0] == 2);
    REQUIRE(term_nine.IdsData()[1] == 2);

    float values[2]{};
    std::memcpy(values, term_two.ValuesData(), sizeof(values));
    REQUIRE(values[0] == 1.25F);
    REQUIRE(values[1] == 2.5F);
    std::memcpy(values, term_five.ValuesData(), sizeof(values));
    REQUIRE(values[0] == 0.5F);
    REQUIRE(values[1] == 4.0F);
    std::memcpy(values, term_nine.ValuesData(), sizeof(values));
    REQUIRE(values[0] == 3.0F);
    REQUIRE(values[1] == 5.0F);

    SparseVector restored_first;
    restored->GetSparseVector(1, &restored_first, common_param.allocator_.get());
    REQUIRE(restored_first.len_ == 2);
    REQUIRE(restored_first.ids_[0] == 2);
    REQUIRE(restored_first.ids_[1] == 5);
    REQUIRE(restored_first.vals_[0] == 1.25F);
    REQUIRE(restored_first.vals_[1] == 0.5F);
    common_param.allocator_->Deallocate(restored_first.ids_);
    common_param.allocator_->Deallocate(restored_first.vals_);
}
#endif

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
        std::make_shared<MutableSindiTermDataCell>(term_id_limit,
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

    auto restored = DiskSindiTermDataCellInterface::MakeInstance(term_id_limit,
                                                                 common_param.allocator_.get(),
                                                                 SparseValueQuantizationType::FP32,
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
        DiskSindiTermDataCellInterface::MakeInstance(10,
                                                     common_param.allocator_.get(),
                                                     SparseValueQuantizationType::FP32,
                                                     nullptr,
                                                     10000,
                                                     io_param,
                                                     common_param),
        Catch::Matchers::ContainsSubstring("unsupported SINDIV2 term io type"));
}
