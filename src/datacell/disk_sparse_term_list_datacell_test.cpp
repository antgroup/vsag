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

#include "disk_sparse_term_list_datacell.h"

#include <fmt/format.h>

#include <cstring>
#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "io/io_parameter.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("DiskSparseTermListDataCell restores payload io", "[ut][DiskSparseTermListDataCell]") {
    fixtures::TempDir dir("disk_sparse_term_list_datacell");
    auto io_type = GENERATE(IO_TYPE_VALUE_MMAP_IO, IO_TYPE_VALUE_ASYNC_IO);
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

    auto data_cell =
        DiskSparseTermListDataCellInterface::MakeInstance(1.0F,
                                                          term_id_limit,
                                                          common_param.allocator_.get(),
                                                          false,
                                                          nullptr,
                                                          window_size,
                                                          io_param,
                                                          common_param);

    SparseVector vector;
    uint32_t ids[] = {target_term_id};
    float vals[] = {target_value};
    vector.len_ = 1;
    vector.ids_ = ids;
    vector.vals_ = vals;
    data_cell->InsertVector(vector, target_inner_id);
    data_cell->FinalizeTermBuffers(1);

    auto term_dict_size = static_cast<uint64_t>(term_id_limit + 1) * sizeof(DiskTermEntry);
    std::stringstream stream;
    IOStreamWriter writer(stream);
    data_cell->WriteTermDictAndPayload(writer);
    stream.seekg(0, std::ios::beg);

    auto restored = DiskSparseTermListDataCellInterface::MakeInstance(1.0F,
                                                                      term_id_limit,
                                                                      common_param.allocator_.get(),
                                                                      false,
                                                                      nullptr,
                                                                      window_size,
                                                                      io_param,
                                                                      common_param);
    IOStreamReader reader(stream);
    restored->Deserialize(reader, term_dict_size, 1);
    restored->InitIO(io_param);
    restored->WritePayloadToIO(reader, term_dict_size, writer.GetCursor() - term_dict_size);

    Vector<uint32_t> query_terms(common_param.allocator_.get());
    query_terms.push_back(target_term_id);
    auto query_term_buffers = restored->LoadQueryTermBuffers(query_terms);

    REQUIRE(query_term_buffers.size() == 1);
    REQUIRE(query_term_buffers.count(target_term_id) == 1);
    const auto& term_buffer = query_term_buffers[target_term_id];
    REQUIRE(term_buffer.window_offsets.size() == 2);
    REQUIRE(term_buffer.window_offsets[0] == 0);
    REQUIRE(term_buffer.window_offsets[1] == 1);
    REQUIRE(term_buffer.ids.size() == 1);
    REQUIRE(term_buffer.ids[0] == target_inner_id);
    REQUIRE(term_buffer.values.size() == sizeof(float));
    float restored_value = 0.0F;
    std::memcpy(&restored_value, term_buffer.values.data(), sizeof(float));
    REQUIRE(restored_value == target_value);
}

TEST_CASE("DiskSparseTermListDataCell rejects memory io", "[ut][DiskSparseTermListDataCell]") {
    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    auto io_param = IOParameter::GetIOParameterByJson(JsonType::Parse(R"({"type":"memory_io"})"));

    REQUIRE_THROWS_WITH(
        DiskSparseTermListDataCellInterface::MakeInstance(
            1.0F, 10, common_param.allocator_.get(), false, nullptr, 10000, io_param, common_param),
        Catch::Matchers::ContainsSubstring("unsupported DiskSINDI term io type: memory_io"));
}
