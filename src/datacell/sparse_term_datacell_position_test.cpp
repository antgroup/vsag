
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

#include <cstring>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "sparse_term_datacell.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("SparseTermDataCell Position Storage Basic", "[ut][SparseTermDatacell][Position]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    float doc_retain_ratio = 1.0f;  // no pruning
    uint32_t term_id_limit = 100;
    bool use_quantization = false;
    bool store_positions = true;
    uint32_t max_positions_per_term = 64;

    SparseTermDataCell cell(doc_retain_ratio,
                            term_id_limit,
                            allocator.get(),
                            use_quantization,
                            nullptr,
                            store_positions,
                            max_positions_per_term);

    // Document: "A(10) B(20) C(30) A(10) D(40)"
    // Sparse vector: ids=[10,20,30,40], vals=[1,1,1,1]
    // token_seq: [10, 20, 30, 10, 40]
    uint32_t ids[] = {10, 20, 30, 40};
    float vals[] = {1.0f, 1.0f, 1.0f, 1.0f};
    uint32_t token_seq[] = {10, 20, 30, 10, 40};

    SparseVector sv;
    sv.len_ = 4;
    sv.ids_ = ids;
    sv.vals_ = vals;
    sv.token_seq_len_ = 5;
    sv.token_sequence_ = token_seq;

    cell.InsertVector(sv, 0);

    // Verify positions stored correctly
    // term 10: positions [0, 3]
    auto pos_10 = cell.GetPositions(10, 0);
    REQUIRE(pos_10.size() == 2);
    REQUIRE(pos_10[0] == 0);
    REQUIRE(pos_10[1] == 3);

    // term 20: positions [1]
    auto pos_20 = cell.GetPositions(20, 0);
    REQUIRE(pos_20.size() == 1);
    REQUIRE(pos_20[0] == 1);

    // term 30: positions [2]
    auto pos_30 = cell.GetPositions(30, 0);
    REQUIRE(pos_30.size() == 1);
    REQUIRE(pos_30[0] == 2);

    // term 40: positions [4]
    auto pos_40 = cell.GetPositions(40, 0);
    REQUIRE(pos_40.size() == 1);
    REQUIRE(pos_40[0] == 4);
}

TEST_CASE("SparseTermDataCell Position Storage Cap", "[ut][SparseTermDatacell][Position]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    bool store_positions = true;
    uint32_t max_positions_per_term = 3;  // very small cap

    SparseTermDataCell cell(
        1.0f, 100, allocator.get(), false, nullptr, store_positions, max_positions_per_term);

    // term 10 appears 5 times but cap is 3
    uint32_t ids[] = {10};
    float vals[] = {1.0f};
    uint32_t token_seq[] = {10, 10, 10, 10, 10};

    SparseVector sv;
    sv.len_ = 1;
    sv.ids_ = ids;
    sv.vals_ = vals;
    sv.token_seq_len_ = 5;
    sv.token_sequence_ = token_seq;

    cell.InsertVector(sv, 0);

    auto pos = cell.GetPositions(10, 0);
    REQUIRE(pos.size() == 3);  // capped
    REQUIRE(pos[0] == 0);
    REQUIRE(pos[1] == 1);
    REQUIRE(pos[2] == 2);
}

TEST_CASE("SparseTermDataCell Position Storage Disabled", "[ut][SparseTermDatacell][Position]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    bool store_positions = false;

    SparseTermDataCell cell(1.0f, 100, allocator.get(), false, nullptr, store_positions, 64);

    uint32_t ids[] = {10, 20};
    float vals[] = {1.0f, 1.0f};
    uint32_t token_seq[] = {10, 20, 10};

    SparseVector sv;
    sv.len_ = 2;
    sv.ids_ = ids;
    sv.vals_ = vals;
    sv.token_seq_len_ = 3;
    sv.token_sequence_ = token_seq;

    cell.InsertVector(sv, 0);

    // No positions stored
    auto pos = cell.GetPositions(10, 0);
    REQUIRE(pos.empty());
}

TEST_CASE("SparseTermDataCell Position Storage No TokenSeq", "[ut][SparseTermDatacell][Position]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    bool store_positions = true;

    SparseTermDataCell cell(1.0f, 100, allocator.get(), false, nullptr, store_positions, 64);

    uint32_t ids[] = {10, 20};
    float vals[] = {1.0f, 1.0f};

    SparseVector sv;
    sv.len_ = 2;
    sv.ids_ = ids;
    sv.vals_ = vals;
    sv.token_seq_len_ = 0;
    sv.token_sequence_ = nullptr;

    cell.InsertVector(sv, 0);

    // No positions stored because token_sequence is null
    auto pos = cell.GetPositions(10, 0);
    REQUIRE(pos.empty());
}

TEST_CASE("SparseTermDataCell Position Storage Multiple Docs",
          "[ut][SparseTermDatacell][Position]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    bool store_positions = true;

    SparseTermDataCell cell(1.0f, 100, allocator.get(), false, nullptr, store_positions, 64);

    // Doc 0: term 10 at pos [0, 2]
    {
        uint32_t ids[] = {10, 20};
        float vals[] = {1.0f, 1.0f};
        uint32_t token_seq[] = {10, 20, 10};
        SparseVector sv;
        sv.len_ = 2;
        sv.ids_ = ids;
        sv.vals_ = vals;
        sv.token_seq_len_ = 3;
        sv.token_sequence_ = token_seq;
        cell.InsertVector(sv, 0);
    }

    // Doc 1: term 10 at pos [1]
    {
        uint32_t ids[] = {10, 30};
        float vals[] = {1.0f, 1.0f};
        uint32_t token_seq[] = {30, 10};
        SparseVector sv;
        sv.len_ = 2;
        sv.ids_ = ids;
        sv.vals_ = vals;
        sv.token_seq_len_ = 2;
        sv.token_sequence_ = token_seq;
        cell.InsertVector(sv, 1);
    }

    // Doc 0, term 10: [0, 2]
    auto pos0 = cell.GetPositions(10, 0);
    REQUIRE(pos0.size() == 2);
    REQUIRE(pos0[0] == 0);
    REQUIRE(pos0[1] == 2);

    // Doc 1, term 10: [1]
    auto pos1 = cell.GetPositions(10, 1);
    REQUIRE(pos1.size() == 1);
    REQUIRE(pos1[0] == 1);
}

TEST_CASE("SparseTermDataCell Position Serialize Deserialize",
          "[ut][SparseTermDatacell][Position]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    bool store_positions = true;

    SparseTermDataCell cell(1.0f, 100, allocator.get(), false, nullptr, store_positions, 64);

    // Insert two docs
    {
        uint32_t ids[] = {10, 20};
        float vals[] = {1.0f, 2.0f};
        uint32_t token_seq[] = {10, 20, 10};
        SparseVector sv;
        sv.len_ = 2;
        sv.ids_ = ids;
        sv.vals_ = vals;
        sv.token_seq_len_ = 3;
        sv.token_sequence_ = token_seq;
        cell.InsertVector(sv, 0);
    }
    {
        uint32_t ids[] = {10};
        float vals[] = {3.0f};
        uint32_t token_seq[] = {10};
        SparseVector sv;
        sv.len_ = 1;
        sv.ids_ = ids;
        sv.vals_ = vals;
        sv.token_seq_len_ = 1;
        sv.token_sequence_ = token_seq;
        cell.InsertVector(sv, 1);
    }

    // Serialize
    std::vector<char> buffer(1024 * 1024);
    BufferStreamWriter writer(buffer.data());
    cell.Serialize(writer);

    // Deserialize into a new cell
    SparseTermDataCell cell2(1.0f, 100, allocator.get(), false, nullptr, store_positions, 64);
    auto reader_func = [&buffer](uint64_t offset, uint64_t size, void* dest) {
        memcpy(dest, buffer.data() + offset, size);
    };
    ReadFuncStreamReader reader(reader_func, 0, writer.GetCursor());
    cell2.Deserialize(reader);

    // Verify positions survive round-trip
    auto pos0 = cell2.GetPositions(10, 0);
    REQUIRE(pos0.size() == 2);
    REQUIRE(pos0[0] == 0);
    REQUIRE(pos0[1] == 2);

    auto pos1 = cell2.GetPositions(10, 1);
    REQUIRE(pos1.size() == 1);
    REQUIRE(pos1[0] == 0);

    auto pos_20 = cell2.GetPositions(20, 0);
    REQUIRE(pos_20.size() == 1);
    REQUIRE(pos_20[0] == 1);
}
