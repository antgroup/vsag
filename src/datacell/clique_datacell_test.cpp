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

#include "clique_datacell.h"

#include <catch2/catch_test_macros.hpp>
#include <sstream>

#include "allocator/memory_record_allocator.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"

TEST_CASE("MCI memory usage matches live vector allocations", "[ut][mci][datacell]") {
    fixtures::MemoryRecordAllocator storage;
    fixtures::MemoryRecordAllocator inputs;
    vsag::CliqueDataCell cell(&storage);
    auto check_usage = [&] {
        REQUIRE(cell.GetMemoryUsage() == sizeof(cell) + storage.GetCurrentMemory());
    };
    check_usage();
    cell.Clear(128);
    check_usage();

    // Repeatedly grow the inverse row at node 0 as well as the outer clique vector.
    vsag::Vector<vsag::InnerIdType> members({0, 0}, &inputs);
    for (vsag::InnerIdType id = 1; id < 96; ++id) {
        members[1] = id;
        cell.AppendNewClique(members, 128);
        check_usage();
    }
    REQUIRE(cell.AppendNodeToClique(100, 0, 128, 50));
    check_usage();

    // A load replaces existing delta rows and can throw after allocating a row.
    std::stringstream snapshot;
    vsag::IOStreamWriter writer(snapshot);
    cell.Serialize(writer);
    const auto bytes = snapshot.str();
    std::istringstream truncated(bytes.substr(0, bytes.size() / 2));
    vsag::IOStreamReader partial(truncated);
    REQUIRE_THROWS(cell.Deserialize(partial, 2, 128));
    check_usage();
    vsag::IOStreamReader complete(snapshot);
    cell.Deserialize(complete, 2, 128);
    check_usage();

    cell.Flush(128);
    check_usage();
    REQUIRE(cell.AppendNodeToClique(101, 0, 128, 50));
    check_usage();

    // An invalid clique can still have caused storage to grow for the new node.
    REQUIRE_FALSE(cell.AppendNodeToClique(128, 1000, 129, 50));
    check_usage();
    cell.Clear(0);
    check_usage();
}
