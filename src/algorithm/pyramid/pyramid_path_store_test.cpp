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

#include "pyramid_path_store.h"

#include <array>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"

TEST_CASE("PyramidPathStore inserts and reorders paths", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidPathStore store(allocator.get());
    const std::array<std::string, 4> source = {"root/a", "", "root/c", "root/d"};

    {
        auto writer = store.AcquireWriter();
        for (uint64_t slot = 0; slot < source.size(); ++slot) {
            writer.Insert(static_cast<vsag::InnerIdType>(slot), source[slot]);
        }
    }
    vsag::Vector<vsag::InnerIdType> inner_ids(allocator.get());
    inner_ids.push_back(2);
    inner_ids.push_back(0);
    inner_ids.push_back(1);
    std::array<std::string, 3> restored;

    REQUIRE(store.GetPaths(inner_ids, restored.data()));
    REQUIRE(restored == std::array<std::string, 3>{"root/c", "root/a", ""});

    {
        auto writer = store.AcquireWriter();
        REQUIRE_THROWS(writer.Insert(0, "duplicate"));
    }
}

TEST_CASE("PyramidPathStore stores zero, one, and multiple paths", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidPathStore store(allocator.get());
    const std::array<std::string, 3> multiple_paths = {"root/a", "root/b", "root/a"};
    {
        auto writer = store.AcquireWriter();
        writer.Prepare(6, 4);
        writer.Insert(5, multiple_paths.data(), multiple_paths.size());
        writer.Insert(2, nullptr, 0);
        writer.Insert(4, "single");
    }
    vsag::Vector<vsag::InnerIdType> inner_ids(allocator.get());
    inner_ids.push_back(5);
    inner_ids.push_back(2);
    inner_ids.push_back(4);
    std::vector<std::vector<std::string>> restored_rows;
    REQUIRE(store.GetPathRows(inner_ids, restored_rows));
    REQUIRE(restored_rows ==
            std::vector<std::vector<std::string>>{{"root/a", "root/b", "root/a"}, {}, {"single"}});

    std::array<std::string, 1> restored_single;
    vsag::Vector<vsag::InnerIdType> single_id(allocator.get());
    single_id.push_back(4);
    REQUIRE(store.GetPaths(single_id, restored_single.data()));
    REQUIRE(restored_single[0] == "single");

    vsag::Vector<vsag::InnerIdType> multiple_id(allocator.get());
    multiple_id.push_back(5);
    REQUIRE_FALSE(store.GetPaths(multiple_id, restored_single.data()));

    vsag::Vector<vsag::InnerIdType> empty_id(allocator.get());
    empty_id.push_back(2);
    REQUIRE_FALSE(store.GetPaths(empty_id, restored_single.data()));

    vsag::Vector<vsag::InnerIdType> hole_ids(allocator.get());
    hole_ids.push_back(3);
    REQUIRE_FALSE(store.GetPaths(hole_ids, restored_single.data()));
    REQUIRE_FALSE(store.GetPathRows(hole_ids, restored_rows));

    vsag::Vector<vsag::InnerIdType> out_of_range_ids(allocator.get());
    out_of_range_ids.push_back(6);
    REQUIRE_FALSE(store.GetPaths(out_of_range_ids, restored_single.data()));

    {
        auto writer = store.AcquireWriter();
        REQUIRE_THROWS(writer.Insert(2, "duplicate"));
        REQUIRE_THROWS(writer.Insert(1, nullptr, 1));
        const std::string path = "too-many";
        REQUIRE_THROWS(writer.Insert(
            1, &path, static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1));
        writer.Insert(1, "valid-after-rejected-insert");
    }
}

TEST_CASE("PyramidPathStore preparation does not shrink slots", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidPathStore store(allocator.get());
    {
        auto writer = store.AcquireWriter();
        writer.Prepare(8, 2);
        writer.Insert(7, "last");
    }
    {
        auto writer = store.AcquireWriter();
        writer.Prepare(3, 1);
        writer.Insert(2, "earlier");
    }
    vsag::Vector<vsag::InnerIdType> present_ids(allocator.get());
    present_ids.push_back(7);
    present_ids.push_back(2);
    std::array<std::string, 2> restored;
    REQUIRE(store.GetPaths(present_ids, restored.data()));
    REQUIRE(restored == std::array<std::string, 2>{"last", "earlier"});

    vsag::Vector<vsag::InnerIdType> hole_ids(allocator.get());
    hole_ids.push_back(3);
    REQUIRE_FALSE(store.GetPaths(hole_ids, restored.data()));
}

TEST_CASE("PyramidPathStore serializes direct dense fields", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    const std::array<std::string, 3> multiple_paths = {"root/a", "root/b", "root/a"};

    vsag::PyramidPathStore store(allocator.get());
    {
        auto writer = store.AcquireWriter();
        writer.Insert(0, multiple_paths.data(), multiple_paths.size());
        writer.Insert(2, nullptr, 0);
        writer.Insert(4, "single");
    }

    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    store.Serialize(writer);

    std::stringstream expected_stream;
    vsag::IOStreamWriter expected_writer(expected_stream);
    const std::vector<uint64_t> expected_offsets = {
        0, std::numeric_limits<uint64_t>::max(), 3, std::numeric_limits<uint64_t>::max(), 3};
    const std::vector<uint16_t> expected_counts = {3, 0, 0, 0, 1};
    vsag::StreamWriter::WriteVector(expected_writer, expected_offsets);
    vsag::StreamWriter::WriteVector(expected_writer, expected_counts);
    vsag::StreamWriter::WriteObj(expected_writer, uint64_t{4});
    for (const auto& path : multiple_paths) {
        vsag::StreamWriter::WriteString(expected_writer, path);
    }
    vsag::StreamWriter::WriteString(expected_writer, "single");
    REQUIRE(stream.str() == expected_stream.str());

    vsag::PyramidPathStore restored_store(allocator.get());
    vsag::IOStreamReader reader(stream);
    restored_store.Deserialize(reader, 5);
    vsag::Vector<vsag::InnerIdType> inner_ids(allocator.get());
    inner_ids.push_back(0);
    inner_ids.push_back(2);
    inner_ids.push_back(4);
    std::vector<std::vector<std::string>> restored_rows;
    REQUIRE(restored_store.GetPathRows(inner_ids, restored_rows));
    REQUIRE(restored_rows ==
            std::vector<std::vector<std::string>>{{"root/a", "root/b", "root/a"}, {}, {"single"}});

    vsag::Vector<vsag::InnerIdType> hole_ids(allocator.get());
    hole_ids.push_back(1);
    REQUIRE_FALSE(restored_store.GetPathRows(hole_ids, restored_rows));
}

TEST_CASE("PyramidPathStore rejects malformed serialization", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    const auto write_payload = [](std::stringstream& stream,
                                  const std::vector<uint64_t>& offsets,
                                  const std::vector<uint16_t>& counts,
                                  const std::vector<std::string>& paths) {
        vsag::IOStreamWriter writer(stream);
        vsag::StreamWriter::WriteVector(writer, offsets);
        vsag::StreamWriter::WriteVector(writer, counts);
        vsag::StreamWriter::WriteObj(writer, static_cast<uint64_t>(paths.size()));
        for (const auto& path : paths) {
            vsag::StreamWriter::WriteString(writer, path);
        }
    };

    SECTION("vector size exceeds maximum") {
        std::stringstream stream;
        vsag::IOStreamWriter writer(stream);
        vsag::StreamWriter::WriteObj(writer, uint64_t{3});

        vsag::PyramidPathStore store(allocator.get());
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 2));
    }

    SECTION("vector size exceeds remaining payload") {
        std::stringstream stream;
        vsag::IOStreamWriter writer(stream);
        vsag::StreamWriter::WriteObj(writer, uint64_t{2});
        vsag::StreamWriter::WriteObj(writer, uint64_t{0});

        vsag::PyramidPathStore store(allocator.get());
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 2));
    }

    SECTION("slot vector sizes do not match") {
        std::stringstream stream;
        write_payload(stream, {0}, {}, {"path"});

        vsag::PyramidPathStore store(allocator.get());
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 1));
    }

    SECTION("hole has a nonzero path count") {
        std::stringstream stream;
        write_payload(stream, {std::numeric_limits<uint64_t>::max()}, {1}, {});

        vsag::PyramidPathStore store(allocator.get());
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 1));
    }

    SECTION("path range is invalid") {
        std::stringstream stream;
        write_payload(stream, {1}, {1}, {"path"});

        vsag::PyramidPathStore store(allocator.get());
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 1));
    }

    SECTION("path count exceeds remaining payload") {
        std::stringstream stream;
        vsag::IOStreamWriter writer(stream);
        vsag::StreamWriter::WriteVector(writer, std::vector<uint64_t>{0});
        vsag::StreamWriter::WriteVector(writer, std::vector<uint16_t>{1});
        vsag::StreamWriter::WriteObj(writer, uint64_t{1});

        vsag::PyramidPathStore store(allocator.get());
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 1));
    }

    SECTION("path string exceeds remaining payload") {
        std::stringstream stream;
        vsag::IOStreamWriter writer(stream);
        vsag::StreamWriter::WriteVector(writer, std::vector<uint64_t>{0});
        vsag::StreamWriter::WriteVector(writer, std::vector<uint16_t>{1});
        vsag::StreamWriter::WriteObj(writer, uint64_t{1});
        vsag::StreamWriter::WriteObj(writer, std::numeric_limits<uint64_t>::max());

        vsag::PyramidPathStore store(allocator.get());
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 1));
    }
}
