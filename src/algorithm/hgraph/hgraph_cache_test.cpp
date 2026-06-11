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

#include "hgraph_cache.h"

#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"

TEST_CASE("HGraphCache Serialize & Deserialize", "[ut][hgraph_cache]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    SECTION("empty cache") {
        vsag::HGraphCache cache1(allocator.get());
        std::stringstream ss;
        vsag::IOStreamWriter writer(ss);
        cache1.Serialize(writer);

        vsag::HGraphCache cache2(allocator.get());
        vsag::IOStreamReader reader(ss);
        cache2.Deserialize(reader);

        REQUIRE(cache2.source_ids_.empty());
        REQUIRE(cache2.neighbors_.empty());
    }

    SECTION("cache with source_ids of different lengths") {
        vsag::HGraphCache cache1(allocator.get());
        cache1.source_ids_.push_back("a");
        cache1.source_ids_.push_back("bb");
        cache1.source_ids_.push_back("ccc");
        cache1.source_ids_.push_back("long_source_id_12345");

        std::stringstream ss;
        vsag::IOStreamWriter writer(ss);
        cache1.Serialize(writer);

        vsag::HGraphCache cache2(allocator.get());
        vsag::IOStreamReader reader(ss);
        cache2.Deserialize(reader);

        REQUIRE(cache2.source_ids_.size() == 4);
        REQUIRE(cache2.source_ids_[0] == "a");
        REQUIRE(cache2.source_ids_[1] == "bb");
        REQUIRE(cache2.source_ids_[2] == "ccc");
        REQUIRE(cache2.source_ids_[3] == "long_source_id_12345");
    }

    SECTION("cache with neighbors") {
        vsag::HGraphCache cache1(allocator.get());
        cache1.source_ids_.push_back("a");
        cache1.source_ids_.push_back("bb");
        cache1.source_ids_.push_back("ccc");
        cache1.source_ids_.push_back("dddd");
        cache1.source_ids_.push_back("eeeee");

        vsag::Vector<vsag::InnerIdType> neighbors1(allocator.get());
        neighbors1.push_back(0);
        neighbors1.push_back(1);
        neighbors1.push_back(2);
        cache1.neighbors_.emplace("a", std::move(neighbors1));

        vsag::Vector<vsag::InnerIdType> neighbors2(allocator.get());
        neighbors2.push_back(3);
        neighbors2.push_back(4);
        cache1.neighbors_.emplace("bb", std::move(neighbors2));

        std::stringstream ss;
        vsag::IOStreamWriter writer(ss);
        cache1.Serialize(writer);

        vsag::HGraphCache cache2(allocator.get());
        vsag::IOStreamReader reader(ss);
        cache2.Deserialize(reader);

        REQUIRE(cache2.neighbors_.size() == 2);
        REQUIRE(cache2.neighbors_.at("a").size() == 3);
        REQUIRE(cache2.neighbors_.at("a")[0] == 0);
        REQUIRE(cache2.neighbors_.at("a")[1] == 1);
        REQUIRE(cache2.neighbors_.at("a")[2] == 2);
        REQUIRE(cache2.neighbors_.at("bb").size() == 2);
        REQUIRE(cache2.neighbors_.at("bb")[0] == 3);
        REQUIRE(cache2.neighbors_.at("bb")[1] == 4);
    }

    SECTION("cache with both source_ids and neighbors") {
        vsag::HGraphCache cache1(allocator.get());
        cache1.source_ids_.push_back("x");
        cache1.source_ids_.push_back("yy");

        vsag::Vector<vsag::InnerIdType> neighbors1(allocator.get());
        neighbors1.push_back(0);
        neighbors1.push_back(1);
        cache1.neighbors_.emplace("x", std::move(neighbors1));

        std::stringstream ss;
        vsag::IOStreamWriter writer(ss);
        cache1.Serialize(writer);

        vsag::HGraphCache cache2(allocator.get());
        vsag::IOStreamReader reader(ss);
        cache2.Deserialize(reader);

        REQUIRE(cache2.source_ids_.size() == 2);
        REQUIRE(cache2.source_ids_[0] == "x");
        REQUIRE(cache2.neighbors_.size() == 1);
        REQUIRE(cache2.neighbors_.at("x").size() == 2);
    }
}

TEST_CASE("HGraphCache GetNeighbors", "[ut][hgraph_cache]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    SECTION("source_id not found") {
        vsag::HGraphCache cache(allocator.get());
        auto result = cache.GetNeighbors("nonexistent");
        REQUIRE(result.empty());
    }

    SECTION("get neighbors for existing source_id") {
        vsag::HGraphCache cache(allocator.get());
        cache.source_ids_.push_back("a");
        cache.source_ids_.push_back("bb");
        cache.source_ids_.push_back("ccc");
        cache.source_ids_.push_back("dddddd");

        vsag::Vector<vsag::InnerIdType> neighbors(allocator.get());
        neighbors.push_back(0);
        neighbors.push_back(1);
        neighbors.push_back(2);
        cache.neighbors_.emplace("a", std::move(neighbors));

        auto result = cache.GetNeighbors("a");
        REQUIRE(result.size() == 2);
        REQUIRE(result[0] == "bb");
        REQUIRE(result[1] == "ccc");
    }

    SECTION("get neighbors after deserialize") {
        vsag::HGraphCache cache1(allocator.get());
        cache1.source_ids_.push_back("key_a");
        cache1.source_ids_.push_back("key_bb");
        cache1.source_ids_.push_back("key_ccc");

        vsag::Vector<vsag::InnerIdType> neighbors(allocator.get());
        neighbors.push_back(0);
        neighbors.push_back(1);
        neighbors.push_back(2);
        cache1.neighbors_.emplace("key_a", std::move(neighbors));

        std::stringstream ss;
        vsag::IOStreamWriter writer(ss);
        cache1.Serialize(writer);

        vsag::HGraphCache cache2(allocator.get());
        vsag::IOStreamReader reader(ss);
        cache2.Deserialize(reader);

        auto result = cache2.GetNeighbors("key_a");
        REQUIRE(result.size() == 2);
        REQUIRE(result[0] == "key_bb");
        REQUIRE(result[1] == "key_ccc");
    }
}
