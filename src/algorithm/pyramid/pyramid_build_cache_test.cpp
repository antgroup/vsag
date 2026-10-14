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

#include "algorithm/pyramid/pyramid_build_cache.h"

#include <limits>
#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"

namespace {

void
PopulateCache(vsag::BuildCache& cache,
              vsag::Allocator* allocator,
              const std::string& first,
              const std::string& second) {
    cache.source_ids_.push_back(first);
    cache.source_ids_.push_back(second);
    vsag::Vector<vsag::InnerIdType> neighbors(allocator);
    neighbors.push_back(0);
    neighbors.push_back(1);
    cache.neighbors_.emplace(first, std::move(neighbors));
}

}  // namespace

TEST_CASE("PyramidBuildCache Serialize & Deserialize", "[ut][pyramid_build_cache]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidBuildCache cache(allocator.get());
    PopulateCache(
        cache.CreateGraphCache("site", "continent/country"), allocator.get(), "site-a", "site-b");
    PopulateCache(
        cache.CreateGraphCache("taxonomy", "continent/country"), allocator.get(), "tax-a", "tax-b");

    REQUIRE_FALSE(cache.Empty());
    REQUIRE(cache.GetGraphCache("missing", "continent/country") == nullptr);

    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    cache.Serialize(writer);

    vsag::PyramidBuildCache restored(allocator.get());
    vsag::IOStreamReader reader(stream);
    restored.Deserialize(reader);

    auto* site_cache = restored.GetGraphCache("site", "continent/country");
    REQUIRE(site_cache != nullptr);
    REQUIRE(site_cache->GetNeighbors("site-a") == std::vector<std::string>{"site-b"});

    auto* taxonomy_cache = restored.GetGraphCache("taxonomy", "continent/country");
    REQUIRE(taxonomy_cache != nullptr);
    REQUIRE(taxonomy_cache->GetNeighbors("tax-a") == std::vector<std::string>{"tax-b"});
    REQUIRE(restored.GetGraphCache("site", "missing") == nullptr);
}

TEST_CASE("PyramidBuildCache graph keys are unambiguous", "[ut][pyramid_build_cache]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidBuildCache cache(allocator.get());
    PopulateCache(cache.CreateGraphCache("a", "bc"), allocator.get(), "first", "first-neighbor");
    PopulateCache(cache.CreateGraphCache("ab", "c"), allocator.get(), "second", "second-neighbor");

    REQUIRE(cache.GetGraphCache("a", "bc")->GetNeighbors("first") ==
            std::vector<std::string>{"first-neighbor"});
    REQUIRE(cache.GetGraphCache("ab", "c")->GetNeighbors("second") ==
            std::vector<std::string>{"second-neighbor"});
}

TEST_CASE("PyramidBuildCache counts matched source IDs", "[ut][pyramid_build_cache]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidBuildCache cache(allocator.get());
    PopulateCache(cache.CreateGraphCache("site", ""), allocator.get(), "a", "b");
    PopulateCache(cache.CreateGraphCache("site", "child"), allocator.get(), "a", "c");

    const std::string source_ids[]{"a", "b", "missing"};
    REQUIRE(cache.CountMatchedSourceIds(source_ids, 3) == 2);
    REQUIRE(cache.CountMatchedSourceIds(source_ids, 0) == 0);
}

TEST_CASE("PyramidBuildCache empty cache remains empty", "[ut][pyramid_build_cache]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidBuildCache cache(allocator.get());
    cache.CreateGraphCache("site", "empty");
    REQUIRE(cache.Empty());
}

TEST_CASE("PyramidBuildCache rejects truncated declared allocations", "[ut][pyramid_build_cache]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    vsag::StreamWriter::WriteObj(writer, std::numeric_limits<uint64_t>::max());

    vsag::PyramidBuildCache cache(allocator.get());
    vsag::IOStreamReader reader(stream);
    REQUIRE_THROWS(cache.Deserialize(reader));
}

TEST_CASE("PyramidBuildCache group owners and legacy replacement", "[ut][pyramid_build_cache]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidBuildCache cache(allocator.get());
    PopulateCache(cache.CreateGraphCache("site", ""), allocator.get(), "a", "b");
    cache.SetGroupOwners("site", "", {1, 1});
    REQUIRE_THROWS(cache.SetGroupOwners("site", "", {2, 2}));
    REQUIRE_THROWS(cache.SetGroupOwners("site", "", {1, 0}));
    REQUIRE_THROWS(cache.SetGroupOwners("site", "", {0}));
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    cache.Serialize(writer);
    vsag::PyramidBuildCache restored(allocator.get());
    vsag::IOStreamReader reader(stream);
    restored.Deserialize(reader);
    REQUIRE(restored.GetGroupOwners("site", "") != nullptr);
    REQUIRE(*restored.GetGroupOwners("site", "") == vsag::PyramidBuildCache::GroupOwners{1, 1});

    std::stringstream truncated;
    vsag::IOStreamWriter truncated_writer(truncated);
    vsag::StreamWriter::WriteObj(truncated_writer, std::numeric_limits<uint64_t>::max());
    vsag::IOStreamReader truncated_reader(truncated);
    REQUIRE_THROWS(restored.Deserialize(truncated_reader));
    REQUIRE(*restored.GetGroupOwners("site", "") == vsag::PyramidBuildCache::GroupOwners{1, 1});

    // Independent old-format fixture, deliberately not PyramidBuildCache::Serialize.
    std::stringstream legacy;
    vsag::IOStreamWriter legacy_writer(legacy);
    vsag::StreamWriter::WriteObj(legacy_writer, uint64_t{1});
    vsag::StreamWriter::WriteString(legacy_writer, "4:site");
    cache.GetGraphCache("site", "")->Serialize(legacy_writer);
    vsag::IOStreamReader legacy_reader(legacy);
    restored.Deserialize(legacy_reader);
    REQUIRE(restored.GetGroupOwners("site", "") == nullptr);
    REQUIRE(restored.GetGraphCache("site", "")->GetNeighbors("a") == std::vector<std::string>{"b"});
    std::stringstream ordinary;
    vsag::IOStreamWriter ordinary_writer(ordinary);
    restored.Serialize(ordinary_writer);
    REQUIRE(ordinary.str() == legacy.str());
}

TEST_CASE("PyramidBuildCache rejects invalid serialized groups", "[ut][pyramid_build_cache]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    const auto owner = GENERATE(vsag::InnerIdType{2}, vsag::InnerIdType{0});
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    vsag::StreamWriter::WriteObj(writer, std::numeric_limits<uint64_t>::max());
    vsag::StreamWriter::WriteObj(writer, uint64_t{1});
    vsag::StreamWriter::WriteObj(writer, uint64_t{1});
    vsag::StreamWriter::WriteString(writer, "4:site");
    vsag::BuildCache graph(allocator.get());
    PopulateCache(graph, allocator.get(), "a", "b");
    graph.Serialize(writer);
    vsag::StreamWriter::WriteObj(writer, uint64_t{2});
    vsag::StreamWriter::WriteObj(writer, vsag::InnerIdType{1});
    vsag::StreamWriter::WriteObj(writer, owner);
    vsag::PyramidBuildCache cache(allocator.get());
    vsag::IOStreamReader reader(stream);
    REQUIRE_THROWS(cache.Deserialize(reader));
}
