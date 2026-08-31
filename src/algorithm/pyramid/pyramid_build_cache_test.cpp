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
