
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

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <limits>
#include <sstream>

#include "product_quantizer.h"
#include "safe_allocator.h"
#include "vsag_exception.h"

using namespace vsag;

TEST_CASE("ProductQuantizer Deserialize Valid Geometry", "[ut][ProductQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int dim = 32;
    constexpr int64_t pq_dim = 8;
    ProductQuantizer<> original(dim, pq_dim, allocator.get());
    for (size_t i = 0; i < original.codebooks_.size(); ++i) {
        original.codebooks_[i] = static_cast<float>(i);
    }
    std::stringstream stream;
    IOStreamWriter writer(stream);
    original.Serialize(writer);
    ProductQuantizer<> restored(dim, pq_dim, allocator.get());
    IOStreamReader reader(stream);
    REQUIRE_NOTHROW(restored.Deserialize(reader));
    REQUIRE(restored.codebooks_ == original.codebooks_);
    REQUIRE(restored.GetDim() == dim);
    REQUIRE(restored.GetCodeSize() == pq_dim);
    for (int64_t i = 0; i < pq_dim; ++i) {
        for (int64_t j = 0; j < original.CENTROIDS_PER_SUBSPACE; ++j) {
            for (int64_t k = 0; k < original.subspace_dim_; ++k) {
                REQUIRE(restored.reverse_codebooks_[j * dim + i * original.subspace_dim_ + k] ==
                        original.codebooks_[(i * original.CENTROIDS_PER_SUBSPACE + j) *
                                                original.subspace_dim_ +
                                            k]);
            }
        }
    }
    uint8_t codes[pq_dim] = {0, 1, 2, 3, 4, 5, 6, 7};
    float expected[dim];
    float actual[dim];
    original.DecodeOne(codes, expected);
    restored.DecodeOne(codes, actual);
    for (int i = 0; i < dim; ++i) {
        REQUIRE(actual[i] == expected[i]);
    }
    std::stringstream roundtrip;
    IOStreamWriter roundtrip_writer(roundtrip);
    restored.Serialize(roundtrip_writer);
    REQUIRE(roundtrip.str() == stream.str());
}

TEST_CASE("ProductQuantizer Deserialize Rejects Tampered Geometry", "[ut][ProductQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int dim = 32;
    constexpr int64_t pq_dim = 8;
    ProductQuantizer<> original(dim, pq_dim, allocator.get());
    std::stringstream stream;
    IOStreamWriter writer(stream);
    original.Serialize(writer);
    const auto blob = stream.str();

    // Read the base header in its serialized order to locate the PQ geometry.
    IOStreamReader header_reader(stream);
    uint64_t stored_dim = 0;
    MetricType stored_metric;
    uint64_t stored_code_size = 0;
    bool stored_trained = false;
    StreamReader::ReadObj(header_reader, stored_dim);
    StreamReader::ReadObj(header_reader, stored_metric);
    StreamReader::ReadObj(header_reader, stored_code_size);
    StreamReader::ReadObj(header_reader, stored_trained);
    const auto geometry_offset = header_reader.GetCursor();
    REQUIRE(stored_dim == dim);
    REQUIRE(stored_code_size == pq_dim);

    auto expect_invalid = [&](size_t offset, auto value) {
        auto tampered = blob;
        REQUIRE(offset + sizeof(value) <= tampered.size());
        std::memcpy(tampered.data() + offset, &value, sizeof(value));
        ProductQuantizer<> restored(dim, pq_dim, allocator.get());
        std::stringstream input(tampered);
        IOStreamReader reader(input);
        try {
            restored.Deserialize(reader);
            FAIL("malformed PQ geometry was accepted");
        } catch (const VsagException& error) {
            REQUIRE(error.error_.type == ErrorType::INVALID_BINARY);
        }
        REQUIRE(restored.pq_dim_ == pq_dim);
        REQUIRE(restored.subspace_dim_ == dim / pq_dim);
        REQUIRE(restored.codebooks_.size() == original.codebooks_.size());
        REQUIRE(reader.GetCursor() == geometry_offset + 3 * sizeof(uint64_t));
    };

    SECTION("pq dimension") {
        for (int64_t value : {int64_t{0},
                              int64_t{-1},
                              pq_dim / 2,
                              pq_dim + 1,
                              std::numeric_limits<int64_t>::max()}) {
            CAPTURE(value);
            expect_invalid(geometry_offset, value);
        }
    }
    SECTION("subspace dimension") {
        for (int64_t value : {int64_t{0},
                              int64_t{-1},
                              int64_t{dim / pq_dim / 2},
                              int64_t{dim / pq_dim + 1},
                              std::numeric_limits<int64_t>::max()}) {
            CAPTURE(value);
            expect_invalid(geometry_offset + sizeof(int64_t), value);
        }
    }
    SECTION("codebook length") {
        for (uint64_t value : {uint64_t{0},
                               uint64_t{original.codebooks_.size() / 2},
                               uint64_t{original.codebooks_.size() + 1},
                               uint64_t{1} << 32,
                               std::numeric_limits<uint64_t>::max()}) {
            CAPTURE(value);
            expect_invalid(geometry_offset + 2 * sizeof(int64_t), value);
        }
    }
    SECTION("base dimension") {
        for (uint64_t value : {uint64_t{0},
                               uint64_t{dim / 2},
                               uint64_t{dim * 2},
                               std::numeric_limits<uint64_t>::max()}) {
            CAPTURE(value);
            expect_invalid(0, value);
        }
    }
}
