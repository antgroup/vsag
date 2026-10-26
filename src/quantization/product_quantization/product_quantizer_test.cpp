
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

#include "product_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_test.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag_exception.h"

using namespace vsag;

const auto dims = {128, 256};
const auto counts = {300};

template <MetricType metric>
void
TestQuantizerEncodeDecodeMetricPQ(
    uint64_t dim, int64_t pq_dim, int count, float error = 1e-5, float error_same = 1e-2) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    ProductQuantizer<metric> quantizer(dim, pq_dim, allocator.get());
    TestQuantizerEncodeDecode(quantizer, dim, count, error);
    TestQuantizerEncodeDecodeSame(quantizer, dim, count, 255, error_same);
}

TEST_CASE("ProductQuantizer Encode and Decode", "[ut][ProductQuantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 8.0F / 255.0F;
    int64_t pq_dim;
    for (auto dim : dims) {
        if (dim % 2 == 0) {
            pq_dim = dim / 2;
        } else {
            pq_dim = dim;
        }
        for (auto count : counts) {
            auto error_same = (float)(dim * 255 * 0.01);
            error_same *= (dim / pq_dim);
            error *= (dim / pq_dim);
            TestQuantizerEncodeDecodeMetricPQ<metrics[0]>(dim, pq_dim, count, error, error_same);
            TestQuantizerEncodeDecodeMetricPQ<metrics[1]>(dim, pq_dim, count, error, error_same);
        }
    }
}

template <MetricType metric>
void
TestComputeMetricPQ(uint64_t dim, int64_t pq_dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    ProductQuantizer<metric> quantizer(dim, pq_dim, allocator.get());
    TestComputer<ProductQuantizer<metric>, metric>(quantizer, dim, count, error);
    TestComputeCodes<ProductQuantizer<metric>, metric>(quantizer, dim, count, error * dim);
}

TEST_CASE("ProductQuantizer Compute", "[ut][ProductQuantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR,
        MetricType::METRIC_TYPE_IP,
        MetricType::METRIC_TYPE_COSINE,
    };
    float error = 8.0F / 255.0F;
    int64_t pq_dim;
    for (auto dim : dims) {
        if (dim % 2 == 0) {
            pq_dim = dim / 2;
        } else {
            pq_dim = dim;
        }
        error *= (dim / pq_dim);
        for (auto count : counts) {
            TestComputeMetricPQ<metrics[0]>(dim, pq_dim, count, error);
            TestComputeMetricPQ<metrics[1]>(dim, pq_dim, count, error);
            TestComputeMetricPQ<metrics[2]>(dim, pq_dim, count, error);
        }
    }
}

template <MetricType metric>
void
TestSerializeAndDeserializeMetricPQ(uint64_t dim, int64_t pq_dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    ProductQuantizer<metric> quantizer1(dim, pq_dim, allocator.get());
    ProductQuantizer<metric> quantizer2(dim, pq_dim, allocator.get());
    TestSerializeAndDeserialize<ProductQuantizer<metric>, metric, false>(
        quantizer1, quantizer2, dim, count, error);
}

TEST_CASE("ProductQuantizer Serialize and Deserialize", "[ut][ProductQuantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 8.0F / 255.0F;
    int64_t pq_dim;
    for (auto dim : dims) {
        if (dim % 2 == 0) {
            pq_dim = dim / 2;
        } else {
            pq_dim = dim;
        }
        error *= (dim / pq_dim);
        for (auto count : counts) {
            TestSerializeAndDeserializeMetricPQ<metrics[0]>(dim, pq_dim, count, error);
            TestSerializeAndDeserializeMetricPQ<metrics[1]>(dim, pq_dim, count, error);
            TestSerializeAndDeserializeMetricPQ<metrics[2]>(dim, pq_dim, count, error);
        }
    }
}

TEST_CASE("ProductQuantizer Deserialize Rejects Tampered Geometry", "[ut][ProductQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 32;
    constexpr int64_t pq_dim = 8;
    constexpr int64_t subspace_dim = dim / pq_dim;
    constexpr uint64_t codebook_len = dim * 256;
    using QuantizerT = ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>;

    QuantizerT original(dim, pq_dim, allocator.get());
    std::stringstream output;
    IOStreamWriter writer(output);
    original.Serialize(writer);
    const auto blob = output.str();

    // Consume the base header using its serialized types to locate the PQ metadata.
    std::stringstream header(blob);
    IOStreamReader header_reader(header);
    uint64_t stored_dim = 0;
    MetricType stored_metric;
    uint64_t stored_code_size = 0;
    bool stored_trained = false;
    StreamReader::ReadObj(header_reader, stored_dim);
    StreamReader::ReadObj(header_reader, stored_metric);
    StreamReader::ReadObj(header_reader, stored_code_size);
    StreamReader::ReadObj(header_reader, stored_trained);
    const auto pq_offset = header_reader.GetCursor();
    REQUIRE(stored_dim == dim);
    REQUIRE(blob.size() == pq_offset + 3 * sizeof(uint64_t) + codebook_len * sizeof(float));

    SECTION("untampered roundtrip") {
        QuantizerT loaded(dim, pq_dim, allocator.get());
        std::stringstream input(blob);
        IOStreamReader reader(input);
        REQUIRE_NOTHROW(loaded.Deserialize(reader));
        std::stringstream roundtrip;
        IOStreamWriter roundtrip_writer(roundtrip);
        loaded.Serialize(roundtrip_writer);
        REQUIRE(roundtrip.str() == blob);
    }

    auto expect_invalid = [&](uint64_t offset, auto value) {
        std::string tampered = blob;
        std::memcpy(tampered.data() + offset, &value, sizeof(value));
        QuantizerT loaded(dim, pq_dim, allocator.get());
        std::stringstream input(tampered);
        IOStreamReader reader(input);
        try {
            loaded.Deserialize(reader);
            FAIL("malformed PQ metadata was accepted");
        } catch (const VsagException& error) {
            REQUIRE(error.error_.type == ErrorType::INVALID_BINARY);
            // Rejection must precede reading the codebook payload.
            REQUIRE(reader.GetCursor() == pq_offset + 3 * sizeof(uint64_t));
        }
    };

    SECTION("shrunk pq dimension") {
        expect_invalid(pq_offset, pq_dim - 1);
    }
    SECTION("negative pq dimension") {
        expect_invalid(pq_offset, int64_t{-1});
    }
    SECTION("oversized pq dimension") {
        expect_invalid(pq_offset, std::numeric_limits<int64_t>::max());
    }
    SECTION("shrunk subspace dimension") {
        expect_invalid(pq_offset + sizeof(int64_t), subspace_dim - 1);
    }
    SECTION("negative subspace dimension") {
        expect_invalid(pq_offset + sizeof(int64_t), int64_t{-1});
    }
    SECTION("oversized subspace dimension") {
        expect_invalid(pq_offset + sizeof(int64_t), std::numeric_limits<int64_t>::max());
    }
    SECTION("shrunk codebook length") {
        expect_invalid(pq_offset + 2 * sizeof(int64_t), codebook_len / 2);
    }
    SECTION("oversized codebook length") {
        expect_invalid(pq_offset + 2 * sizeof(int64_t), std::numeric_limits<uint64_t>::max());
    }
    SECTION("shrunk header dimension") {
        expect_invalid(0, uint64_t{dim / 2});
    }
    SECTION("oversized header dimension") {
        expect_invalid(0, std::numeric_limits<uint64_t>::max());
    }
}
