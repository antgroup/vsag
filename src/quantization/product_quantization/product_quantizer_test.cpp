
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
    float error = 20;
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
    float error = 20;
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
    float error = 20;
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
    using QuantizerT = ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>;
    constexpr uint64_t codebook_len = dim * QuantizerT::CENTROIDS_PER_SUBSPACE;

    QuantizerT original(dim, pq_dim, allocator.get());
    std::stringstream stream;
    IOStreamWriter writer(stream);
    original.Serialize(writer);
    const auto blob = stream.str();

    // Follow the base wrapper's serialized field types to locate the PQ payload.
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
    REQUIRE(stored_code_size == pq_dim);

    SECTION("untampered") {
        QuantizerT loaded(dim, pq_dim, allocator.get());
        std::stringstream input(blob);
        IOStreamReader reader(input);
        REQUIRE_NOTHROW(loaded.Deserialize(reader));
        REQUIRE(loaded.codebooks_ == original.codebooks_);
        REQUIRE(reader.GetCursor() == blob.size());
    }

    auto reject = [&](uint64_t offset, auto value) {
        auto tampered = blob;
        REQUIRE(offset + sizeof(value) <= tampered.size());
        std::memcpy(tampered.data() + offset, &value, sizeof(value));
        QuantizerT loaded(dim, pq_dim, allocator.get());
        std::stringstream input(tampered);
        IOStreamReader reader(input);
        try {
            loaded.Deserialize(reader);
            FAIL("malformed geometry was accepted");
        } catch (const VsagException& error) {
            REQUIRE(error.error_.type == ErrorType::INVALID_BINARY);
        }
        // Rejection must precede reading the codebook payload or resizing its vector.
        REQUIRE(reader.GetCursor() == pq_offset + 2 * sizeof(int64_t) + sizeof(uint64_t));
        REQUIRE(loaded.codebooks_.size() == codebook_len);
        REQUIRE(loaded.pq_dim_ == pq_dim);
        REQUIRE(loaded.subspace_dim_ == subspace_dim);
    };

    SECTION("pq_dim zero") {
        reject(pq_offset, int64_t{0});
    }
    SECTION("pq_dim malformed") {
        for (auto value :
             {int64_t{-1}, pq_dim - 1, pq_dim + 1, std::numeric_limits<int64_t>::max()}) {
            reject(pq_offset, value);
        }
    }
    SECTION("subspace_dim malformed") {
        for (auto value : {int64_t{-1},
                           int64_t{0},
                           subspace_dim - 1,
                           subspace_dim + 1,
                           std::numeric_limits<int64_t>::max()}) {
            reject(pq_offset + sizeof(int64_t), value);
        }
    }
    SECTION("codebook length malformed") {
        for (auto value : {uint64_t{0},
                           codebook_len / 2,
                           codebook_len + 1,
                           uint64_t{1} << 32,
                           std::numeric_limits<uint64_t>::max()}) {
            reject(pq_offset + 2 * sizeof(int64_t), value);
        }
    }
    SECTION("dim malformed") {
        for (auto value : {uint64_t{0},
                           uint64_t{dim - 1},
                           uint64_t{dim + 1},
                           std::numeric_limits<uint64_t>::max()}) {
            reject(0, value);
        }
    }
}
