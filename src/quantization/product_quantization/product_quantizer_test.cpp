
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

#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_test.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"
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
    using QuantizerT = ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>;
    constexpr uint64_t codebook_len = dim * QuantizerT::CENTROIDS_PER_SUBSPACE;
    QuantizerT original(dim, pq_dim, allocator.get());
    std::stringstream output;
    IOStreamWriter writer(output);
    original.Serialize(writer);
    const auto blob = output.str();

    // Offsets intentionally follow the 1.0 Quantizer::Serialize field order:
    // dim (at offset zero), metric, code size, then training state. Update these
    // offsets if that layout changes; the blob-size assertion checks its total size.
    constexpr uint64_t geometry_offset =
        sizeof(uint64_t) + sizeof(MetricType) + sizeof(uint64_t) + sizeof(bool);
    REQUIRE(blob.size() == geometry_offset + 3 * sizeof(uint64_t) + codebook_len * sizeof(float));

    SECTION("untampered blob") {
        QuantizerT restored(dim, pq_dim, allocator.get());
        std::stringstream input(blob);
        IOStreamReader reader(input);
        REQUIRE_NOTHROW(restored.Deserialize(reader));
        std::stringstream roundtrip;
        IOStreamWriter roundtrip_writer(roundtrip);
        restored.Serialize(roundtrip_writer);
        REQUIRE(roundtrip.str() == blob);
    }

    auto expect_invalid = [&](uint64_t offset, auto value) {
        auto tampered = blob;
        std::memcpy(tampered.data() + offset, &value, sizeof(value));
        QuantizerT restored(dim, pq_dim, allocator.get());
        std::stringstream input(tampered);
        IOStreamReader reader(input);
        try {
            restored.Deserialize(reader);
            FAIL("Malformed PQ metadata was accepted");
        } catch (const VsagException& error) {
            REQUIRE(error.error_.type == ErrorType::INVALID_BINARY);
            // Reject metadata before reading the codebook payload.
            REQUIRE(reader.GetCursor() == geometry_offset + 3 * sizeof(uint64_t));
        }
    };

    SECTION("pq dimension") {
        for (int64_t value : {int64_t{0},
                              int64_t{-1},
                              pq_dim / 2,
                              pq_dim * 2,
                              std::numeric_limits<int64_t>::max()}) {
            expect_invalid(geometry_offset, value);
        }
    }
    SECTION("subspace dimension") {
        for (int64_t value : {int64_t{0},
                              int64_t{-1},
                              int64_t{2},
                              int64_t{5},
                              std::numeric_limits<int64_t>::max()}) {
            expect_invalid(geometry_offset + sizeof(int64_t), value);
        }
    }
    SECTION("codebook length") {
        for (uint64_t value : {uint64_t{0},
                               codebook_len / 2,
                               codebook_len + 1,
                               uint64_t{1} << 32,
                               std::numeric_limits<uint64_t>::max()}) {
            expect_invalid(geometry_offset + 2 * sizeof(int64_t), value);
        }
    }
    SECTION("base dimension") {
        for (uint64_t value : {uint64_t{0},
                               uint64_t{dim / 2},
                               uint64_t{dim * 2},
                               std::numeric_limits<uint64_t>::max()}) {
            // dim is the first field in the 1.0 base wrapper (see layout above).
            expect_invalid(0, value);
        }
    }
}
