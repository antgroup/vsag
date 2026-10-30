
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
    using QuantizerT = ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>;
    constexpr uint64_t codebook_len = dim * QuantizerT::CENTROIDS_PER_SUBSPACE;

    QuantizerT original(dim, pq_dim, allocator.get());
    for (size_t i = 0; i < original.codebooks_.size(); ++i) {
        original.codebooks_[i] = static_cast<float>(i % 17);
    }
    std::stringstream output;
    IOStreamWriter writer(output);
    original.Serialize(writer);
    const auto blob = output.str();

    SECTION("untampered roundtrip") {
        QuantizerT restored(dim, pq_dim, allocator.get());
        std::stringstream input(blob);
        IOStreamReader reader(input);
        REQUIRE_NOTHROW(restored.Deserialize(reader));
        std::stringstream roundtrip;
        IOStreamWriter roundtrip_writer(roundtrip);
        restored.Serialize(roundtrip_writer);
        REQUIRE(roundtrip.str() == blob);
        std::vector<float> query(dim, 1.0F);
        std::vector<uint8_t> original_codes(pq_dim);
        std::vector<uint8_t> restored_codes(pq_dim);
        REQUIRE(original.EncodeOne(query.data(), original_codes.data()));
        REQUIRE(restored.EncodeOne(query.data(), restored_codes.data()));
        REQUIRE(restored_codes == original_codes);
        auto original_computer = original.FactoryComputer();
        auto restored_computer = restored.FactoryComputer();
        original.ProcessQuery(query.data(), *original_computer);
        restored.ProcessQuery(query.data(), *restored_computer);
        REQUIRE(original.ComputeDist(*original_computer, original_codes.data()) ==
                restored.ComputeDist(*restored_computer, restored_codes.data()));
    }

    // Follow the base header's serialized field types to locate the PQ metadata.
    const size_t pq_offset =
        sizeof(uint64_t) + sizeof(MetricType) + sizeof(uint64_t) + sizeof(bool);
    const size_t subspace_offset = pq_offset + sizeof(int64_t);
    const size_t length_offset = subspace_offset + sizeof(int64_t);
    REQUIRE(blob.size() == length_offset + sizeof(uint64_t) + codebook_len * sizeof(float));
    auto reject = [&](size_t offset, auto value) {
        auto tampered = blob;
        std::memcpy(tampered.data() + offset, &value, sizeof(value));
        QuantizerT restored(dim, pq_dim, allocator.get());
        std::stringstream input(tampered);
        IOStreamReader reader(input);
        try {
            restored.Deserialize(reader);
            FAIL("malformed PQ metadata was accepted");
        } catch (const VsagException& error) {
            REQUIRE(error.error_.type == ErrorType::INVALID_BINARY);
        }
        // Reject metadata before reading the codebook payload or changing its storage.
        REQUIRE(reader.GetCursor() == length_offset + sizeof(uint64_t));
        REQUIRE(restored.codebooks_.size() == codebook_len);
        REQUIRE(restored.pq_dim_ == pq_dim);
        REQUIRE(restored.subspace_dim_ == dim / pq_dim);
    };

    SECTION("zero pq dimension") {
        reject(pq_offset, int64_t{0});
    }
    SECTION("negative pq dimension") {
        reject(pq_offset, int64_t{-1});
    }
    SECTION("shrunk pq dimension") {
        reject(pq_offset, pq_dim - 1);
    }
    SECTION("oversized pq dimension") {
        reject(pq_offset, std::numeric_limits<int64_t>::max());
    }
    SECTION("zero subspace dimension") {
        reject(subspace_offset, int64_t{0});
    }
    SECTION("negative subspace dimension") {
        reject(subspace_offset, int64_t{-1});
    }
    SECTION("shrunk subspace dimension") {
        reject(subspace_offset, dim / pq_dim - 1);
    }
    SECTION("oversized subspace dimension") {
        reject(subspace_offset, std::numeric_limits<int64_t>::max());
    }
    SECTION("shrunk codebook length") {
        reject(length_offset, codebook_len / 2);
    }
    SECTION("oversized codebook length") {
        reject(length_offset, std::numeric_limits<uint64_t>::max());
    }
    SECTION("shrunk header dimension") {
        reject(0, uint64_t{dim / 2});
    }
    SECTION("oversized header dimension") {
        reject(0, std::numeric_limits<uint64_t>::max());
    }
}
