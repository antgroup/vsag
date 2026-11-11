
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
    constexpr int64_t subspace_dim = dim / pq_dim;
    constexpr uint64_t codebook_len = static_cast<uint64_t>(dim) * 256;  // CENTROIDS_PER_SUBSPACE
    using QuantizerT = ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>;

    std::string blob;
    {
        QuantizerT quantizer(dim, pq_dim, allocator.get());
        std::stringstream stream;
        IOStreamWriter writer(stream);
        quantizer.Serialize(writer);
        blob = stream.str();
    }

    // control: an untampered blob must deserialize cleanly
    {
        QuantizerT quantizer(dim, pq_dim, allocator.get());
        std::stringstream stream(blob);
        IOStreamReader reader(stream);
        REQUIRE_NOTHROW(quantizer.Deserialize(reader));
    }

    auto expect_throw = [&](const std::string& tampered) {
        QuantizerT quantizer(dim, pq_dim, allocator.get());
        std::stringstream stream(tampered);
        IOStreamReader reader(stream);
        REQUIRE_THROWS_AS(quantizer.Deserialize(reader), VsagException);
    };

    // locate the quantizer blob triple: int64 pq_dim, int64 subspace_dim, uint64 codebook_len
    uint64_t hit = blob.size();
    for (uint64_t i = 0; i + 24 <= blob.size(); ++i) {
        int64_t pq = 0;
        int64_t sub = 0;
        uint64_t len = 0;
        std::memcpy(&pq, blob.data() + i, 8);
        std::memcpy(&sub, blob.data() + i + 8, 8);
        std::memcpy(&len, blob.data() + i + 16, 8);
        if (pq == pq_dim and sub == subspace_dim and len == codebook_len) {
            hit = i;
            break;
        }
    }
    REQUIRE(hit != blob.size());

    // tampered pq_dim: transpose_codebooks() loop bound overflows the codebook buffers
    {
        std::string tampered = blob;
        auto evil = static_cast<int64_t>(4096);
        std::memcpy(tampered.data() + hit, &evil, 8);
        expect_throw(tampered);
    }

    // tampered subspace_dim
    {
        std::string tampered = blob;
        auto evil = static_cast<int64_t>(subspace_dim + 1);
        std::memcpy(tampered.data() + hit + 8, &evil, 8);
        expect_throw(tampered);
    }

    // tampered codebook length: shorter than the construction-time geometry
    {
        std::string tampered = blob;
        auto evil = static_cast<uint64_t>(codebook_len / 2);
        std::memcpy(tampered.data() + hit + 16, &evil, 8);
        expect_throw(tampered);
    }

    // oversized codebook length: must be rejected before any allocation is attempted
    {
        std::string tampered = blob;
        auto evil = static_cast<uint64_t>(1) << 32;  // ~16 GiB of floats if honored
        std::memcpy(tampered.data() + hit + 16, &evil, 8);
        expect_throw(tampered);
    }

    // tampered dim_: locate it by value instead of assuming a fixed offset, so the
    // test stays correct if the base-class serialization order ever changes
    uint64_t dim_pos = blob.size();
    for (uint64_t i = 0; i + 8 <= hit; ++i) {
        int64_t val = 0;
        std::memcpy(&val, blob.data() + i, 8);
        if (val == dim) {
            dim_pos = i;
            break;
        }
    }
    REQUIRE(dim_pos != blob.size());
    {
        std::string tampered = blob;
        auto evil = static_cast<int64_t>(dim * 2);
        std::memcpy(tampered.data() + dim_pos, &evil, 8);
        expect_throw(tampered);
    }
}
