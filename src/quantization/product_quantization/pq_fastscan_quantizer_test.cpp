
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

#include "pq_fastscan_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_test.h"

using namespace vsag;

const auto dims = {128, 256};
const auto counts = {300};

template <MetricType metric>
void
TestQuantizerEncodeDecodeMetricPQFS(uint64_t dim, int64_t pq_dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    PQFastScanQuantizer<metric> quantizer(dim, pq_dim, allocator.get());
    TestQuantizerEncodeDecode(quantizer, dim, count, error);
}

TEST_CASE("PQFSQuantizer Encode and Decode", "[ut][PQFSQuantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 1.0F / 255.0F;
    for (auto dim : dims) {
        int64_t pq_dim = dim;
        for (auto count : counts) {
            TestQuantizerEncodeDecodeMetricPQFS<metrics[0]>(dim, pq_dim, count, error);
            TestQuantizerEncodeDecodeMetricPQFS<metrics[1]>(dim, pq_dim, count, error);
        }
    }
}

template <MetricType metric>
void
TestPackageUnpackMetricPQFS(uint64_t dim, int64_t pq_dim) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    PQFastScanQuantizer<metric> quantizer(dim, pq_dim, allocator.get());
    constexpr int count = PQFastScanQuantizer<metric>::BLOCK_SIZE_PACKAGE;
    size_t code_size = quantizer.GetCodeSize();
    std::vector<uint8_t> original_codes(count * code_size);
    for (size_t i = 0; i < original_codes.size(); ++i) {
        original_codes[i] = static_cast<uint8_t>(rand() % 256);
    }

    std::vector<uint8_t> packaged(code_size * count);
    quantizer.Package32(original_codes.data(), packaged.data(), -1);
    std::vector<uint8_t> unpacked_codes(count * code_size);
    quantizer.Unpack32(packaged.data(), unpacked_codes.data());
    for (size_t i = 0; i < original_codes.size(); ++i) {
        REQUIRE(original_codes[i] == unpacked_codes[i]);
    }
}

TEST_CASE("PQFSQuantizer Package32 and Unpack32", "[ut][PQFSQuantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (auto dim : dims) {
        int64_t pq_dim = dim;
        TestPackageUnpackMetricPQFS<metrics[0]>(dim, pq_dim);
        TestPackageUnpackMetricPQFS<metrics[0]>(dim, pq_dim / 2);
        TestPackageUnpackMetricPQFS<metrics[1]>(dim, pq_dim);
        TestPackageUnpackMetricPQFS<metrics[1]>(dim, pq_dim / 2);
    }
}

template <MetricType metric>
void
TestComputerBatchPQFS(PQFastScanQuantizer<metric>& quant,
                      size_t dim,
                      uint32_t count,
                      float error = 1e-5F,
                      bool retrain = true) {
    auto query_count = 100;
    bool need_normalize = true;
    if constexpr (metric == vsag::MetricType::METRIC_TYPE_COSINE) {
        need_normalize = false;
    }
    auto vecs = fixtures::generate_vectors(count, dim, need_normalize);
    auto queries = fixtures::generate_vectors(query_count, dim, need_normalize);
    if (retrain) {
        quant.ReTrain(vecs.data(), count);
    }

    auto gt_func = [&](int base_idx, int query_idx) -> float {
        if constexpr (metric == vsag::MetricType::METRIC_TYPE_IP) {
            return 1 - InnerProduct(
                           vecs.data() + base_idx * dim, queries.data() + query_idx * dim, &dim);
        } else if constexpr (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            return L2Sqr(vecs.data() + base_idx * dim, queries.data() + query_idx * dim, &dim);
        } else if constexpr (metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            std::vector<float> v1(dim);
            std::vector<float> v2(dim);
            Normalize(vecs.data() + base_idx * dim, v1.data(), dim);
            Normalize(queries.data() + query_idx * dim, v2.data(), dim);
            return 1 - InnerProduct(v1.data(), v2.data(), &dim);
        }
    };

    int64_t new_count = (count + 31) / 32 * 32;
    std::vector<uint8_t> codes(quant.GetCodeSize() * new_count);
    std::vector<uint8_t> packaged_codes(quant.GetCodeSize() * new_count);
    quant.EncodeBatch(vecs.data(), codes.data(), count);
    for (int64_t i = 0; i < new_count; i += 32) {
        quant.Package32(codes.data() + i * quant.GetCodeSize(),
                        packaged_codes.data() + i * quant.GetCodeSize(),
                        -1);
    }

    for (int i = 0; i < query_count; ++i) {
        std::shared_ptr<Computer<PQFastScanQuantizer<metric>>> computer;
        computer = quant.FactoryComputer();
        computer->SetQuery(queries.data() + i * dim);
        std::vector<float> dists(count);

        quant.ScanBatchDists(*computer, count, packaged_codes.data(), dists.data());
        for (int j = 0; j < count; ++j) {
            auto gt = gt_func(j, i);
            REQUIRE(std::abs(dists[j] - gt) <= error);
        }
        REQUIRE_THROWS(quant.ComputeDistImpl(*computer, packaged_codes.data(), dists.data()));
        REQUIRE_THROWS(
            quant.ComputeImpl(packaged_codes.data(), packaged_codes.data() + quant.GetCodeSize()));
    }
}

template <MetricType metric>
void
TestComputeMetricPQFS(int64_t dim, int64_t pq_dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    PQFastScanQuantizer<metric> quantizer(dim, pq_dim, allocator.get());
    TestComputerBatchPQFS(quantizer, dim, count, error);
}

TEST_CASE("PQFSQuantizer Compute", "[ut][PQFSQuantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR,
        MetricType::METRIC_TYPE_IP,
        MetricType::METRIC_TYPE_COSINE,
    };
    float error = 0.08F;
    for (auto dim : dims) {
        int64_t pq_dim = dim;
        for (auto count : counts) {
            TestComputeMetricPQFS<metrics[0]>(dim, pq_dim, count, error);
            TestComputeMetricPQFS<metrics[1]>(dim, pq_dim, count, error);
            TestComputeMetricPQFS<metrics[2]>(dim, pq_dim, count, error);
        }
    }
}

template <MetricType metric>
void
TestSerializeAndDeserializeMetricPQFS(uint64_t dim, int64_t pq_dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    PQFastScanQuantizer<metric> quantizer1(dim, pq_dim, allocator.get());
    PQFastScanQuantizer<metric> quantizer2(dim, pq_dim, allocator.get());
    TestComputerBatchPQFS(quantizer1, dim, count, error);
    fixtures::TempDir dir("quantizer");
    auto filename = dir.GenerateRandomFile();
    std::ofstream outfile(filename.c_str(), std::ios::binary);
    IOStreamWriter writer(outfile);
    quantizer1.Serialize(writer);
    outfile.close();

    std::ifstream infile(filename.c_str(), std::ios::binary);
    IOStreamReader reader(infile);
    quantizer2.Deserialize(reader);
    infile.close();
    TestComputerBatchPQFS(quantizer2, dim, count, error, false);
}

TEST_CASE("PQFSQuantizer Serialize and Deserialize", "[ut][PQFSQuantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 0.08F;
    int64_t pq_dim;
    for (auto dim : dims) {
        pq_dim = dim;
        for (auto count : counts) {
            TestSerializeAndDeserializeMetricPQFS<metrics[0]>(dim, pq_dim, count, error);
            TestSerializeAndDeserializeMetricPQFS<metrics[1]>(dim, pq_dim, count, error);
            TestSerializeAndDeserializeMetricPQFS<metrics[2]>(dim, pq_dim, count, error);
        }
    }
}
