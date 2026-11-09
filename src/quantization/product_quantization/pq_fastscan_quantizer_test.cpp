
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

#include <algorithm>
#include <cmath>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_test.h"
#include "simd/fp32_simd.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"
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
    uint64_t code_size = quantizer.GetCodeSize();
    std::vector<uint8_t> original_codes(count * code_size);
    for (uint64_t i = 0; i < original_codes.size(); ++i) {
        original_codes[i] = static_cast<uint8_t>(rand() % 256);
    }

    std::vector<uint8_t> packaged(code_size * count);
    quantizer.Package32(original_codes.data(), packaged.data(), -1);
    std::vector<uint8_t> unpacked_codes(count * code_size);
    quantizer.Unpack32(packaged.data(), unpacked_codes.data());
    for (uint64_t i = 0; i < original_codes.size(); ++i) {
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
                      uint64_t dim,
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
        computer = std::dynamic_pointer_cast<Computer<PQFastScanQuantizer<metric>>>(
            quant.FactoryComputer());
        computer->SetQuery(queries.data() + i * dim);
        std::vector<float> dists(count);

        quant.ScanBatchDists(*computer, count, packaged_codes.data(), dists.data());
        for (int j = 0; j < count; ++j) {
            auto gt = gt_func(j, i);
            REQUIRE(std::abs(dists[j] - gt) <= error);
        }
        std::vector<float> row_dists(count);
        quant.ScanRowBatchDists(*computer, count, codes.data(), row_dists.data());
        for (int j = 0; j < count; ++j) {
            REQUIRE(row_dists[j] == dists[j]);
        }
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
TestRowDistancesPQFS() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 12;
    for (int64_t pq_dim : {3, 4}) {
        PQFastScanQuantizer<metric> quant(dim, pq_dim, allocator.get());
        // Fixed, nonzero codebooks exercise LUT scaling and the lower offset without training.
        for (uint64_t i = 0; i < quant.codebooks_.size(); ++i) {
            quant.codebooks_[i] = static_cast<float>((i * 13) % 29 + 1) / 31.0F;
        }
        std::vector<float> query(dim);
        for (int64_t i = 0; i < dim; ++i) {
            query[i] = static_cast<float>(i + 1) / 17.0F;
        }
        auto computer = quant.FactoryComputer();
        computer->SetQuery(query.data());
        const auto code_size = quant.GetCodeSize();
        for (uint64_t count : {0, 1, 7, 31, 32, 33, 63, 65}) {
            // Deliberately no row padding: the row API must not read a full packed tail block.
            std::vector<uint8_t> rows(count * code_size);
            for (uint64_t i = 0; i < rows.size(); ++i) {
                rows[i] = static_cast<uint8_t>((i * 37 + 23) % 256);
            }
            const auto padded_count = (count + 31) / 32 * 32;
            std::vector<uint8_t> packed(padded_count * code_size);
            for (uint64_t i = 0; i < count; i += 32) {
                quant.Package32(rows.data() + i * code_size,
                                packed.data() + i * code_size,
                                std::min<uint64_t>(32, count - i));
            }
            constexpr float sentinel = -12345.0F;
            std::vector<float> row_dists(count + 2, sentinel);
            std::vector<float> packed_dists(count + 2, sentinel);
            quant.ScanRowBatchDists(*computer, count, rows.data(), row_dists.data() + 1);
            quant.ScanBatchDists(*computer, count, packed.data(), packed_dists.data() + 1);
            REQUIRE(row_dists.front() == sentinel);
            REQUIRE(row_dists.back() == sentinel);
            REQUIRE(packed_dists.front() == sentinel);
            REQUIRE(packed_dists.back() == sentinel);
            for (uint64_t i = 0; i < count; ++i) {
                REQUIRE(row_dists[i + 1] == packed_dists[i + 1]);
                REQUIRE(quant.ComputeDist(*computer, rows.data() + i * code_size) ==
                        packed_dists[i + 1]);
            }
            std::vector<float> decoded1(dim);
            std::vector<float> decoded2(dim);
            for (uint64_t i = 1; i < count; ++i) {
                const auto* code1 = rows.data() + (i - 1) * code_size;
                const auto* code2 = rows.data() + i * code_size;
                quant.DecodeOne(code1, decoded1.data());
                quant.DecodeOne(code2, decoded2.data());
                float expected;
                if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
                    expected = FP32ComputeL2Sqr(decoded1.data(), decoded2.data(), dim);
                } else {
                    expected = 1.0F - FP32ComputeIP(decoded1.data(), decoded2.data(), dim);
                }
                REQUIRE(std::abs(quant.Compute(code1, code2) - expected) <= 1e-5F);
            }
        }
    }
}

TEST_CASE("PQFSQuantizer row and packed distance contracts", "[ut][PQFSQuantizer]") {
    TestRowDistancesPQFS<MetricType::METRIC_TYPE_L2SQR>();
    TestRowDistancesPQFS<MetricType::METRIC_TYPE_IP>();
    TestRowDistancesPQFS<MetricType::METRIC_TYPE_COSINE>();
}

template <MetricType metric>
void
TestAffineLUTDistancesPQFS() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 12;
    constexpr uint64_t train_count = 64;
    std::vector<float> training(train_count * dim);
    for (uint64_t i = 0; i < training.size(); ++i) {
        training[i] = static_cast<float>((i * 13 + i / dim) % 97 + 1) / 97.0F;
    }
    for (int64_t pq_dim : {3, 4}) {
        PQFastScanQuantizer<metric> quant(dim, pq_dim, allocator.get());
        REQUIRE(quant.Train(training.data(), train_count));
        auto computer = quant.FactoryComputer();
        const auto lut_size = pq_dim * quant.CENTROIDS_PER_SUBSPACE;
        // The native computer owns this buffer and releases it through ReleaseComputer.
        computer->buf_ = static_cast<uint8_t*>(
            allocator->Allocate(lut_size * sizeof(uint8_t) + 2 * sizeof(float)));
        for (int64_t subspace = 0; subspace < pq_dim; ++subspace) {
            for (int64_t centroid = 0; centroid < quant.CENTROIDS_PER_SUBSPACE; ++centroid) {
                computer->buf_[subspace * quant.CENTROIDS_PER_SUBSPACE + centroid] =
                    static_cast<uint8_t>((subspace * 53 + centroid * 17) % 256);
            }
        }
        auto* affine = reinterpret_cast<float*>(computer->buf_ + lut_size);
        constexpr float lower = 1.75F;
        affine[1] = lower;
        for (float diff : {0.0F, 3.25F}) {
            affine[0] = diff;
            for (uint64_t count : {1, 7, 31, 32, 33, 65}) {
                CAPTURE(pq_dim, diff, count);
                const auto code_size = quant.GetCodeSize();
                std::vector<uint8_t> rows(count * code_size, 0);
                std::vector<float> expected(count, 0.0F);
                for (uint64_t row = 0; row < count; ++row) {
                    int32_t sum = 0;
                    for (int64_t subspace = 0; subspace < pq_dim; ++subspace) {
                        const auto centroid = (row * 7 + subspace * 3) % 16;
                        rows[row * code_size + subspace / 2] |=
                            static_cast<uint8_t>(centroid << ((subspace % 2) * 4));
                        // Independent oracle: sum the known LUT, then restore every offset.
                        sum += (subspace * 53 + centroid * 17) % 256;
                    }
                    expected[row] = static_cast<float>(sum) * (diff / 255.0F) +
                                    static_cast<float>(pq_dim) * lower;
                    if constexpr (metric != MetricType::METRIC_TYPE_L2SQR) {
                        expected[row] = 1.0F - expected[row];
                    }
                }
                const auto padded_count = (count + 31) / 32 * 32;
                std::vector<uint8_t> packed(padded_count * code_size);
                for (uint64_t row = 0; row < count; row += 32) {
                    quant.Package32(rows.data() + row * code_size,
                                    packed.data() + row * code_size,
                                    std::min<uint64_t>(32, count - row));
                }
                std::vector<float> row_dists(count);
                std::vector<float> packed_dists(count);
                quant.ScanRowBatchDists(*computer, count, rows.data(), row_dists.data());
                quant.ScanBatchDists(*computer, count, packed.data(), packed_dists.data());
                for (uint64_t row = 0; row < count; ++row) {
                    CAPTURE(row);
                    const auto scalar = quant.ComputeDist(*computer, rows.data() + row * code_size);
                    REQUIRE(std::isfinite(scalar));
                    REQUIRE(std::abs(scalar - expected[row]) <= 1e-5F);
                    REQUIRE(std::abs(row_dists[row] - expected[row]) <= 1e-5F);
                    REQUIRE(std::abs(packed_dists[row] - expected[row]) <= 1e-5F);
                }
            }
        }
    }
}

TEST_CASE("PQFSQuantizer restores nonzero lower for every subspace", "[ut][PQFSQuantizer]") {
    TestAffineLUTDistancesPQFS<MetricType::METRIC_TYPE_L2SQR>();
    TestAffineLUTDistancesPQFS<MetricType::METRIC_TYPE_IP>();
    TestAffineLUTDistancesPQFS<MetricType::METRIC_TYPE_COSINE>();
}

TEST_CASE("PQFSQuantizer zero IP query produces a constant LUT", "[ut][PQFSQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 12;
    constexpr uint64_t count = 33;
    auto training = fixtures::generate_vectors(64, dim, true);
    for (int64_t pq_dim : {3, 4}) {
        CAPTURE(pq_dim);
        PQFastScanQuantizer<MetricType::METRIC_TYPE_IP> quant(dim, pq_dim, allocator.get());
        REQUIRE(quant.Train(training.data(), 64));
        auto computer = quant.FactoryComputer();
        // Reuse an existing query buffer so stale, nonconstant LUT entries cannot survive.
        computer->SetQuery(training.data());
        const auto lut_size = pq_dim * quant.CENTROIDS_PER_SUBSPACE;
        std::fill_n(computer->buf_, lut_size, uint8_t{255});
        std::vector<float> query(dim, 0.0F);
        quant.ProcessQuery(query.data(), *computer);
        for (int64_t i = 0; i < lut_size; ++i) {
            REQUIRE(computer->buf_[i] == 0);
        }
        const auto* affine = reinterpret_cast<const float*>(computer->buf_ + lut_size);
        REQUIRE(affine[0] == 0.0F);
        REQUIRE(affine[1] == 0.0F);
        const auto code_size = quant.GetCodeSize();
        std::vector<uint8_t> rows(count * code_size);
        for (uint64_t i = 0; i < rows.size(); ++i) {
            rows[i] = static_cast<uint8_t>((i * 37 + 23) % 256);
        }
        std::vector<uint8_t> packed(64 * code_size);
        quant.Package32(rows.data(), packed.data(), 32);
        quant.Package32(rows.data() + 32 * code_size, packed.data() + 32 * code_size, 1);
        std::vector<float> row_dists(count);
        std::vector<float> packed_dists(count);
        quant.ScanRowBatchDists(*computer, count, rows.data(), row_dists.data());
        quant.ScanBatchDists(*computer, count, packed.data(), packed_dists.data());
        for (uint64_t row = 0; row < count; ++row) {
            const auto scalar = quant.ComputeDist(*computer, rows.data() + row * code_size);
            REQUIRE(std::isfinite(scalar));
            REQUIRE(scalar == 1.0F);
            REQUIRE(std::isfinite(row_dists[row]));
            REQUIRE(row_dists[row] == 1.0F);
            REQUIRE(std::isfinite(packed_dists[row]));
            REQUIRE(packed_dists[row] == 1.0F);
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

    test_serializion(quantizer1, quantizer2);
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
