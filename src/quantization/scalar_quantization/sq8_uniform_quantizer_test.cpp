
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

#include "sq8_uniform_quantizer.h"

#include <array>
#include <catch2/catch_approx.hpp>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_test.h"
#include "scalar_quantizer_test_utils.h"
#include "unittest.h"
using namespace vsag;

const auto dims = fixtures::get_common_used_dims();
const auto counts = {10, 101};

template <MetricType metric>
void
TestQuantizerEncodeDecodeMetricSQ8Uniform(uint64_t dim,
                                          int count,
                                          float error = 1e-5,
                                          float error_same = 1e-2) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SQ8UniformQuantizer<metric> quantizer(dim, allocator.get());
    TestQuantizerEncodeDecode(quantizer, dim, count, error);
    TestQuantizerEncodeDecodeSame(quantizer, dim, count, 255, error_same);
}

TEST_CASE("SQ8 Uniform Encode and Decode", "[ut][SQ8UniformQuantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 2 * 1.0f / 255.0f;
    for (auto dim : dims) {
        for (auto count : counts) {
            auto error_same = (float)(dim * 255 * 0.01);
            TestQuantizerEncodeDecodeMetricSQ8Uniform<metrics[0]>(dim, count, error, error_same);
            TestQuantizerEncodeDecodeMetricSQ8Uniform<metrics[1]>(dim, count, error, error_same);
        }
    }
}

TEST_CASE("SQ8 Uniform encodes zero range to zero", "[ut][SQ8UniformQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 4;
    SQ8UniformQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(dim, allocator.get());
    TestUniformZeroRangeEncodesToZero(quantizer, dim, dim);
}

template <MetricType metric>
void
TestComputeMetricSQ8Uniform(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SQ8UniformQuantizer<metric> quantizer(dim, allocator.get());
    TestComputeCodesSame<SQ8UniformQuantizer<metric>, metric>(quantizer, dim, count, error);
}

TEST_CASE("SQ8 Uniform Compute", "[ut][SQ8UniformQuantizer]") {
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 4 * 1.0f / 255.0f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestComputeMetricSQ8Uniform<metrics[0]>(dim, count, error);
            TestComputeMetricSQ8Uniform<metrics[1]>(dim, count, error);
        }
    }
}

template <MetricType metric>
void
test_unaligned_sq8_uniform_codes() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 7;
    SQ8UniformQuantizer<metric> quantizer(dim, allocator.get());
    const std::array<float, dim * 2> training = {
        -1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 2.0F, 1.5F, 1.0F, 0.5F, 0.0F, -0.5F, -1.0F};
    REQUIRE(quantizer.Train(training.data(), 2));

    const auto code_size = quantizer.GetCodeSize();
    std::vector<uint8_t> aligned_first(code_size);
    std::vector<uint8_t> aligned_second(code_size);
    std::vector<uint8_t> unaligned_first_storage(code_size + 1);
    std::vector<uint8_t> unaligned_second_storage(code_size + 1);
    auto* unaligned_first = unaligned_first_storage.data() + 1;
    auto* unaligned_second = unaligned_second_storage.data() + 1;

    REQUIRE(quantizer.EncodeOne(training.data(), aligned_first.data()));
    REQUIRE(quantizer.EncodeOne(training.data() + dim, aligned_second.data()));
    REQUIRE(quantizer.EncodeOne(training.data(), unaligned_first));
    REQUIRE(quantizer.EncodeOne(training.data() + dim, unaligned_second));
    CHECK(std::equal(aligned_first.begin(), aligned_first.end(), unaligned_first));
    CHECK(std::equal(aligned_second.begin(), aligned_second.end(), unaligned_second));
    CHECK(quantizer.Compute(unaligned_first, unaligned_second) ==
          quantizer.Compute(aligned_first.data(), aligned_second.data()));
}

TEST_CASE("SQ8 Uniform supports unaligned code buffers", "[ut][SQ8UniformQuantizer]") {
    test_unaligned_sq8_uniform_codes<MetricType::METRIC_TYPE_L2SQR>();
    test_unaligned_sq8_uniform_codes<MetricType::METRIC_TYPE_IP>();
}

template <MetricType metric>
void
TestSerializeAndDeserializeMetricSQ8Uniform(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SQ8UniformQuantizer<metric> quantizer1(dim, allocator.get());
    SQ8UniformQuantizer<metric> quantizer2(dim, allocator.get());
    TestSerializeAndDeserialize<SQ8UniformQuantizer<metric>, metric, true>(
        quantizer1, quantizer2, dim, count, error);
}

TEST_CASE("SQ8 Uniform Serialize and Deserialize", "[ut][SQ8UniformQuantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    for (auto dim : dims) {
        float error = 4 * 1.0f / 255.0f;
        for (auto count : counts) {
            TestSerializeAndDeserializeMetricSQ8Uniform<metrics[0]>(dim, count, error);
            // TestSerializeAndDeserializeMetricSQ8Uniform<metrics[1]>(dim, count, error);
            TestSerializeAndDeserializeMetricSQ8Uniform<metrics[2]>(dim, count, error);
        }
    }
}

namespace {

// A packed batch API must not be selected when the caller supplies ordinary rows.
class RowFallbackProbeQuantizer : public Quantizer<RowFallbackProbeQuantizer> {
public:
    explicit RowFallbackProbeQuantizer(Allocator* allocator)
        : Quantizer<RowFallbackProbeQuantizer>(7, allocator) {
        code_size_ = 7;
    }

    void
    ComputeDistImpl(Computer<RowFallbackProbeQuantizer>&, const uint8_t* codes, float* dist) const {
        ++scalar_calls;
        *dist = static_cast<float>(codes[0]) + 2.0F * static_cast<float>(codes[6]);
    }

    void
    ScanBatchDistImpl(Computer<RowFallbackProbeQuantizer>&,
                      uint64_t,
                      const uint8_t*,
                      float*) const {
        ++packed_calls;
    }

    void
    ReleaseComputerImpl(Computer<RowFallbackProbeQuantizer>&) const {
    }

    std::string
    NameImpl() const {
        return "RowFallbackProbeQuantizer";
    }

    mutable uint64_t scalar_calls{0};
    mutable uint64_t packed_calls{0};
};

TEST_CASE("Quantizer row fallback does not dispatch packed batches",
          "[ut][SQ8UniformQuantizer][RowBatchDists]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    RowFallbackProbeQuantizer quantizer(allocator.get());
    auto computer = quantizer.FactoryComputer();
    constexpr float sentinel = -12345.0F;
    for (uint64_t count : {0, 1, 7, 33}) {
        CAPTURE(count);
        std::vector<uint8_t> storage(count * 7 + 2, 0xA5);
        auto* codes = storage.data() + 1;
        for (uint64_t i = 0; i < count; ++i) {
            for (uint64_t j = 0; j < 7; ++j) {
                codes[i * 7 + j] = static_cast<uint8_t>(i * 3 + j);
            }
        }
        const auto original = storage;
        std::vector<float> distances(count + 2, sentinel);
        quantizer.scalar_calls = 0;
        quantizer.ScanRowBatchDists(*computer, count, codes, distances.data() + 1);
        CHECK(quantizer.scalar_calls == count);
        CHECK(quantizer.packed_calls == 0);
        CHECK(distances.front() == sentinel);
        CHECK(distances.back() == sentinel);
        CHECK(storage == original);
        for (uint64_t i = 0; i < count; ++i) {
            CHECK(distances[i + 1] ==
                  static_cast<float>(codes[i * 7]) + 2.0F * static_cast<float>(codes[i * 7 + 6]));
        }
    }
}

template <MetricType metric>
void
test_sq8_uniform_row_batch() {
    using QuantT = SQ8UniformQuantizer<metric>;
    using RowMember = void (QuantT::*)(Computer<QuantT>&, uint64_t, const uint8_t*, float*) const;
    // Inheriting the default loop would yield a pointer-to-base member instead.
    static_assert(std::is_same_v<decltype(&QuantT::ScanRowBatchDistImpl), RowMember>);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 7;
    constexpr uint64_t max_count = 33;
    QuantT quantizer(dim, allocator.get());
    std::array<float, dim * max_count> training{};
    for (uint64_t i = 0; i < max_count; ++i) {
        for (uint64_t j = 0; j < dim; ++j) {
            training[i * dim + j] =
                static_cast<float>(static_cast<int>((i * 13 + j * 7) % 41) - 20) / 8.0F;
        }
    }
    REQUIRE(quantizer.Train(training.data(), max_count));
    const std::array<float, dim> query = {0.25F, -0.5F, 0.75F, 1.0F, -1.25F, 0.5F, 0.125F};
    auto computer = quantizer.FactoryComputer();
    computer->SetQuery(query.data());
    const auto code_size = quantizer.GetCodeSize();
    constexpr float sentinel = -12345.0F;
    for (uint64_t count : {0, 1, 7, 31, 32, 33}) {
        CAPTURE(metric, count);
        // Deliberately offset codes, and allocate no extra rows for a 32-wide tail.
        std::vector<uint8_t> storage(count * code_size + 2, 0xA5);
        auto* codes = storage.data() + 1;
        for (uint64_t i = 0; i < count; ++i) {
            REQUIRE(quantizer.EncodeOne(training.data() + i * dim, codes + i * code_size));
        }
        const auto original = storage;
        std::vector<float> rows(count + 2, sentinel);
        std::vector<float> native(count + 2, sentinel);
        quantizer.ScanRowBatchDists(*computer, count, codes, rows.data() + 1);
        quantizer.ScanBatchDists(*computer, count, codes, native.data() + 1);
        CHECK(rows.front() == sentinel);
        CHECK(rows.back() == sentinel);
        CHECK(native.front() == sentinel);
        CHECK(native.back() == sentinel);
        CHECK(storage == original);
        for (uint64_t i = 0; i < count; ++i) {
            float scalar = sentinel;
            computer->ComputeDist(codes + i * code_size, &scalar);
            CHECK(rows[i + 1] == Catch::Approx(native[i + 1]).epsilon(1e-5));
            CHECK(rows[i + 1] == Catch::Approx(scalar).epsilon(1e-5));
        }
    }
}

TEST_CASE("SQ8 Uniform row batches preserve native and scalar distances",
          "[ut][SQ8UniformQuantizer][RowBatchDists]") {
    test_sq8_uniform_row_batch<MetricType::METRIC_TYPE_L2SQR>();
    test_sq8_uniform_row_batch<MetricType::METRIC_TYPE_IP>();
    test_sq8_uniform_row_batch<MetricType::METRIC_TYPE_COSINE>();
}

}  // namespace
