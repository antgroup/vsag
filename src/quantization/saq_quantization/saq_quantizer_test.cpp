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

#include "saq_quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_test.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"
#include "vsag_exception.h"

using namespace vsag;

namespace {

std::vector<float>
GenerateAnisotropicData(uint64_t count, uint64_t dim) {
    auto data = fixtures::generate_vectors(count, dim, false, 2026);
    for (uint64_t i = 0; i < count; ++i) {
        for (uint64_t d = 0; d < dim; ++d) {
            const float scale = d < 64 ? 100.0F : (d < 128 ? 1.0F : 0.01F);
            data[i * dim + d] *= scale;
        }
    }
    return data;
}

float
Cosine(const float* left, const float* right, uint64_t dim) {
    float inner = 0.0F;
    float left_norm = 0.0F;
    float right_norm = 0.0F;
    for (uint64_t d = 0; d < dim; ++d) {
        inner += left[d] * right[d];
        left_norm += left[d] * left[d];
        right_norm += right[d] * right[d];
    }
    return inner / std::sqrt(left_norm * right_norm);
}

template <MetricType metric>
void
TestMetricSerialization(const std::vector<float>& data,
                        uint64_t count,
                        uint64_t dim,
                        Allocator* allocator,
                        bool use_pca = true) {
    SAQQuantizer<metric> original(dim, 6.0F, 0, 4, use_pca, false, allocator);
    SAQQuantizer<metric> restored(dim / 2, 4.0F, 1, 0, false, true, allocator);
    REQUIRE(original.Train(data.data(), count));

    std::stringstream stream;
    IOStreamWriter writer(stream);
    original.Serialize(writer);
    stream.seekg(0, std::ios::beg);
    IOStreamReader reader(stream);
    restored.Deserialize(reader);

    std::vector<uint8_t> original_code(original.GetCodeSize());
    std::vector<uint8_t> restored_code(restored.GetCodeSize());
    REQUIRE(original.EncodeOne(data.data(), original_code.data()));
    REQUIRE(restored.EncodeOne(data.data(), restored_code.data()));
    REQUIRE(original_code == restored_code);

    auto original_computer = original.FactoryComputer();
    auto restored_computer = restored.FactoryComputer();
    original_computer->SetQuery(data.data() + dim);
    restored_computer->SetQuery(data.data() + dim);
    const float original_distance = original.ComputeDist(*original_computer, original_code.data());
    const float restored_distance = restored.ComputeDist(*restored_computer, restored_code.data());
    REQUIRE(std::isfinite(original_distance));
    REQUIRE(original_distance == restored_distance);
}

template <MetricType metric>
void
TestPairDistanceMatchesDecodedVectors(const std::vector<float>& data,
                                      uint64_t count,
                                      uint64_t dim,
                                      Allocator* allocator) {
    SAQQuantizer<metric> quantizer(dim, 6.0F, 0, 4, true, true, allocator);
    REQUIRE(quantizer.Train(data.data(), count));

    std::vector<uint8_t> code1(quantizer.GetCodeSize());
    std::vector<uint8_t> code2(quantizer.GetCodeSize());
    std::vector<float> decoded1(dim);
    std::vector<float> decoded2(dim);
    REQUIRE(quantizer.EncodeOne(data.data(), code1.data()));
    REQUIRE(quantizer.EncodeOne(data.data() + dim, code2.data()));
    REQUIRE(quantizer.DecodeOne(code1.data(), decoded1.data()));
    REQUIRE(quantizer.DecodeOne(code2.data(), decoded2.data()));

    float expected = 0.0F;
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        for (uint64_t d = 0; d < dim; ++d) {
            const float difference = decoded1[d] - decoded2[d];
            expected += difference * difference;
        }
    } else if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        expected = 1.0F - Cosine(decoded1.data(), decoded2.data(), dim);
    } else {
        expected = 1.0F;
        for (uint64_t d = 0; d < dim; ++d) {
            expected -= decoded1[d] * decoded2[d];
        }
    }

    const float actual = quantizer.Compute(code1.data(), code2.data());
    const float tolerance = 1e-3F * std::max(1.0F, std::abs(expected));
    REQUIRE(std::abs(actual - expected) <= tolerance);
}

}  // namespace

TEST_CASE("SAQ dynamic segmentation and code budget", "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 192;
    constexpr uint64_t count = 256;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = GenerateAnisotropicData(count, dim);
    SAQQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, 4.0F, 0, 6, false, false, allocator.get());

    REQUIRE_FALSE(quantizer.TrainImpl(nullptr, count));
    REQUIRE_FALSE(quantizer.TrainImpl(data.data(), 1));
    REQUIRE(quantizer.TrainImpl(data.data(), count));
    REQUIRE(quantizer.GetSegments().size() >= 2);
    REQUIRE(quantizer.GetCodeSize() == (dim * 4 + 3 * sizeof(float) * 8) / 8);

    uint64_t covered = 0;
    for (const auto& segment : quantizer.GetSegments()) {
        REQUIRE(segment.begin == covered);
        REQUIRE(segment.bits >= 1);
        REQUIRE(segment.bits <= 13);
        covered += segment.length;
    }
    REQUIRE(covered == dim);
}

TEST_CASE("SAQ operation failures follow their API return contracts", "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 64;
    constexpr uint64_t count = 32;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = fixtures::generate_vectors(count, dim, false, 37);
    SAQQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, 4.0F, 1, 2, false, false, allocator.get());
    std::vector<uint8_t> code(quantizer.GetCodeSize());
    std::vector<float> decoded(dim);

    auto require_compute_failure = [&quantizer](const uint8_t* left, const uint8_t* right) {
        try {
            (void)quantizer.Compute(left, right);
            FAIL("invalid SAQ codes were accepted by Compute");
        } catch (const VsagException& error) {
            REQUIRE(error.error_.type == ErrorType::INTERNAL_ERROR);
            REQUIRE(std::string(error.what()) ==
                    "cannot compute distance from untrained or null SAQ codes");
        }
    };

    REQUIRE_FALSE(quantizer.EncodeOne(data.data(), code.data()));
    REQUIRE_FALSE(quantizer.DecodeOne(code.data(), decoded.data()));
    require_compute_failure(code.data(), code.data());

    REQUIRE(quantizer.Train(data.data(), count));
    REQUIRE_FALSE(quantizer.EncodeOne(nullptr, code.data()));
    REQUIRE_FALSE(quantizer.EncodeOne(data.data(), nullptr));
    REQUIRE_FALSE(quantizer.DecodeOne(nullptr, decoded.data()));
    REQUIRE_FALSE(quantizer.DecodeOne(code.data(), nullptr));
    require_compute_failure(nullptr, code.data());
    require_compute_failure(code.data(), nullptr);
}

TEST_CASE("SAQ fixed segmentation and CAQ adjustment", "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 128;
    constexpr uint64_t count = 128;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = fixtures::generate_vectors(count, dim, false, 47);

    SAQQuantizer<MetricType::METRIC_TYPE_L2SQR> plain(
        dim, 4.0F, 2, 0, false, false, allocator.get());
    SAQQuantizer<MetricType::METRIC_TYPE_L2SQR> adjusted(
        dim, 4.0F, 2, 6, false, false, allocator.get());
    REQUIRE(plain.Train(data.data(), count));
    REQUIRE(adjusted.Train(data.data(), count));
    REQUIRE(adjusted.GetSegments().size() == 2);

    std::vector<uint8_t> plain_code(plain.GetCodeSize());
    std::vector<uint8_t> adjusted_code(adjusted.GetCodeSize());
    std::vector<float> plain_decoded(dim);
    std::vector<float> adjusted_decoded(dim);
    REQUIRE(plain.EncodeOne(data.data(), plain_code.data()));
    REQUIRE(adjusted.EncodeOne(data.data(), adjusted_code.data()));
    REQUIRE(plain.DecodeOne(plain_code.data(), plain_decoded.data()));
    REQUIRE(adjusted.DecodeOne(adjusted_code.data(), adjusted_decoded.data()));
    REQUIRE(Cosine(data.data(), adjusted_decoded.data(), dim) + 1e-6F >=
            Cosine(data.data(), plain_decoded.data(), dim));

    auto computer = adjusted.FactoryComputer();
    computer->SetQuery(data.data());
    REQUIRE(std::abs(adjusted.ComputeDist(*computer, adjusted_code.data())) < 1e-3F);
}

TEST_CASE("SAQ supports metrics and serialization", "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 64;
    constexpr uint64_t count = 96;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = fixtures::generate_vectors(count, dim, false, 11);

    TestMetricSerialization<MetricType::METRIC_TYPE_L2SQR>(data, count, dim, allocator.get());
    TestMetricSerialization<MetricType::METRIC_TYPE_L2SQR>(
        data, count, dim, allocator.get(), false);
    TestMetricSerialization<MetricType::METRIC_TYPE_IP>(data, count, dim, allocator.get());
    TestMetricSerialization<MetricType::METRIC_TYPE_COSINE>(data, count, dim, allocator.get());
}

TEST_CASE("SAQ random rotations are reproducible", "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 127;
    constexpr uint64_t count = 96;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = fixtures::generate_vectors(count, dim, false, 19);
    SAQQuantizer<MetricType::METRIC_TYPE_L2SQR> first(dim, 4.0F, 0, 6, true, true, allocator.get());
    SAQQuantizer<MetricType::METRIC_TYPE_L2SQR> second(
        dim, 4.0F, 0, 6, true, true, allocator.get());

    REQUIRE(first.Train(data.data(), count));
    REQUIRE(second.Train(data.data(), count));
    REQUIRE(first.GetSegments().size() == second.GetSegments().size());
    std::vector<uint8_t> first_code(first.GetCodeSize());
    std::vector<uint8_t> second_code(second.GetCodeSize());
    REQUIRE(first.EncodeOne(data.data(), first_code.data()));
    REQUIRE(second.EncodeOne(data.data(), second_code.data()));
    REQUIRE(first_code == second_code);
}

TEST_CASE("SAQ computes code-pair distance in the orthogonal projection space",
          "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 127;
    constexpr uint64_t count = 96;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = fixtures::generate_vectors(count, dim, false, 29);

    TestPairDistanceMatchesDecodedVectors<MetricType::METRIC_TYPE_L2SQR>(
        data, count, dim, allocator.get());
    TestPairDistanceMatchesDecodedVectors<MetricType::METRIC_TYPE_IP>(
        data, count, dim, allocator.get());
    TestPairDistanceMatchesDecodedVectors<MetricType::METRIC_TYPE_COSINE>(
        data, count, dim, allocator.get());
}

TEST_CASE("SAQ cosine uses finite zero-norm and consistent code/query semantics",
          "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 64;
    constexpr uint64_t count = 96;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = fixtures::generate_vectors(count, dim, false, 211);
    std::fill(data.begin(), data.begin() + dim, 0.0F);
    SAQQuantizer<MetricType::METRIC_TYPE_COSINE> quantizer(
        dim, 1.0F, 1, 4, true, true, allocator.get());
    REQUIRE(quantizer.Train(data.data(), count));

    std::vector<uint8_t> zero_code(quantizer.GetCodeSize());
    std::vector<uint8_t> first_code(quantizer.GetCodeSize());
    std::vector<uint8_t> second_code(quantizer.GetCodeSize());
    REQUIRE(quantizer.EncodeOne(data.data(), zero_code.data()));
    REQUIRE(quantizer.EncodeOne(data.data() + dim, first_code.data()));
    REQUIRE(quantizer.EncodeOne(data.data() + 2 * dim, second_code.data()));

    for (const float distance : {quantizer.Compute(zero_code.data(), zero_code.data()),
                                 quantizer.Compute(zero_code.data(), first_code.data()),
                                 quantizer.Compute(first_code.data(), zero_code.data())}) {
        REQUIRE(std::isfinite(distance));
        REQUIRE(std::abs(distance - 1.0F) <= 1e-6F);
    }

    auto computer = quantizer.FactoryComputer();
    computer->SetQuery(data.data());
    const float zero_query_distance = quantizer.ComputeDist(*computer, first_code.data());
    REQUIRE(std::isfinite(zero_query_distance));
    REQUIRE(std::abs(zero_query_distance - 1.0F) <= 1e-6F);

    std::vector<float> decoded_first(dim);
    REQUIRE(quantizer.DecodeOne(first_code.data(), decoded_first.data()));
    computer->SetQuery(decoded_first.data());
    const float query_to_code_distance = quantizer.ComputeDist(*computer, second_code.data());
    const float code_to_code_distance = quantizer.Compute(first_code.data(), second_code.data());
    REQUIRE(std::isfinite(query_to_code_distance));
    REQUIRE(std::abs(query_to_code_distance - code_to_code_distance) <= 1e-4F);

    computer->SetQuery(data.data());
    float batch_distances[4]{};
    quantizer.ComputeDistsBatch4(*computer,
                                 zero_code.data(),
                                 first_code.data(),
                                 second_code.data(),
                                 zero_code.data(),
                                 batch_distances[0],
                                 batch_distances[1],
                                 batch_distances[2],
                                 batch_distances[3]);
    for (const float distance : batch_distances) {
        REQUIRE(std::isfinite(distance));
        REQUIRE(std::abs(distance - 1.0F) <= 1e-6F);
    }

    std::vector<std::vector<float>> non_finite_vectors(3, std::vector<float>(dim, 1.0F));
    non_finite_vectors[0][0] = std::numeric_limits<float>::quiet_NaN();
    non_finite_vectors[1][0] = std::numeric_limits<float>::infinity();
    std::fill(non_finite_vectors[2].begin(),
              non_finite_vectors[2].end(),
              std::numeric_limits<float>::max());
    for (uint64_t i = 0; i < non_finite_vectors.size(); ++i) {
        CAPTURE(i);
        std::vector<uint8_t> code(quantizer.GetCodeSize());
        REQUIRE(quantizer.EncodeOne(non_finite_vectors[i].data(), code.data()));
        REQUIRE(code == zero_code);
        REQUIRE(std::isfinite(quantizer.Compute(code.data(), first_code.data())));

        computer->SetQuery(non_finite_vectors[i].data());
        const float query_distance = quantizer.ComputeDist(*computer, first_code.data());
        REQUIRE(std::isfinite(query_distance));
        REQUIRE(std::abs(query_distance - zero_query_distance) <= 1e-6F);
    }
}

TEST_CASE("SAQ cosine code-pair distance uses stored reconstructed norms", "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 64;
    constexpr uint64_t count = 96;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = fixtures::generate_vectors(count, dim, false, 229);
    SAQQuantizer<MetricType::METRIC_TYPE_COSINE> quantizer(
        dim, 4.0F, 1, 4, false, false, allocator.get());
    REQUIRE(quantizer.Train(data.data(), count));
    REQUIRE(quantizer.GetSegments().size() == 1);

    std::vector<uint8_t> first_code(quantizer.GetCodeSize());
    std::vector<uint8_t> second_code(quantizer.GetCodeSize());
    REQUIRE(quantizer.EncodeOne(data.data(), first_code.data()));
    REQUIRE(quantizer.EncodeOne(data.data() + dim, second_code.data()));

    const float baseline = quantizer.Compute(first_code.data(), second_code.data());
    const uint64_t norm_offset = quantizer.GetSegments()[0].metadata_offset + sizeof(float);
    float stored_norm = 0.0F;
    std::memcpy(&stored_norm, first_code.data() + norm_offset, sizeof(float));
    REQUIRE(stored_norm > 0.0F);
    stored_norm *= 4.0F;
    std::memcpy(first_code.data() + norm_offset, &stored_norm, sizeof(float));

    const float scaled = quantizer.Compute(first_code.data(), second_code.data());
    const float expected = 1.0F - (1.0F - baseline) / 2.0F;
    REQUIRE(std::abs(scaled - expected) <= 1e-6F);
}

TEST_CASE("SAQ rejects a serialized rotation with the wrong dimensions", "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 64;
    constexpr uint64_t count = 32;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = fixtures::generate_vectors(count, dim, false, 223);
    SAQQuantizer<MetricType::METRIC_TYPE_L2SQR> original(
        dim, 4.0F, 1, 2, false, true, allocator.get());
    REQUIRE(original.Train(data.data(), count));

    std::stringstream stream;
    IOStreamWriter writer(stream);
    original.Serialize(writer);
    std::string serialized = stream.str();
    const uint64_t matrix_elements = dim * dim;
    const uint64_t matrix_bytes = matrix_elements * sizeof(float);
    REQUIRE(serialized.size() >= matrix_bytes + sizeof(uint64_t));
    const uint64_t matrix_size_offset = serialized.size() - matrix_bytes - sizeof(uint64_t);
    const uint64_t malformed_elements = matrix_elements - 1;
    std::memcpy(
        serialized.data() + matrix_size_offset, &malformed_elements, sizeof(malformed_elements));

    std::stringstream malformed_stream(serialized);
    IOStreamReader reader(malformed_stream);
    SAQQuantizer<MetricType::METRIC_TYPE_L2SQR> restored(
        dim, 4.0F, 1, 2, false, true, allocator.get());
    REQUIRE_THROWS_AS(restored.Deserialize(reader), VsagException);
}

TEST_CASE("SAQ L2 projection restores the training mean", "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 64;
    constexpr uint64_t count = 128;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = fixtures::generate_vectors(count, dim, false, 113);
    for (uint64_t i = 0; i < data.size(); ++i) {
        data[i] = data[i] * static_cast<float>(i % dim + 1) + 1'000.0F;
    }

    for (bool use_pca : {false, true}) {
        SAQQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
            dim, 8.0F, 1, 6, use_pca, false, allocator.get());
        REQUIRE(quantizer.Train(data.data(), count));

        std::vector<uint8_t> code(quantizer.GetCodeSize());
        std::vector<float> decoded(dim);
        REQUIRE(quantizer.EncodeOne(data.data(), code.data()));
        REQUIRE(quantizer.DecodeOne(code.data(), decoded.data()));
        double mean_absolute_error = 0.0;
        for (uint64_t d = 0; d < dim; ++d) {
            mean_absolute_error += std::abs(decoded[d] - data[d]);
        }
        mean_absolute_error /= static_cast<double>(dim);
        REQUIRE(mean_absolute_error < 1.0);
    }
}

TEST_CASE("SAQ progressive L2 distance stops above threshold", "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 192;
    constexpr uint64_t count = 128;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = GenerateAnisotropicData(count, dim);
    SAQQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, 4.0F, 0, 6, false, false, allocator.get());
    REQUIRE(quantizer.Train(data.data(), count));

    std::vector<uint8_t> code(quantizer.GetCodeSize());
    REQUIRE(quantizer.EncodeOne(data.data(), code.data()));
    auto computer = quantizer.FactoryComputer();
    computer->SetQuery(data.data() + dim);
    float distance = 0.0F;
    REQUIRE(quantizer.ComputeDistWithThreshold(*computer, code.data(), 0.0F, &distance));
    REQUIRE(distance > 0.0F);
}

TEST_CASE("SAQ byte-aligns non-integral record budgets", "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 127;
    constexpr uint64_t count = 32;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = fixtures::generate_vectors(count, dim, false, 83);
    SAQQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, 3.5F, 0, 2, false, false, allocator.get());

    REQUIRE(quantizer.GetCodeSize() == 68);
    REQUIRE(quantizer.Train(data.data(), count));
    std::vector<uint8_t> code(quantizer.GetCodeSize());
    REQUIRE(quantizer.EncodeOne(data.data(), code.data()));
}

TEST_CASE("SAQ packs every supported scalar bit width", "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 64;
    constexpr uint64_t count = 32;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = fixtures::generate_vectors(count, dim, false, 97);

    for (uint64_t bits = 1; bits <= 8; ++bits) {
        SAQQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
            dim, static_cast<float>(bits), 1, 2, false, false, allocator.get());
        REQUIRE(quantizer.Train(data.data(), count));
        REQUIRE(quantizer.GetSegments().size() == 1);
        REQUIRE(quantizer.GetSegments()[0].bits == bits);

        std::vector<uint8_t> code(quantizer.GetCodeSize());
        std::vector<float> decoded(dim);
        REQUIRE(quantizer.EncodeOne(data.data(), code.data()));
        REQUIRE(quantizer.DecodeOne(code.data(), decoded.data()));
        REQUIRE(std::all_of(
            decoded.begin(), decoded.end(), [](float value) { return std::isfinite(value); }));

        auto computer = quantizer.FactoryComputer();
        computer->SetQuery(data.data());
        REQUIRE(std::abs(quantizer.ComputeDist(*computer, code.data())) < 1e-3F);
    }
}

TEST_CASE("SAQ packs trained segment widths from 9 through 13 bits", "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 128;
    constexpr uint64_t count = 128;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const auto base = fixtures::generate_vectors(count, dim / 2, false, 227);

    for (uint64_t target_bits = 9; target_bits <= 13; ++target_bits) {
        const uint64_t other_bits = 15 - target_bits;
        const float scale = std::pow(2.0F, static_cast<float>(target_bits - other_bits) / 2.0F);
        std::vector<float> data(count * dim);
        for (uint64_t i = 0; i < count; ++i) {
            for (uint64_t d = 0; d < dim / 2; ++d) {
                data[i * dim + d] = base[i * dim / 2 + d] * scale;
                data[i * dim + dim / 2 + d] = base[i * dim / 2 + d];
            }
        }

        SAQQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
            dim, 8.0F, 2, 2, false, false, allocator.get());
        REQUIRE(quantizer.Train(data.data(), count));
        REQUIRE(quantizer.GetSegments().size() == 2);
        REQUIRE(quantizer.GetSegments()[0].bits == target_bits);
        REQUIRE(quantizer.GetSegments()[1].bits == other_bits);

        std::vector<uint8_t> code1(quantizer.GetCodeSize());
        std::vector<uint8_t> code2(quantizer.GetCodeSize());
        std::vector<float> decoded1(dim);
        std::vector<float> decoded2(dim);
        REQUIRE(quantizer.EncodeOne(data.data(), code1.data()));
        REQUIRE(quantizer.EncodeOne(data.data() + dim, code2.data()));
        REQUIRE(quantizer.DecodeOne(code1.data(), decoded1.data()));
        REQUIRE(quantizer.DecodeOne(code2.data(), decoded2.data()));
        REQUIRE(std::all_of(
            decoded1.begin(), decoded1.end(), [](float value) { return std::isfinite(value); }));
        float decoded_distance = 0.0F;
        for (uint64_t d = 0; d < dim; ++d) {
            const float difference = decoded1[d] - decoded2[d];
            decoded_distance += difference * difference;
        }
        const float actual_distance = quantizer.Compute(code1.data(), code2.data());
        const float tolerance = 1e-3F * std::max(1.0F, decoded_distance);
        REQUIRE(std::abs(actual_distance - decoded_distance) <= tolerance);
    }
}

TEST_CASE("SAQ batch-four distances match scalar distances", "[ut][SAQQuantizer]") {
    constexpr uint64_t dim = 128;
    constexpr uint64_t count = 64;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = fixtures::generate_vectors(count, dim, false, 131);
    SAQQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, 4.0F, 1, 2, false, false, allocator.get());
    REQUIRE(quantizer.Train(data.data(), count));

    std::vector<uint8_t> codes(4 * quantizer.GetCodeSize());
    for (uint64_t i = 0; i < 4; ++i) {
        REQUIRE(
            quantizer.EncodeOne(data.data() + i * dim, codes.data() + i * quantizer.GetCodeSize()));
    }
    auto computer = quantizer.FactoryComputer();
    computer->SetQuery(data.data() + 4 * dim);
    std::vector<float> scalar(4);
    for (uint64_t i = 0; i < 4; ++i) {
        scalar[i] = quantizer.ComputeDist(*computer, codes.data() + i * quantizer.GetCodeSize());
    }
    std::vector<float> batch(4);
    quantizer.ComputeDistsBatch4(*computer,
                                 codes.data(),
                                 codes.data() + quantizer.GetCodeSize(),
                                 codes.data() + 2 * quantizer.GetCodeSize(),
                                 codes.data() + 3 * quantizer.GetCodeSize(),
                                 batch[0],
                                 batch[1],
                                 batch[2],
                                 batch[3]);
    REQUIRE(batch == scalar);
}
