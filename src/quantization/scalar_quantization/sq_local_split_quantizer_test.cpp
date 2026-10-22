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

#include "sq_local_split_quantizer.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "impl/transform/fht_kac_rotate_transformer.h"
#include "simd/sq_local_split_simd.h"
#include "unittest.h"

using namespace vsag;

namespace {

using Quantizer = SQLocalSplitQuantizer<MetricType::METRIC_TYPE_L2SQR>;
using IPQuantizer = SQLocalSplitQuantizer<MetricType::METRIC_TYPE_IP>;

std::vector<float>
random_vectors(int64_t count, int64_t dim, uint32_t seed) {
    std::vector<float> values(static_cast<size_t>(count * dim));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto& value : values) {
        value = dist(rng);
    }
    return values;
}

// Recomputes the code-domain quantities from the packed records so the tests do not trust the
// kernels they are validating.
struct Decoded {
    float minimum{0.0F};
    float delta{0.0F};
    float norm_sqr{0.0F};
    std::vector<uint32_t> codes;
};

template <typename QuantizerT>
Decoded
decode_codes(const QuantizerT& quantizer,
             const uint8_t* filter_code,
             const uint8_t* supplement_code) {
    Decoded decoded;
    decoded.minimum = QuantizerT::MetaMinimum(filter_code);
    decoded.delta = QuantizerT::MetaDelta(filter_code);
    decoded.norm_sqr = QuantizerT::MetaNormSqr(filter_code);
    const auto dim = quantizer.GetDim();
    decoded.codes.resize(static_cast<size_t>(dim));
    const auto x = quantizer.FilterBits();
    const auto y = quantizer.SupplementBits();
    for (int64_t i = 0; i < dim; ++i) {
        const uint32_t hi = QuantizerT::GetBits(filter_code + QuantizerT::META_SIZE, i * x, x);
        const uint32_t lo = QuantizerT::GetBits(supplement_code, i * y, y);
        decoded.codes[static_cast<size_t>(i)] = (hi << y) | lo;
    }
    return decoded;
}

}  // namespace

TEST_CASE("sq_local_split code sizes and parameter validation", "[ut][SQLocalSplit]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 17;
    Quantizer quantizer(dim, 3, 8, 0, allocator.get());

    REQUIRE(quantizer.GetDim() == dim);
    REQUIRE(quantizer.FilterBits() == 3);
    REQUIRE(quantizer.BaseBits() == 8);
    REQUIRE(quantizer.SupplementBits() == 5);
    REQUIRE(quantizer.GetFilterCodeSize() == Quantizer::META_SIZE + (dim * 3 + 7) / 8 + 1);
    REQUIRE(quantizer.GetSupplementCodeSize() == (dim * 5 + 7) / 8 + 1);
    REQUIRE(quantizer.GetScalarCodeSize() == Quantizer::META_SIZE + dim);

    REQUIRE_THROWS(Quantizer(0, 3, 8, 0, allocator.get()));
    REQUIRE_THROWS(Quantizer(dim, 8, 8, 0, allocator.get()));  // x must be smaller than x + y
    REQUIRE_THROWS(Quantizer(dim, 0, 8, 0, allocator.get()));  // x >= 1
    REQUIRE_THROWS(Quantizer(dim, 3, 9, 0, allocator.get()));  // x + y <= 8
    REQUIRE_THROWS(Quantizer(dim, 3, 8, 4, allocator.get()));  // query bits are 0 or 8

    auto parameter = std::make_shared<SQLocalSplitQuantizerParameter>();
    parameter->FromJson(JsonType::Parse(R"({"type":"sq_local_split",
                                            "sq_local_bits_per_dim_filter":3,
                                            "sq_local_bits_per_dim_base":8,
                                            "sq_local_bits_per_dim_query":8})"));
    REQUIRE(parameter->FilterBits() == 3);
    REQUIRE(parameter->BaseBits() == 8);
    REQUIRE(parameter->QueryBits() == 8);
    REQUIRE(parameter->ToJson().Contains(SQ_LOCAL_BITS_PER_DIM_FILTER));

    auto invalid = std::make_shared<SQLocalSplitQuantizerParameter>();
    REQUIRE_THROWS(invalid->FromJson(JsonType::Parse(
        R"({"type":"sq_local_split","sq_local_bits_per_dim_filter":6,
            "sq_local_bits_per_dim_base":5})")));
}

TEST_CASE("sq_local_split encode and decode stay inside the quantization cell",
          "[ut][SQLocalSplit]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (int64_t dim : {1, 3, 7, 8, 17, 64, 960}) {
        Quantizer quantizer(dim, 3, 8, 0, allocator.get());
        const auto vectors = random_vectors(6, dim, 37 + static_cast<uint32_t>(dim));
        std::vector<uint8_t> filter_code(quantizer.GetFilterCodeSize());
        std::vector<uint8_t> supplement_code(quantizer.GetSupplementCodeSize());
        std::vector<float> rotated(static_cast<size_t>(dim));
        std::vector<float> decoded(static_cast<size_t>(dim));
        FhtKacRotator rotator(allocator.get(), dim);
        rotator.Train();
        for (int64_t row = 0; row < 6; ++row) {
            const float* vector = vectors.data() + row * dim;
            REQUIRE(quantizer.Encode(vector, filter_code.data(), supplement_code.data()));
            REQUIRE(quantizer.Decode(filter_code.data(), supplement_code.data(), decoded.data()));

            // Per-component error in the rotated domain is <= step / 2.
            rotator.Transform(vector, rotated.data(), nullptr);
            const auto decoded_codes =
                decode_codes(quantizer, filter_code.data(), supplement_code.data());
            const auto step = static_cast<double>(Quantizer::MetaDelta(filter_code.data()));
            for (int64_t i = 0; i < dim; ++i) {
                const double reconstructed = static_cast<double>(decoded_codes.minimum) +
                                             static_cast<double>(decoded_codes.delta) *
                                                 decoded_codes.codes[static_cast<size_t>(i)];
                REQUIRE(std::abs(reconstructed -
                                 static_cast<double>(rotated[static_cast<size_t>(i)])) <=
                        step / 2.0 + 1e-5);
            }
            const auto tolerance = step / 2.0 * std::sqrt(static_cast<double>(dim)) + 1e-5;
            for (int64_t i = 0; i < dim; ++i) {
                REQUIRE(std::abs(static_cast<double>(decoded[static_cast<size_t>(i)]) -
                                 static_cast<double>(vector[i])) <= tolerance);
            }
        }
        // The zero vector is the only input that stays constant after the rotation: it has a
        // zero step, all-zero codes and decodes back to zero.
        std::vector<float> zero(static_cast<size_t>(dim), 0.0F);
        REQUIRE(quantizer.Encode(zero.data(), filter_code.data(), supplement_code.data()));
        REQUIRE(Quantizer::MetaDelta(filter_code.data()) == 0.0F);
        REQUIRE(Quantizer::MetaSumCodes(filter_code.data()) == 0U);
        REQUIRE(quantizer.Decode(filter_code.data(), supplement_code.data(), decoded.data()));
        for (int64_t i = 0; i < dim; ++i) {
            REQUIRE(std::abs(decoded[static_cast<size_t>(i)]) <= 1e-5F);
        }
    }
}

TEST_CASE("sq_local_split scalar codes pack into the split records bit for bit",
          "[ut][SQLocalSplit]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (auto [x, y] : std::vector<std::pair<uint64_t, uint64_t>>{
             {1, 7}, {2, 6}, {3, 5}, {4, 4}, {5, 3}, {6, 2}, {7, 1}}) {
        constexpr int64_t dim = 65;
        Quantizer quantizer(dim, x, x + y, 0, allocator.get());
        const auto vectors = random_vectors(4, dim, 91);
        std::vector<uint8_t> filter_code(quantizer.GetFilterCodeSize());
        std::vector<uint8_t> supplement_code(quantizer.GetSupplementCodeSize());
        std::vector<uint8_t> scalar_code(quantizer.GetScalarCodeSize());
        std::vector<uint8_t> packed_filter(filter_code.size());
        std::vector<uint8_t> packed_supplement(supplement_code.size());
        for (int64_t row = 0; row < 4; ++row) {
            const float* vector = vectors.data() + row * dim;
            REQUIRE(quantizer.Encode(vector, filter_code.data(), supplement_code.data()));
            REQUIRE(quantizer.EncodeScalar(vector, scalar_code.data()));
            quantizer.PackScalarToSplit(
                scalar_code.data(), packed_filter.data(), packed_supplement.data());
            REQUIRE(std::memcmp(filter_code.data(), packed_filter.data(), filter_code.size()) == 0);
            REQUIRE(std::memcmp(supplement_code.data(),
                                packed_supplement.data(),
                                supplement_code.size()) == 0);
        }
    }
}

TEST_CASE("sq_local_split distances match the reconstructed codes", "[ut][SQLocalSplit]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 96;
    Quantizer quantizer(dim, 3, 8, 0, allocator.get());
    const auto vectors = random_vectors(2, dim, 13);
    std::vector<uint8_t> filter1(quantizer.GetFilterCodeSize());
    std::vector<uint8_t> supplement1(quantizer.GetSupplementCodeSize());
    std::vector<uint8_t> filter2(quantizer.GetFilterCodeSize());
    std::vector<uint8_t> supplement2(quantizer.GetSupplementCodeSize());
    REQUIRE(quantizer.Encode(vectors.data(), filter1.data(), supplement1.data()));
    REQUIRE(quantizer.Encode(vectors.data() + dim, filter2.data(), supplement2.data()));

    const auto query = random_vectors(1, dim, 29);
    auto computer = quantizer.FactoryComputer(query.data());

    const auto decoded1 = decode_codes(quantizer, filter1.data(), supplement1.data());
    const auto decoded2 = decode_codes(quantizer, filter2.data(), supplement2.data());

    // Rebuild the rotated query the same way the computer does.
    const auto computer_query = computer->rotated_query_;
    double expected_ip = 0.0;
    double norm_q = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
        const double value =
            static_cast<double>(decoded2.minimum) +
            static_cast<double>(decoded2.delta) * decoded2.codes[static_cast<size_t>(i)];
        expected_ip += static_cast<double>(computer_query[static_cast<size_t>(i)]) * value;
        norm_q += static_cast<double>(computer_query[static_cast<size_t>(i)]) *
                  static_cast<double>(computer_query[static_cast<size_t>(i)]);
    }
    const double expected_distance = norm_q - 2.0 * expected_ip + decoded2.norm_sqr;
    REQUIRE(quantizer.ComputeFullDistance(*computer, filter2.data(), supplement2.data()) ==
            Catch::Approx(expected_distance).epsilon(1e-4));

    // Code-to-code: both sides reconstructed.
    double expected_pair = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
        const double value1 =
            static_cast<double>(decoded1.minimum) +
            static_cast<double>(decoded1.delta) * decoded1.codes[static_cast<size_t>(i)];
        const double value2 =
            static_cast<double>(decoded2.minimum) +
            static_cast<double>(decoded2.delta) * decoded2.codes[static_cast<size_t>(i)];
        expected_pair += (value1 - value2) * (value1 - value2);
    }
    REQUIRE(quantizer.ComputePairDistance(
                filter1.data(), supplement1.data(), filter2.data(), supplement2.data()) ==
            Catch::Approx(expected_pair).epsilon(1e-4));

    // The filter-only bound never exceeds the full distance.
    REQUIRE(quantizer.ComputeFilterDistanceLowerBound(*computer, filter2.data()) <=
            quantizer.ComputeFullDistance(*computer, filter2.data(), supplement2.data()) + 1e-3F);
}

TEST_CASE("sq_local_split quantized query keeps the sum-of-codes term", "[ut][SQLocalSplit]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 64;
    IPQuantizer quantizer(dim, 3, 8, 8, allocator.get());
    const auto vectors = random_vectors(1, dim, 5);
    std::vector<uint8_t> filter_code(quantizer.GetFilterCodeSize());
    std::vector<uint8_t> supplement_code(quantizer.GetSupplementCodeSize());
    REQUIRE(quantizer.Encode(vectors.data(), filter_code.data(), supplement_code.data()));

    const auto query = random_vectors(1, dim, 17);
    auto computer = quantizer.FactoryComputer(query.data());
    const auto decoded = decode_codes(quantizer, filter_code.data(), supplement_code.data());

    double expected_ip = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
        const double q_value =
            static_cast<double>(computer->q_minimum) +
            static_cast<double>(computer->q_delta) *
                static_cast<double>(computer->query_codes_[static_cast<size_t>(i)]);
        const double d_value =
            static_cast<double>(decoded.minimum) +
            static_cast<double>(decoded.delta) * decoded.codes[static_cast<size_t>(i)];
        expected_ip += q_value * d_value;
    }
    const float actual =
        quantizer.ComputeFullDistance(*computer, filter_code.data(), supplement_code.data());
    INFO("actual=" << actual << " expected=" << (1.0 - expected_ip)
                   << " q_min=" << computer->q_minimum << " q_delta=" << computer->q_delta
                   << " sum_cq=" << computer->q_sum_codes);
    // The affine terms cancel to O(1e-1), so compare with a relative tolerance of 1e-3.
    REQUIRE(actual == Catch::Approx(1.0 - expected_ip).epsilon(1e-3));
}

TEST_CASE("sq_local_split SIMD kernels agree with the generic implementation",
          "[ut][SQLocalSplit]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (int64_t dim : {1, 7, 8, 17, 65, 960}) {
        for (auto [x, y] :
             std::vector<std::pair<uint64_t, uint64_t>>{{1, 7}, {3, 5}, {4, 4}, {7, 1}}) {
            Quantizer quantizer(dim, x, x + y, 0, allocator.get());
            const auto vectors = random_vectors(2, dim, 3 + static_cast<uint32_t>(dim));
            std::vector<uint8_t> filter1(quantizer.GetFilterCodeSize());
            std::vector<uint8_t> supplement1(quantizer.GetSupplementCodeSize());
            std::vector<uint8_t> filter2(quantizer.GetFilterCodeSize());
            std::vector<uint8_t> supplement2(quantizer.GetSupplementCodeSize());
            std::vector<uint8_t> scalar1(quantizer.GetScalarCodeSize());
            std::vector<uint8_t> scalar2(quantizer.GetScalarCodeSize());
            REQUIRE(quantizer.Encode(vectors.data(), filter1.data(), supplement1.data()));
            REQUIRE(quantizer.Encode(vectors.data() + dim, filter2.data(), supplement2.data()));
            REQUIRE(quantizer.EncodeScalar(vectors.data(), scalar1.data()));
            REQUIRE(quantizer.EncodeScalar(vectors.data() + dim, scalar2.data()));

            const auto query = random_vectors(1, dim, 43);
            const auto expected_float_ip = generic::FloatPackedIP(
                query.data(), filter1.data() + Quantizer::META_SIZE, supplement1.data(), dim, x, y);
            REQUIRE(SQFloatPackedIP(query.data(),
                                    filter1.data() + Quantizer::META_SIZE,
                                    supplement1.data(),
                                    dim,
                                    x,
                                    y) == Catch::Approx(expected_float_ip).epsilon(1e-4));

            const auto expected_pair =
                generic::PackedPackedIP(filter1.data() + Quantizer::META_SIZE,
                                        supplement1.data(),
                                        filter2.data() + Quantizer::META_SIZE,
                                        supplement2.data(),
                                        dim,
                                        x,
                                        y);
            REQUIRE(SQPackedPackedIP(filter1.data() + Quantizer::META_SIZE,
                                     supplement1.data(),
                                     filter2.data() + Quantizer::META_SIZE,
                                     supplement2.data(),
                                     dim,
                                     x,
                                     y) == Catch::Approx(expected_pair).epsilon(1e-4));

            const auto expected_byte_ip =
                generic::FloatByteIP(query.data(), scalar1.data() + Quantizer::META_SIZE, dim);
            REQUIRE(SQFloatByteIP(query.data(), scalar1.data() + Quantizer::META_SIZE, dim) ==
                    Catch::Approx(expected_byte_ip).epsilon(1e-4));

            const auto expected_byte_pair = generic::ByteByteIP(
                scalar1.data() + Quantizer::META_SIZE, scalar2.data() + Quantizer::META_SIZE, dim);
            REQUIRE(SQByteByteIP(scalar1.data() + Quantizer::META_SIZE,
                                 scalar2.data() + Quantizer::META_SIZE,
                                 dim) == Catch::Approx(expected_byte_pair).epsilon(1e-4));
        }
    }
}
