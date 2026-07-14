
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

#include "rabitq_quantizer.h"

#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "impl/logger/logger.h"
#include "quantization/quantizer_test.h"
#include "quantization/scalar_quantization/sq4_uniform_quantizer.h"
#include "unittest.h"
#include "utils/util_functions.h"
using namespace vsag;

const auto dims = fixtures::get_common_used_dims(6, 129);
const auto counts = {100};

TEST_CASE("RaBitQ Basic Test", "[ut][RaBitQuantizer]") {
    bool use_fht = GENERATE(true, false);
    auto num_bits_per_dim_query = GENERATE(4, 32);
    auto num_bits_per_dim_base = GENERATE(1, 2, 4, 8);
    for (auto dim : dims) {
        uint64_t pca_dim = dim;
        if (dim >= 1500) {
            pca_dim = dim / 2;
        }
        if (num_bits_per_dim_query == 4 and num_bits_per_dim_base != 1) {
            WARN("num_bits_per_dim_query=4 only supports num_bits_per_dim_base=1");
            continue;
        }
        for (auto count : counts) {
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            auto vecs = fixtures::generate_vectors(count, dim);
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(dim,
                                                                    pca_dim,
                                                                    num_bits_per_dim_query,
                                                                    num_bits_per_dim_base,
                                                                    use_fht,
                                                                    false,
                                                                    allocator.get());

            // name
            REQUIRE(quantizer.NameImpl() == QUANTIZATION_TYPE_VALUE_RABITQ);

            // train
            REQUIRE(quantizer.TrainImpl(vecs.data(), 0) == false);
            REQUIRE(quantizer.TrainImpl(vecs.data(), count) == true);
            REQUIRE(quantizer.TrainImpl(vecs.data(), count) == true);
        }
    }
}

TEST_CASE("Extend RaBitQ Basic Test", "[ut][RaBitQuantizer]") {
    auto count = 1000;
    auto dim = 32;
    auto num_bits_per_dim_query = 32;
    auto num_bits_per_dim_base = GENERATE(2, 4, 8);
    auto use_fht = false;
    auto pca_dim = dim;

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto vecs = fixtures::generate_vectors(count, dim);
    RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(dim,
                                                            pca_dim,
                                                            num_bits_per_dim_query,
                                                            num_bits_per_dim_base,
                                                            use_fht,
                                                            false,
                                                            allocator.get());
    quantizer.TrainImpl(vecs.data(), count);

    for (auto i = 0; i < count; i++) {
        auto base = vecs.data() + i * dim;

        auto computer = quantizer.FactoryComputer();
        computer->SetQuery(base);

        std::vector<uint8_t> base_code(quantizer.GetCodeSize());
        quantizer.EncodeOne(base, base_code.data());

        auto dist = quantizer.ComputeDist(*computer, base_code.data());
        REQUIRE(std::abs(dist) <= 1e-3);
    }
}

TEST_CASE("RaBitQ Split Code Storage", "[ut][RaBitQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr auto dim = 64;
    constexpr auto count = 32;
    auto vecs = fixtures::generate_vectors(count, dim);

    std::vector<std::pair<uint64_t, uint64_t>> split_cases;
    for (uint64_t base_bits = 1; base_bits <= 8; ++base_bits) {
        split_cases.emplace_back(base_bits, 1);
    }
    split_cases.emplace_back(8, 2);
    split_cases.emplace_back(8, 3);
    split_cases.emplace_back(8, 8);

    for (const auto& [base_bits, filter_bits] : split_cases) {
        RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
            dim,
            dim,
            32,
            base_bits,
            false,
            false,
            allocator.get(),
            RaBitQuantizerParameter::RABITQ_VERSION_SPLIT,
            RaBitQuantizerParameter::DEFAULT_RABITQ_ERROR_RATE,
            filter_bits);
        REQUIRE(quantizer.SupportSplitCodeStorage());
        REQUIRE(quantizer.FilterBits() == filter_bits);
        REQUIRE(quantizer.ReorderBits() == base_bits - filter_bits);
        REQUIRE(quantizer.FilterPlanesSize() == quantizer.PlaneBytes() * filter_bits);
        REQUIRE(quantizer.SupplementPlanesSize() ==
                quantizer.PlaneBytes() * (base_bits - filter_bits));
        REQUIRE(quantizer.GetOneBitCodeSize() >= quantizer.FilterPlanesSize());
        REQUIRE(quantizer.GetSupplementCodeSize() >= quantizer.SupplementPlanesSize());
        REQUIRE(quantizer.GetSupplementCodeSize() < quantizer.GetCodeSize());
        quantizer.TrainImpl(vecs.data(), count);

        std::vector<uint8_t> full_code(quantizer.GetCodeSize());
        std::vector<uint8_t> merged_code(quantizer.GetCodeSize());
        std::vector<uint8_t> one_bit_code(quantizer.GetOneBitCodeSize());
        std::vector<uint8_t> supplement_code(quantizer.GetSupplementCodeSize());

        const auto* base = vecs.data();
        REQUIRE(quantizer.EncodeOne(base, full_code.data()));
        quantizer.SplitCode(full_code.data(), one_bit_code.data(), supplement_code.data());
        quantizer.MergeSplitCode(one_bit_code.data(), supplement_code.data(), merged_code.data());
        REQUIRE(std::memcmp(full_code.data(), merged_code.data(), full_code.size()) == 0);

        auto computer = quantizer.FactoryComputer();
        computer->SetQuery(base);
        auto full_dist = quantizer.ComputeDist(*computer, full_code.data());
        float split_dist = 0.0F;
        REQUIRE(quantizer.ComputeDistWithSplitCode(
            *computer, one_bit_code.data(), supplement_code.data(), &split_dist));
        REQUIRE(std::abs(full_dist - split_dist) <= 1e-6F);

        float one_bit_dist = 0.0F;
        float lower_bound = std::numeric_limits<float>::max();
        REQUIRE(quantizer.ComputeDistWithOneBitLowerBound(
            *computer, one_bit_code.data(), &one_bit_dist, &lower_bound));
        REQUIRE(std::isfinite(one_bit_dist));
        REQUIRE(std::isfinite(lower_bound));
        REQUIRE(lower_bound <= one_bit_dist + 1e-5F);

        if (filter_bits > 1) {
            float stored_filter_norm_code = 0.0F;
            std::memcpy(&stored_filter_norm_code,
                        one_bit_code.data() + quantizer.OneBitRecordNormCodeOffset(),
                        sizeof(stored_filter_norm_code));
            const float filter_center = 0.5F * static_cast<float>((1U << filter_bits) - 1U);
            double filter_norm_sqr = 0.0;
            for (uint64_t d = 0; d < dim; ++d) {
                const uint64_t byte_idx = d >> 3;
                const auto bit_mask = static_cast<uint8_t>(1U << (d & 7));
                uint32_t filter_code = 0;
                for (uint32_t bit = 0; bit < filter_bits; ++bit) {
                    const auto* plane = one_bit_code.data() + bit * quantizer.PlaneBytes();
                    if ((plane[byte_idx] & bit_mask) != 0U) {
                        filter_code += 1U << (filter_bits - bit - 1);
                    }
                }
                const float centered = static_cast<float>(filter_code) - filter_center;
                filter_norm_sqr += static_cast<double>(centered) * centered;
            }
            const auto expected_filter_norm_code = static_cast<float>(std::sqrt(filter_norm_sqr));
            REQUIRE(std::abs(stored_filter_norm_code - expected_filter_norm_code) <= 1e-5F);

            if (filter_bits == 2 or filter_bits == 3) {
                float hinted_split_dist = 0.0F;
                REQUIRE(quantizer.ComputeDistWithSplitCodeAndFilterDist(*computer,
                                                                        one_bit_code.data(),
                                                                        supplement_code.data(),
                                                                        one_bit_dist,
                                                                        &hinted_split_dist));
                REQUIRE(std::abs(split_dist - hinted_split_dist) <= 1e-5F);
            }
        }

        if (filter_bits == 2) {
            std::vector<std::vector<uint8_t>> batch_one_bit_codes(4);
            std::vector<uint8_t> batch_supplement_code(quantizer.GetSupplementCodeSize());
            std::vector<float> single_dists(4, 0.0F);
            std::vector<float> single_lower_bounds(4, std::numeric_limits<float>::max());
            for (uint32_t i = 0; i < 4; ++i) {
                batch_one_bit_codes[i].resize(quantizer.GetOneBitCodeSize());
                quantizer.EncodeOne(vecs.data() + i * dim, full_code.data());
                quantizer.SplitCode(
                    full_code.data(), batch_one_bit_codes[i].data(), batch_supplement_code.data());
                REQUIRE(quantizer.ComputeDistWithOneBitLowerBound(*computer,
                                                                  batch_one_bit_codes[i].data(),
                                                                  &single_dists[i],
                                                                  &single_lower_bounds[i]));
            }

            float batch_dist1 = 0.0F;
            float batch_dist2 = 0.0F;
            float batch_dist3 = 0.0F;
            float batch_dist4 = 0.0F;
            float batch_lower_bound1 = std::numeric_limits<float>::max();
            float batch_lower_bound2 = std::numeric_limits<float>::max();
            float batch_lower_bound3 = std::numeric_limits<float>::max();
            float batch_lower_bound4 = std::numeric_limits<float>::max();
            bool computed1 = false;
            bool computed2 = false;
            bool computed3 = false;
            bool computed4 = false;
            quantizer.ComputeDistsWithOneBitLowerBoundBatch4(*computer,
                                                             batch_one_bit_codes[0].data(),
                                                             batch_one_bit_codes[1].data(),
                                                             batch_one_bit_codes[2].data(),
                                                             batch_one_bit_codes[3].data(),
                                                             batch_dist1,
                                                             batch_dist2,
                                                             batch_dist3,
                                                             batch_dist4,
                                                             &batch_lower_bound1,
                                                             &batch_lower_bound2,
                                                             &batch_lower_bound3,
                                                             &batch_lower_bound4,
                                                             computed1,
                                                             computed2,
                                                             computed3,
                                                             computed4);

            REQUIRE(computed1);
            REQUIRE(computed2);
            REQUIRE(computed3);
            REQUIRE(computed4);
            const float batch_dists[4] = {batch_dist1, batch_dist2, batch_dist3, batch_dist4};
            const float batch_lower_bounds[4] = {
                batch_lower_bound1, batch_lower_bound2, batch_lower_bound3, batch_lower_bound4};
            for (uint32_t i = 0; i < 4; ++i) {
                REQUIRE(std::abs(batch_dists[i] - single_dists[i]) <= 1e-5F);
                REQUIRE(std::abs(batch_lower_bounds[i] - single_lower_bounds[i]) <= 1e-5F);
            }
        }
    }
}

TEST_CASE("RaBitQ Split IP Batch4 and Reorder Hint", "[ut][RaBitQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr uint64_t count = 32;
    constexpr uint64_t base_bits = 8;
    constexpr uint64_t filter_bits = 3;
    auto vecs = fixtures::generate_vectors(count, dim);

    RaBitQuantizer<MetricType::METRIC_TYPE_IP> quantizer(
        dim,
        dim,
        32,
        base_bits,
        false,
        false,
        allocator.get(),
        RaBitQuantizerParameter::RABITQ_VERSION_SPLIT,
        RaBitQuantizerParameter::DEFAULT_RABITQ_ERROR_RATE,
        filter_bits);
    REQUIRE(quantizer.SupportSplitCodeStorage());
    quantizer.TrainImpl(vecs.data(), count);

    auto computer = quantizer.FactoryComputer();
    computer->SetQuery(vecs.data() + 8 * dim);

    std::vector<uint8_t> full_code(quantizer.GetCodeSize());
    std::vector<uint8_t> supplement_code(quantizer.GetSupplementCodeSize());
    std::vector<std::vector<uint8_t>> one_bit_codes(
        4, std::vector<uint8_t>(quantizer.GetOneBitCodeSize()));
    float single_dists[4] = {};
    float single_lower_bounds[4] = {};

    for (uint64_t i = 0; i < 4; ++i) {
        REQUIRE(quantizer.EncodeOne(vecs.data() + i * dim, full_code.data()));
        quantizer.SplitCode(full_code.data(), one_bit_codes[i].data(), supplement_code.data());

        REQUIRE(quantizer.ComputeDistWithOneBitLowerBound(
            *computer, one_bit_codes[i].data(), single_dists + i, single_lower_bounds + i));
        float split_dist = 0.0F;
        REQUIRE(quantizer.ComputeDistWithSplitCode(
            *computer, one_bit_codes[i].data(), supplement_code.data(), &split_dist));
        float hinted_split_dist = 0.0F;
        REQUIRE(quantizer.ComputeDistWithSplitCodeAndFilterDist(*computer,
                                                                one_bit_codes[i].data(),
                                                                supplement_code.data(),
                                                                single_dists[i],
                                                                &hinted_split_dist));
        REQUIRE(std::abs(split_dist - hinted_split_dist) <= 1e-5F);
    }

    float batch_dists[4] = {};
    float batch_lower_bounds[4] = {};
    bool computed[4] = {};
    quantizer.ComputeDistsWithOneBitLowerBoundBatch4(*computer,
                                                     one_bit_codes[0].data(),
                                                     one_bit_codes[1].data(),
                                                     one_bit_codes[2].data(),
                                                     one_bit_codes[3].data(),
                                                     batch_dists[0],
                                                     batch_dists[1],
                                                     batch_dists[2],
                                                     batch_dists[3],
                                                     batch_lower_bounds,
                                                     batch_lower_bounds + 1,
                                                     batch_lower_bounds + 2,
                                                     batch_lower_bounds + 3,
                                                     computed[0],
                                                     computed[1],
                                                     computed[2],
                                                     computed[3]);
    for (uint64_t i = 0; i < 4; ++i) {
        REQUIRE(computed[i]);
        REQUIRE(std::abs(single_dists[i] - batch_dists[i]) <= 1e-5F);
        REQUIRE(std::abs(single_lower_bounds[i] - batch_lower_bounds[i]) <= 1e-5F);
    }
}

TEST_CASE("RaBitQ Encode and Decode", "[ut][RaBitQuantizer]") {
    bool use_fht = GENERATE(true, false);
    auto num_bits_per_dim_query = GENERATE(4, 32);
    auto num_bits_per_dim_base = GENERATE(1, 2, 4, 8);
    for (auto dim : dims) {
        auto pca_dim = dim;
        bool use_mrq = false;
        for (auto count : counts) {
            if (num_bits_per_dim_query == 4 and num_bits_per_dim_base != 1) {
                continue;
            }
            if (dim < 900) {
                WARN(
                    "RaBitQ encode/decode tests only run on high-dimensional data (dim >= 900), "
                    "skipping dim="
                    << dim);
                continue;
            }
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(dim,
                                                                    pca_dim,
                                                                    num_bits_per_dim_query,
                                                                    num_bits_per_dim_base,
                                                                    use_fht,
                                                                    use_mrq,
                                                                    allocator.get());

            TestEncodeDecodeRaBitQ<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>>(
                quantizer, dim, count);

            RaBitQuantizer<MetricType::METRIC_TYPE_IP> quantizer_ip(dim,
                                                                    dim,
                                                                    num_bits_per_dim_query,
                                                                    num_bits_per_dim_base,
                                                                    use_fht,
                                                                    use_mrq,
                                                                    allocator.get());

            TestEncodeDecodeRaBitQ<RaBitQuantizer<MetricType::METRIC_TYPE_IP>>(
                quantizer_ip, dim, count);

            RaBitQuantizer<MetricType::METRIC_TYPE_COSINE> quantizer_cos(dim,
                                                                         dim,
                                                                         num_bits_per_dim_query,
                                                                         num_bits_per_dim_base,
                                                                         use_fht,
                                                                         use_mrq,
                                                                         allocator.get());

            TestEncodeDecodeRaBitQ<RaBitQuantizer<MetricType::METRIC_TYPE_COSINE>>(
                quantizer_cos, dim, count);
        }
    }
}

TEST_CASE("RaBitQ Compute", "[ut][RaBitQuantizer]") {
    auto use_fht = GENERATE(true, false);
    auto num_bits_per_dim_query = GENERATE(4, 32);
    auto num_bits_per_dim_base = GENERATE(2, 4);
    for (auto dim : dims) {
        float numeric_error = 1 / std::sqrt(dim) * dim;
        float related_error = 0.05F;
        float unbounded_numeric_error_rate = 0.05F;
        float unbounded_related_error_rate = 0.1F;
        if (num_bits_per_dim_query == 4) {
            unbounded_related_error_rate = 0.12F;
            if (num_bits_per_dim_base != 1) {
                WARN("num_bits_per_dim_query=4 only supports num_bits_per_dim_base=1");
                continue;
            }
        }
        if (use_fht) {
            unbounded_related_error_rate += 0.05F;
        }
        if (dim < 900) {
            continue;
        }
        bool use_mrq = false;
        for (auto count : counts) {
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            RaBitQuantizer<MetricType::METRIC_TYPE_COSINE> quantizer(dim,
                                                                     dim,
                                                                     num_bits_per_dim_query,
                                                                     num_bits_per_dim_base,
                                                                     use_fht,
                                                                     use_mrq,
                                                                     allocator.get());

            TestComputer<RaBitQuantizer<MetricType::METRIC_TYPE_COSINE>,
                         MetricType::METRIC_TYPE_COSINE>(quantizer,
                                                         dim,
                                                         count,
                                                         numeric_error,
                                                         related_error,
                                                         true,
                                                         unbounded_numeric_error_rate,
                                                         unbounded_related_error_rate);
            REQUIRE_THROWS(TestComputeCodes<RaBitQuantizer<MetricType::METRIC_TYPE_COSINE>,
                                            MetricType::METRIC_TYPE_COSINE>(
                quantizer, dim, count, numeric_error, false));

            RaBitQuantizer<MetricType::METRIC_TYPE_IP> quantizer_ip(dim,
                                                                    dim,
                                                                    num_bits_per_dim_query,
                                                                    num_bits_per_dim_base,
                                                                    use_fht,
                                                                    use_mrq,
                                                                    allocator.get());

            TestComputer<RaBitQuantizer<MetricType::METRIC_TYPE_IP>, MetricType::METRIC_TYPE_IP>(
                quantizer_ip,
                dim,
                count,
                numeric_error,
                related_error,
                true,
                unbounded_numeric_error_rate,
                unbounded_related_error_rate);
            REQUIRE_THROWS(TestComputeCodes<RaBitQuantizer<MetricType::METRIC_TYPE_IP>,
                                            MetricType::METRIC_TYPE_IP>(
                quantizer_ip, dim, count, numeric_error, false));

            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer_l2(dim,
                                                                       dim,
                                                                       num_bits_per_dim_query,
                                                                       num_bits_per_dim_base,
                                                                       use_fht,
                                                                       use_mrq,
                                                                       allocator.get());

            TestComputer<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                         MetricType::METRIC_TYPE_L2SQR>(quantizer_l2,
                                                        dim,
                                                        count,
                                                        numeric_error,
                                                        related_error,
                                                        true,
                                                        unbounded_numeric_error_rate,
                                                        unbounded_related_error_rate);
            REQUIRE_THROWS(TestComputeCodes<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                                            MetricType::METRIC_TYPE_L2SQR>(
                quantizer_l2, dim, count, numeric_error, false));
        }
    }
}

TEST_CASE("RaBitQ Serialize and Deserialize", "[ut][RaBitQuantizer]") {
    bool use_fht = GENERATE(true, false);
    auto num_bits_per_dim_query = GENERATE(4, 32);
    auto num_bits_per_dim_base = GENERATE(1, 4);
    auto dim = 1024;
    float numeric_error = 1 / std::sqrt(dim) * dim;
    float related_error = 0.05F;
    float unbounded_numeric_error_rate = 0.05F;
    float unbounded_related_error_rate = 0.1F;

    bool use_mrq = false;
    for (auto count : counts) {
        if (num_bits_per_dim_query == 4) {
            if (num_bits_per_dim_base != 1) {
                WARN("num_bits_per_dim_query=4 only supports num_bits_per_dim_base=1");
                continue;
            }
            unbounded_related_error_rate = 0.15F;
        }
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer1(dim,
                                                                 dim,
                                                                 num_bits_per_dim_query,
                                                                 num_bits_per_dim_base,
                                                                 use_fht,
                                                                 use_mrq,
                                                                 allocator.get());
        RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer2(dim,
                                                                 dim,
                                                                 num_bits_per_dim_query,
                                                                 num_bits_per_dim_base,
                                                                 use_fht,
                                                                 use_mrq,
                                                                 allocator.get());

        TestSerializeAndDeserialize<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                                    MetricType::METRIC_TYPE_L2SQR>(quantizer1,
                                                                   quantizer2,
                                                                   dim,
                                                                   count,
                                                                   numeric_error,
                                                                   related_error,
                                                                   unbounded_numeric_error_rate,
                                                                   unbounded_related_error_rate,
                                                                   true);
        RaBitQuantizer<MetricType::METRIC_TYPE_IP> quantizer_ip1(dim,
                                                                 dim,
                                                                 num_bits_per_dim_query,
                                                                 num_bits_per_dim_base,
                                                                 use_fht,
                                                                 use_mrq,
                                                                 allocator.get());
        RaBitQuantizer<MetricType::METRIC_TYPE_IP> quantizer_ip2(dim,
                                                                 dim,
                                                                 num_bits_per_dim_query,
                                                                 num_bits_per_dim_base,
                                                                 use_fht,
                                                                 use_mrq,
                                                                 allocator.get());

        TestSerializeAndDeserialize<RaBitQuantizer<MetricType::METRIC_TYPE_IP>,
                                    MetricType::METRIC_TYPE_IP>(quantizer_ip1,
                                                                quantizer_ip2,
                                                                dim,
                                                                count,
                                                                numeric_error,
                                                                related_error,
                                                                unbounded_numeric_error_rate,
                                                                unbounded_related_error_rate,
                                                                true);
        RaBitQuantizer<MetricType::METRIC_TYPE_COSINE> quantizer_cos1(dim,
                                                                      dim,
                                                                      num_bits_per_dim_query,
                                                                      num_bits_per_dim_base,
                                                                      use_fht,
                                                                      use_mrq,
                                                                      allocator.get());
        RaBitQuantizer<MetricType::METRIC_TYPE_COSINE> quantizer_cos2(dim,
                                                                      dim,
                                                                      num_bits_per_dim_query,
                                                                      num_bits_per_dim_base,
                                                                      use_fht,
                                                                      use_mrq,
                                                                      allocator.get());

        TestSerializeAndDeserialize<RaBitQuantizer<MetricType::METRIC_TYPE_COSINE>,
                                    MetricType::METRIC_TYPE_COSINE>(quantizer_cos1,
                                                                    quantizer_cos2,
                                                                    dim,
                                                                    count,
                                                                    numeric_error,
                                                                    related_error,
                                                                    unbounded_numeric_error_rate,
                                                                    unbounded_related_error_rate,
                                                                    true);
    }
}

TEST_CASE("RaBitQ Query SQ4 Transform", "[ut][RaBitQuantizer]") {
    bool use_fht = GENERATE(true, false);
    int dim = 6;
    uint64_t num_bits_per_dim_query = 4;
    uint64_t num_bits_per_dim_base = 1;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, dim, num_bits_per_dim_query, num_bits_per_dim_base, use_fht, false, allocator.get());

    std::vector<float> original_data = {1, 2, 4, 8, 15, 0};
    // input  [0010 0001, 1000 0100, 0000 1111]
    std::vector<uint8_t> input = {0x21, 0x84, 0x0f};
    std::vector<uint8_t> sq_data(4 + 4 + 4, 0);

    // test sq
    SQ4UniformQuantizer<MetricType::METRIC_TYPE_IP> sq4_quantizer(dim, allocator.get(), 0.0F);
    sq4_quantizer.Train(original_data.data(), 1);
    sq4_quantizer.EncodeOneImpl(original_data.data(), sq_data.data());
    auto is_consistent = std::memcmp(sq_data.data(), input.data(), input.size());
    REQUIRE(is_consistent == 0);
    REQUIRE(std::abs(*(float*)(&sq_data[4]) - 30) < 1e-5);
    REQUIRE(std::abs(*(float*)(&sq_data[8]) - 30) < 1e-5);

    // test reorder
    // output  [0001 0001, 0001 0010, 0001 0100, 0001 1000]
    std::vector<uint8_t> expected_output;
    expected_output.reserve(64 * 4);
    for (auto i = 0; i < 64 * 4; i++) {
        if (i == 0) {
            expected_output.push_back(0x11);
        } else if (i == 64) {
            expected_output.push_back(0x12);
        } else if (i == 128) {
            expected_output.push_back(0x14);
        } else if (i == 192) {
            expected_output.push_back(0x18);
        } else {
            expected_output.push_back(0);
        }
    }
    std::vector<uint8_t> output(64 * 4, 0);
    std::vector<uint8_t> recovered_input(3, 0);

    // reorder the input
    quantizer.ReOrderSQ4(input.data(), output.data());
    is_consistent = std::memcmp(expected_output.data(), output.data(), output.size());
    REQUIRE(is_consistent == 0);

    // recover the original order
    quantizer.RecoverOrderSQ4(output.data(), recovered_input.data());
    is_consistent = std::memcmp(recovered_input.data(), input.data(), input.size());
    REQUIRE(is_consistent == 0);
}

TEST_CASE("RaBitQ Query SQ4 Transform dim=15", "[ut][RaBitQuantizer]") {
    bool use_fht = GENERATE(true, false);
    int dim = 15;
    int aligned_dim = ((dim + 511) / 512) * 512;
    uint64_t num_bits_per_dim_query = 4;
    uint64_t num_bits_per_dim_base = 1;
    int sq_code_size = aligned_dim / 8 * num_bits_per_dim_query;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, dim, num_bits_per_dim_query, num_bits_per_dim_base, use_fht, false, allocator.get());

    std::vector<float> original_data = {1, 2, 4, 8, 0, 3, 11, 15, 9, 13, 10, 6, 7, 12, 14};
    // input  [0010 0001, 1000 0100, 0011 0000, 1111 1011, 1101 1001, 0110 1010, 1100 0111, 0000 1110]
    std::vector<uint8_t> input = {0x21, 0x84, 0x30, 0xfb, 0xd9, 0x6a, 0xc7, 0x0e};
    std::vector<uint8_t> sq_data(dim + 4 + 4, 0);

    // test sq
    SQ4UniformQuantizer<MetricType::METRIC_TYPE_IP> sq4_quantizer(dim, allocator.get(), 0.0F);
    sq4_quantizer.Train(original_data.data(), 1);
    sq4_quantizer.EncodeOneImpl(original_data.data(), sq_data.data());

    auto is_consistent = std::memcmp(sq_data.data(), input.data(), input.size());
    REQUIRE(is_consistent == 0);

    // test reorder
    // output:
    //     1110 0001 0001 0011 000000000...
    //     1110 0010 0101 1100 000000000...
    //     1000 0100 0111 1010 000000000...
    //     1100 1000 0110 0111 000000000...
    std::vector<uint8_t> expected_output(sq_code_size, 0);
    expected_output[0] = 0xe1;
    expected_output[1] = 0x13;

    expected_output[aligned_dim / 8] = 0xe2;
    expected_output[aligned_dim / 8 * 1 + 1] = 0x5c;

    expected_output[aligned_dim / 8 * 2] = 0x84;
    expected_output[aligned_dim / 8 * 2 + 1] = 0x7a;

    expected_output[aligned_dim / 8 * 3] = 0xc8;
    expected_output[aligned_dim / 8 * 3 + 1] = 0x67;

    std::vector<uint8_t> output(sq_code_size, 0);
    std::vector<uint8_t> recovered_input(dim, 0);

    // reorder the input
    quantizer.ReOrderSQ4(input.data(), output.data());
    is_consistent = std::memcmp(expected_output.data(), output.data(), output.size());
    REQUIRE(is_consistent == 0);

    // recover the original order
    quantizer.RecoverOrderSQ4(output.data(), recovered_input.data());
    is_consistent = std::memcmp(recovered_input.data(), input.data(), input.size());
    REQUIRE(is_consistent == 0);
}

TEST_CASE("RaBitQ Query SQ Encode Decode", "[ut][RaBitQuantizer]") {
    bool use_fht = GENERATE(true, false);
    int dim = 6;
    uint64_t num_bits_per_dim_query = 4;
    uint64_t num_bits_per_dim_base = 1;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, dim, num_bits_per_dim_query, num_bits_per_dim_base, use_fht, false, allocator.get());

    std::vector<float> original_data = {1, 2, 4, 8, 15, 0};
    std::vector<uint8_t> sq_data(dim, 0);
    std::vector<uint8_t> expected_data = {1, 2, 4, 8, 15, 0};

    float upper_bound = std::numeric_limits<float>::max();
    float lower_bound = std::numeric_limits<float>::max();
    float delta = 0.0F;
    sum_type query_sum = 0;
    // test sq encode
    quantizer.EncodeSQ(
        original_data.data(), sq_data.data(), upper_bound, lower_bound, delta, query_sum);
    auto is_consistent = std::memcmp(sq_data.data(), expected_data.data(), expected_data.size());
    REQUIRE(is_consistent == 0);

    // test sq decode
    std::vector<float> decode_data(dim, 0);
    quantizer.DecodeSQ(sq_data.data(), decode_data.data(), upper_bound, lower_bound);
    for (int i = 0; i < dim; ++i) {
        REQUIRE(is_approx_zero(original_data[i] - decode_data[i]));
    }

    // test reorder
    // output  [0001 0001, 0001 0010, 0001 0100, 0001 1000]
    std::vector<uint8_t> expected_output;
    expected_output.reserve(64 * 4);
    for (auto i = 0; i < 64 * 4; i++) {
        if (i == 0) {
            expected_output.push_back(0x11);
        } else if (i == 64) {
            expected_output.push_back(0x12);
        } else if (i == 128) {
            expected_output.push_back(0x14);
        } else if (i == 192) {
            expected_output.push_back(0x18);
        } else {
            expected_output.push_back(0);
        }
    }

    std::vector<uint8_t> output(64 * 4, 0);
    std::vector<uint8_t> recovered_input(dim, 0);
    // reorder the input
    quantizer.ReOrderSQ(sq_data.data(), output.data());
    is_consistent = std::memcmp(expected_output.data(), output.data(), output.size());
    REQUIRE(is_consistent == 0);

    // recover the original order
    quantizer.RecoverOrderSQ(output.data(), recovered_input.data());
    is_consistent = std::memcmp(recovered_input.data(), sq_data.data(), sq_data.size());
    REQUIRE(is_consistent == 0);
}

TEST_CASE("RaBitQ Query SQ Transform dim=15", "[ut][RaBitQuantizer]") {
    bool use_fht = GENERATE(true, false);
    int dim = 15;
    int aligned_dim = ((dim + 511) / 512) * 512;
    uint64_t num_bits_per_dim_query = 4;
    uint64_t num_bits_per_dim_base = 1;
    int sq_code_size = aligned_dim / 8 * num_bits_per_dim_query;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, dim, num_bits_per_dim_query, num_bits_per_dim_base, use_fht, false, allocator.get());

    std::vector<float> original_data = {1, 2, 4, 8, 0, 3, 11, 15, 9, 13, 10, 6, 7, 12, 14};
    std::vector<uint8_t> expected_data = {1, 2, 4, 8, 0, 3, 11, 15, 9, 13, 10, 6, 7, 12, 14};
    std::vector<uint8_t> sq_data(dim, 0);

    // test sq
    float upper_bound = std::numeric_limits<float>::max();
    float lower_bound = std::numeric_limits<float>::max();
    float delta = 0.0F;
    sum_type query_sum = 0;
    // test sq encode
    quantizer.EncodeSQ(
        original_data.data(), sq_data.data(), upper_bound, lower_bound, delta, query_sum);
    auto is_consistent = std::memcmp(sq_data.data(), expected_data.data(), expected_data.size());
    REQUIRE(is_consistent == 0);

    // test sq decode
    std::vector<float> decode_data(dim, 0);
    quantizer.DecodeSQ(sq_data.data(), decode_data.data(), upper_bound, lower_bound);
    for (int i = 0; i < dim; ++i) {
        REQUIRE(is_approx_zero(original_data[i] - decode_data[i]));
    }

    // test reorder
    // output:
    //     1110 0001 0001 0011 000000000...
    //     1110 0010 0101 1100 000000000...
    //     1000 0100 0111 1010 000000000...
    //     1100 1000 0110 0111 000000000...
    std::vector<uint8_t> expected_output(sq_code_size, 0);
    expected_output[0] = 0xe1;
    expected_output[1] = 0x13;

    expected_output[aligned_dim / 8] = 0xe2;
    expected_output[aligned_dim / 8 * 1 + 1] = 0x5c;

    expected_output[aligned_dim / 8 * 2] = 0x84;
    expected_output[aligned_dim / 8 * 2 + 1] = 0x7a;

    expected_output[aligned_dim / 8 * 3] = 0xc8;
    expected_output[aligned_dim / 8 * 3 + 1] = 0x67;

    std::vector<uint8_t> output(sq_code_size, 0);
    std::vector<uint8_t> recovered_input(dim, 0);

    // reorder the input
    quantizer.ReOrderSQ(sq_data.data(), output.data());
    is_consistent = std::memcmp(expected_output.data(), output.data(), output.size());
    REQUIRE(is_consistent == 0);

    // recover the original order
    quantizer.RecoverOrderSQ(output.data(), recovered_input.data());
    is_consistent = std::memcmp(recovered_input.data(), sq_data.data(), sq_data.size());
    REQUIRE(is_consistent == 0);
}

TEST_CASE("RaBitQ Query SQ Transform With All Same Element", "[ut][RaBitQuantizer]") {
    bool use_fht = GENERATE(true, false);
    int dim = 15;
    int aligned_dim = ((dim + 511) / 512) * 512;
    uint64_t num_bits_per_dim_query = 4;
    uint64_t num_bits_per_dim_base = 1;
    int sq_code_size = aligned_dim / 8 * num_bits_per_dim_query;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, dim, num_bits_per_dim_query, num_bits_per_dim_base, use_fht, false, allocator.get());

    std::vector<float> original_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<uint8_t> sq_data(dim, 0);

    // test sq
    float upper_bound = std::numeric_limits<float>::max();
    float lower_bound = std::numeric_limits<float>::max();
    float delta = 0.0F;
    sum_type query_sum = 0;
    quantizer.EncodeSQ(
        original_data.data(), sq_data.data(), upper_bound, lower_bound, delta, query_sum);
    std::vector<float> decode_data(dim, 0);
    quantizer.DecodeSQ(sq_data.data(), decode_data.data(), upper_bound, lower_bound);
    for (int i = 0; i < dim; ++i) {
        REQUIRE(is_approx_zero(original_data[i] - decode_data[i]));
    }
}
