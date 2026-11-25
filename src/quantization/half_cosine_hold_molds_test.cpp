// Copyright 2026-present the vsag project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "fp32_quantizer.h"
#include "fp32_quantizer_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "simd/fp32_simd.h"
#include "unittest.h"

namespace {
using Quant = vsag::FP32Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>;
uint32_t
Bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
bool
Finite(float value) {
    return (Bits(value) & 0x7f800000U) != 0x7f800000U;
}
void
CheckRows(int64_t dim, const std::vector<float>& rows) {
    vsag::IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto raw_param = std::make_shared<vsag::FP32QuantizerParameter>();
    raw_param->hold_molds = true;
    auto normal_param = std::make_shared<vsag::FP32QuantizerParameter>();
    Quant raw(raw_param, common);
    Quant normal(normal_param, common);
    const uint64_t count = rows.size() / dim;
    std::vector<uint8_t> raw_codes(count * raw.GetCodeSize());
    std::vector<uint8_t> normal_codes(count * normal.GetCodeSize());
    REQUIRE(raw.EncodeBatch(rows.data(), raw_codes.data(), count));
    REQUIRE(normal.EncodeBatch(rows.data(), normal_codes.data(), count));
    std::vector<float> decoded(dim);
    for (uint64_t i = 0; i < count; ++i) {
        REQUIRE(raw.DecodeOne(raw_codes.data() + i * raw.GetCodeSize(), decoded.data()));
        for (int64_t j = 0; j < dim; ++j) {
            REQUIRE(Bits(decoded[j]) == Bits(rows[i * dim + j]));
        }
        auto raw_comp = raw.FactoryComputer();
        auto normal_comp = normal.FactoryComputer();
        raw_comp->SetQuery(rows.data() + i * dim);
        normal_comp->SetQuery(rows.data() + i * dim);
        std::vector<float> scan(count);
        raw.ScanBatchDists(*raw_comp, count, raw_codes.data(), scan.data());
        float batch[4]{};
        raw.ComputeDistsBatch4Impl(*raw_comp,
                                   raw_codes.data(),
                                   raw_codes.data() + raw.GetCodeSize(),
                                   raw_codes.data() + 2 * raw.GetCodeSize(),
                                   raw_codes.data() + 3 * raw.GetCodeSize(),
                                   batch[0],
                                   batch[1],
                                   batch[2],
                                   batch[3]);
        for (uint64_t j = 0; j < count; ++j) {
            CAPTURE(dim, i, j);
            auto* rc = raw_codes.data() + j * raw.GetCodeSize();
            auto* nc = normal_codes.data() + j * normal.GetCodeSize();
            const auto expected = normal.ComputeDist(*normal_comp, nc);
            const auto actual = raw.ComputeDist(*raw_comp, rc);
            INFO("query actual bits=" << Bits(actual) << " expected bits=" << Bits(expected));
            REQUIRE(Finite(actual));
            REQUIRE(std::abs(actual - expected) <= 1e-5F);
            REQUIRE(std::abs(scan[j] - expected) <= 1e-5F);
            if (j < 4)
                REQUIRE(std::abs(batch[j] - expected) <= 1e-5F);
            const auto pair = raw.Compute(raw_codes.data() + i * raw.GetCodeSize(), rc);
            const auto ref_pair =
                normal.Compute(normal_codes.data() + i * normal.GetCodeSize(), nc);
            INFO("pair actual bits=" << Bits(pair) << " expected bits=" << Bits(ref_pair));
            REQUIRE(Finite(pair));
            REQUIRE(std::abs(pair - ref_pair) <= 1e-5F);
        }
    }
}
}  // namespace

TEST_CASE("Half cosine raw molds finite and zero oracle", "[half_cosine_oracle]") {
    for (int64_t dim : {7, 33}) {
        std::vector<float> rows(4 * dim);
        for (int64_t j = 0; j < dim; ++j) {
            rows[j] = (j % 2 == 0) ? 0.0F : -0.0F;
            rows[dim + j] = static_cast<float>(j + 1) / 16.0F;
            rows[2 * dim + j] = static_cast<float>(j - 3) / 8.0F;
            // Smallest BF16 subnormal has a norm that underflows in FP32.
            uint32_t bits = 0x00010000U;
            std::memcpy(rows.data() + 3 * dim + j, &bits, sizeof(bits));
        }
        CheckRows(dim, rows);
    }
}

TEST_CASE("Half cosine finite BF16 overflow pair oracle", "[half_cosine_extreme]") {
    constexpr int64_t dim = 7;
    std::vector<float> rows(4 * dim);
    for (int64_t i = 0; i < 4 * dim; ++i) {
        uint32_t bits = 0x60ad0000U;  // Finite BF16 value about 1e20; norm-square overflows.
        std::memcpy(rows.data() + i, &bits, sizeof(bits));
    }
    CheckRows(dim, rows);
}

TEST_CASE("Half cosine finite boundary rows oracle", "[half_cosine_extreme]") {
    for (int64_t dim : {7, 33}) {
        // Mixed Batch4 lanes: overflowing norms, ordinary, signed zero, tiny.
        std::vector<float> rows(4 * dim);
        for (int64_t j = 0; j < dim; ++j) {
            uint32_t huge = 0x60ad0000U;
            std::memcpy(rows.data() + j, &huge, sizeof(huge));
            rows[dim + j] = static_cast<float>(j - 2) / 8.0F;
            rows[2 * dim + j] = j % 2 == 0 ? 0.0F : -0.0F;
            rows[3 * dim + j] = 1e-20F;
        }
        CheckRows(dim, rows);
    }
}

TEST_CASE("Half cosine legacy molds and accumulator boundaries", "[half_cosine_extreme]") {
    constexpr int64_t dim = 7;
    vsag::IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto param = std::make_shared<vsag::FP32QuantizerParameter>();
    param->hold_molds = true;
    Quant raw(param, common);
    auto normal_param = std::make_shared<vsag::FP32QuantizerParameter>();
    Quant normal(normal_param, common);
    std::vector<float> rows(4 * dim, 0.25F);
    std::fill(rows.begin(), rows.begin() + dim, 0.0F);
    std::vector<uint8_t> codes(4 * raw.GetCodeSize());
    std::vector<uint8_t> reference(4 * normal.GetCodeSize());
    REQUIRE(raw.EncodeBatch(rows.data(), codes.data(), 4));
    REQUIRE(normal.EncodeBatch(rows.data(), reference.data(), 4));
    // Zero norm occurs in historical zero-row payloads. The two finite
    // synthetic stored norms exercise consumer product overflow/underflow;
    // these are not claimed to be emitted by the current encoder.
    const float norms[4] = {0.0F, 2e19F, 2e19F, 1e-30F};
    for (uint64_t i = 0; i < 4; ++i) {
        std::memcpy(
            codes.data() + i * raw.GetCodeSize() + dim * sizeof(float), norms + i, sizeof(float));
    }
    for (uint64_t i = 0; i < 4; ++i) {
        for (uint64_t j = 0; j < 4; ++j) {
            if (i == 0 or j == 0 or (i == 1 and j == 2) or (i == 3 and j == 3)) {
                const auto actual = raw.Compute(codes.data() + i * raw.GetCodeSize(),
                                                codes.data() + j * raw.GetCodeSize());
                const auto expected = normal.Compute(reference.data() + i * normal.GetCodeSize(),
                                                     reference.data() + j * normal.GetCodeSize());
                REQUIRE(Finite(actual));
                REQUIRE(std::abs(actual - expected) <= 1e-5F);
            }
        }
    }
    auto computer = raw.FactoryComputer();
    computer->SetQuery(rows.data() + dim);
    float batch[4] = {2.0F, -3.0F, 0.5F, 4.0F};
    float expected[4] = {2.0F, -3.0F, 0.5F, 4.0F};
    vsag::FP32ComputeIPBatch4(reinterpret_cast<const float*>(computer->buf_),
                              dim,
                              reinterpret_cast<const float*>(codes.data()),
                              reinterpret_cast<const float*>(codes.data() + raw.GetCodeSize()),
                              reinterpret_cast<const float*>(codes.data() + 2 * raw.GetCodeSize()),
                              reinterpret_cast<const float*>(codes.data() + 3 * raw.GetCodeSize()),
                              expected[0],
                              expected[1],
                              expected[2],
                              expected[3]);
    for (uint64_t i = 0; i < 4; ++i) expected[i] = 1.0F - expected[i] / norms[i];
    raw.ComputeDistsBatch4Impl(*computer,
                               codes.data(),
                               codes.data() + raw.GetCodeSize(),
                               codes.data() + 2 * raw.GetCodeSize(),
                               codes.data() + 3 * raw.GetCodeSize(),
                               batch[0],
                               batch[1],
                               batch[2],
                               batch[3]);
    // Nonzero-entry low-level accumulation keeps its old result, including
    // the historical nonfinite boundary. It is not the zeroed Flatten API.
    for (uint64_t i = 0; i < 4; ++i) REQUIRE(Bits(batch[i]) == Bits(expected[i]));
}

TEST_CASE("Half cosine old zero norm query paths", "[half_cosine_extreme]") {
    constexpr int64_t dim = 7;
    vsag::IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto param = std::make_shared<vsag::FP32QuantizerParameter>();
    param->hold_molds = true;
    Quant raw(param, common);
    std::vector<float> rows(4 * dim, 0.0F);
    std::vector<uint8_t> codes(4 * raw.GetCodeSize());
    REQUIRE(raw.EncodeBatch(rows.data(), codes.data(), 4));
    const float old_norm = 0.0F;
    for (uint64_t i = 0; i < 4; ++i) {
        std::memcpy(codes.data() + i * raw.GetCodeSize() + dim * sizeof(float),
                    &old_norm,
                    sizeof(old_norm));
    }
    std::vector<float> query(dim, 0.25F);
    auto computer = raw.FactoryComputer();
    computer->SetQuery(query.data());
    std::vector<float> scan(4);
    raw.ScanBatchDists(*computer, 4, codes.data(), scan.data());
    float batch[4]{};
    raw.ComputeDistsBatch4Impl(*computer,
                               codes.data(),
                               codes.data() + raw.GetCodeSize(),
                               codes.data() + 2 * raw.GetCodeSize(),
                               codes.data() + 3 * raw.GetCodeSize(),
                               batch[0],
                               batch[1],
                               batch[2],
                               batch[3]);
    for (uint64_t i = 0; i < 4; ++i) {
        REQUIRE(raw.ComputeDist(*computer, codes.data() + i * raw.GetCodeSize()) == 1.0F);
        REQUIRE(scan[i] == 1.0F);
        REQUIRE(batch[i] == 1.0F);
    }
}

TEST_CASE("Half cosine nonfinite follows Normalize classification", "[half_cosine_extreme]") {
    constexpr int64_t dim = 7;
    vsag::IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto param = std::make_shared<vsag::FP32QuantizerParameter>();
    param->hold_molds = true;
    Quant raw(param, common);
    auto normal_param = std::make_shared<vsag::FP32QuantizerParameter>();
    Quant normal(normal_param, common);
    std::vector<float> rows(4 * dim, 0.25F);
    const uint32_t special[3] = {0x7f800000U, 0xff800000U, 0x7fc10000U};
    for (uint64_t i = 0; i < 3; ++i) std::memcpy(rows.data() + i * dim, special + i, 4);
    std::vector<uint8_t> codes(4 * raw.GetCodeSize());
    std::vector<uint8_t> refs(4 * normal.GetCodeSize());
    REQUIRE(raw.EncodeBatch(rows.data(), codes.data(), 4));
    REQUIRE(normal.EncodeBatch(rows.data(), refs.data(), 4));
    auto same_class = [](float actual, float expected) {
        const uint32_t a = Bits(actual), e = Bits(expected);
        REQUIRE((a & 0x7f800000U) == (e & 0x7f800000U));
        if ((e & 0x7f800000U) == 0x7f800000U) {
            REQUIRE(((a & 0x007fffffU) == 0U) == ((e & 0x007fffffU) == 0U));
        } else {
            REQUIRE(std::abs(actual - expected) <= 1e-5F);
        }
    };
    for (uint64_t i = 0; i < 4; ++i) {
        auto rc = raw.FactoryComputer();
        auto nc = normal.FactoryComputer();
        rc->SetQuery(rows.data() + i * dim);
        nc->SetQuery(rows.data() + i * dim);
        float scan[4];
        raw.ScanBatchDists(*rc, 4, codes.data(), scan);
        float batch[4]{};
        raw.ComputeDistsBatch4Impl(*rc,
                                   codes.data(),
                                   codes.data() + raw.GetCodeSize(),
                                   codes.data() + 2 * raw.GetCodeSize(),
                                   codes.data() + 3 * raw.GetCodeSize(),
                                   batch[0],
                                   batch[1],
                                   batch[2],
                                   batch[3]);
        for (uint64_t j = 0; j < 4; ++j) {
            CAPTURE(i, j);
            const auto expected = normal.ComputeDist(*nc, refs.data() + j * normal.GetCodeSize());
            same_class(raw.ComputeDist(*rc, codes.data() + j * raw.GetCodeSize()), expected);
            same_class(scan[j], expected);
            same_class(batch[j], expected);
            same_class(raw.Compute(codes.data() + i * raw.GetCodeSize(),
                                   codes.data() + j * raw.GetCodeSize()),
                       normal.Compute(refs.data() + i * normal.GetCodeSize(),
                                      refs.data() + j * normal.GetCodeSize()));
        }
    }
}
