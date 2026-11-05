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

#include "binary_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "binary_quantizer_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantizer_test.h"
#include "unittest.h"

using namespace vsag;

namespace {

uint64_t
ReferenceHamming(const uint8_t* x, const uint8_t* y, uint64_t num_bytes) {
    uint64_t result = 0;
    for (uint64_t i = 0; i < num_bytes; ++i) {
        result += static_cast<uint64_t>(__builtin_popcount(static_cast<unsigned int>(x[i] ^ y[i])));
    }
    return result;
}

// Generate packed binary vectors of `bits` logical bits each. Unless keep_dirty_tail is set,
// the unused high bits of the final byte are cleared so that ground-truth and stored codes agree.
std::vector<uint8_t>
GeneratePackedBinary(uint64_t count, uint64_t bits, uint32_t seed, bool keep_dirty_tail = false) {
    const uint64_t num_bytes = (bits + 7) / 8;
    std::vector<uint8_t> data(count * num_bytes);
    std::mt19937 rng(seed);
    for (auto& byte : data) {
        byte = static_cast<uint8_t>(rng());
    }
    if (not keep_dirty_tail and bits % 8 != 0) {
        const auto mask = static_cast<uint8_t>((1U << (bits % 8U)) - 1U);
        for (uint64_t i = 0; i < count; ++i) {
            data[(i + 1) * num_bytes - 1] &= mask;
        }
    }
    return data;
}

uint64_t
TailMaskedByte(uint8_t byte, uint64_t bits, uint64_t byte_index, uint64_t num_bytes) {
    if (bits % 8 != 0 and byte_index + 1 == num_bytes) {
        return byte & static_cast<uint8_t>((1U << (bits % 8U)) - 1U);
    }
    return byte;
}

uint64_t
ReferenceHammingMasked(const uint8_t* x, const uint8_t* y, uint64_t num_bytes, uint64_t bits) {
    uint64_t result = 0;
    for (uint64_t i = 0; i < num_bytes; ++i) {
        const auto diff = static_cast<unsigned int>(TailMaskedByte(x[i], bits, i, num_bytes) ^
                                                    TailMaskedByte(y[i], bits, i, num_bytes));
        result += static_cast<uint64_t>(__builtin_popcount(diff));
    }
    return result;
}

}  // namespace

TEST_CASE("BinaryQuantizer Code Size and Parameter", "[ut][binary_quantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (uint64_t bits : {1, 7, 8, 9, 63, 64, 65, 127, 128, 256}) {
        BinaryQuantizer quantizer(static_cast<int>(bits), allocator.get());
        REQUIRE(quantizer.GetCodeSize() == (bits + 7) / 8);
        REQUIRE(quantizer.GetQueryCodeSize() == (bits + 7) / 8);
        REQUIRE(quantizer.GetDim() == static_cast<int>(bits));
        REQUIRE(quantizer.Metric() == MetricType::METRIC_TYPE_HAMMING);
        REQUIRE(quantizer.Name() == QUANTIZATION_TYPE_VALUE_BINARY);
    }

    // Parameter type round-trips through the quantizer parameter factory.
    auto param = std::make_shared<BinaryQuantizerParameter>();
    auto json = param->ToJson();
    REQUIRE(json[TYPE_KEY].GetString() == QUANTIZATION_TYPE_VALUE_BINARY);
    auto parsed = QuantizerParameter::GetQuantizerParameterByJson(json);
    REQUIRE(parsed != nullptr);
    REQUIRE(parsed->GetTypeName() == QUANTIZATION_TYPE_VALUE_BINARY);
    REQUIRE(QuantizerParameter::IsValidQuantizationType(QUANTIZATION_TYPE_VALUE_BINARY));

    // Constructing via IndexCommonParam matches the direct-dimension constructor.
    IndexCommonParam common_param;
    common_param.dim_ = 100;
    common_param.allocator_ = allocator;
    BinaryQuantizer from_common(param, common_param);
    REQUIRE(from_common.GetCodeSize() == 13);
}

TEST_CASE("BinaryQuantizer Encode Decode Compute", "[ut][binary_quantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const uint64_t count = 17;
    for (uint64_t bits : {9, 64, 127, 256}) {
        const uint64_t num_bytes = (bits + 7) / 8;
        auto data = GeneratePackedBinary(count, bits, 47);
        BinaryQuantizer quantizer(static_cast<int>(bits), allocator.get());
        REQUIRE(quantizer.Train(nullptr, count));

        std::vector<uint8_t> codes(num_bytes * count, 0);
        std::vector<uint8_t> single(num_bytes, 0);
        for (uint64_t i = 0; i < count; ++i) {
            const auto* source = data.data() + i * num_bytes;
            REQUIRE(quantizer.EncodeOne(reinterpret_cast<const float*>(source), single.data()));
            REQUIRE(std::memcmp(single.data(), source, num_bytes) == 0);
            REQUIRE(quantizer.Compute(single.data(), codes.data() + i * num_bytes) ==
                    static_cast<float>(ReferenceHammingMasked(
                        source, codes.data() + i * num_bytes, num_bytes, bits)));
        }

        REQUIRE(quantizer.EncodeBatch(
            reinterpret_cast<const float*>(data.data()), codes.data(), count));
        for (uint64_t i = 0; i < count * num_bytes; ++i) {
            REQUIRE(codes[i] == data[i]);
        }

        // Self distance is zero; pairwise distance is the exact differing-bit count.
        for (uint64_t i = 0; i < count; ++i) {
            REQUIRE(quantizer.Compute(codes.data() + i * num_bytes, codes.data() + i * num_bytes) ==
                    0.0F);
            for (uint64_t j = 0; j < count; ++j) {
                auto expected = static_cast<float>(ReferenceHamming(
                    codes.data() + i * num_bytes, codes.data() + j * num_bytes, num_bytes));
                REQUIRE(quantizer.Compute(codes.data() + i * num_bytes,
                                          codes.data() + j * num_bytes) == expected);
            }
        }

        // Decode preserves packed bytes so decoded vectors can be encoded again unchanged.
        std::vector<uint8_t> decoded(count * num_bytes, 0xFFU);
        REQUIRE(quantizer.DecodeBatch(
            codes.data(), reinterpret_cast<float*>(decoded.data()), count));
        REQUIRE(decoded == codes);
        std::vector<uint8_t> reencoded(count * num_bytes, 0);
        REQUIRE(quantizer.EncodeBatch(
            reinterpret_cast<const float*>(decoded.data()), reencoded.data(), count));
        REQUIRE(reencoded == codes);
    }
}

TEST_CASE("BinaryQuantizer Ignores Unused Tail Bits", "[ut][binary_quantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t bits = 9;  // two bytes, last byte uses 1 bit
    const uint64_t num_bytes = 2;
    BinaryQuantizer quantizer(static_cast<int>(bits), allocator.get());
    REQUIRE(quantizer.Train(nullptr, 1));

    const std::vector<uint8_t> clean = {0x02U, 0x00U};
    const std::vector<uint8_t> dirty = {0x02U, 0xFCU};  // differs only in unused bits

    std::vector<uint8_t> codes1(num_bytes, 0);
    std::vector<uint8_t> codes2(num_bytes, 0);
    quantizer.EncodeOne(reinterpret_cast<const float*>(clean.data()), codes1.data());
    quantizer.EncodeOne(reinterpret_cast<const float*>(dirty.data()), codes2.data());

    // Stored codes have their unused tail bits cleared.
    REQUIRE(codes2[1] == 0x00U);
    REQUIRE(std::memcmp(codes1.data(), codes2.data(), num_bytes) == 0);
    REQUIRE(quantizer.Compute(codes1.data(), codes2.data()) == 0.0F);

    // A dirty-tail query is masked during preprocessing as well.
    auto computer = quantizer.FactoryComputer();
    computer->SetQuery(reinterpret_cast<const float*>(dirty.data()));
    float dist = -1.0F;
    computer->ComputeDist(codes1.data(), &dist);
    REQUIRE(dist == 0.0F);
}

TEST_CASE("BinaryQuantizer Computer Distances", "[ut][binary_quantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const uint64_t count = 20;
    for (uint64_t bits : {128, 257}) {
        const uint64_t num_bytes = (bits + 7) / 8;
        auto data = GeneratePackedBinary(count, bits, 1234);
        auto query = GeneratePackedBinary(1, bits, 99);
        BinaryQuantizer quantizer(static_cast<int>(bits), allocator.get());
        REQUIRE(quantizer.Train(nullptr, count));
        std::vector<uint8_t> codes(num_bytes * count, 0);
        REQUIRE(quantizer.EncodeBatch(
            reinterpret_cast<const float*>(data.data()), codes.data(), count));

        auto computer = quantizer.FactoryComputer();
        computer->SetQuery(reinterpret_cast<const float*>(query.data()));

        std::vector<float> dists(count, -1.0F);
        computer->ScanBatchDists(count, codes.data(), dists.data());
        for (uint64_t i = 0; i < count; ++i) {
            auto expected = static_cast<float>(
                ReferenceHamming(query.data(), codes.data() + i * num_bytes, num_bytes));
            REQUIRE(dists[i] == expected);
            float single = -1.0F;
            computer->ComputeDist(codes.data() + i * num_bytes, &single);
            REQUIRE(single == expected);
        }

        float d1 = -1.0F;
        float d2 = -1.0F;
        float d3 = -1.0F;
        float d4 = -1.0F;
        if (count >= 4) {
            computer->ComputeDistsBatch4(codes.data(),
                                         codes.data() + num_bytes,
                                         codes.data() + 2 * num_bytes,
                                         codes.data() + 3 * num_bytes,
                                         d1,
                                         d2,
                                         d3,
                                         d4);
            for (uint64_t i = 0; i < 4; ++i) {
                auto expected = static_cast<float>(
                    ReferenceHamming(query.data(), codes.data() + i * num_bytes, num_bytes));
                const float actual = (i == 0) ? d1 : (i == 1) ? d2 : (i == 2) ? d3 : d4;
                REQUIRE(actual == expected);
            }
        }
    }
}

TEST_CASE("BinaryQuantizer Serialize and Deserialize", "[ut][binary_quantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const uint64_t bits = 100;
    const uint64_t num_bytes = (bits + 7) / 8;
    const uint64_t count = 12;
    auto data = GeneratePackedBinary(count, bits, 2026);

    BinaryQuantizer quantizer1(static_cast<int>(bits), allocator.get());
    BinaryQuantizer quantizer2(64, allocator.get());
    REQUIRE(quantizer1.Train(nullptr, count));
    std::vector<uint8_t> codes(num_bytes * count, 0);
    REQUIRE(
        quantizer1.EncodeBatch(reinterpret_cast<const float*>(data.data()), codes.data(), count));
    test_serializion(quantizer1, quantizer2);

    REQUIRE(quantizer2.GetCodeSize() == quantizer1.GetCodeSize());
    REQUIRE(quantizer2.GetQueryCodeSize() == quantizer1.GetQueryCodeSize());
    REQUIRE(quantizer2.GetDim() == quantizer1.GetDim());
    REQUIRE(quantizer2.Metric() == MetricType::METRIC_TYPE_HAMMING);

    auto computer = quantizer2.FactoryComputer();
    computer->SetQuery(reinterpret_cast<const float*>(data.data()));
    float query_distance = -1.0F;
    computer->ComputeDist(codes.data(), &query_distance);
    REQUIRE(query_distance == 0.0F);

    std::vector<uint8_t> restored_codes(num_bytes * count, 0);
    REQUIRE(quantizer2.EncodeBatch(
        reinterpret_cast<const float*>(data.data()), restored_codes.data(), count));
    REQUIRE(std::memcmp(codes.data(), restored_codes.data(), codes.size()) == 0);
    for (uint64_t i = 0; i < count; ++i) {
        for (uint64_t j = 0; j < count; ++j) {
            REQUIRE(quantizer2.Compute(codes.data() + i * num_bytes,
                                       restored_codes.data() + j * num_bytes) ==
                    static_cast<float>(ReferenceHamming(codes.data() + i * num_bytes,
                                                        restored_codes.data() + j * num_bytes,
                                                        num_bytes)));
        }
    }

    BinaryQuantizer smaller_quantizer(64, allocator.get());
    test_serializion(smaller_quantizer, quantizer2);
    REQUIRE(quantizer2.GetDim() == smaller_quantizer.GetDim());
    REQUIRE(quantizer2.GetQueryCodeSize() == smaller_quantizer.GetCodeSize());
}
