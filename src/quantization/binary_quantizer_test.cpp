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

#include "quantization/binary_quantizer.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "data_type.h"
#include "impl/allocator/safe_allocator.h"
#include "inner_string_params.h"
#include "quantization/binary_quantizer_parameter.h"
#include "quantization/computer.h"
#include "quantization/quantizer_parameter.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

// Independent reference: compares the first `dim` bits one by one (MSB-first packing).
float
reference_hamming(const uint8_t* x, const uint8_t* y, uint64_t dim) {
    uint64_t distance = 0;
    for (uint64_t i = 0; i < dim; ++i) {
        const auto shift = 7 - (i % 8);
        distance += ((x[i / 8] >> shift) & 1U) != ((y[i / 8] >> shift) & 1U) ? 1 : 0;
    }
    return static_cast<float>(distance);
}

std::vector<uint8_t>
random_codes(uint64_t count, uint64_t dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<uint8_t> bytes(count * BinaryVectorBytes(dim));
    for (auto& value : bytes) {
        value = static_cast<uint8_t>(rng());
    }
    return bytes;
}

const float*
as_input(const uint8_t* bytes) {
    // The quantizer API carries packed binary input through `const float*`.
    return reinterpret_cast<const float*>(bytes);
}

}  // namespace

TEST_CASE("Binary Quantizer basic properties", "[ut][BinaryQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (int dim : {1, 7, 8, 9, 63, 64, 65, 127, 128, 1000}) {
        BinaryQuantizer quantizer(dim, allocator.get());
        REQUIRE(quantizer.GetDim() == dim);
        REQUIRE(quantizer.GetCodeSize() == BinaryVectorBytes(dim));
        REQUIRE(quantizer.GetQueryCodeSize() == BinaryVectorBytes(dim));
        REQUIRE(quantizer.Metric() == MetricType::METRIC_TYPE_HAMMING);
        REQUIRE(quantizer.Name() == QUANTIZATION_TYPE_VALUE_BINARY);
        // Training is a no-op: the packed bits are stored as they are.
        REQUIRE(quantizer.Train(nullptr, 0));
    }
}

TEST_CASE("Binary Quantizer parameter", "[ut][BinaryQuantizer]") {
    auto param = QuantizerParameter::CreateDefault(QUANTIZATION_TYPE_VALUE_BINARY);
    REQUIRE(param->GetTypeName() == QUANTIZATION_TYPE_VALUE_BINARY);
    REQUIRE(std::dynamic_pointer_cast<BinaryQuantizerParameter>(param) != nullptr);
    REQUIRE(QuantizerParameter::IsValidQuantizationType(QUANTIZATION_TYPE_VALUE_BINARY));
    auto json = param->ToJson();
    REQUIRE(json[TYPE_KEY].GetString() == QUANTIZATION_TYPE_VALUE_BINARY);
    auto restored = QuantizerParameter::GetQuantizerParameterByJson(json);
    REQUIRE(restored->GetTypeName() == QUANTIZATION_TYPE_VALUE_BINARY);
    REQUIRE(restored->ToJson().Dump() == json.Dump());
}

TEST_CASE("Binary Quantizer encode, decode and padding", "[ut][BinaryQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (int dim : {1, 7, 8, 9, 63, 64, 65, 127, 128, 1000}) {
        BinaryQuantizer quantizer(dim, allocator.get());
        const auto code_size = BinaryVectorBytes(dim);
        constexpr uint64_t count = 17;
        auto input = random_codes(count, dim, dim);

        std::vector<uint8_t> batch_codes(count * code_size);
        REQUIRE(quantizer.EncodeBatch(as_input(input.data()), batch_codes.data(), count));
        for (uint64_t i = 0; i < count; ++i) {
            const uint8_t* source = input.data() + i * code_size;
            std::vector<uint8_t> code(code_size);
            REQUIRE(quantizer.EncodeOne(as_input(source), code.data()));
            // Batch encoding reads rows ceil(dim / 8) bytes apart.
            REQUIRE(std::equal(code.begin(), code.end(), batch_codes.begin() + i * code_size));
            // Whole bytes are kept as they are; padding bits of the last byte are cleared.
            for (uint64_t b = 0; b + 1 < code_size; ++b) {
                REQUIRE(code[b] == source[b]);
            }
            const auto valid_mask =
                dim % 8 == 0 ? 0xFFU : static_cast<uint32_t>((0xFF00U >> (dim % 8)) & 0xFFU);
            REQUIRE(code[code_size - 1] == (source[code_size - 1] & valid_mask));
            REQUIRE(reference_hamming(code.data(), source, dim) == 0.0F);

            std::vector<uint8_t> decoded(code_size);
            REQUIRE(quantizer.DecodeOne(code.data(), reinterpret_cast<float*>(decoded.data())));
            REQUIRE(decoded == code);
        }
        std::vector<uint8_t> decoded_batch(count * code_size);
        REQUIRE(quantizer.DecodeBatch(
            batch_codes.data(), reinterpret_cast<float*>(decoded_batch.data()), count));
        REQUIRE(decoded_batch == batch_codes);
    }
}

TEST_CASE("Binary Quantizer distances", "[ut][BinaryQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (int dim : {1, 7, 8, 9, 63, 64, 65, 127, 128, 1000, 4096}) {
        BinaryQuantizer quantizer(dim, allocator.get());
        const auto code_size = BinaryVectorBytes(dim);
        constexpr uint64_t count = 23;
        auto input = random_codes(count, dim, 1000 + dim);
        std::vector<uint8_t> codes(count * code_size);
        quantizer.EncodeBatch(as_input(input.data()), codes.data(), count);
        // Raw, unmasked query bytes: padding garbage must not change any distance.
        auto query = random_codes(1, dim, 2000 + dim);

        auto computer = quantizer.FactoryComputer();
        quantizer.ProcessQuery(as_input(query.data()), *computer);

        std::vector<float> expected(count);
        for (uint64_t i = 0; i < count; ++i) {
            const uint8_t* code = codes.data() + i * code_size;
            expected[i] = reference_hamming(query.data(), code, dim);
            REQUIRE(quantizer.ComputeDist(*computer, code) == expected[i]);
            REQUIRE(quantizer.Compute(code, codes.data()) ==
                    reference_hamming(code, codes.data(), dim));
        }

        // Batch scans group codes by four and finish with single distances.
        for (uint64_t scan = 0; scan <= count; ++scan) {
            std::vector<float> dists(scan, -1.0F);
            quantizer.ScanBatchDists(*computer, scan, codes.data(), dists.data());
            for (uint64_t i = 0; i < scan; ++i) {
                REQUIRE(dists[i] == expected[i]);
            }
        }

        float d1 = -1.0F;
        float d2 = -1.0F;
        float d3 = -1.0F;
        float d4 = -1.0F;
        quantizer.ComputeDistsBatch4(*computer,
                                     codes.data(),
                                     codes.data() + code_size,
                                     codes.data() + 2 * code_size,
                                     codes.data() + 3 * code_size,
                                     d1,
                                     d2,
                                     d3,
                                     d4);
        REQUIRE(d1 == expected[0]);
        REQUIRE(d2 == expected[1]);
        REQUIRE(d3 == expected[2]);
        REQUIRE(d4 == expected[3]);
    }
}

TEST_CASE("Binary Quantizer reports a failed query buffer allocation", "[ut][BinaryQuantizer]") {
    class NullAllocator : public Allocator {
    public:
        std::string
        Name() override {
            return "NullAllocator";
        }

        void*
        Allocate(uint64_t size) override {
            (void)size;
            return nullptr;
        }

        void
        Deallocate(void* p) override {
            (void)p;
        }

        void*
        Reallocate(void* p, uint64_t size) override {
            (void)p;
            (void)size;
            return nullptr;
        }
    };
    // SafeAllocator turns the failed allocation into std::bad_alloc, which the quantizer reports
    // as an out-of-memory error.
    SafeAllocator allocator(std::make_shared<NullAllocator>());
    BinaryQuantizer quantizer(64, &allocator);
    auto computer = quantizer.FactoryComputer();
    const auto query = random_codes(1, 64, 5);
    REQUIRE_THROWS_AS(quantizer.ProcessQuery(as_input(query.data()), *computer), VsagException);
}

TEST_CASE("Binary Quantizer serialize and deserialize", "[ut][BinaryQuantizer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (int dim : {9, 128, 1000}) {
        BinaryQuantizer original(dim, allocator.get());
        original.Train(nullptr, 0);
        BinaryQuantizer restored(1, allocator.get());
        test_serializion(original, restored);
        REQUIRE(restored.GetDim() == dim);
        REQUIRE(restored.GetCodeSize() == original.GetCodeSize());
        REQUIRE(restored.Metric() == MetricType::METRIC_TYPE_HAMMING);

        auto input = random_codes(2, dim, 3000 + dim);
        const auto code_size = BinaryVectorBytes(dim);
        std::vector<uint8_t> codes(2 * code_size);
        restored.EncodeBatch(as_input(input.data()), codes.data(), 2);
        REQUIRE(restored.Compute(codes.data(), codes.data() + code_size) ==
                reference_hamming(input.data(), input.data() + code_size, dim));
    }
}

}  // namespace vsag
