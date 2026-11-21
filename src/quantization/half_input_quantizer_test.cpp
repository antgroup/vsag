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

#include <catch2/catch_approx.hpp>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/quantizer_adapter.h"
#include "quantization/saq_quantization/saq_quantizer.h"
#include "quantization/scalar_quantization/half_precision_quantizer.h"
#include "quantization/scalar_quantization/scalar_quantizer.h"
#include "quantization/transform_quantization/transform_quantizer.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"
#include "utils/half_input.h"

using namespace vsag;

namespace {
template <typename Format>
constexpr DataTypes InputType =
    std::is_same_v<Format, FP16Format> ? DataTypes::DATA_TYPE_FP16 : DataTypes::DATA_TYPE_BF16;

template <typename Format, MetricType metric>
void
CheckPreservedBits() {
    using Inner = HalfPrecisionQuantizer<Format, metric>;
    using Adapter = QuantizerAdapter<Inner, uint16_t>;
    IndexCommonParam common;
    common.dim_ = 8;
    common.metric_ = metric;
    common.data_type_ = InputType<Format>;
    common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    Adapter adapter(nullptr, common);
    // Include signed zero, subnormals, infinities, and NaN payloads: a float
    // round trip is not an acceptable replacement for bit-preserving storage.
    const std::vector<uint16_t> bits = {0,
                                        0x8000,
                                        1,
                                        0x8001,
                                        0x7c01,
                                        0x7f81,
                                        0x7fff,
                                        0xffff,
                                        0x3c00,
                                        0xbc00,
                                        0x0080,
                                        0x8080,
                                        0x7c00,
                                        0xfc00,
                                        2,
                                        3};
    std::vector<uint8_t> codes(bits.size() * sizeof(uint16_t));
    REQUIRE(adapter.GetCodeSize() == common.dim_ * sizeof(uint16_t));
    REQUIRE(adapter.EncodeBatchImpl(reinterpret_cast<const float*>(bits.data()), codes.data(), 2));
    REQUIRE(std::memcmp(codes.data(), bits.data(), codes.size()) == 0);
    REQUIRE(adapter.EncodeOneImpl(reinterpret_cast<const float*>(bits.data() + common.dim_),
                                  codes.data()));
    REQUIRE(std::memcmp(codes.data(), bits.data() + common.dim_, adapter.GetCodeSize()) == 0);
    Computer<Adapter> computer(&adapter, common.allocator_.get());
    adapter.ProcessQueryImpl(reinterpret_cast<const float*>(bits.data()), computer);
    REQUIRE(std::memcmp(static_cast<const Computer<Inner>*>(computer.inner_computer_.get())->buf_,
                        bits.data(),
                        adapter.GetCodeSize()) == 0);
    // Reusing the computer must overwrite the old query without reallocating semantics.
    adapter.ProcessQueryImpl(reinterpret_cast<const float*>(bits.data() + common.dim_), computer);
    REQUIRE(std::memcmp(static_cast<const Computer<Inner>*>(computer.inner_computer_.get())->buf_,
                        bits.data() + common.dim_,
                        adapter.GetCodeSize()) == 0);
}

template <typename InputFormat, typename Inner, MetricType metric>
void
CheckConvertedReference() {
    using Adapter = QuantizerAdapter<Inner, uint16_t>;
    IndexCommonParam common;
    common.dim_ = 4;
    common.metric_ = metric;
    common.data_type_ = InputType<InputFormat>;
    common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    QuantizerParamPtr param;
    if constexpr (std::is_same_v<Inner, FP32Quantizer<metric>>) {
        param = std::make_shared<FP32QuantizerParameter>();
    }
    Adapter adapter(param, common);
    Inner reference(param, common);
    const std::vector<float> original = {3, 4, 0, 0, -2, 1, 2, -3, 0, 0, 0, 0};
    std::vector<uint16_t> bits(original.size());
    std::vector<float> expanded(original.size());
    for (uint64_t i = 0; i < bits.size(); ++i) {
        bits[i] = InputFormat::FloatToHalf(original[i]);
        if constexpr (std::is_same_v<InputFormat, FP16Format>) {
            expanded[i] = half_input::FP16ToFloat(bits[i]);
        } else {
            expanded[i] = InputFormat::HalfToFloat(bits[i]);
        }
    }
    REQUIRE(adapter.TrainImpl(reinterpret_cast<const float*>(bits.data()), 3));
    REQUIRE(reference.TrainImpl(expanded.data(), 3));
    const auto size = adapter.GetCodeSize();
    std::vector<uint8_t> actual(size * 3), expected(size * 3), one(size);
    REQUIRE(adapter.EncodeBatchImpl(reinterpret_cast<const float*>(bits.data()), actual.data(), 3));
    REQUIRE(reference.EncodeBatchImpl(expanded.data(), expected.data(), 3));
    REQUIRE(actual == expected);
    Computer<Adapter> computer(&adapter, common.allocator_.get());
    Computer<Inner> reference_computer(&reference, common.allocator_.get());
    for (uint64_t i = 0; i < 3; ++i) {
        REQUIRE(
            adapter.EncodeOneImpl(reinterpret_cast<const float*>(bits.data() + i * 4), one.data()));
        REQUIRE(std::memcmp(one.data(), expected.data() + i * size, size) == 0);
        adapter.ProcessQueryImpl(reinterpret_cast<const float*>(bits.data() + i * 4), computer);
        reference.ProcessQueryImpl(expanded.data() + i * 4, reference_computer);
        REQUIRE(
            std::memcmp(static_cast<const Computer<Inner>*>(computer.inner_computer_.get())->buf_,
                        reference_computer.buf_,
                        adapter.GetQueryCodeSize()) == 0);
        float distance = 0, expected_distance = 0;
        adapter.ComputeDistImpl(computer, actual.data(), &distance);
        reference.ComputeDistImpl(reference_computer, expected.data(), &expected_distance);
        REQUIRE(distance == Catch::Approx(expected_distance));
    }
}

template <typename InputFormat, typename Inner, MetricType metric>
void
CheckBatch4Reference() {
    using Adapter = QuantizerAdapter<Inner, uint16_t>;
    // Require the adapter's fused entry point, rather than silently accepting
    // Quantizer's inherited four-single-distance fallback.
    static_assert(std::is_member_function_pointer_v<decltype(&Adapter::ComputeDistsBatch4Impl)>);
    for (const uint64_t dim : {1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129}) {
        CAPTURE(dim);
        IndexCommonParam common;
        common.dim_ = dim;
        common.metric_ = metric;
        common.data_type_ = InputType<InputFormat>;
        common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
        QuantizerParamPtr param;
        if constexpr (std::is_same_v<Inner, FP32Quantizer<metric>>) {
            param = std::make_shared<FP32QuantizerParameter>();
        }
        Adapter adapter(param, common);
        auto reference_common = common;
        reference_common.data_type_ = DataTypes::DATA_TYPE_FLOAT;
        Inner reference(param, reference_common);
        // Four distinct codes and two queries, with non-integral values that
        // expose differences in fused versus single-distance accumulation.
        std::vector<uint16_t> bits(6 * dim);
        std::vector<float> expanded(bits.size());
        for (uint64_t i = 0; i < bits.size(); ++i) {
            const float value = static_cast<float>(static_cast<int>((i * 37 + 11) % 251) - 125) /
                                static_cast<float>(3 + (i % 7));
            bits[i] = InputFormat::FloatToHalf(value);
            if constexpr (std::is_same_v<InputFormat, FP16Format>) {
                expanded[i] = half_input::FP16ToFloat(bits[i]);
            } else {
                expanded[i] = InputFormat::HalfToFloat(bits[i]);
            }
        }
        REQUIRE(adapter.TrainImpl(reinterpret_cast<const float*>(bits.data()), 6));
        REQUIRE(reference.TrainImpl(expanded.data(), 6));
        const auto code_size = adapter.GetCodeSize();
        REQUIRE(code_size == reference.GetCodeSize());
        std::vector<uint8_t> codes(4 * code_size), reference_codes(4 * code_size);
        REQUIRE(
            adapter.EncodeBatchImpl(reinterpret_cast<const float*>(bits.data()), codes.data(), 4));
        REQUIRE(reference.EncodeBatchImpl(expanded.data(), reference_codes.data(), 4));
        REQUIRE(codes == reference_codes);
        Computer<Adapter> computer(&adapter, common.allocator_.get());
        Computer<Inner> reference_computer(&reference, common.allocator_.get());
        for (uint64_t query = 4; query < 6; ++query) {
            CAPTURE(query);
            adapter.ProcessQueryImpl(reinterpret_cast<const float*>(bits.data() + query * dim),
                                     computer);
            reference.ProcessQueryImpl(expanded.data() + query * dim, reference_computer);
            REQUIRE(adapter.GetQueryCodeSize() == reference.GetQueryCodeSize());
            REQUIRE(std::memcmp(
                        static_cast<const Computer<Inner>*>(computer.inner_computer_.get())->buf_,
                        reference_computer.buf_,
                        adapter.GetQueryCodeSize()) == 0);
            // Reverse the code order on reuse to detect stale queries or lane mixups.
            const uint8_t* lanes[4];
            for (uint64_t lane = 0; lane < 4; ++lane) {
                lanes[lane] = codes.data() + (query == 4 ? lane : 3 - lane) * code_size;
            }
            float actual[4] = {}, expected[4] = {};
            adapter.ComputeDistsBatch4(computer,
                                       lanes[0],
                                       lanes[1],
                                       lanes[2],
                                       lanes[3],
                                       actual[0],
                                       actual[1],
                                       actual[2],
                                       actual[3]);
            reference.ComputeDistsBatch4(reference_computer,
                                         lanes[0],
                                         lanes[1],
                                         lanes[2],
                                         lanes[3],
                                         expected[0],
                                         expected[1],
                                         expected[2],
                                         expected[3]);
            for (uint64_t lane = 0; lane < 4; ++lane) {
                CAPTURE(lane);
                // Approx would hide the accumulation-order regression that
                // forwarding the inner public batch4 entry point fixes.
                REQUIRE(actual[lane] == expected[lane]);
            }
        }
    }
}

template <typename Format, MetricType metric>
void
CheckCommonQuantizers() {
    CheckConvertedReference<Format, FP32Quantizer<metric>, metric>();
    CheckConvertedReference<Format, SQ8Quantizer<metric>, metric>();
}
}  // namespace

TEST_CASE("Matched half L2 and IP preserve input bits", "[ut][HalfInputQuantizer]") {
    CheckPreservedBits<FP16Format, MetricType::METRIC_TYPE_L2SQR>();
    CheckPreservedBits<FP16Format, MetricType::METRIC_TYPE_IP>();
    CheckPreservedBits<BF16Format, MetricType::METRIC_TYPE_L2SQR>();
    CheckPreservedBits<BF16Format, MetricType::METRIC_TYPE_IP>();
}

TEST_CASE("Matched half cosine retains float normalization", "[ut][HalfInputQuantizer]") {
    CheckConvertedReference<FP16Format,
                            FP16Quantizer<MetricType::METRIC_TYPE_COSINE>,
                            MetricType::METRIC_TYPE_COSINE>();
    CheckConvertedReference<BF16Format,
                            BF16Quantizer<MetricType::METRIC_TYPE_COSINE>,
                            MetricType::METRIC_TYPE_COSINE>();
}

TEST_CASE("Cross format half inputs retain conversion", "[ut][HalfInputQuantizer]") {
    CheckConvertedReference<FP16Format,
                            BF16Quantizer<MetricType::METRIC_TYPE_L2SQR>,
                            MetricType::METRIC_TYPE_L2SQR>();
    CheckConvertedReference<BF16Format,
                            FP16Quantizer<MetricType::METRIC_TYPE_IP>,
                            MetricType::METRIC_TYPE_IP>();
    CheckConvertedReference<FP16Format,
                            BF16Quantizer<MetricType::METRIC_TYPE_COSINE>,
                            MetricType::METRIC_TYPE_COSINE>();
    CheckConvertedReference<BF16Format,
                            FP16Quantizer<MetricType::METRIC_TYPE_COSINE>,
                            MetricType::METRIC_TYPE_COSINE>();
}

TEST_CASE("Half inputs support FP32 and SQ8 conversion", "[ut][HalfInputQuantizer]") {
    CheckCommonQuantizers<FP16Format, MetricType::METRIC_TYPE_L2SQR>();
    CheckCommonQuantizers<FP16Format, MetricType::METRIC_TYPE_IP>();
    CheckCommonQuantizers<FP16Format, MetricType::METRIC_TYPE_COSINE>();
    CheckCommonQuantizers<BF16Format, MetricType::METRIC_TYPE_L2SQR>();
    CheckCommonQuantizers<BF16Format, MetricType::METRIC_TYPE_IP>();
    CheckCommonQuantizers<BF16Format, MetricType::METRIC_TYPE_COSINE>();
}

TEST_CASE("Half input FP32 batch4 retains fused distances", "[ut][HalfInputQuantizer]") {
    CheckBatch4Reference<FP16Format,
                         FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>,
                         MetricType::METRIC_TYPE_L2SQR>();
    CheckBatch4Reference<FP16Format,
                         FP32Quantizer<MetricType::METRIC_TYPE_IP>,
                         MetricType::METRIC_TYPE_IP>();
    CheckBatch4Reference<FP16Format,
                         FP32Quantizer<MetricType::METRIC_TYPE_COSINE>,
                         MetricType::METRIC_TYPE_COSINE>();
    CheckBatch4Reference<BF16Format,
                         FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>,
                         MetricType::METRIC_TYPE_L2SQR>();
    CheckBatch4Reference<BF16Format,
                         FP32Quantizer<MetricType::METRIC_TYPE_IP>,
                         MetricType::METRIC_TYPE_IP>();
    CheckBatch4Reference<BF16Format,
                         FP32Quantizer<MetricType::METRIC_TYPE_COSINE>,
                         MetricType::METRIC_TYPE_COSINE>();
}

TEST_CASE("Half input SQ8 batch4 retains fallback distances", "[ut][HalfInputQuantizer]") {
    CheckBatch4Reference<FP16Format,
                         SQ8Quantizer<MetricType::METRIC_TYPE_L2SQR>,
                         MetricType::METRIC_TYPE_L2SQR>();
    CheckBatch4Reference<BF16Format,
                         SQ8Quantizer<MetricType::METRIC_TYPE_IP>,
                         MetricType::METRIC_TYPE_IP>();
}

TEST_CASE("FP16 input expansion follows IEEE values", "[ut][HalfInputQuantizer]") {
    const std::vector<uint16_t> bits = {
        0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x8400, 0x3c00};
    const std::vector<uint32_t> expected = {0x00000000,
                                            0x80000000,
                                            0x33800000,
                                            0xb3800000,
                                            0x387fc000,
                                            0x38800000,
                                            0xb8800000,
                                            0x3f800000};
    for (size_t i = 0; i < bits.size(); ++i) {
        const float value = half_input::FP16ToFloat(bits[i]);
        uint32_t actual;
        std::memcpy(&actual, &value, sizeof(actual));
        REQUIRE(actual == expected[i]);
    }
    using Inner = FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>;
    using Adapter = QuantizerAdapter<Inner, uint16_t>;
    IndexCommonParam common;
    common.dim_ = bits.size();
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common.data_type_ = DataTypes::DATA_TYPE_FP16;
    common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    Adapter adapter(std::make_shared<FP32QuantizerParameter>(), common);
    REQUIRE(adapter.TrainImpl(reinterpret_cast<const float*>(bits.data()), 1));
    std::vector<uint8_t> codes(adapter.GetCodeSize());
    REQUIRE(adapter.EncodeOneImpl(reinterpret_cast<const float*>(bits.data()), codes.data()));
    REQUIRE(std::memcmp(codes.data(), expected.data(), codes.size()) == 0);
    REQUIRE(adapter.EncodeBatchImpl(reinterpret_cast<const float*>(bits.data()), codes.data(), 1));
    REQUIRE(std::memcmp(codes.data(), expected.data(), codes.size()) == 0);
    Computer<Adapter> computer(&adapter, common.allocator_.get());
    adapter.ProcessQueryImpl(reinterpret_cast<const float*>(bits.data()), computer);
    REQUIRE(std::memcmp(static_cast<const Computer<Inner>*>(computer.inner_computer_.get())->buf_,
                        expected.data(),
                        codes.size()) == 0);
}

TEST_CASE("FP16 cosine input uses IEEE direction", "[ut][HalfInputQuantizer]") {
    const uint16_t input[] = {0x0000, 0x0001};
    const float reference_values[] = {0.0F, 0x1p-24F};
    IndexCommonParam common;
    common.dim_ = 2;
    common.metric_ = MetricType::METRIC_TYPE_COSINE;
    common.data_type_ = DataTypes::DATA_TYPE_FP16;
    common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    SECTION("FP32 storage") {
        using Inner = FP32Quantizer<MetricType::METRIC_TYPE_COSINE>;
        using Adapter = QuantizerAdapter<Inner, uint16_t>;
        auto param = std::make_shared<FP32QuantizerParameter>();
        Adapter adapter(param, common);
        Inner reference(param, common);
        std::vector<uint8_t> actual(adapter.GetCodeSize()), expected(reference.GetCodeSize());
        REQUIRE(adapter.EncodeOneImpl(reinterpret_cast<const float*>(input), actual.data()));
        REQUIRE(reference.EncodeOneImpl(reference_values, expected.data()));
        REQUIRE(actual == expected);
        Computer<Adapter> computer(&adapter, common.allocator_.get());
        Computer<Inner> ref_computer(&reference, common.allocator_.get());
        adapter.ProcessQueryImpl(reinterpret_cast<const float*>(input), computer);
        reference.ProcessQueryImpl(reference_values, ref_computer);
        REQUIRE(
            std::memcmp(static_cast<const Computer<Inner>*>(computer.inner_computer_.get())->buf_,
                        ref_computer.buf_,
                        adapter.GetQueryCodeSize()) == 0);
    }
    SECTION("matched FP16 storage") {
        using Inner = FP16Quantizer<MetricType::METRIC_TYPE_COSINE>;
        using Adapter = QuantizerAdapter<Inner, uint16_t>;
        Adapter adapter(nullptr, common);
        Inner reference(nullptr, common);
        std::vector<uint8_t> actual(adapter.GetCodeSize()), expected(reference.GetCodeSize());
        REQUIRE(adapter.EncodeOneImpl(reinterpret_cast<const float*>(input), actual.data()));
        REQUIRE(reference.EncodeOneImpl(reference_values, expected.data()));
        REQUIRE(actual == expected);
        Computer<Adapter> computer(&adapter, common.allocator_.get());
        adapter.ProcessQueryImpl(reinterpret_cast<const float*>(input), computer);
        REQUIRE(
            std::memcmp(static_cast<const Computer<Inner>*>(computer.inner_computer_.get())->buf_,
                        expected.data(),
                        expected.size()) == 0);
    }
}

TEST_CASE("Half TQ Train is idempotent", "[ut][HalfInputQuantizer][quantizer_matrix]") {
    const bool bf16 = GENERATE(false, true);
    using Inner = TransformQuantizer<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>,
                                     MetricType::METRIC_TYPE_L2SQR>;
    using Adapter = QuantizerAdapter<Inner, uint16_t>;
    IndexCommonParam common;
    common.dim_ = 8;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common.data_type_ = bf16 ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16;
    common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    auto param = std::make_shared<TransformQuantizerParameter>();
    param->FromJson(JsonType::Parse(R"({"tq_chain":"pca, fp32","pca_dim":4,"mrle_dim":4})"));
    Adapter adapter(param, common);
    std::vector<uint16_t> first(64 * common.dim_), second(first.size());
    for (size_t i = 0; i < first.size(); ++i) {
        const float a = static_cast<float>(static_cast<int>((i * 37) % 127) - 63) / 64.0F;
        const float b = static_cast<float>(static_cast<int>((i * 17) % 101) - 50) / 16.0F;
        first[i] = bf16 ? BF16Format::FloatToHalf(a) : FP16Format::FloatToHalf(a);
        second[i] = bf16 ? BF16Format::FloatToHalf(b) : FP16Format::FloatToHalf(b);
    }
    REQUIRE(adapter.Train(reinterpret_cast<const float*>(first.data()), 64));
    std::vector<uint8_t> before(adapter.GetCodeSize()), after(before.size());
    REQUIRE(adapter.EncodeOne(reinterpret_cast<const float*>(first.data()), before.data()));
    REQUIRE(adapter.Train(reinterpret_cast<const float*>(second.data()), 64));
    REQUIRE(adapter.EncodeOne(reinterpret_cast<const float*>(first.data()), after.data()));
    REQUIRE(before == after);
    Adapter restored(param, common);
    test_serializion(adapter, restored);
    REQUIRE(restored.Train(reinterpret_cast<const float*>(second.data()), 64));
    REQUIRE(restored.EncodeOne(reinterpret_cast<const float*>(first.data()), after.data()));
    REQUIRE(before == after);
}

namespace {
class AdapterReviewAllocator : public DefaultAllocator {
public:
    void*
    Allocate(uint64_t size) override {
        auto* ptr = DefaultAllocator::Allocate(size);
        live.insert(ptr);
        return ptr;
    }
    void
    Deallocate(void* ptr) override {
        if (ptr == nullptr) {
            return;
        }
        if (live.erase(ptr) != 1) {
            invalid_free = true;
            return;
        }
        DefaultAllocator::Deallocate(ptr);
    }
    void*
    Reallocate(void* ptr, uint64_t size) override {
        if (ptr == nullptr) {
            return Allocate(size);
        }
        auto* result = DefaultAllocator::Reallocate(ptr, size);
        live.erase(ptr);
        live.insert(result);
        return result;
    }
    bool invalid_free{false};
    std::unordered_set<void*> live;
};

template <typename Format, typename Inner>
void
CheckAdapterReviewOwnership(const QuantizerParamPtr& param) {
    using Adapter = QuantizerAdapter<Inner, uint16_t>;
    auto allocator = std::make_shared<AdapterReviewAllocator>();
    {
        IndexCommonParam common;
        common.dim_ = 8;
        common.metric_ = MetricType::METRIC_TYPE_L2SQR;
        common.data_type_ = InputType<Format>;
        common.allocator_ = allocator;
        Adapter adapter(param, common);
        auto reference_common = common;
        reference_common.data_type_ = DataTypes::DATA_TYPE_FLOAT;
        Inner reference(param, reference_common);
        std::vector<uint16_t> bits(64 * common.dim_);
        std::vector<float> values(bits.size());
        for (uint64_t i = 0; i < bits.size(); ++i) {
            const float value = static_cast<float>(static_cast<int>((i * 37) % 127) - 63) / 16;
            bits[i] = Format::FloatToHalf(value);
            if constexpr (std::is_same_v<Format, FP16Format>) {
                values[i] = half_input::FP16ToFloat(bits[i]);
            } else {
                values[i] = Format::HalfToFloat(bits[i]);
            }
        }
        REQUIRE(adapter.Train(reinterpret_cast<const float*>(bits.data()), 64));
        REQUIRE(reference.Train(values.data(), 64));
        std::vector<uint8_t> code(adapter.GetCodeSize()), expected(reference.GetCodeSize());
        REQUIRE(adapter.EncodeOne(reinterpret_cast<const float*>(bits.data()), code.data()));
        REQUIRE(reference.EncodeOne(values.data(), expected.data()));
        REQUIRE(code == expected);
        {
            // Both quantizers and their allocator outlive both computers.
            Computer<Adapter> computer(&adapter, allocator.get());
            Computer<Inner> ref_computer(&reference, allocator.get());
            auto* outer_buffer = static_cast<uint8_t*>(allocator->Allocate(37));
            std::memset(outer_buffer, 0x5a, 37);
            computer.buf_ = outer_buffer;
            REQUIRE(computer.inner_computer_ == nullptr);
            ComputerInterface* first_inner = nullptr;
            for (const uint64_t row : {uint64_t{1}, uint64_t{2}, uint64_t{1}}) {
                computer.SetQuery(reinterpret_cast<const float*>(bits.data() + row * common.dim_));
                ref_computer.SetQuery(values.data() + row * common.dim_);
                REQUIRE(computer.inner_computer_ != nullptr);
                auto* nested = dynamic_cast<Computer<Inner>*>(computer.inner_computer_.get());
                REQUIRE(nested != nullptr);
                REQUIRE(nested->allocator_ == allocator.get());
                if (first_inner == nullptr) {
                    first_inner = nested;
                }
                REQUIRE(computer.inner_computer_.get() == first_inner);
                REQUIRE(computer.buf_ == outer_buffer);
                for (uint64_t i = 0; i < 37; ++i) {
                    REQUIRE(outer_buffer[i] == 0x5a);
                }
                float actual = 0, distance = 0;
                computer.ComputeDist(code.data(), &actual);
                ref_computer.ComputeDist(expected.data(), &distance);
                REQUIRE(actual == Catch::Approx(distance).margin(1e-5));
            }
        }
        REQUIRE_FALSE(allocator->invalid_free);
    }
    // Includes nested query buffers, conversion scratch, transform inner
    // computers, and the outer allocator-owned buffer; no raw malloc counts.
    REQUIRE(allocator->live.empty());
    REQUIRE_FALSE(allocator->invalid_free);
}
}  // namespace

TEST_CASE("Half adapter owns nested computers separately from allocator query buffers",
          "[ut][HalfInputQuantizer][maintenance_review][adapter_ownership]") {
    constexpr auto metric = MetricType::METRIC_TYPE_L2SQR;
    CheckAdapterReviewOwnership<FP16Format, FP16Quantizer<metric>>(nullptr);
    CheckAdapterReviewOwnership<BF16Format, BF16Quantizer<metric>>(nullptr);
    CheckAdapterReviewOwnership<FP16Format, BF16Quantizer<metric>>(nullptr);
    CheckAdapterReviewOwnership<BF16Format, FP16Quantizer<metric>>(nullptr);
    auto fp32 = std::make_shared<FP32QuantizerParameter>();
    CheckAdapterReviewOwnership<FP16Format, FP32Quantizer<metric>>(fp32);
    CheckAdapterReviewOwnership<BF16Format, FP32Quantizer<metric>>(fp32);
    CheckAdapterReviewOwnership<FP16Format, SQ8Quantizer<metric>>(nullptr);
    CheckAdapterReviewOwnership<BF16Format, SQ8Quantizer<metric>>(nullptr);
    auto tq = std::make_shared<TransformQuantizerParameter>();
    tq->FromJson(JsonType::Parse(R"({"tq_chain":"pca, fp32","pca_dim":4,"mrle_dim":4})"));
    using Transform = TransformQuantizer<FP32Quantizer<metric>, metric>;
    CheckAdapterReviewOwnership<FP16Format, Transform>(tq);
    CheckAdapterReviewOwnership<BF16Format, Transform>(tq);
}

TEST_CASE("Half SAQ adapters preserve native training and query ownership",
          "[ut][HalfInputQuantizer][saq_half_rebase]") {
    constexpr auto metric = MetricType::METRIC_TYPE_L2SQR;
    auto saq = std::make_shared<SAQQuantizerParameter>();
    CheckAdapterReviewOwnership<FP16Format, SAQQuantizer<metric>>(saq);
    CheckAdapterReviewOwnership<BF16Format, SAQQuantizer<metric>>(saq);
    auto tq = std::make_shared<TransformQuantizerParameter>();
    tq->FromJson(JsonType::Parse(R"({"tq_chain":"pca, saq","pca_dim":4,"mrle_dim":4})"));
    using Transform = TransformQuantizer<SAQQuantizer<metric>, metric>;
    CheckAdapterReviewOwnership<FP16Format, Transform>(tq);
    CheckAdapterReviewOwnership<BF16Format, Transform>(tq);
}
