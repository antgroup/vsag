
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
#include <catch2/catch_template_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_test.h"
#include "simd/normalize.h"
#include "simd/simd.h"
#include "simd/sq4_local_simd.h"
#include "simd/sq8_local_simd.h"
#include "sq8_local_quantizer.h"
#include "unittest.h"
#include "vsag/vsag.h"

using namespace vsag;

// Query the compiled library: quantizer_test does not inherit SIMD compile definitions.
// Inline SimdStatus::Support* checks here could silently skip native ISA coverage.
static SimdStatus
LocalSQTestSIMDStatus() {
    SimdStatus::Init();
    return setup_simd();
}

TEMPLATE_TEST_CASE("Local SQ decoded reference and SIMD",
                   "[ut][LocalSQ][simd]",
                   SQ8LocalQuantizer,
                   SQ4LocalQuantizer) {
    const auto simd_status = LocalSQTestSIMDStatus();
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    REQUIRE_THROWS(TestType(0, allocator.get()));
    REQUIRE_THROWS(TestType(-1, allocator.get()));
    IndexCommonParam unsupported;
    unsupported.dim_ = 17;
    unsupported.allocator_ = allocator;
    unsupported.metric_ = static_cast<MetricType>(99);
    REQUIRE_THROWS(TestType(QuantizerParameter::CreateDefault(TestType::NameImpl()), unsupported));
    for (int dim : {1, 3, 7, 16, 31, 64, 65, 960}) {
        TestType quant(dim, allocator.get());
        REQUIRE(quant.GetCodeSize() == (dim * TestType::BIT_PER_DIM + 7) / 8 + 8);
        REQUIRE(quant.GetQueryCodeSize() == dim * sizeof(float));
        REQUIRE(quant.Train(nullptr, 0));
        std::vector<float> a(dim), b(dim), query_storage(dim + 1), da(dim), db(dim);
        auto* query = query_storage.data() + 1;  // Float-aligned, deliberately not SIMD-aligned.
        // Deliberately unaligned starts and odd strides exercise metadata copies.
        std::vector<uint8_t> storage(2 * quant.GetCodeSize() + 1);
        auto* ca = storage.data() + 1;
        auto* cb = ca + quant.GetCodeSize();
        for (int pattern = 0; pattern < 4; ++pattern) {
            for (int i = 0; i < dim; ++i) {
                a[i] = pattern == 0   ? 0.0F
                       : pattern == 1 ? -3.5F
                                      : std::sin(i * 1.7F) * 2.0F - 5.0F;
                b[i] = pattern < 2 ? 8.0F : std::cos(i * 0.9F) * 10.0F + 20.0F;
                query[i] = std::sin(i * 0.7F) * 5.0F;
            }
            REQUIRE(quant.EncodeOne(a.data(), ca));
            REQUIRE(quant.EncodeOne(b.data(), cb));
            REQUIRE(quant.DecodeOne(ca, da.data()));
            REQUIRE(quant.DecodeOne(cb, db.data()));
            double code_ref = 0, query_ref = 0;
            for (int i = 0; i < dim; ++i) {
                REQUIRE(
                    std::abs(da[i] - a[i]) <=
                    (pattern < 2 ? 0.0F : 4.0F / (2 * ((1 << TestType::BIT_PER_DIM) - 1)) + 1e-5F));
                REQUIRE(std::abs(db[i] - b[i]) <=
                        (pattern < 2 ? 0.0F
                                     : 20.0F / (2 * ((1 << TestType::BIT_PER_DIM) - 1)) + 1e-5F));
                code_ref += std::pow(static_cast<double>(da[i]) - db[i], 2);
                query_ref += std::pow(static_cast<double>(query[i]) - db[i], 2);
            }
            std::vector<SQ8LocalComputeType> adc{
                (TestType::BIT_PER_DIM == 8 ? generic::SQ8LocalComputeL2Sqr
                                            : generic::SQ4LocalComputeL2Sqr),
                (TestType::BIT_PER_DIM == 8 ? SQ8LocalComputeL2Sqr : SQ4LocalComputeL2Sqr)};
            std::vector<SQ8LocalComputeCodesType> sdc{
                (TestType::BIT_PER_DIM == 8 ? generic::SQ8LocalComputeCodesL2Sqr
                                            : generic::SQ4LocalComputeCodesL2Sqr),
                (TestType::BIT_PER_DIM == 8 ? SQ8LocalComputeCodesL2Sqr
                                            : SQ4LocalComputeCodesL2Sqr)};
#define CHECK_ISA(isa, check)                                                                      \
    if ((simd_status.dist_support_##check && simd_status.runtime_has_##check)) {                   \
        adc.push_back(                                                                             \
            (TestType::BIT_PER_DIM == 8 ? isa::SQ8LocalComputeL2Sqr : isa::SQ4LocalComputeL2Sqr)); \
        sdc.push_back((TestType::BIT_PER_DIM == 8 ? isa::SQ8LocalComputeCodesL2Sqr                 \
                                                  : isa::SQ4LocalComputeCodesL2Sqr));              \
    }
            CHECK_ISA(sse, sse)
            CHECK_ISA(avx, avx)
            CHECK_ISA(avx2, avx2)
            CHECK_ISA(avx512, avx512f)
            CHECK_ISA(neon, neon)
            CHECK_ISA(sve, sve)
#undef CHECK_ISA
            for (auto fn : adc) {
                REQUIRE(fn(query, cb, dim) == Catch::Approx(query_ref).epsilon(2e-5));
                // ADC must reconstruct exactly the values exposed by DecodeOne, even on
                // backends without FMA and when minimum/scale contributions cancel.
                REQUIRE(fn(da.data(), ca, dim) == 0.0F);
            }
            for (auto fn : sdc) {
                REQUIRE(fn(ca, cb, dim) == Catch::Approx(code_ref).epsilon(2e-5));
                REQUIRE(fn(cb, ca, dim) == Catch::Approx(code_ref).epsilon(2e-5));
                REQUIRE(fn(ca, ca, dim) == 0.0F);
            }
            auto computer = quant.FactoryComputer();
            computer->SetQuery(query);
            float distance;
            computer->ComputeDist(cb, &distance);
            REQUIRE(distance == Catch::Approx(query_ref).epsilon(2e-5));
            REQUIRE(quant.Compute(ca, cb) == Catch::Approx(code_ref).epsilon(2e-5));
            TestType restored(1, allocator.get());
            test_serializion(quant, restored);
            REQUIRE(restored.GetDim() == dim);
            REQUIRE(restored.GetQueryCodeSize() == quant.GetQueryCodeSize());
            REQUIRE(restored.Name() == TestType::NameImpl());
            REQUIRE(restored.Compute(ca, cb) == Catch::Approx(code_ref).epsilon(2e-5));
            auto restored_computer = restored.FactoryComputer();
            restored_computer->SetQuery(query);
            restored_computer->ComputeDist(cb, &distance);
            REQUIRE(distance == Catch::Approx(query_ref).epsilon(2e-5));
        }
        a[0] = std::numeric_limits<float>::infinity();
        REQUIRE_THROWS(quant.EncodeOne(a.data(), ca));
    }
}

TEMPLATE_TEST_CASE("Local SQ HGraph and metric validation",
                   "[ut][LocalSQ]",
                   SQ8LocalQuantizer,
                   SQ4LocalQuantizer) {
    std::string config = R"({"dtype":"float32","metric_type":"l2","dim":17,
        "index_param":{"base_quantization_type":"sq8_local","use_reorder":false,
        "max_degree":16,"ef_construction":100,"build_thread_count":1}})";
    config.replace(config.find("sq8_local"), 9, TestType::NameImpl());
    auto result = Factory::CreateIndex("hgraph", config);
    REQUIRE(result.has_value());
    auto index = result.value();
    auto vectors = fixtures::generate_vectors(100, 17);
    std::vector<int64_t> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    auto base = Dataset::Make()
                    ->NumElements(100)
                    ->Dim(17)
                    ->Float32Vectors(vectors.data())
                    ->Ids(ids.data())
                    ->Owner(false);
    REQUIRE(index->Build(base).has_value());
    auto query =
        Dataset::Make()->NumElements(1)->Dim(17)->Float32Vectors(vectors.data())->Owner(false);
    const std::string search = R"({"hgraph":{"ef_search":100}})";
    auto before = index->KnnSearch(query, 10, search);
    REQUIRE(before.has_value());
    REQUIRE(before.value()->GetIds()[0] == 0);
    auto serialized = index->Serialize();
    REQUIRE(serialized.has_value());
    auto restored = Factory::CreateIndex("hgraph", config).value();
    REQUIRE(restored->Deserialize(serialized.value()).has_value());
    auto after = restored->KnnSearch(query, 10, search);
    REQUIRE(after.has_value());
    for (int i = 0; i < 10; ++i) {
        REQUIRE(before.value()->GetIds()[i] == after.value()->GetIds()[i]);
        REQUIRE(before.value()->GetDistances()[i] == after.value()->GetDistances()[i]);
    }
    for (auto metric : {"ip", "cosine"}) {
        auto unsupported = config;
        unsupported.replace(unsupported.find("l2"), 2, metric);
        REQUIRE(Factory::CreateIndex("hgraph", unsupported).has_value());
    }
    auto param = QuantizerParameter::CreateDefault(TestType::NameImpl());
    REQUIRE(QuantizerParameter::IsValidQuantizationType(param->GetTypeName()));
    REQUIRE(QuantizerParameter::GetQuantizerParameterByJson(param->ToJson())->GetTypeName() ==
            TestType::NameImpl());
}

TEMPLATE_TEST_CASE("Local SQ finite boundaries and training",
                   "[ut][LocalSQ][simd]",
                   SQ8LocalQuantizer,
                   SQ4LocalQuantizer) {
    const auto simd_status = LocalSQTestSIMDStatus();
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    TestType quant(17, allocator.get());
    std::vector<float> input(17, -7.0F), decoded(17);
    std::vector<uint8_t> codes(quant.GetCodeSize()), before(codes.size());
    // No shared calibration: encoding works before training and retraining is a no-op.
    REQUIRE(quant.EncodeOne(input.data(), codes.data()));
    before = codes;
    REQUIRE(quant.Train(nullptr, 0));
    REQUIRE(quant.ReTrain(input.data(), 1));
    REQUIRE(quant.EncodeOne(input.data(), codes.data()));
    REQUIRE(codes == before);
    auto computer = quant.FactoryComputer();
    for (uint32_t bits : {0x7f800000U, 0xff800000U, 0x7fc00000U, 0x7f800001U}) {
        memcpy(input.data() + 16, &bits, sizeof(bits));
        REQUIRE_THROWS(quant.EncodeOne(input.data(), codes.data()));
        REQUIRE(codes == before);
        REQUIRE_THROWS(computer->SetQuery(input.data()));
    }
    const float maximum = std::numeric_limits<float>::max();
    for (auto extreme : {maximum, -maximum}) {
        std::fill(input.begin(), input.end(), extreme);
        input[0] = -maximum;
        input[1] = maximum;
        REQUIRE(quant.EncodeOne(input.data(), codes.data()));
        REQUIRE(quant.DecodeOne(codes.data(), decoded.data()));
        for (auto value : decoded) {
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            REQUIRE((bits & 0x7f800000U) != 0x7f800000U);
        }
        REQUIRE((TestType::BIT_PER_DIM == 8
                     ? generic::SQ8LocalComputeCodesL2Sqr
                     : generic::SQ4LocalComputeCodesL2Sqr)(codes.data(), codes.data(), 17) == 0);
        REQUIRE((TestType::BIT_PER_DIM == 8
                     ? SQ8LocalComputeCodesL2Sqr
                     : SQ4LocalComputeCodesL2Sqr)(codes.data(), codes.data(), 17) == 0);
        REQUIRE((TestType::BIT_PER_DIM == 8 ? SQ8LocalComputeL2Sqr : SQ4LocalComputeL2Sqr)(
                    decoded.data(), codes.data(), 17) == 0);
#define CHECK_EXTREME_ISA(isa, check)                                                         \
    if ((simd_status.dist_support_##check && simd_status.runtime_has_##check)) {              \
        REQUIRE((TestType::BIT_PER_DIM == 8                                                   \
                     ? isa::SQ8LocalComputeCodesL2Sqr                                         \
                     : isa::SQ4LocalComputeCodesL2Sqr)(codes.data(), codes.data(), 17) == 0); \
        REQUIRE((TestType::BIT_PER_DIM == 8                                                   \
                     ? isa::SQ8LocalComputeL2Sqr                                              \
                     : isa::SQ4LocalComputeL2Sqr)(decoded.data(), codes.data(), 17) == 0);    \
    }
        CHECK_EXTREME_ISA(sse, sse)
        CHECK_EXTREME_ISA(avx, avx)
        CHECK_EXTREME_ISA(avx2, avx2)
        CHECK_EXTREME_ISA(avx512, avx512f)
        CHECK_EXTREME_ISA(neon, neon)
        CHECK_EXTREME_ISA(sve, sve)
#undef CHECK_EXTREME_ISA
        std::fill(input.begin(), input.end(), 0);
        float distance = (TestType::BIT_PER_DIM == 8 ? SQ8LocalComputeL2Sqr : SQ4LocalComputeL2Sqr)(
            input.data(), codes.data(), 17);
        uint32_t bits;
        memcpy(&bits, &distance, sizeof(bits));
        REQUIRE(bits == 0x7f800000U);  // Overflow is positive infinity, never NaN.
    }
    std::fill(input.begin(), input.end(), std::numeric_limits<float>::denorm_min());
    input[0] = -std::numeric_limits<float>::denorm_min();
    REQUIRE(quant.EncodeOne(input.data(), codes.data()));
    REQUIRE(quant.DecodeOne(codes.data(), decoded.data()));
    REQUIRE(quant.Compute(codes.data(), codes.data()) == 0);
    IndexCommonParam common;
    common.allocator_ = allocator;
    auto param = QuantizerParameter::CreateDefault(TestType::NameImpl());
    for (auto dim : {int64_t{0}, int64_t{-1}, int64_t{1} << 32}) {
        common.dim_ = dim;
        REQUIRE_THROWS(TestType(param, common));
    }
    common.dim_ = 17;
    for (auto type : {DataTypes::DATA_TYPE_INT8,
                      DataTypes::DATA_TYPE_FP16,
                      DataTypes::DATA_TYPE_BF16,
                      DataTypes::DATA_TYPE_SPARSE}) {
        common.data_type_ = type;
        REQUIRE_THROWS(TestType(param, common));
    }
}

TEMPLATE_TEST_CASE("Local SQ HGraph rerank and Tune persistence",
                   "[ut][LocalSQ]",
                   SQ8LocalQuantizer,
                   SQ4LocalQuantizer) {
    const auto metric = GENERATE(
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_COSINE);
    const std::string metric_name = metric == MetricType::METRIC_TYPE_L2SQR ? "l2"
                                    : metric == MetricType::METRIC_TYPE_IP  ? "ip"
                                                                            : "cosine";
    CAPTURE(metric_name, TestType::NameImpl());
    constexpr int64_t dim = 17;
    constexpr int64_t count = 100;
    auto vectors = fixtures::generate_vectors(count, dim);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    auto base = Dataset::Make()
                    ->NumElements(count)
                    ->Dim(dim)
                    ->Float32Vectors(vectors.data())
                    ->Ids(ids.data())
                    ->Owner(false);
    auto query =
        Dataset::Make()->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);
    const std::string search = R"({"hgraph":{"ef_search":100}})";
    auto config = [&](const std::string& base_type, const std::string& precise_type) {
        auto json = JsonType::Parse(R"({"dtype":"float32","metric_type":"l2","dim":17,
            "index_param":{"max_degree":16,"ef_construction":100,"build_thread_count":1}})");
        json["metric_type"].SetString(metric_name);
        json["index_param"]["base_quantization_type"].SetString(base_type);
        json["index_param"]["precise_quantization_type"].SetString(precise_type);
        json["index_param"]["use_reorder"].SetBool(true);
        return json.Dump();
    };
    for (const auto& base_type : std::vector<std::string>{"fp32", "rabitq", TestType::NameImpl()}) {
        const auto parameters = config(base_type, TestType::NameImpl());
        auto created = Factory::CreateIndex("hgraph", parameters);
        REQUIRE(created.has_value());
        auto index = created.value();
        REQUIRE(index->Build(base).has_value());
        auto before = index->KnnSearch(query, 10, search);
        REQUIRE(before.has_value());
        if (metric == MetricType::METRIC_TYPE_L2SQR) {
            REQUIRE(before.value()->GetIds()[0] == 0);
        }
        auto binary = index->Serialize();
        REQUIRE(binary.has_value());
        auto restored = Factory::CreateIndex("hgraph", parameters).value();
        REQUIRE(restored->Deserialize(binary.value()).has_value());
        auto after = restored->KnnSearch(query, 10, search);
        REQUIRE(after.has_value());
        for (int i = 0; i < 10; ++i) {
            REQUIRE(after.value()->GetIds()[i] == before.value()->GetIds()[i]);
            REQUIRE(after.value()->GetDistances()[i] == before.value()->GetDistances()[i]);
        }
        // Reranking must use per-vector ADC, including both offsets and scales.
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        IndexCommonParam common;
        common.dim_ = dim;
        common.metric_ = metric;
        common.allocator_ = allocator;
        TestType quant(QuantizerParameter::CreateDefault(TestType::NameImpl()), common);
        auto computer = quant.FactoryComputer();
        computer->SetQuery(vectors.data());
        std::vector<uint8_t> code(quant.GetCodeSize());
        for (int i = 0; i < 10; ++i) {
            REQUIRE(
                quant.EncodeOne(vectors.data() + after.value()->GetIds()[i] * dim, code.data()));
            float expected;
            computer->ComputeDist(code.data(), &expected);
            REQUIRE(after.value()->GetDistances()[i] == Catch::Approx(expected).margin(1e-5));
        }
        auto invalid_values = vectors;
        uint32_t invalid_bits = 0x7fc00000U;
        memcpy(invalid_values.data(), &invalid_bits, sizeof(invalid_bits));
        auto invalid_query = Dataset::Make()
                                 ->NumElements(1)
                                 ->Dim(dim)
                                 ->Float32Vectors(invalid_values.data())
                                 ->Owner(false);
        REQUIRE_FALSE(index->KnnSearch(invalid_query, 10, search).has_value());
        auto invalid_base = Dataset::Make()
                                ->NumElements(count)
                                ->Dim(dim)
                                ->Float32Vectors(invalid_values.data())
                                ->Ids(ids.data())
                                ->Owner(false);
        auto invalid_index = Factory::CreateIndex("hgraph", parameters).value();
        REQUIRE_FALSE(invalid_index->Build(invalid_base).has_value());
        for (auto dtype : {"int8", "float16", "bfloat16"}) {
            auto invalid = JsonType::Parse(parameters);
            invalid["dtype"].SetString(dtype);
            REQUIRE_FALSE(Factory::CreateIndex("hgraph", invalid.Dump()).has_value());
        }
    }
    for (const auto& target_base : std::vector<std::string>{TestType::NameImpl(), "rabitq"}) {
        auto index = Factory::CreateIndex("hgraph", config("fp32", "fp32")).value();
        REQUIRE(index->Build(base).has_value());
        const auto target = config(target_base, TestType::NameImpl());
        auto tuned = index->Tune(target, false);
        REQUIRE(tuned.has_value());
        REQUIRE(tuned.value());
        auto binary = index->Serialize();
        REQUIRE(binary.has_value());
        auto restored = Factory::CreateIndex("hgraph", target).value();
        REQUIRE(restored->Deserialize(binary.value()).has_value());
        auto result = restored->KnnSearch(query, 10, search);
        REQUIRE(result.has_value());
        if (metric == MetricType::METRIC_TYPE_L2SQR) {
            REQUIRE(result.value()->GetIds()[0] == 0);
        }
        // Lossy codes alone cannot supply a later precision-changing Tune.
        auto back = index->Tune(config("fp32", "fp32"), false);
        REQUIRE(back.has_value());
        REQUIRE_FALSE(back.value());
    }
}

TEMPLATE_TEST_CASE("Local SQ batch layout and invalid metadata",
                   "[ut][LocalSQ]",
                   SQ8LocalQuantizer,
                   SQ4LocalQuantizer) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    TestType quant(3, allocator.get());
    const std::vector<float> input{-3, 2, 7, 100, 100, 100};
    std::vector<uint8_t> codes(2 * quant.GetCodeSize() + 2, 0xa5);
    std::vector<float> decoded(input.size());
    REQUIRE(quant.EncodeBatch(input.data(), codes.data() + 1, 2));
    REQUIRE(quant.DecodeBatch(codes.data() + 1, decoded.data(), 2));
    REQUIRE(codes.front() == 0xa5);
    REQUIRE(codes.back() == 0xa5);
    REQUIRE(decoded[0] == -3);
    REQUIRE(decoded[2] == Catch::Approx(7).margin(1e-6));
    REQUIRE(decoded[3] == 100);
    REQUIRE(decoded[4] == 100);
    REQUIRE(decoded[5] == 100);
    REQUIRE(quant.EncodeBatch(nullptr, nullptr, 0));
    REQUIRE(quant.DecodeBatch(nullptr, nullptr, 0));
    // Quantizer metadata is the common header only: dim, metric, size, trained.
    std::stringstream stream;
    IOStreamWriter writer(stream);
    quant.Serialize(writer);
    auto serialized = stream.str();
    for (uint64_t length = 0; length < serialized.size(); ++length) {
        std::stringstream truncated(serialized.substr(0, length));
        IOStreamReader reader(truncated);
        TestType restored(3, allocator.get());
        REQUIRE_THROWS(restored.Deserialize(reader));
    }
    for (const auto offset : {uint64_t{0},
                              uint64_t{sizeof(uint64_t)},
                              uint64_t{sizeof(uint64_t) + sizeof(MetricType)}}) {
        auto corrupt = serialized;
        corrupt.at(offset) = static_cast<char>(0xff);
        std::stringstream bad(corrupt);
        IOStreamReader reader(bad);
        TestType restored(3, allocator.get());
        REQUIRE_THROWS(restored.Deserialize(reader));
    }
}

TEMPLATE_TEST_CASE("Local SQ HGraph metadata accounting and transform rejection",
                   "[ut][LocalSQ]",
                   SQ8LocalQuantizer,
                   SQ4LocalQuantizer) {
    auto params = JsonType::Parse(R"({"dtype":"float32","metric_type":"l2","dim":17,
        "index_param":{"base_quantization_type":"sq8","use_reorder":false}})");
    params["index_param"]["base_quantization_type"].SetString(TestType::BIT_PER_DIM == 8 ? "sq8"
                                                                                         : "sq4");
    auto existing = Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(existing.has_value());
    params["index_param"]["base_quantization_type"].SetString(TestType::NameImpl());
    auto per_vector = Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(per_vector.has_value());
    constexpr uint64_t count = 1ULL << 20;
    const auto overhead =
        per_vector.value()->EstimateMemory(count) - existing.value()->EstimateMemory(count);
    // Rounded allocation blocks may add slack, but the eight metadata bytes are counted.
    const auto block_size = Options::Instance().block_size_limit();
    REQUIRE(overhead + block_size >= count * 8);
    REQUIRE(overhead <= count * 8 + block_size);
    params["index_param"]["base_quantization_type"].SetString("tq");
    params["index_param"]["tq_chain"].SetString("rom," + TestType::NameImpl());
    REQUIRE_FALSE(Factory::CreateIndex("hgraph", params.Dump()).has_value());
}

TEMPLATE_TEST_CASE("Local SQ metric references and normalization",
                   "[ut][LocalSQ][simd]",
                   SQ8LocalQuantizer,
                   SQ4LocalQuantizer) {
    const auto simd_status = LocalSQTestSIMDStatus();
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (auto metric : {MetricType::METRIC_TYPE_L2SQR,
                        MetricType::METRIC_TYPE_IP,
                        MetricType::METRIC_TYPE_COSINE}) {
        for (int dim : {1, 3, 7, 16, 31, 65, 960}) {
            CAPTURE(metric, dim, TestType::BIT_PER_DIM);
            IndexCommonParam common;
            common.dim_ = dim;
            common.allocator_ = allocator;
            common.metric_ = metric;
            common.data_type_ = DataTypes::DATA_TYPE_FLOAT;
            TestType quant(QuantizerParameter::CreateDefault(TestType::NameImpl()), common);
            std::vector<float> a(dim), b(dim), query(dim), nq(dim), na(dim), da(dim), db(dim);
            for (int pattern = 0; pattern < 4; ++pattern) {
                for (int i = 0; i < dim; ++i) {
                    a[i] = pattern == 0 ? 0 : pattern == 1 ? -7 : std::sin(i * 1.3F) * 3 - 2;
                    b[i] = pattern == 0 ? 0 : std::cos(i * 0.7F) * 2 + 4;
                    query[i] = pattern == 0 ? 0 : std::sin(i * 0.9F) * 5 + 1;
                }
                nq = query;
                na = a;
                if (metric == MetricType::METRIC_TYPE_COSINE) {
                    Normalize(query.data(), nq.data(), dim);
                    Normalize(a.data(), na.data(), dim);
                }
                std::vector<uint8_t> storage(2 * quant.GetCodeSize() + 2, 0xa5);
                auto* ca = storage.data() + 1;
                auto* cb = ca + quant.GetCodeSize();
                REQUIRE(quant.EncodeOne(a.data(), ca));
                REQUIRE(quant.EncodeOne(b.data(), cb));
                REQUIRE(storage.front() == 0xa5);
                REQUIRE(storage.back() == 0xa5);
                if (TestType::BIT_PER_DIM == 4 && dim % 2) {
                    REQUIRE((ca[quant.GetCodeSize() - 1] & 0xf0) == 0);
                    REQUIRE((cb[quant.GetCodeSize() - 1] & 0xf0) == 0);
                }
                REQUIRE(quant.DecodeOne(ca, da.data()));
                REQUIRE(quant.DecodeOne(cb, db.data()));
                auto limits = std::minmax_element(na.begin(), na.end());
                double error =
                    (*limits.second - *limits.first) / (2.0 * ((1 << TestType::BIT_PER_DIM) - 1));
                double dot_adc = 0, dot_sdc = 0, l2_adc = 0, l2_sdc = 0;
                for (int i = 0; i < dim; ++i) {
                    REQUIRE(da[i] == Catch::Approx(na[i]).margin(error + 1e-6));
                    dot_adc += double(nq[i]) * db[i];
                    dot_sdc += double(da[i]) * db[i];
                    l2_adc += std::pow(double(nq[i]) - db[i], 2);
                    l2_sdc += std::pow(double(da[i]) - db[i], 2);
                }
                const bool l2 = metric == MetricType::METRIC_TYPE_L2SQR;
                auto computer = quant.FactoryComputer();
                computer->SetQuery(query.data());
                float distance;
                computer->ComputeDist(cb, &distance);
                REQUIRE(distance ==
                        Catch::Approx(l2 ? l2_adc : 1 - dot_adc).epsilon(2e-5).margin(1e-5));
                REQUIRE(quant.Compute(ca, cb) ==
                        Catch::Approx(l2 ? l2_sdc : 1 - dot_sdc).epsilon(2e-5).margin(1e-5));
                std::vector<SQ8LocalComputeType> adc;
                std::vector<SQ8LocalComputeCodesType> sdc;
#define ADD_LOCAL_ISA(isa)                                                                       \
    adc.push_back(TestType::BIT_PER_DIM == 8 ? isa::SQ8LocalComputeIP : isa::SQ4LocalComputeIP); \
    sdc.push_back(TestType::BIT_PER_DIM == 8 ? isa::SQ8LocalComputeCodesIP                       \
                                             : isa::SQ4LocalComputeCodesIP)
                ADD_LOCAL_ISA(generic);
                if ((simd_status.dist_support_sse && simd_status.runtime_has_sse)) {
                    ADD_LOCAL_ISA(sse);
                }
                if ((simd_status.dist_support_avx && simd_status.runtime_has_avx)) {
                    ADD_LOCAL_ISA(avx);
                }
                if ((simd_status.dist_support_avx2 && simd_status.runtime_has_avx2)) {
                    ADD_LOCAL_ISA(avx2);
                }
                if ((simd_status.dist_support_avx512f && simd_status.runtime_has_avx512f)) {
                    ADD_LOCAL_ISA(avx512);
                }
                if ((simd_status.dist_support_neon && simd_status.runtime_has_neon)) {
                    ADD_LOCAL_ISA(neon);
                }
                if ((simd_status.dist_support_sve && simd_status.runtime_has_sve)) {
                    ADD_LOCAL_ISA(sve);
                }
#undef ADD_LOCAL_ISA
                for (auto fn : adc) {
                    REQUIRE(fn(nq.data(), cb, dim) ==
                            Catch::Approx(dot_adc).epsilon(2e-5).margin(1e-5));
                }
                for (auto fn : sdc) {
                    REQUIRE(fn(ca, cb, dim) == Catch::Approx(dot_sdc).epsilon(2e-5).margin(1e-5));
                    REQUIRE(fn(cb, ca, dim) == Catch::Approx(dot_sdc).epsilon(2e-5).margin(1e-5));
                }
                TestType restored(1, allocator.get());
                test_serializion(quant, restored);
                REQUIRE(restored.Compute(ca, cb) == quant.Compute(ca, cb));
                auto restored_computer = restored.FactoryComputer();
                restored_computer->SetQuery(query.data());
                float restored_distance;
                restored_computer->ComputeDist(cb, &restored_distance);
                REQUIRE(restored_distance == distance);
            }
            for (uint32_t bits : {0x7f800000U, 0xff800000U, 0x7fc00000U}) {
                memcpy(a.data(), &bits, sizeof(bits));
                std::vector<uint8_t> code(quant.GetCodeSize());
                REQUIRE_THROWS(quant.EncodeOne(a.data(), code.data()));
                auto computer = quant.FactoryComputer();
                REQUIRE_THROWS(computer->SetQuery(a.data()));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Local SQ finite IP overflow and cosine extremes",
                   "[ut][LocalSQ][simd]",
                   SQ8LocalQuantizer,
                   SQ4LocalQuantizer) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (auto metric : {MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_COSINE}) {
        IndexCommonParam common;
        common.dim_ = 17;
        common.allocator_ = allocator;
        common.metric_ = metric;
        TestType quant(QuantizerParameter::CreateDefault(TestType::NameImpl()), common);
        for (float magnitude : {std::numeric_limits<float>::max(),
                                1e20F,
                                std::numeric_limits<float>::min(),
                                std::numeric_limits<float>::denorm_min()}) {
            CAPTURE(metric, magnitude);
            std::vector<float> a(17, magnitude), b(17, -magnitude), q(17), da(17), db(17), nq(17);
            // Exact cancellation of large query products must not become NaN.
            for (int i = 0; i < 16; ++i) {
                q[i] = i % 2 ? -magnitude : magnitude;
            }
            std::vector<uint8_t> ca(quant.GetCodeSize()), cb(quant.GetCodeSize());
            REQUIRE(quant.EncodeOne(a.data(), ca.data()));
            REQUIRE(quant.EncodeOne(b.data(), cb.data()));
            quant.DecodeOne(ca.data(), da.data());
            quant.DecodeOne(cb.data(), db.data());
            nq = q;
            if (metric == MetricType::METRIC_TYPE_COSINE) {
                Normalize(q.data(), nq.data(), 17);
            }
            double ref_adc = 0, ref_sdc = 0;
            for (int i = 0; i < 17; ++i) {
                ref_adc += double(nq[i]) * db[i];
                ref_sdc += double(da[i]) * db[i];
            }
            auto computer = quant.FactoryComputer();
            computer->SetQuery(q.data());
            float distance;
            computer->ComputeDist(cb.data(), &distance);
            REQUIRE(distance == Catch::Approx(1 - ref_adc).margin(1e-5));
            const float sdc = quant.Compute(ca.data(), cb.data());
            if (1 - ref_sdc > std::numeric_limits<float>::max()) {
                uint32_t bits;
                memcpy(&bits, &sdc, sizeof(bits));
                REQUIRE(bits == 0x7f800000U);
            } else {
                REQUIRE(sdc == Catch::Approx(1 - ref_sdc).margin(1e-5));
            }
        }
        // Query-only overflow exercises the SIMD-result fallback with ordinary records.
        std::vector<float> b(17, 8), q(17, 0);
        for (int i = 0; i < 16; ++i) {
            q[i] = i % 2 ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        }
        std::vector<uint8_t> code(quant.GetCodeSize());
        quant.EncodeOne(b.data(), code.data());
        auto computer = quant.FactoryComputer();
        computer->SetQuery(q.data());
        float distance;
        computer->ComputeDist(code.data(), &distance);
        REQUIRE(distance == Catch::Approx(1).margin(1e-5));
    }
}

TEMPLATE_TEST_CASE("Local SQ HGraph needs no training",
                   "[ut][LocalSQ]",
                   SQ8LocalQuantizer,
                   SQ4LocalQuantizer) {
    for (const auto* metric : {"l2", "ip", "cosine"}) {
        auto params = JsonType::Parse(R"({"dtype":"float32","metric_type":"l2","dim":17,
            "index_param":{"base_quantization_type":"sq8_local","use_reorder":false}})");
        params["metric_type"].SetString(metric);
        params["index_param"]["base_quantization_type"].SetString(TestType::NameImpl());
        auto created = Factory::CreateIndex("hgraph", params.Dump());
        REQUIRE(created.has_value());
        REQUIRE_FALSE(created.value()->CheckFeature(IndexFeature::NEED_TRAIN));
        REQUIRE(created.value()->CheckFeature(IndexFeature::SUPPORT_RANGE_SEARCH));
        REQUIRE(created.value()->CheckFeature(IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER));
        // Globally calibrated SQ still needs training.
        params["index_param"]["base_quantization_type"].SetString("sq8");
        auto calibrated = Factory::CreateIndex("hgraph", params.Dump());
        REQUIRE(calibrated.has_value());
        REQUIRE(calibrated.value()->CheckFeature(IndexFeature::NEED_TRAIN));
    }
}
