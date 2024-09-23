
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

#include "fp32_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "fixtures.h"
#include "simd/simd.h"
using namespace vsag;

template <typename T>
void
TestQuantizerEncodeDecode(Quantizer<T>& quant, int64_t dim, int count) {
    auto vecs = fixtures::generate_vectors(count, dim);
    quant.Train(vecs.data(), count);

    // Test EncodeOne
    auto idx = int(count / 3);
    auto* codes = new uint8_t[quant.GetCodeSize()];
    quant.EncodeOne(vecs.data() + idx * dim, codes);
    auto values = (float*)(codes);
    for (int i = 0; i < dim; ++i) {
        REQUIRE(std::abs(vecs[idx * dim + i] - values[i]) < 1e-5);
    }

    // Test DecodeOne
    auto* outVec = new float[dim];
    quant.DecodeOne(codes, outVec);
    for (int i = 0; i < dim; ++i) {
        REQUIRE(std::abs(vecs[idx * dim + i] - outVec[i]) < 1e-5);
    }

    // Test EncodeBatch
    delete[] codes;
    delete[] outVec;
    codes = new uint8_t[quant.GetCodeSize() * count];
    quant.EncodeBatch(vecs.data(), codes, count);
    values = (float*)(codes);
    for (int64_t i = 0; i < dim * count; ++i) {
        REQUIRE(std::abs(vecs[i] - values[i]) < 1e-5);
    }

    // Test DecodeBatch
    outVec = new float[dim * count];
    quant.DecodeBatch(codes, outVec, count);
    for (int i = 0; i < dim; ++i) {
        REQUIRE(fixtures::dist_t(vecs[i]) == fixtures::dist_t(outVec[i]));
    }

    delete[] outVec;
    delete[] codes;
}

template <typename T>
void
TestComputeCodes(Quantizer<T>& quant, int64_t dim, int count, const MetricType& metric) {
    auto vecs = fixtures::generate_vectors(count, dim);
    quant.Train(vecs.data(), count);
    for (int i = 0; i < 100; ++i) {
        auto idx1 = random() % count;
        auto idx2 = random() % count;
        auto* codes1 = new uint8_t[quant.GetCodeSize()];
        auto* codes2 = new uint8_t[quant.GetCodeSize()];
        quant.EncodeOne(vecs.data() + idx1 * dim, codes1);
        quant.EncodeOne(vecs.data() + idx2 * dim, codes2);
        float gt = 0.;
        float value = quant.Compute(codes2, codes1);
        if (metric == vsag::MetricType::METRIC_TYPE_IP ||
            metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            gt = InnerProduct(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
        } else if (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            gt = L2Sqr(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
        }
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(value));
        delete[] codes1;
        delete[] codes2;
    }
}

template <typename T>
void
TestComputer(Quantizer<T>& quant, int64_t dim, int count, const MetricType& metric) {
    auto vecs = fixtures::generate_vectors(count, dim);
    auto querys = fixtures::generate_vectors(100, dim);
    auto* codes = new uint8_t[quant.GetCodeSize() * dim];
    quant.Train(vecs.data(), count);
    for (int i = 0; i < 100; ++i) {
        auto computer = quant.FactoryComputer();
        computer->SetQuery(querys.data() + i * dim);
        auto idx1 = random() % count;
        auto* codes1 = new uint8_t[quant.GetCodeSize()];
        quant.EncodeOne(vecs.data() + idx1 * dim, codes1);
        float gt = 0.;
        float value = 0.;
        computer->ComputeDist(codes1, &value);
        if (metric == vsag::MetricType::METRIC_TYPE_IP ||
            metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            gt = InnerProduct(vecs.data() + idx1 * dim, querys.data() + i * dim, &dim);
        } else if (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            gt = L2Sqr(vecs.data() + idx1 * dim, querys.data() + i * dim, &dim);
        }
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(value));
        delete[] codes1;
    }
    delete[] codes;
}

TEST_CASE("encode&decode", "[ut][fp32_quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    auto dim = 32;
    {
        FP32Quantizer<metrics[0]> quantizer{dim};
        TestQuantizerEncodeDecode<>(quantizer, dim, 100);
    }
    {
        FP32Quantizer<metrics[1]> quantizer{dim};
        TestQuantizerEncodeDecode<>(quantizer, dim, 100);
    }
    {
        FP32Quantizer<metrics[2]> quantizer{dim};
        TestQuantizerEncodeDecode<>(quantizer, dim, 100);
    }
}

TEST_CASE("compute_codes[ut][fp32_quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    auto dim = 32;
    {
        FP32Quantizer<metrics[0]> quantizer{dim};
        TestComputeCodes<>(quantizer, dim, 100, metrics[0]);
    }
    {
        FP32Quantizer<metrics[1]> quantizer{dim};
        TestComputeCodes<>(quantizer, dim, 100, metrics[1]);
    }
    {
        FP32Quantizer<metrics[2]> quantizer{dim};
        TestComputeCodes<>(quantizer, dim, 100, metrics[2]);
    }
}

TEST_CASE("computer[ut][fp32_quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    auto dim = 32;
    {
        FP32Quantizer<metrics[0]> quantizer{dim};
        TestComputer<>(quantizer, dim, 100, metrics[0]);
    }
    {
        FP32Quantizer<metrics[1]> quantizer{dim};
        TestComputer<>(quantizer, dim, 100, metrics[1]);
    }
    {
        FP32Quantizer<metrics[2]> quantizer{dim};
        TestComputer<>(quantizer, dim, 100, metrics[2]);
    }
}
