
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

#include "sq8_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "../../tests/fixtures/fixtures.h"
#include "quantizer_test.h"
#include "simd/simd.h"

using namespace vsag;

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
        REQUIRE(std::abs(gt - value) < 1e-2);
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
        REQUIRE(std::abs(gt - value) < 2e-2);
        delete[] codes1;
    }
    delete[] codes;
}

TEST_CASE("encode&decode", "[SQ8Quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    auto dim = 256;
    auto size = 1000;
    float error = 0.01;
    float error_same = (float)dim * 255 * 0.01;
    {
        SQ8Quantizer<metrics[0]> quantizer{dim};
        TestQuantizerEncodeDecode<>(quantizer, dim, size, error);
        TestQuantizerEncodeDecodeSame(quantizer, dim, size, 255, error_same);
    }
    {
        SQ8Quantizer<metrics[1]> quantizer{dim};
        TestQuantizerEncodeDecode<>(quantizer, dim, size, error);
        TestQuantizerEncodeDecodeSame(quantizer, dim, size, 255, error_same);
    }
    {
        SQ8Quantizer<metrics[2]> quantizer{dim};
        TestQuantizerEncodeDecode<>(quantizer, dim, size, error);
        TestQuantizerEncodeDecodeSame(quantizer, dim, size, 255, error_same);
    }
}

TEST_CASE("compute_codes", "[SQ8Quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    auto dim = 32;
    {
        SQ8Quantizer<metrics[0]> quantizer{dim};
        TestComputeCodes<>(quantizer, dim, 100, metrics[0]);
    }
    {
        SQ8Quantizer<metrics[1]> quantizer{dim};
        TestComputeCodes<>(quantizer, dim, 100, metrics[1]);
    }
    {
        SQ8Quantizer<metrics[2]> quantizer{dim};
        TestComputeCodes<>(quantizer, dim, 100, metrics[2]);
    }
}

TEST_CASE("computer", "[SQ8Quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    auto dim = 32;
    {
        SQ8Quantizer<metrics[0]> quantizer{dim};
        TestComputer(quantizer, dim, 100, metrics[0]);
    }
    {
        SQ8Quantizer<metrics[1]> quantizer{dim};
        TestComputer(quantizer, dim, 100, metrics[1]);
    }
    {
        SQ8Quantizer<metrics[2]> quantizer{dim};
        TestComputer(quantizer, dim, 100, metrics[2]);
    }
}
