
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

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "quantizer.h"
#include "simd/simd.h"
using namespace vsag;

template <typename T>
void
TestQuantizerEncodeDecode(Quantizer<T>& quant, int64_t dim, int count, float error = 1e-5) {
    auto vecs = fixtures::generate_vectors(count, dim);
    quant.ReTrain(vecs.data(), count);

    // Test EncodeOne & DecodeOne
    auto idx = random() % count;
    auto* codes = new uint8_t[quant.GetCodeSize()];
    quant.EncodeOne(vecs.data() + idx * dim, codes);
    auto* outVec = new float[dim];
    quant.DecodeOne(codes, outVec);
    for (int i = 0; i < dim; ++i) {
        REQUIRE(std::abs(vecs[idx * dim + i] - outVec[i]) < error);
    }

    // Test EncodeBatch & DecodeBatch
    delete[] codes;
    delete[] outVec;

    codes = new uint8_t[quant.GetCodeSize() * count];
    quant.EncodeBatch(vecs.data(), codes, count);
    outVec = new float[dim * count];
    quant.DecodeBatch(codes, outVec, count);
    for (int64_t i = 0; i < dim * count; ++i) {
        REQUIRE(std::abs(vecs[i] - outVec[i]) < error);
    }

    delete[] outVec;
    delete[] codes;
}

template <typename T>
void
TestQuantizerEncodeDecodeSame(
    Quantizer<T>& quant, int64_t dim, int count, int code_max = 15, float error = 1e-5) {
    float* data = new float[dim * count];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, code_max);

    for (int i = 0; i < dim * count; i++) {
        data[i] = int(dis(gen));
    }
    quant.ReTrain(data, count);

    // Test EncodeOne & DecodeOne
    for (int k = 0; k < count; k++) {
        auto idx = random() % count;
        auto* codes = new uint8_t[quant.GetCodeSize()];
        quant.EncodeOne(data + idx * dim, codes);
        auto* outVec = new float[dim];
        quant.DecodeOne(codes, outVec);
        for (int i = 0; i < dim; ++i) {
            REQUIRE(std::abs(data[idx * dim + i] - outVec[i]) < error);
        }
        delete[] codes;
        delete[] outVec;
    }

    // Test EncodeBatch & DecodeBatch
    {
        auto codes = new uint8_t[quant.GetCodeSize() * count];
        quant.EncodeBatch(data, codes, count);
        auto outVec = new float[dim * count];
        quant.DecodeBatch(codes, outVec, count);
        for (int64_t i = 0; i < dim * count; ++i) {
            REQUIRE(std::abs(data[i] - outVec[i]) < error);
        }

        delete[] outVec;
        delete[] codes;
    }
}

template <typename T>
void
TestComputeCodes(Quantizer<T>& quantizer, size_t dim, uint32_t size, const MetricType& metric) {
    auto vecs = fixtures::generate_vectors(size, dim);

    quantizer.ReTrain(vecs.data(), size);
    for (int i = 0; i < size; ++i) {
        auto idx1 = random() % size;
        auto idx2 = random() % size;
        auto* codes1 = new uint8_t[quantizer.GetCodeSize()];
        auto* codes2 = new uint8_t[quantizer.GetCodeSize()];
        quantizer.EncodeOne(vecs.data() + idx1 * dim, codes1);
        quantizer.EncodeOne(vecs.data() + idx2 * dim, codes2);
        float gt = 0.;
        float value = quantizer.Compute(codes1, codes2);
        if (metric == vsag::MetricType::METRIC_TYPE_IP ||
            metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            gt = InnerProduct(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
        } else if (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            gt = L2Sqr(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
        }
        REQUIRE(std::abs(gt - value) < 1e-4);
        delete[] codes1;
        delete[] codes2;
    }
}

template <typename T>
void
TestComputeCodesSame(Quantizer<T>& quantizer, size_t dim, uint32_t size, const MetricType& metric) {
    float* data = new float[dim * size];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    for (int i = 0; i < dim * size; i++) {
        data[i] = int(dis(gen));
    }

    quantizer.ReTrain(data, size);
    for (int i = 0; i < size; ++i) {
        auto idx1 = random() % size;
        auto idx2 = random() % size;
        auto* codes1 = new uint8_t[quantizer.GetCodeSize()];
        auto* codes2 = new uint8_t[quantizer.GetCodeSize()];
        quantizer.EncodeOne(data + idx1 * dim, codes1);
        quantizer.EncodeOne(data + idx2 * dim, codes2);
        float gt = 0.;
        float value = quantizer.Compute(codes1, codes2);
        if (metric == vsag::MetricType::METRIC_TYPE_IP ||
            metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            gt = InnerProduct(data + idx1 * dim, data + idx2 * dim, &dim);
        } else if (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            gt = L2Sqr(data + idx1 * dim, data + idx2 * dim, &dim);
        }
        REQUIRE(std::abs(gt - value) < 1e-4);
        delete[] codes1;
        delete[] codes2;
    }

    delete[] data;
}

template <typename T>
void
TestComputerSame(Quantizer<T>& quant, size_t dim, uint32_t size, const MetricType& metric) {
    float* data = new float[dim * size];
    float* query = new float[dim * size];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    for (int i = 0; i < dim * size; i++) {
        data[i] = int(dis(gen));
        query[i] = int(dis(gen));
    }

    auto* codes = new uint8_t[quant.GetCodeSize() * dim];
    quant.Train(data, size);
    for (int i = 0; i < size; ++i) {
        auto computer = quant.FactoryComputer();
        computer->SetQuery(query + i * dim);
        auto idx1 = random() % size;
        auto* codes1 = new uint8_t[quant.GetCodeSize()];
        quant.EncodeOne(data + idx1 * dim, codes1);
        float gt = 0.;
        float value = 0.;
        computer->ComputeDist(codes1, &value);
        if (metric == vsag::MetricType::METRIC_TYPE_IP ||
            metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            gt = InnerProduct(data + idx1 * dim, query + i * dim, &dim);
        } else if (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            gt = L2Sqr(data + idx1 * dim, query + i * dim, &dim);
        }
        REQUIRE(std::abs(gt - value) < 1e-4);
    }
    delete[] codes;

    delete[] data;
    delete[] query;
}
