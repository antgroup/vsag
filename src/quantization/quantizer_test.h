
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
    std::vector<uint8_t> codes(quant.GetCodeSize());
    quant.EncodeOne(vecs.data() + idx * dim, codes.data());
    std::vector<float> out_vec(dim);
    quant.DecodeOne(codes.data(), out_vec.data());
    for (int i = 0; i < dim; ++i) {
        REQUIRE(std::abs(vecs[idx * dim + i] - out_vec[i]) < error);
    }
    // Test EncodeBatch & DecodeBatch
    codes.resize(quant.GetCodeSize() * count);
    quant.EncodeBatch(vecs.data(), codes.data(), count);
    std::vector<float> outVecBatch(dim * count);
    quant.DecodeBatch(codes.data(), outVecBatch.data(), count);
    for (int64_t i = 0; i < dim * count; ++i) {
        REQUIRE(std::abs(vecs[i] - outVecBatch[i]) < error);
    }
}

template <typename T>
void
TestQuantizerEncodeDecodeSame(
    Quantizer<T>& quant, int64_t dim, int count, int code_max = 15, float error = 1e-5) {
    std::vector<float> data(dim * count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, code_max);

    for (int i = 0; i < dim * count; i++) {
        data[i] = int(dis(gen));
    }
    quant.ReTrain(data.data(), count);

    // Test EncodeOne & DecodeOne
    for (int k = 0; k < count; k++) {
        auto idx = random() % count;
        std::vector<uint8_t> codes(quant.GetCodeSize());
        quant.EncodeOne(data.data() + idx * dim, codes.data());
        std::vector<float> out_vec(dim);
        quant.DecodeOne(codes.data(), out_vec.data());
        for (int i = 0; i < dim; ++i) {
            REQUIRE(std::abs(data[idx * dim + i] - out_vec[i]) < error);
        }
    }

    // Test EncodeBatch & DecodeBatch
    {
        std::vector<uint8_t> codes(quant.GetCodeSize() * count);
        quant.EncodeBatch(data.data(), codes.data(), count);

        std::vector<float> out_vec(dim * count);
        quant.DecodeBatch(codes.data(), out_vec.data(), count);

        for (int64_t i = 0; i < dim * count; ++i) {
            REQUIRE(std::abs(data[i] - out_vec[i]) < error);
        }
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
        std::vector<uint8_t> codes1(quantizer.GetCodeSize());
        std::vector<uint8_t> codes2(quantizer.GetCodeSize());
        quantizer.EncodeOne(vecs.data() + idx1 * dim, codes1.data());
        quantizer.EncodeOne(vecs.data() + idx2 * dim, codes2.data());
        float gt = 0.0;
        float value = quantizer.Compute(codes1.data(), codes2.data());
        if (metric == vsag::MetricType::METRIC_TYPE_IP ||
            metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            gt = InnerProduct(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
        } else if (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            gt = L2Sqr(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
        }
        REQUIRE(std::abs(gt - value) < 1e-4);
    }
}

template <typename T>
void
TestComputeCodesSame(Quantizer<T>& quantizer, size_t dim, uint32_t size, const MetricType& metric) {
    std::vector<float> data(dim * size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    for (int i = 0; i < dim * size; i++) {
        data[i] = dis(gen);
    }
    quantizer.ReTrain(data.data(), size);
    for (int i = 0; i < size; ++i) {
        auto idx1 = random() % size;
        auto idx2 = random() % size;
        std::vector<uint8_t> codes1(quantizer.GetCodeSize());
        std::vector<uint8_t> codes2(quantizer.GetCodeSize());
        quantizer.EncodeOne(data.data() + idx1 * dim, codes1.data());
        quantizer.EncodeOne(data.data() + idx2 * dim, codes2.data());
        float gt = 0.0f;
        float value = quantizer.Compute(codes1.data(), codes2.data());
        if (metric == vsag::MetricType::METRIC_TYPE_IP ||
            metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            gt = InnerProduct(data.data() + idx1 * dim, data.data() + idx2 * dim, &dim);
        } else if (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            gt = L2Sqr(data.data() + idx1 * dim, data.data() + idx2 * dim, &dim);
        }
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(value));
    }
}

template <typename T>
void
TestComputerSame(Quantizer<T>& quant, size_t dim, uint32_t size, const MetricType& metric) {
    std::vector<float> data(dim * size);
    std::vector<float> query(dim * size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    for (int i = 0; i < dim * size; i++) {
        data[i] = dis(gen);
        query[i] = dis(gen);
    }
    std::vector<uint8_t> codes(quant.GetCodeSize() * dim);
    quant.Train(data.data(), size);
    for (int i = 0; i < size; ++i) {
        auto computer = quant.FactoryComputer();
        computer->SetQuery(query.data() + i * dim);
        auto idx1 = random() % size;
        std::vector<uint8_t> codes1(quant.GetCodeSize());
        quant.EncodeOne(data.data() + idx1 * dim, codes1.data());
        float gt = 0.0;
        float value = 0.0;
        computer->ComputeDist(codes1.data(), &value);
        if (metric == vsag::MetricType::METRIC_TYPE_IP ||
            metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            gt = InnerProduct(data.data() + idx1 * dim, query.data() + i * dim, &dim);
        } else if (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            gt = L2Sqr(data.data() + idx1 * dim, query.data() + i * dim, &dim);
        }
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(value));
    }
}
