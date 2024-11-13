
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
#include <fstream>

#include "fixtures.h"
#include "quantizer.h"
#include "simd/normalize.h"
#include "simd/simd.h"

using namespace vsag;

template <typename T>
void
TestQuantizerEncodeDecode(
    Quantizer<T>& quant, int64_t dim, int count, float error = 1e-5, bool retrain = true) {
    auto vecs = fixtures::generate_vectors(count, dim, true);
    if (retrain) {
        quant.ReTrain(vecs.data(), count);
    }
    // Test EncodeOne & DecodeOne
    for (uint64_t i = 0; i < count; ++i) {
        std::vector<uint8_t> codes(quant.GetCodeSize());
        quant.EncodeOne(vecs.data() + i * dim, codes.data());
        std::vector<float> out_vec(dim);
        quant.DecodeOne(codes.data(), out_vec.data());
        for (int j = 0; j < dim; ++j) {
            REQUIRE(std::abs(vecs[i * dim + j] - out_vec[j]) < error);
        }
    }

    // Test EncodeBatch & DecodeBatch
    std::vector<uint8_t> codes(quant.GetCodeSize() * count);
    quant.EncodeBatch(vecs.data(), codes.data(), count);
    std::vector<float> out_vec(dim * count);
    quant.DecodeBatch(codes.data(), out_vec.data(), count);
    for (int64_t i = 0; i < dim * count; ++i) {
        REQUIRE(std::abs(vecs[i] - out_vec[i]) < error);
    }
}

template <typename T>
void
TestQuantizerEncodeDecodeSame(Quantizer<T>& quant,
                              int64_t dim,
                              int count,
                              int code_max = 15,
                              float error = 1e-5,
                              bool retrain = true) {
    auto data = fixtures::generate_vectors(count, dim);
    for (auto& val : data) {
        val = uint8_t(val * code_max);
    }
    if (retrain) {
        quant.ReTrain(data.data(), count);
    }

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

template <typename T, MetricType metric>
void
TestComputeCodes(
    Quantizer<T>& quantizer, size_t dim, uint32_t count, float error = 1e-4f, bool retrain = true) {
    auto vecs = fixtures::generate_vectors(count, dim, true);
    if (retrain) {
        quantizer.ReTrain(vecs.data(), count);
    }
    for (int i = 0; i < count; ++i) {
        auto idx1 = random() % count;
        auto idx2 = random() % count;
        std::vector<uint8_t> codes1(quantizer.GetCodeSize());
        std::vector<uint8_t> codes2(quantizer.GetCodeSize());
        quantizer.EncodeOne(vecs.data() + idx1 * dim, codes1.data());
        quantizer.EncodeOne(vecs.data() + idx2 * dim, codes2.data());
        float gt = 0.0;
        float value = quantizer.Compute(codes1.data(), codes2.data());
        if constexpr (metric == vsag::MetricType::METRIC_TYPE_IP ||
                      metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            gt = 1 - InnerProduct(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
        } else if constexpr (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            gt = L2Sqr(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
        }
        REQUIRE(std::abs(gt - value) < error);
    }
}

template <typename T, MetricType metric>
void
TestComputeCodesSame(Quantizer<T>& quantizer,
                     size_t dim,
                     uint32_t count,
                     uint32_t code_max = 15,
                     float error = 1e-5f,
                     bool retrain = true) {
    auto data = fixtures::generate_vectors(count, dim, false);
    for (auto& val : data) {
        val = uint8_t(val * code_max);
    }
    if (retrain) {
        quantizer.ReTrain(data.data(), count);
    }
    for (int i = 0; i < count; ++i) {
        auto idx1 = random() % count;
        auto idx2 = random() % count;
        std::vector<uint8_t> codes1(quantizer.GetCodeSize());
        std::vector<uint8_t> codes2(quantizer.GetCodeSize());
        quantizer.EncodeOne(data.data() + idx1 * dim, codes1.data());
        quantizer.EncodeOne(data.data() + idx2 * dim, codes2.data());
        float gt = 0.0f;
        float value = quantizer.Compute(codes1.data(), codes2.data());
        if constexpr (metric == vsag::MetricType::METRIC_TYPE_IP) {
            gt = 1 - InnerProduct(data.data() + idx1 * dim, data.data() + idx2 * dim, &dim);
        } else if constexpr (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            gt = L2Sqr(data.data() + idx1 * dim, data.data() + idx2 * dim, &dim);
        } else if constexpr (metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            std::vector<float> v1(dim);
            std::vector<float> v2(dim);
            Normalize(data.data() + idx1 * dim, v1.data(), dim);
            Normalize(data.data() + idx2 * dim, v2.data(), dim);
            gt = 1 - InnerProduct(v1.data(), v2.data(), &dim);
        }
        REQUIRE(std::abs(gt - value) <= error);
    }
}

template <typename T, MetricType metric>
void
TestComputer(
    Quantizer<T>& quant, size_t dim, uint32_t count, float error = 1e-5f, bool retrain = true) {
    auto query_count = 100;
    bool need_normalize = true;
    if constexpr (metric == vsag::MetricType::METRIC_TYPE_COSINE) {
        need_normalize = false;
    }
    auto vecs = fixtures::generate_vectors(count, dim, need_normalize);
    auto querys = fixtures::generate_vectors(query_count, dim, need_normalize, 165);
    if (retrain) {
        quant.ReTrain(vecs.data(), count);
    }
    for (int i = 0; i < query_count; ++i) {
        std::shared_ptr<Computer<T>> computer;
        computer = quant.FactoryComputer();
        computer->SetQuery(querys.data() + i * dim);
        for (int j = 0; j < 100; ++j) {
            auto idx1 = random() % count;
            auto* codes1 = new uint8_t[quant.GetCodeSize()];
            quant.EncodeOne(vecs.data() + idx1 * dim, codes1);
            float gt = 0.0f;
            float value = 0.0f;
            quant.ComputeDist(*computer, codes1, &value);
            REQUIRE(quant.ComputeDist(*computer, codes1) == value);
            if constexpr (metric == vsag::MetricType::METRIC_TYPE_IP) {
                gt = 1 - InnerProduct(vecs.data() + idx1 * dim, querys.data() + i * dim, &dim);
            } else if constexpr (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
                gt = L2Sqr(vecs.data() + idx1 * dim, querys.data() + i * dim, &dim);
            } else if constexpr (metric == vsag::MetricType::METRIC_TYPE_COSINE) {
                std::vector<float> v1(dim);
                std::vector<float> v2(dim);
                Normalize(vecs.data() + idx1 * dim, v1.data(), dim);
                Normalize(querys.data() + i * dim, v2.data(), dim);
                gt = 1 - InnerProduct(v1.data(), v2.data(), &dim);
            }
            REQUIRE(std::abs(gt - value) < error);
            delete[] codes1;
        }
    }
}

template <typename T, MetricType metric, bool uniform = false>
void
TestSerializeAndDeserialize(
    Quantizer<T>& quant1, Quantizer<T>& quant2, size_t dim, uint32_t count, float error = 1e-5f) {
    auto vecs = fixtures::generate_vectors(count, dim);
    quant1.ReTrain(vecs.data(), count);
    std::string dirname = "/tmp/quantizer_TestSerializeAndDeserialize_" + std::to_string(random());
    std::filesystem::create_directory(dirname);
    auto filename = dirname + "/file_" + std::to_string(random());
    std::ofstream outfile(filename.c_str(), std::ios::binary);
    IOStreamWriter writer(outfile);
    quant1.Serialize(writer);
    outfile.close();

    std::ifstream infile(filename.c_str(), std::ios::binary);
    IOStreamReader reader(infile);
    quant2.Deserialize(reader);

    REQUIRE(quant1.GetCodeSize() == quant2.GetCodeSize());
    REQUIRE(quant1.GetDim() == quant2.GetDim());

    TestQuantizerEncodeDecode<T>(quant2, dim, count, error, false);
    if constexpr (uniform == false) {
        TestComputer<T, metric>(quant2, dim, count, error, false);
        TestComputeCodes<T, metric>(quant2, dim, count, error, false);
    } else {
        TestComputeCodesSame<T, metric>(quant2, dim, count, error, false);
    }

    infile.close();
    std::filesystem::remove_all(dirname);
}
