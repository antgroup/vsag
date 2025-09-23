
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
#include <cmath>
#include <iostream>
#include <vector>

#include "fixtures.h"
#include "quantization/computer.h"
#include "quantizer.h"
#include "simd/basic_func.h"

using namespace vsag;

template <typename T, typename DataT = float>
void
TestQuantizerAdapterEncodeDecode(
    Quantizer<T>& quant, int64_t dim, int count, float error = 1e-5, bool retrain = true) {
    std::vector<DataT> vecs;
    if constexpr (std::is_same<DataT, float>::value == true) {
        vecs = fixtures::generate_vectors(count, dim, true);
    } else if constexpr (std::is_same<DataT, int8_t>::value == true) {
        vecs = fixtures::generate_int8_codes(count, dim, true);
    } else {
        static_assert("Unsupported DataT type");
    }
    if (retrain) {
        quant.ReTrain(reinterpret_cast<DataType*>(vecs.data()), count);
    }
    // Test EncodeOne & DecodeOne
    for (uint64_t i = 0; i < count; ++i) {
        std::vector<uint8_t> codes(quant.GetCodeSize());
        quant.EncodeOne(reinterpret_cast<DataType*>(vecs.data() + i * dim), codes.data());
        std::vector<DataT> out_vec(dim);
        quant.DecodeOne(codes.data(), reinterpret_cast<DataType*>(out_vec.data()));
        float sum = 0.0F;
        for (int j = 0; j < dim; ++j) {
            sum += std::abs(static_cast<DataType>(vecs[i * dim + j]) -
                            static_cast<DataType>(out_vec[j]));
        }
        REQUIRE(sum < error * dim);
    }

    // Test EncodeBatch & DecodeBatch
    std::vector<uint8_t> codes(quant.GetCodeSize() * count);
    quant.EncodeBatch(reinterpret_cast<DataType*>(vecs.data()), codes.data(), count);
    std::vector<DataT> out_vec(dim * count);
    quant.DecodeBatch(codes.data(), reinterpret_cast<DataType*>(out_vec.data()), count);
    for (int64_t i = 0; i < count; ++i) {
        float sum = 0.0F;
        for (int j = 0; j < dim; ++j) {
            sum += std::abs(static_cast<DataType>(vecs[i * dim + j]) -
                            static_cast<DataType>(out_vec[i * dim + j]));
        }
        REQUIRE(sum < error * dim);
    }
}

template <typename T, MetricType metric, typename DataT = float>
void
TestQuantizerAdapterComputeCodes(
    Quantizer<T>& quantizer, size_t dim, uint32_t count, float error = 1e-4f, bool retrain = true) {
    std::vector<DataT> vecs;
    if constexpr (std::is_same<DataT, float>::value == true) {
        vecs = fixtures::generate_vectors(count, dim, false);
    } else if constexpr (std::is_same<DataT, int8_t>::value == true) {
        vecs = fixtures::generate_int8_codes(count, dim, false);
    } else {
        static_assert("Unsupported DataT type");
    }

    if (retrain) {
        quantizer.ReTrain(reinterpret_cast<DataType*>(vecs.data()), count);
    }

    for (int64_t i = 0; i < count; ++i) {
        auto idx1 = random() % count;
        auto idx2 = random() % count;
        std::vector<uint8_t> codes1(quantizer.GetCodeSize());
        std::vector<uint8_t> codes2(quantizer.GetCodeSize());
        quantizer.EncodeOne(reinterpret_cast<DataType*>(vecs.data() + idx1 * dim), codes1.data());
        quantizer.EncodeOne(reinterpret_cast<DataType*>(vecs.data() + idx2 * dim), codes2.data());
        float gt = 0.0;
        float value = quantizer.Compute(codes1.data(), codes2.data());
        if constexpr (metric == MetricType::METRIC_TYPE_IP ||
                      metric == MetricType::METRIC_TYPE_COSINE) {
            gt = 1 - INT8InnerProduct(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
        } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
            gt = INT8L2Sqr(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
        }
        if (gt != 0.0) {
            REQUIRE(std::abs((gt - value) / gt) < error);
        } else {
            REQUIRE(std::abs(gt - value) < error);
        }
    }
}
