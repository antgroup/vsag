
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
#include <vector>

#include "fixtures.h"
#include "quantization/computer.h"
#include "quantizer.h"

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
