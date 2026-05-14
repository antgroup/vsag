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

#pragma once

#include <cmath>
#include <vector>

#include "unittest.h"

namespace vsag {

template <typename QuantizerT>
void
TestUniformZeroRangeEncodesToZero(QuantizerT& quantizer,
                                  uint64_t dim,
                                  uint64_t encoded_code_size,
                                  uint64_t code_offset = 0,
                                  float tolerance = 1e-5F) {
    std::vector<float> train(dim * 3, 3.0F);
    std::vector<float> query(dim, 4.0F);
    std::vector<uint8_t> codes(quantizer.GetCodeSize());
    std::vector<float> decoded(dim);

    REQUIRE(quantizer.Train(train.data(), 3));
    REQUIRE(quantizer.EncodeOne(query.data(), codes.data()));
    for (uint64_t i = 0; i < encoded_code_size; ++i) {
        REQUIRE(codes[code_offset + i] == 0);
    }
    REQUIRE(quantizer.DecodeOne(codes.data(), decoded.data()));
    for (auto value : decoded) {
        REQUIRE(std::abs(value - 3.0F) <= tolerance);
    }
}

}  // namespace vsag
