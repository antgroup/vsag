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

#include "caq_encoder.h"

#include <array>
#include <vector>

#include "unittest.h"

using namespace vsag;

TEST_CASE("CAQ preserves established RaBitQ scalar codes", "[ut][CAQ]") {
    const std::vector<float> data{0.37F,
                                  -0.91F,
                                  0.08F,
                                  0.72F,
                                  -0.24F,
                                  0.53F,
                                  -0.66F,
                                  0.12F,
                                  -0.43F,
                                  0.99F,
                                  -0.03F,
                                  0.28F,
                                  -0.57F,
                                  0.81F,
                                  -0.16F,
                                  0.46F};
    const uint64_t bits = GENERATE(2, 4, 8);
    const uint64_t rounds = GENERATE(1, 6, 32);
    // Fixed outputs from the original RaBitQ coordinate-adjustment encoder.
    const std::vector<uint8_t> expected =
        bits == 2 ? std::vector<uint8_t>{2, 0, 2, 3, 1, 3, 0, 2, 1, 3, 1, 2, 0, 3, 1, 2}
        : bits == 4
            ? std::vector<uint8_t>{10, 0, 8, 13, 6, 12, 2, 8, 4, 15, 7, 10, 3, 14, 6, 11}
            : std::vector<uint8_t>{
                  175, 10, 138, 220, 97, 196, 42, 143, 72, 255, 124, 164, 54, 232, 107, 187};
    const float expected_norm = bits == 2 ? 4.2426405F : bits == 4 ? 17.2626762F : 280.830902F;
    std::vector<uint8_t> codes(data.size());
    const float norm = CodeAdjustmentQuantize(data.data(), data.size(), bits, codes.data(), rounds);
    REQUIRE(codes == expected);
    REQUIRE(std::abs(norm - expected_norm) <= 1e-5F);
}

TEST_CASE("CAQ supports every SAQ bit width and improves alignment", "[ut][CAQ]") {
    constexpr uint64_t dim = 67;
    const uint64_t bits = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
    std::vector<float> data(dim);
    for (uint64_t d = 0; d < dim; ++d) {
        data[d] = std::sin(static_cast<float>(d) * 0.37F) +
                  0.2F * std::cos(static_cast<float>(d) * 0.11F);
    }
    std::vector<uint16_t> initial(dim);
    std::vector<uint16_t> adjusted(dim);
    CodeAdjustmentQuantize(data.data(), dim, bits, initial.data(), 0);
    const float norm = CodeAdjustmentQuantize(data.data(), dim, bits, adjusted.data());
    const uint32_t code_max = (1U << bits) - 1U;
    const double center = 0.5 * code_max;
    auto score = [&](const std::vector<uint16_t>& codes) {
        double inner = 0.0;
        double squared_norm = 0.0;
        for (uint64_t d = 0; d < dim; ++d) {
            const double value = codes[d] - center;
            inner += data[d] * value;
            squared_norm += value * value;
            REQUIRE(codes[d] <= code_max);
        }
        return inner * inner / squared_norm;
    };
    REQUIRE(score(adjusted) + 1e-8 >= score(initial));
    REQUIRE(std::isfinite(norm));
    REQUIRE(norm > 0.0F);

    std::vector<uint16_t> repeated(dim);
    REQUIRE(CodeAdjustmentQuantize(data.data(), dim, bits, repeated.data(), DEFAULT_CAQ_ROUNDS) ==
            norm);
    REQUIRE(repeated == adjusted);
    if (bits <= 8) {
        std::vector<uint8_t> narrow(dim);
        REQUIRE(CodeAdjustmentQuantize(data.data(), dim, bits, narrow.data()) == norm);
        for (uint64_t d = 0; d < dim; ++d) {
            REQUIRE(narrow[d] == adjusted[d]);
        }
    } else {
        REQUIRE(*std::max_element(adjusted.begin(), adjusted.end()) > 255);
    }
}

TEST_CASE("CAQ handles zero vectors and rejects incompatible code widths", "[ut][CAQ]") {
    const std::array<float, 4> data{};
    std::array<uint8_t, 4> narrow{};
    std::array<uint16_t, 4> wide{};
    REQUIRE(CodeAdjustmentQuantize(data.data(), data.size(), 4, narrow.data()) == 1.0F);
    REQUIRE(std::all_of(narrow.begin(), narrow.end(), [](uint8_t code) { return code == 7; }));
    REQUIRE(CodeAdjustmentQuantize(data.data(), data.size(), 13, wide.data()) == 1.0F);
    REQUIRE(std::all_of(wide.begin(), wide.end(), [](uint16_t code) { return code == 4095; }));
    REQUIRE_THROWS_AS(CodeAdjustmentQuantize(data.data(), data.size(), 0, narrow.data()),
                      VsagException);
    REQUIRE_THROWS_AS(CodeAdjustmentQuantize(data.data(), data.size(), 9, narrow.data()),
                      VsagException);
    REQUIRE_THROWS_AS(CodeAdjustmentQuantize(data.data(), data.size(), 17, wide.data()),
                      VsagException);
}
