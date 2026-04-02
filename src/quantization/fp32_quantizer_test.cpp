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

#include "quantization/quantizer_test.h"

using namespace vsag;

TEST_CASE("FP32Quantizer Encode and Decode", "[ut][FP32Quantizer]") {
    const std::vector<int> dims = {64, 128};
    const std::vector<int> counts = {10, 101};
    constexpr float error = 2e-5f;

    auto config = QuantizerTestConfig<FP32Quantizer>()
                      .with_name("FP32Quantizer")
                      .with_error(error)
                      .without_encode_decode_same();

    RunQuantizerEncodeDecodeTests(dims, counts, config);
}

TEST_CASE("FP32Quantizer Compute", "[ut][FP32Quantizer]") {
    const std::vector<int> dims = {64, 128};
    const std::vector<int> counts = {10, 101};
    constexpr float error = 2e-5f;

    auto config = QuantizerTestConfig<FP32Quantizer>()
                      .with_name("FP32Quantizer")
                      .with_error(error)
                      .with_code_max(65536)
                      .with_compute_codes_same();

    RunQuantizerComputeTests(dims, counts, config);
}

TEST_CASE("FP32Quantizer Serialize and Deserialize", "[ut][FP32Quantizer]") {
    const std::vector<int> dims = {64, 128};
    const std::vector<int> counts = {10, 101};
    constexpr float error = 2e-5f;

    auto config = QuantizerTestConfig<FP32Quantizer>().with_name("FP32Quantizer").with_error(error);

    RunQuantizerSerializeTests(dims, counts, config);
}