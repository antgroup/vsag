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

#include "bf16_quantizer.h"

#include "quantization/quantizer_test.h"

using namespace vsag;

TEST_CASE("BF16 Encode and Decode", "[ut][BF16Quantizer]") {
    auto dims = fixtures::get_common_used_dims(3, 225);
    const std::vector<int> counts = {10, 101};

    auto config = QuantizerTestConfig<BF16Quantizer>()
                      .with_name("BF16Quantizer")
                      .with_error(6e-3f)
                      .with_code_max(65536);

    RunQuantizerEncodeDecodeTests(dims, counts, config);
}

TEST_CASE("BF16 Compute", "[ut][BF16Quantizer]") {
    auto dims = fixtures::get_common_used_dims(3, 225);
    const std::vector<int> counts = {10, 101};

    auto config = QuantizerTestConfig<BF16Quantizer>()
                      .with_name("BF16Quantizer")
                      .with_error(6e-3f)
                      .with_code_max(65536)
                      .with_unbounded_flag(true);

    RunQuantizerComputeTests(dims, counts, config);
}

TEST_CASE("BF16 Serialize and Deserialize", "[ut][BF16Quantizer]") {
    auto dims = fixtures::get_common_used_dims(3, 225);
    const std::vector<int> counts = {10, 101};

    auto config = QuantizerTestConfig<BF16Quantizer>()
                      .with_name("BF16Quantizer")
                      .with_error(6e-3f)
                      .with_unbounded_flag(true);

    RunQuantizerSerializeTests(dims, counts, config);
}