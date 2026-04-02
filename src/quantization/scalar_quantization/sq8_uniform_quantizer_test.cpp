
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

#include "sq8_uniform_quantizer.h"

#include "quantization/quantizer_test.h"

using namespace vsag;

TEST_CASE("SQ8UniformQuantizer Encode and Decode", "[ut][SQ8UniformQuantizer]") {
    auto dims = fixtures::get_common_used_dims();
    const std::vector<int> counts = {10, 101};

    auto config = QuantizerTestConfig<SQ8UniformQuantizer>()
                      .with_name("SQ8UniformQuantizer")
                      .with_error_func([](int64_t dim) { return 2 * 1.0f / 255.0f; })
                      .with_error_same_func([](int64_t dim) { return (float)(dim * 255 * 0.01); })
                      .with_code_max(255);

    RunQuantizerEncodeDecodeTests(dims, counts, config);
}

TEST_CASE("SQ8UniformQuantizer Compute", "[ut][SQ8UniformQuantizer]") {
    auto dims = fixtures::get_common_used_dims();
    const std::vector<int> counts = {10, 101};

    auto config = QuantizerTestConfig<SQ8UniformQuantizer>()
                      .with_name("SQ8UniformQuantizer")
                      .with_compute_error(4 * 1.0f / 255.0f)
                      .with_compute_codes_same()
                      .with_code_max(255);

    RunQuantizerComputeTests(dims, counts, config);
}

TEST_CASE("SQ8UniformQuantizer Serialize and Deserialize", "[ut][SQ8UniformQuantizer]") {
    auto dims = fixtures::get_common_used_dims();
    const std::vector<int> counts = {10, 101};

    auto config = QuantizerTestConfig<SQ8UniformQuantizer>()
                      .with_name("SQ8UniformQuantizer")
                      .with_serialize_error_func([](int64_t dim) { return 4 * 1.0f / 255.0f; });

    RunQuantizerSerializeTests(dims, counts, config);
}
