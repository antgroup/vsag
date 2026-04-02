
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

#include "scalar_quantizer.h"

#include "quantization/quantizer_test.h"

using namespace vsag;

TEST_CASE("SQ4Quantizer Encode and Decode", "[ut][SQ4Quantizer]") {
    auto dims = fixtures::get_common_used_dims();
    const std::vector<int> counts = {10, 101};

    auto config = QuantizerTestConfig<SQ4Quantizer>()
                      .with_name("SQ4Quantizer")
                      .with_error_func([](int64_t dim) { return 2 * 1.0f / 15.0f; })
                      .with_error_same_func([](int64_t dim) { return (float)(dim * 15 * 0.01); })
                      .with_code_max(15);

    RunQuantizerEncodeDecodeTests(dims, counts, config);
}

TEST_CASE("SQ4Quantizer Compute", "[ut][SQ4Quantizer]") {
    auto dims = fixtures::get_common_used_dims();
    const std::vector<int> counts = {10, 101};

    auto config = QuantizerTestConfig<SQ4Quantizer>()
                      .with_name("SQ4Quantizer")
                      .with_compute_error_func([](int64_t dim) { return 1.0F * dim; });

    RunQuantizerComputeTests(dims, counts, config);
}

TEST_CASE("SQ4Quantizer Serialize and Deserialize", "[ut][SQ4Quantizer]") {
    auto dims = fixtures::get_common_used_dims();
    const std::vector<int> counts = {10, 101};

    auto config = QuantizerTestConfig<SQ4Quantizer>()
                      .with_name("SQ4Quantizer")
                      .with_serialize_error_func([](int64_t dim) { return 50.0F * dim; })
                      .with_related_error(1.0F);

    RunQuantizerSerializeTests(dims, counts, config);
}

TEST_CASE("SQ8Quantizer Encode and Decode", "[ut][SQ8Quantizer]") {
    auto dims = fixtures::get_common_used_dims();
    const std::vector<int> counts = {10, 101};

    auto config = QuantizerTestConfig<SQ8Quantizer>()
                      .with_name("SQ8Quantizer")
                      .with_error(1e-2f)
                      .with_error_same_func([](int64_t dim) { return (float)(dim * 255 * 0.01); })
                      .with_code_max(255);

    RunQuantizerEncodeDecodeTests(dims, counts, config);
}

TEST_CASE("SQ8Quantizer Compute", "[ut][SQ8Quantizer]") {
    auto dims = fixtures::get_common_used_dims();
    const std::vector<int> counts = {10, 101};

    auto config =
        QuantizerTestConfig<SQ8Quantizer>().with_name("SQ8Quantizer").with_compute_error(10.0F);

    RunQuantizerComputeTests(dims, counts, config);
}

TEST_CASE("SQ8Quantizer Serialize and Deserialize", "[ut][SQ8Quantizer]") {
    auto dims = fixtures::get_common_used_dims();
    const std::vector<int> counts = {10, 101};

    auto config = QuantizerTestConfig<SQ8Quantizer>()
                      .with_name("SQ8Quantizer")
                      .with_serialize_error(10.0F)
                      .with_related_error(1.0F);

    RunQuantizerSerializeTests(dims, counts, config);
}
