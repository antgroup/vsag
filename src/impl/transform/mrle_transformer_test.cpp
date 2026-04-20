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

#include "mrle_transformer.h"

#include <cmath>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"
using namespace vsag;

static constexpr float EPSILON = 1e-5F;

TEST_CASE("MRLETransformer L2 Transform Test", "[ut][MRLETransformer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    int64_t dim = 16;
    MRLETransformer<MetricType::METRIC_TYPE_L2SQR> transformer(allocator.get(), dim, dim);

    std::vector<float> input(dim);
    std::vector<float> output(dim);

    for (int64_t i = 0; i < dim; ++i) {
        input[i] = static_cast<float>(i + 1);
    }

    auto meta = transformer.Transform(input.data(), output.data());
    REQUIRE(meta != nullptr);

    for (int64_t i = 0; i < dim; ++i) {
        REQUIRE(std::abs(output[i] - input[i]) < EPSILON);
    }
}

TEST_CASE("MRLETransformer Cosine Transform Test", "[ut][MRLETransformer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    int64_t dim = 16;
    MRLETransformer<MetricType::METRIC_TYPE_COSINE> transformer(allocator.get(), dim, dim);

    std::vector<float> input(dim);
    std::vector<float> output(dim);

    for (int64_t i = 0; i < dim; ++i) {
        input[i] = static_cast<float>(i + 1);
    }

    auto meta = transformer.Transform(input.data(), output.data());
    REQUIRE(meta != nullptr);

    float sum = 0.0F;
    for (int64_t i = 0; i < dim; ++i) {
        sum += output[i] * output[i];
    }
    REQUIRE(std::abs(sum - 1.0F) < EPSILON);
}

TEST_CASE("MRLETransformer InverseTransform Test", "[ut][MRLETransformer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    int64_t dim = 8;
    MRLETransformer<MetricType::METRIC_TYPE_L2SQR> transformer(allocator.get(), dim, dim);

    std::vector<float> input(dim);
    std::vector<float> output(dim);

    REQUIRE_THROWS_AS(transformer.InverseTransform(input.data(), output.data()), VsagException);
}
