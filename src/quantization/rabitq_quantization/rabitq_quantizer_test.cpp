
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

#include "rabitq_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "default_allocator.h"
#include "fixtures.h"
#include "quantization/quantizer_test.h"
#include "safe_allocator.h"

using namespace vsag;

const auto dims = fixtures::get_common_used_dims();
const auto counts = {10, 100};

TEST_CASE("RaBitQ Encode and Decode", "[ut][RaBitQuantizer]") {
    for (auto dim : dims) {
        for (auto count : counts) {
            // Generate centroid and data
            assert(count % 2 == 0);
            auto centroid = fixtures::generate_vectors(1, dim, false, 114514);
            std::vector<float> vecs(dim * count);
            for (int64_t i = 0; i < count; ++i) {
                for (int64_t d = 0; d < dim; ++d) {
                    vecs[i * dim + d] = centroid[d] + (i % 2 == 0 ? i + 1 : -i);
                }
            }

            // Init quantizer
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(dim, allocator.get());
            quantizer.ReTrain(vecs.data(), count);

            // Test EncodeOne & DecodeOne
            for (uint64_t i = 0; i < count; ++i) {
                std::vector<uint8_t> codes(quantizer.GetCodeSize());
                quantizer.EncodeOne(vecs.data() + i * dim, codes.data());
                for (uint64_t d = 0; d < dim; ++d) {
                    bool ge = vecs[i * dim + d] >= centroid[d];
                    bool bit = ((codes[d / 8] >> (d % 8)) & 1) != 0;
                    REQUIRE(ge == bit);
                }

                std::vector<float> out_vec(dim);
                quantizer.DecodeOne(codes.data(), out_vec.data());
                for (uint64_t d = 0; d < dim; ++d) {
                    REQUIRE(vecs[i * dim + d] * out_vec[d] >= 0);
                }
            }

            // Test EncodeBatch & DecodeBatch
            std::vector<uint8_t> codes(quantizer.GetCodeSize() * count);
            quantizer.EncodeBatch(vecs.data(), codes.data(), count);
            std::vector<float> out_vec(dim * count);
            quantizer.DecodeBatch(codes.data(), out_vec.data(), count);
            for (int64_t i = 0; i < dim * count; ++i) {
                REQUIRE(vecs[i] * out_vec[i] >= 0);
            }
        }
    }
}

TEST_CASE("RaBitQ Compute", "[ut][RaBitQuantizer]") {
    for (auto dim : dims) {
        float numeric_error = 4.0 / std::sqrt(dim) * dim;
        for (auto count : counts) {
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> quantizer(dim, allocator.get());
            TestComputer<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                         MetricType::METRIC_TYPE_L2SQR>(quantizer, dim, count, numeric_error);
        }
    }
}
