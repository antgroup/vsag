

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

#include "impl/transform/mrle_transformer.h"

#include <catch2/catch_test_macros.hpp>

#include "impl/allocator/safe_allocator.h"

TEST_CASE("MRLETransformer Basic Operations", "[ut][mrle_transformer]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    int64_t input_dim = 4;
    int64_t output_dim = 4;

    SECTION("Transform with L2 metric") {
        vsag::MRLETransformer<vsag::MetricType::METRIC_TYPE_L2SQR> transformer(
            allocator.get(), input_dim, output_dim);

        float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float output[4];

        auto meta = transformer.Transform(input, output);
        REQUIRE(meta != nullptr);
        REQUIRE(output[0] == 1.0f);
        REQUIRE(output[1] == 2.0f);
        REQUIRE(output[2] == 3.0f);
        REQUIRE(output[3] == 4.0f);
    }

    SECTION("Transform with Cosine metric") {
        vsag::MRLETransformer<vsag::MetricType::METRIC_TYPE_COSINE> transformer(
            allocator.get(), input_dim, output_dim);

        float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float output[4];

        auto meta = transformer.Transform(input, output);
        REQUIRE(meta != nullptr);
    }

    SECTION("InverseTransform throws exception") {
        vsag::MRLETransformer<> transformer(allocator.get(), input_dim, output_dim);

        float input[4], output[4];
        REQUIRE_THROWS(transformer.InverseTransform(input, output));
    }

    SECTION("Serialize and Deserialize") {
        vsag::MRLETransformer<> transformer(allocator.get(), input_dim, output_dim);

        std::string buffer(1024, '\0');
        auto writer = [&buffer](uint64_t cur, uint64_t size, void* buf) {
            memcpy(buffer.data() + cur, buf, size);
            return;
        };
        WriteFuncStreamWriter stream_writer(writer, 0);
        transformer.Serialize(stream_writer);

        auto reader = [&buffer](uint64_t cur, uint64_t size, void* buf) {
            memcpy(buf, buffer.data() + cur, size);
            return;
        };
        ReadFuncStreamReader stream_reader(reader, 0, 1024);
        transformer.Deserialize(stream_reader);
    }

    SECTION("Train method") {
        vsag::MRLETransformer<> transformer(allocator.get(), input_dim, output_dim);
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        transformer.Train(data, 1);
    }
}
