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

#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "unittest.h"
#include "vsag/factory.h"

namespace vsag {

TEST_CASE("IVF precise quantizers train on the full dataset", "[ut][ivf][full_training]") {
    const auto dtype =
        GENERATE(std::string("float32"), std::string("float16"), std::string("bfloat16"));
    const auto layout = GENERATE(std::string("flat"), std::string("bucket"));
    const int64_t sample_count = GENERATE(int64_t{512}, int64_t{513});
    CAPTURE(dtype, layout, sample_count);

    // Every vector owns one unique coordinate. Any 512-vector sample necessarily misses
    // one coordinate, so sampled SQ8 training loses that vector regardless of the RNG seed.
    // All values (zero and one) are exactly representable in float32, fp16 and bf16.
    constexpr int64_t count = 513;
    constexpr int64_t dim = count;
    std::vector<float> vectors(count * dim, 0.0F);
    std::vector<uint16_t> native_vectors(count * dim, 0);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    for (int64_t i = 0; i < count; ++i) {
        vectors[i * dim + i] = 1.0F;
        native_vectors[i * dim + i] = dtype == "float16" ? 0x3c00 : 0x3f80;
    }
    auto base = Dataset::Make()->Dim(dim)->NumElements(count)->Ids(ids.data())->Owner(false);
    if (dtype == "float32") {
        base->Float32Vectors(vectors.data());
    } else {
        base->Float16Vectors(native_vectors.data());
    }
    const auto parameters = std::string(R"({"dtype":")") + dtype +
                            R"(","metric_type":"l2","dim":513,"index_param":{
            "buckets_count":1,"buckets_per_data":1,"ivf_train_type":"random",
            "base_quantization_type":"sq8","use_reorder":true,
            "precise_quantization_type":"sq8","thread_count":1,
            "precise_codes_layout":")" +
                            layout + R"(","train_sample_count":)" + std::to_string(sample_count) +
                            "}}";
    auto created = Factory::CreateIndex("ivf", parameters);
    REQUIRE(created.has_value());
    auto index = created.value();
    REQUIRE(index->Build(base).has_value());

    int64_t missed_basic_vectors = 0;
    for (int64_t i = 0; i < count; ++i) {
        const auto* query = vectors.data() + i * dim;
        auto precise_distance = index->CalcDistanceById(query, ids[i], true);
        REQUIRE(precise_distance.has_value());
        REQUIRE(std::abs(precise_distance.value()) < 1e-5F);
        auto basic_distance = index->CalcDistanceById(query, ids[i], false);
        REQUIRE(basic_distance.has_value());
        if (basic_distance.value() > 0.5F) {
            ++missed_basic_vectors;
        }
    }
    // The fix must not silently train the basic quantizer on the full dataset as well.
    REQUIRE(missed_basic_vectors == count - sample_count);
}

}  // namespace vsag
