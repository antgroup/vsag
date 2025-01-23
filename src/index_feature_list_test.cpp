
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

#include "index_feature_list.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("IndexFeatureList Set and Check Test", "[ut][IndexFeatureList]") {
    using namespace vsag;
    IndexFeatureList list;
    auto count = static_cast<uint32_t>(IndexFeature::INDEX_FEATURE_COUNT);
    for (uint32_t i = 1; i < count; ++i) {
        REQUIRE(list.CheckFeature(static_cast<IndexFeature>(i)) == false);
        list.SetFeature(static_cast<IndexFeature>(i));
        REQUIRE(list.CheckFeature(static_cast<IndexFeature>(i)) == true);
        list.SetFeature(static_cast<IndexFeature>(i), false);
        REQUIRE(list.CheckFeature(static_cast<IndexFeature>(i)) == false);
    }

    std::vector<int> status(count, 0);

    auto func_set = [&]() {
        int64_t times = random() % 100;
        while (times > 0) {
            bool val = random() % 2;
            int64_t key = random() % (count - 1) + 1;
            list.SetFeature(static_cast<IndexFeature>(key), val);
            status[key] = val ? 1 : 0;
            --times;
        }
    };

    for (uint32_t i = 0; i < 10; ++i) {
        func_set();
        for (uint32_t j = 1; j < count; ++j) {
            bool gt = (status[j] != 0);
            REQUIRE(gt == list.CheckFeature(static_cast<IndexFeature>(j)));
        }
    }
}
