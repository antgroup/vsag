
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

#include "rabitq_quantizer_parameter.h"

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

using namespace vsag;

struct RaBitQDefaultParam {
    int pca_dim = 256;
    int rabitq_bits_per_dim_query = 4;
    bool use_fht = false;
};

std::string
generate_rabitq_param(const RaBitQDefaultParam& param) {
    static constexpr auto param_str = R"(
        {{
            "pca_dim": {},
            "rabitq_bits_per_dim_query": {},
            "use_fht": {}
        }}
    )";
    return fmt::format(param_str, param.pca_dim, param.rabitq_bits_per_dim_query, param.use_fht);
}

#define TEST_COMPATIBILITY_CASE(section_name, param_member, val1, val2, expect_compatible) \
    SECTION(section_name) {                                                                \
        RaBitQDefaultParam param1;                                                         \
        RaBitQDefaultParam param2;                                                         \
        param1.param_member = val1;                                                        \
        param2.param_member = val2;                                                        \
        auto param_str1 = generate_rabitq_param(param1);                                   \
        auto param_str2 = generate_rabitq_param(param2);                                   \
        auto rabitq_param1 = std::make_shared<vsag::RaBitQuantizerParameter>();            \
        auto rabitq_param2 = std::make_shared<vsag::RaBitQuantizerParameter>();            \
        rabitq_param1->FromString(param_str1);                                             \
        rabitq_param2->FromString(param_str2);                                             \
        if (expect_compatible) {                                                           \
            REQUIRE(rabitq_param1->CheckCompatibility(rabitq_param2));                     \
        } else {                                                                           \
            REQUIRE_FALSE(rabitq_param1->CheckCompatibility(rabitq_param2));               \
        }                                                                                  \
    }

TEST_CASE("RaBitQ Quantizer Parameter CheckCompatibility", "[ut][RaBitQuantizerParameter]") {
    SECTION("wrong parameter type") {
        RaBitQDefaultParam default_param;
        auto param_str = generate_rabitq_param(default_param);
        auto param = std::make_shared<vsag::RaBitQuantizerParameter>();
        param->FromString(param_str);
        REQUIRE(param->CheckCompatibility(param));
        REQUIRE_FALSE(param->CheckCompatibility(std::make_shared<vsag::EmptyParameter>()));
    }
    TEST_COMPATIBILITY_CASE("different pac_dim", pca_dim, 256, 512, false)
    TEST_COMPATIBILITY_CASE(
        "different rabitq_bits_per_dim_query", rabitq_bits_per_dim_query, 4, 8, false)
    TEST_COMPATIBILITY_CASE("different use_fht", use_fht, true, false, false)
}
