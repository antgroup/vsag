
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

#include <cmath>

#include "parameter_test.h"
#include "unittest.h"
using namespace vsag;

struct RaBitQDefaultParam {
    int pca_dim = 256;
    std::string rabitq_version = "standard";
    int rabitq_bits_per_dim_query = 4;
    int rabitq_bits_per_dim_base = 1;
    float rabitq_error_rate = RaBitQuantizerParameter::DEFAULT_RABITQ_ERROR_RATE;
    bool use_fht = false;
};

std::string
generate_rabitq_param(const RaBitQDefaultParam& param) {
    static constexpr auto param_str = R"(
        {{
            "pca_dim": {},
            "rabitq_version": "{}",
            "rabitq_bits_per_dim_query": {},
            "rabitq_bits_per_dim_base": {},
            "rabitq_error_rate": {},
            "use_fht": {}
        }}
    )";
    return fmt::format(param_str,
                       param.pca_dim,
                       param.rabitq_version,
                       param.rabitq_bits_per_dim_query,
                       param.rabitq_bits_per_dim_base,
                       param.rabitq_error_rate,
                       param.use_fht);
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

TEST_CASE("RaBitQ Quantizer Parameter CheckCompatibility", "[ut][RaBitQuantizerParameter]"){
    SECTION("wrong parameter type"){RaBitQDefaultParam default_param;
auto param_str = generate_rabitq_param(default_param);
auto param = std::make_shared<vsag::RaBitQuantizerParameter>();
param->FromString(param_str);
REQUIRE(param->CheckCompatibility(param));
REQUIRE_FALSE(param->CheckCompatibility(std::make_shared<vsag::EmptyParameter>()));
}
TEST_COMPATIBILITY_CASE("different pac_dim", pca_dim, 256, 512, false)
TEST_COMPATIBILITY_CASE(
    "different rabitq_bits_per_dim_query", rabitq_bits_per_dim_query, 4, 32, false)
TEST_COMPATIBILITY_CASE("different rabitq_bits_per_dim_base", rabitq_bits_per_dim_base, 1, 4, false)
SECTION("different rabitq_version") {
    RaBitQDefaultParam param1;
    RaBitQDefaultParam param2;
    param2.rabitq_version = "split_1bit_7bit";
    param2.rabitq_bits_per_dim_query = 32;
    param2.rabitq_bits_per_dim_base = 8;
    auto rabitq_param1 = std::make_shared<vsag::RaBitQuantizerParameter>();
    auto rabitq_param2 = std::make_shared<vsag::RaBitQuantizerParameter>();
    rabitq_param1->FromString(generate_rabitq_param(param1));
    rabitq_param2->FromString(generate_rabitq_param(param2));
    REQUIRE_FALSE(rabitq_param1->CheckCompatibility(rabitq_param2));
}
TEST_COMPATIBILITY_CASE("different rabitq_error_rate", rabitq_error_rate, 1.9F, 1.0F, false)
TEST_COMPATIBILITY_CASE("different use_fht", use_fht, true, false, false)
}

TEST_CASE("RaBitQ Quantizer Parameter Defaults", "[ut][RaBitQuantizerParameter]") {
    auto param = std::make_shared<vsag::RaBitQuantizerParameter>();
    param->FromString(R"({"type":"rabitq"})");
    REQUIRE(param->rabitq_version_ == RaBitQuantizerParameter::DEFAULT_RABITQ_VERSION);
    REQUIRE(std::abs(param->rabitq_error_rate_ -
                     RaBitQuantizerParameter::DEFAULT_RABITQ_ERROR_RATE) < 1e-5F);
}

TEST_CASE("RaBitQ Split Version Parameter", "[ut][RaBitQuantizerParameter]") {
    auto rabitq_bits_per_dim_base = GENERATE(1, 8);
    RaBitQDefaultParam default_param;
    default_param.rabitq_version = RaBitQuantizerParameter::RABITQ_VERSION_SPLIT_1BIT_7BIT;
    default_param.rabitq_bits_per_dim_query = 32;
    default_param.rabitq_bits_per_dim_base = rabitq_bits_per_dim_base;
    default_param.rabitq_error_rate = 1.25F;
    auto param = std::make_shared<vsag::RaBitQuantizerParameter>();
    param->FromString(generate_rabitq_param(default_param));
    REQUIRE(param->rabitq_version_ == RaBitQuantizerParameter::RABITQ_VERSION_SPLIT_1BIT_7BIT);
    REQUIRE(param->num_bits_per_dim_base_ == rabitq_bits_per_dim_base);
    REQUIRE(std::abs(param->rabitq_error_rate_ - 1.25F) < 1e-5F);
}

TEST_CASE("Wrong rabitq_bits_per_dim_base parameter", "[ut][RaBitQuantizerParameter]") {
    auto wrong_rabitq_bits_per_dim_base = GENERATE(0, 9);
    RaBitQDefaultParam default_param;
    default_param.rabitq_bits_per_dim_base = wrong_rabitq_bits_per_dim_base;
    auto param_str = generate_rabitq_param(default_param);
    auto param = std::make_shared<vsag::RaBitQuantizerParameter>();
    REQUIRE_THROWS(param->FromString(param_str));
}

TEST_CASE("Wrong rabitq_bits_per_dim_query parameter", "[ut][RaBitQuantizerParameter]") {
    auto wrong_rabitq_bits_per_dim_query = GENERATE(2, 3, 6, 31);
    RaBitQDefaultParam default_param;
    default_param.rabitq_bits_per_dim_query = wrong_rabitq_bits_per_dim_query;
    auto param_str = generate_rabitq_param(default_param);
    auto param = std::make_shared<vsag::RaBitQuantizerParameter>();
    REQUIRE_THROWS(param->FromString(param_str));
}

TEST_CASE("Wrong rabitq_version parameter", "[ut][RaBitQuantizerParameter]") {
    RaBitQDefaultParam default_param;
    default_param.rabitq_version = "unknown";
    auto param = std::make_shared<vsag::RaBitQuantizerParameter>();
    REQUIRE_THROWS(param->FromString(generate_rabitq_param(default_param)));
}

TEST_CASE("Wrong rabitq split version shape", "[ut][RaBitQuantizerParameter]") {
    RaBitQDefaultParam default_param;
    default_param.rabitq_version = RaBitQuantizerParameter::RABITQ_VERSION_SPLIT_1BIT_7BIT;
    default_param.rabitq_bits_per_dim_query = 4;
    default_param.rabitq_bits_per_dim_base = 8;
    auto param = std::make_shared<vsag::RaBitQuantizerParameter>();
    REQUIRE_THROWS(param->FromString(generate_rabitq_param(default_param)));
}

TEST_CASE("Wrong rabitq_error_rate parameter", "[ut][RaBitQuantizerParameter]") {
    auto wrong_rabitq_error_rate = GENERATE(-1.0F, 0.0F);
    RaBitQDefaultParam default_param;
    default_param.rabitq_error_rate = wrong_rabitq_error_rate;
    auto param = std::make_shared<vsag::RaBitQuantizerParameter>();
    REQUIRE_THROWS(param->FromString(generate_rabitq_param(default_param)));
}
