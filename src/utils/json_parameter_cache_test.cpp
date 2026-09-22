// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "utils/json_parameter_cache.h"

#include "unittest.h"
#include "utils/search_threshold.h"

TEST_CASE("JSON parameter cache bounds and caller ownership", "[ut][json_parameter_cache]") {
    const std::string small = R"({"threshold":1})";
    std::optional<vsag::JsonType> first_storage;
    const auto& cached = vsag::GetOrParseJsonParameter(small, first_storage);
    CHECK_FALSE(first_storage.has_value());
    CHECK(&cached == vsag::GetCachedJsonParameter(small));

    // A valid JSON at the byte limit is cached; adding one space must bypass it.
    const auto boundary = small + std::string(4096 - small.size(), ' ');
    CHECK(vsag::GetCachedJsonParameter(boundary) != nullptr);
    CHECK(vsag::GetCachedJsonParameter(boundary + " ") == nullptr);

    const auto large = R"({"threshold":2,"padding":")" + std::string(5000, 'x') + R"("})";
    const auto& first = vsag::GetOrParseJsonParameter(large, first_storage);
    REQUIRE(first_storage.has_value());
    CHECK(&first == &first_storage.value());
    std::optional<vsag::JsonType> second_storage;
    const auto& second = vsag::GetOrParseJsonParameter(boundary + " ", second_storage);
    CHECK(first["threshold"].GetInt() == 2);
    CHECK(second["threshold"].GetInt() == 1);
    CHECK(vsag::ParseSearchThreshold(large).value() == 2.0F);
    CHECK(vsag::ParseSearchThreshold(small).value() == 1.0F);
    CHECK_THROWS(vsag::GetOrParseJsonParameter("{", second_storage));
    CHECK_THROWS(vsag::GetOrParseJsonParameter(std::string(5000, 'x'), second_storage));
    CHECK(vsag::GetOrParseJsonParameter(small, second_storage)["threshold"].GetInt() == 1);
}
