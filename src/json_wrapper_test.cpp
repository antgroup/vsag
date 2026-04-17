#include "json_wrapper.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("JsonWrapper copy constructor preserves valid strings", "[ut][json]") {
    vsag::JsonWrapper param;
    param["index_param"].SetString("{\n    \"window_size\": 20000\n}");

    vsag::JsonWrapper basic_info;
    basic_info.SetJson(param);

    REQUIRE_NOTHROW(basic_info.Dump());
    REQUIRE(basic_info["index_param"].GetString() == "{\n    \"window_size\": 20000\n}");
}
