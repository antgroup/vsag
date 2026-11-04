// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <string>

#include "framework/daily_test_plan.h"

namespace {
class DailyChoicesEnvironment {
public:
    explicit DailyChoicesEnvironment(const char* text) {
        const auto* old = std::getenv("VSAG_DAILY_CHOICES");
        had_old_ = old != nullptr;
        old_ = old == nullptr ? "" : old;
        if (text == nullptr) {
            unsetenv("VSAG_DAILY_CHOICES");
        } else {
            setenv("VSAG_DAILY_CHOICES", text, 1);
        }
    }
    ~DailyChoicesEnvironment() {
        if (had_old_) {
            setenv("VSAG_DAILY_CHOICES", old_.c_str(), 1);
        } else {
            unsetenv("VSAG_DAILY_CHOICES");
        }
    }

private:
    bool had_old_{false};
    std::string old_;
};
}  // namespace

TEST_CASE("Daily choices preserve the default generator product", "[ft][test_budget][daily_plan]") {
    DailyChoicesEnvironment environment(nullptr);
    auto values = fixtures::daily::Choices(10, false, true);
    REQUIRE_FALSE(values.get());
    REQUIRE(values.next());
    REQUIRE(values.get());
    REQUIRE_FALSE(values.next());
}

TEST_CASE("Daily choices select exactly one declared value", "[ft][test_budget][daily_plan]") {
    DailyChoicesEnvironment environment(R"({"10":1})");
    auto values = fixtures::daily::Choices(10, std::string("nsw"), std::string("pipnn"));
    REQUIRE(values.get() == "pipnn");
    REQUIRE_FALSE(values.next());
}

TEST_CASE("Daily choices reject missing or invalid configuration",
          "[ft][test_budget][daily_plan]") {
    for (const auto* text : {"{}", R"({"10":-1})", R"({"10":2})", R"({"10":"1"})", "null"}) {
        DailyChoicesEnvironment environment(text);
        REQUIRE_THROWS(fixtures::daily::Choices(10, false, true));
    }
}
