
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

#include "logger.h"

#include <algorithm>
#include <regex>
#include <string>
#include <vector>

#include "test_env.h"
#include "unittest.h"
#include "vsag/options.h"
#include "vsag/vsag.h"
namespace {

class CollectLogger : public vsag::Logger {
public:
    void
    SetLevel(Level log_level) override {
        level_ = log_level;
    }

    void
    Trace(const std::string& msg) override {
        messages_.emplace_back("trace:" + msg);
    }

    void
    Debug(const std::string& msg) override {
        messages_.emplace_back("debug:" + msg);
    }

    void
    Info(const std::string& msg) override {
        messages_.emplace_back("info:" + msg);
    }

    void
    Warn(const std::string& msg) override {
        messages_.emplace_back("warn:" + msg);
    }

    void
    Error(const std::string& msg) override {
        messages_.emplace_back("error:" + msg);
    }

    void
    Critical(const std::string& msg) override {
        messages_.emplace_back("critical:" + msg);
    }

    Level level_{Level::kINFO};
    std::vector<std::string> messages_;
};

class LoggerReplacer {
public:
    explicit LoggerReplacer(vsag::Logger* logger)
        : origin_logger_(vsag::Options::Instance().logger()) {
        vsag::Options::Instance().set_logger(logger);
    }

    ~LoggerReplacer() {
        vsag::Options::Instance().set_logger(origin_logger_);
    }

private:
    vsag::Logger* origin_logger_;
};

using vsag::test::ScopedEnv;

}  // namespace

TEST_CASE("Logger Test", "[ut][logger]") {
    vsag::logger::set_level(vsag::logger::level::trace);
    vsag::logger::trace("this is a trace level message");
    vsag::logger::debug("this is a debug level message");
    vsag::logger::info("this is a info level message");
    vsag::logger::warn("this is a warn level message");
    vsag::logger::error("this is a error level message");
    vsag::logger::critical("this is a critical level message");
}

TEST_CASE("Logger Format Test", "[ut][logger]") {
    CollectLogger logger;
    LoggerReplacer logger_replacer(&logger);

    vsag::logger::set_level(vsag::logger::level::debug);
    vsag::logger::info("Welcome {}", "logger");
    vsag::logger::warn("Easy padding in numbers like {:08d}", 12);
    vsag::logger::critical("Support for int: {0:d}; hex: {0:x}; oct: {0:o}; bin: {0:b}", 42);

    REQUIRE(logger.level_ == vsag::Logger::Level::kDEBUG);
    REQUIRE(logger.messages_.size() == 3);
    REQUIRE(logger.messages_[0] == "info:Welcome logger");
    REQUIRE(logger.messages_[1] == "warn:Easy padding in numbers like 00000012");
    REQUIRE(logger.messages_[2] == "critical:Support for int: 42; hex: 2a; oct: 52; bin: 101010");
}

TEST_CASE("Init logs startup output by default", "[ut][logger]") {
    ScopedEnv env("VSAG_SUPPRESS_INIT_BANNER");
    CollectLogger logger;
    LoggerReplacer logger_replacer(&logger);

    REQUIRE(vsag::init());

    auto startup_message =
        std::find_if(logger.messages_.begin(), logger.messages_.end(), [](const auto& message) {
            return message.find("info:\n====vsag start init====") != std::string::npos;
        });
    REQUIRE(startup_message != logger.messages_.end());
    REQUIRE(std::regex_search(*startup_message,
                              std::regex(R"(\ninstance spec: [0-9]+C([0-9]+|\?)G\n)")));
    REQUIRE(startup_message->find("\ncpu sve >> ") != std::string::npos);
}

TEST_CASE("Init suppresses startup output when requested", "[ut][logger]") {
    const std::vector<std::string> suppress_values{
        "1", "on", "ON", "On", "oN", "true", "TRUE", "True", "TrUe"};

    for (const auto& suppress_value : suppress_values) {
        ScopedEnv env("VSAG_SUPPRESS_INIT_BANNER", suppress_value.c_str());
        CollectLogger logger;
        LoggerReplacer logger_replacer(&logger);

        REQUIRE(vsag::init());

        auto startup_message =
            std::find_if(logger.messages_.begin(), logger.messages_.end(), [](const auto& message) {
                return message.find("====vsag start init====") != std::string::npos;
            });
        REQUIRE(startup_message == logger.messages_.end());
    }
}
