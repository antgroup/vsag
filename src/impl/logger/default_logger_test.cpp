
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

#include "default_logger.h"

#include <iostream>
#include <sstream>
#include <string>

#include "test_env.h"
#include "unittest.h"
#include "vsag/logger.h"

namespace {

using vsag::test::ScopedEnv;

class ScopedStreamCapture {
public:
    explicit ScopedStreamCapture(std::ostream& stream)
        : stream_(stream), old_buffer_(stream.rdbuf()) {
        stream_.rdbuf(buffer_.rdbuf());
    }

    ~ScopedStreamCapture() {
        stream_.rdbuf(old_buffer_);
    }

    [[nodiscard]] std::string
    str() const {
        return buffer_.str();
    }

private:
    std::ostream& stream_;
    std::streambuf* old_buffer_;
    std::ostringstream buffer_;
};

}  // namespace

TEST_CASE("DefaultLogger Basic Test", "[ut][logger]") {
    vsag::DefaultLogger logger;
    logger.SetLevel(vsag::Logger::Level::kTRACE);
    logger.Trace("this is a trace level message");
    logger.Debug("this is a debug level message");
    logger.Info("this is a info level message");
    logger.Warn("this is a warn level message");
    logger.Error("this is a error level message");
    logger.Critical("this is a critical level message");
}

TEST_CASE("DefaultLogger uses VSAG_LOG_LEVEL", "[ut][logger]") {
    ScopedEnv env("VSAG_LOG_LEVEL", "debug");
    vsag::DefaultLogger logger;

    ScopedStreamCapture capture(std::cout);
    logger.Debug("debug from env");

    REQUIRE(capture.str().find("debug from env") != std::string::npos);
}

TEST_CASE("DefaultLogger accepts warning alias in VSAG_LOG_LEVEL", "[ut][logger]") {
    ScopedEnv env("VSAG_LOG_LEVEL", "warning");
    vsag::DefaultLogger logger;

    ScopedStreamCapture stdout_capture(std::cout);
    logger.Info("info filtered by warning env");
    REQUIRE(stdout_capture.str().find("info filtered by warning env") == std::string::npos);

    ScopedStreamCapture stderr_capture(std::cerr);
    logger.Warn("warn from warning env");
    REQUIRE(stderr_capture.str().find("warn from warning env") != std::string::npos);
}

TEST_CASE("DefaultLogger ignores invalid VSAG_LOG_LEVEL", "[ut][logger]") {
    ScopedEnv env("VSAG_LOG_LEVEL", "verbose");
    vsag::DefaultLogger logger;

    ScopedStreamCapture capture(std::cout);
    logger.Debug("debug filtered by invalid env");
    logger.Info("info from invalid env fallback");

    REQUIRE(capture.str().find("debug filtered by invalid env") == std::string::npos);
    REQUIRE(capture.str().find("info from invalid env fallback") != std::string::npos);
}

TEST_CASE("DefaultLogger supports off in VSAG_LOG_LEVEL", "[ut][logger]") {
    ScopedEnv env("VSAG_LOG_LEVEL", "off");
    vsag::DefaultLogger logger;

    ScopedStreamCapture stdout_capture(std::cout);
    logger.Info("info filtered by off env");
    REQUIRE(stdout_capture.str().find("info filtered by off env") == std::string::npos);

    ScopedStreamCapture stderr_capture(std::cerr);
    logger.Critical("critical filtered by off env");
    REQUIRE(stderr_capture.str().find("critical filtered by off env") == std::string::npos);
}

TEST_CASE("DefaultLogger SetLevel overrides VSAG_LOG_LEVEL", "[ut][logger]") {
    ScopedEnv env("VSAG_LOG_LEVEL", "critical");
    vsag::DefaultLogger logger;
    logger.SetLevel(vsag::Logger::Level::kDEBUG);

    ScopedStreamCapture capture(std::cout);
    logger.Debug("debug from explicit level");

    REQUIRE(capture.str().find("debug from explicit level") != std::string::npos);
}
