
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

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "functest.h"
#include "vsag/vsag.h"

static std::tm*
LocalTime(std::time_t time_value, std::tm* tm_buf) {
#if defined(_WIN32)
    return localtime_s(tm_buf, &time_value) == 0 ? tm_buf : nullptr;
#else
    return localtime_r(&time_value, tm_buf);
#endif
}

static std::string
GetCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tm_buf{};
    if (LocalTime(time_t, &tm_buf) == nullptr) {
        return "[0000-00-00 00:00:00.000]";
    }
    std::ostringstream oss;
    oss << "[" << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0')
        << std::setw(3) << ms.count() << "]";
    return oss.str();
}

struct LogListener : Catch::EventListenerBase {
    using EventListenerBase::EventListenerBase;
    ~LogListener() override {
    }

    void
    testCaseStarting(Catch::TestCaseInfo const& testInfo) override {
        start_ = std::chrono::high_resolution_clock::now();
        std::cout << GetCurrentTimestamp() << " " << testInfo.name << ": Test Begin" << std::endl;
    }

    void
    testCaseEnded(Catch::TestCaseStats const& testCaseStats) override {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << GetCurrentTimestamp() << " " << std::fixed << std::setprecision(3)
                  << testCaseStats.testInfo->name << ": Executed in "
                  << (duration > 10000 ? "\033[31m" : "") << duration / 1000.0F << "\033[0m"
                  << " seconds" << std::endl;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

CATCH_REGISTER_LISTENER(LogListener);

int
main(int argc, char** argv) {
    // your setup ...

    fixtures::logger::test_logger.SetLevel(vsag::Logger::Level::kWARN);
    fixtures::logger::test_logger.OutputDirectly(false);

    vsag::Options::Instance().set_logger(&fixtures::logger::test_logger);

    int result = Catch::Session().run(argc, argv);

    // your clean-up...

    return result;
}
