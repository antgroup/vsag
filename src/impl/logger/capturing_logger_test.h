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

#pragma once

#include <string>
#include <vector>

#include "vsag/logger.h"
#include "vsag/options.h"

/// Reading what a component traced, for the cases where the trace is the only
/// observable difference. Two tests need it, so it lives here rather than in
/// either of them.
namespace vsag::testing {

/// Keeps every line, whatever the level, so a test can look for its own.
class CapturingLogger : public Logger {
public:
    void
    SetLevel(Level /*level*/) override {
    }
    void
    Trace(const std::string& msg) override {
        lines.push_back(msg);
    }
    void
    Debug(const std::string& msg) override {
        lines.push_back(msg);
    }
    void
    Info(const std::string& msg) override {
        lines.push_back(msg);
    }
    void
    Warn(const std::string& msg) override {
        lines.push_back(msg);
    }
    void
    Error(const std::string& msg) override {
        lines.push_back(msg);
    }
    void
    Critical(const std::string& msg) override {
        lines.push_back(msg);
    }

    std::vector<std::string> lines;
};

/// Installs a logger for a scope and puts the previous one back. A destructor
/// rather than a line at the end of the test, because Catch2's REQUIRE throws and
/// would leave Options holding a pointer to a destroyed object.
class ScopedLogger {
public:
    explicit ScopedLogger(Logger* logger) : previous_(Options::Instance().logger()) {
        Options::Instance().set_logger(logger);
    }

    ~ScopedLogger() {
        Options::Instance().set_logger(previous_);
    }

    ScopedLogger(const ScopedLogger&) = delete;
    ScopedLogger&
    operator=(const ScopedLogger&) = delete;

private:
    Logger* previous_{nullptr};
};

}  // namespace vsag::testing
