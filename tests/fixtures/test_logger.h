
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

#include <catch2/catch_message.hpp>
#include <mutex>

#include "vsag/vsag.h"

namespace fixtures {

class TestLogger : public vsag::Logger {
public:
    inline void
    SetLevel(Level log_level) override {
        std::lock_guard<std::mutex> lock(mutex_);
        level_ = log_level - vsag::Logger::Level::kTRACE;
    }

    inline void
    Trace(const std::string& msg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level_ <= 0) {
            UNSCOPED_INFO("[test-logger]::[trace] " + msg);
        }
    }

    inline void
    Debug(const std::string& msg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level_ <= 1) {
            UNSCOPED_INFO("[test-logger]::[debug] " + msg);
        }
    }

    inline void
    Info(const std::string& msg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level_ <= 2) {
            UNSCOPED_INFO("[test-logger]::[info] " + msg);
        }
    }

    inline void
    Warn(const std::string& msg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level_ <= 3) {
            UNSCOPED_INFO("[test-logger]::[warn] " + msg);
        }
    }

    inline void
    Error(const std::string& msg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level_ <= 4) {
            UNSCOPED_INFO("[test-logger]::[error] " + msg);
        }
    }

    void
    Critical(const std::string& msg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level_ <= 5) {
            UNSCOPED_INFO("[test-logger]::[critical] " + msg);
        }
    }

private:
    int64_t level_ = 0;
    std::mutex mutex_;
};

extern TestLogger logger;

}  // namespace fixtures
