
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
#include <sstream>
#include <string>

#include "default_logger.h"
#include "vsag/logger.h"
#include "vsag/vsag.h"

namespace fixtures::logger {

class TestLogger : public vsag::Logger {
public:
    inline void
    Log(const std::string& msg, Level level) {
        switch (level) {
            case Level::kTRACE: {
                Trace(msg);
                break;
            }
            case Level::kDEBUG: {
                Debug(msg);
                break;
            }
            case Level::kINFO: {
                Info(msg);
                break;
            }
            case Level::kWARN: {
                Warn(msg);
                break;
            }
            case Level::kERR: {
                Error(msg);
                break;
            }
            case Level::kCRITICAL: {
                Critical(msg);
                break;
            }
            default: {
                // will not run into here
                break;
            }
        }
    }

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

class LoggerStream : public std::basic_streambuf<char> {
public:
    explicit LoggerStream(TestLogger* logger,
                          vsag::Logger::Level level,
                          uint64_t buffer_size = 1024)
        : logger_(logger), level_(level), buffer_(buffer_size + 1) {
        auto base = &buffer_.front();
        this->setp(base, base + buffer_size);
    }

    virtual ~LoggerStream() {
        logger_ = nullptr;
    }

public:
    virtual int
    overflow(int ch) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ch != EOF) {
            *this->pptr() = (char)ch;
            this->pbump(1);
        }
        this->flush();
        return ch;
    }

    virtual int
    sync() override {
        std::lock_guard<std::mutex> lock(mutex_);
        this->flush();
        return 0;
    }

private:
    void
    flush() {
        std::ptrdiff_t n = this->pptr() - this->pbase();
        std::string msg(this->pbase(), n);
        this->pbump(-n);
        if (logger_) {
            logger_->Log(msg, level_);
        }
    }

private:
    TestLogger* logger_ = nullptr;
    vsag::Logger::Level level_;
    std::mutex mutex_;
    std::vector<char> buffer_;
    uint64_t size_;
};

extern TestLogger test_logger;
extern std::basic_ostream<char> trace;
extern std::basic_ostream<char> debug;
extern std::basic_ostream<char> info;
extern std::basic_ostream<char> warn;
extern std::basic_ostream<char> error;
extern std::basic_ostream<char> critical;

// catch2 logger is NOT supported to be used in multi-threading tests, so
//  we need to replace it at the start of all the test cases in this file
class LoggerReplacer {
public:
    LoggerReplacer() {
        origin_logger_ = vsag::Options::Instance().logger();
        vsag::Options::Instance().set_logger(&logger_);
    }

    ~LoggerReplacer() {
        vsag::Options::Instance().set_logger(origin_logger_);
    }

private:
    vsag::Logger* origin_logger_;
    vsag::DefaultLogger logger_;
};

}  // namespace fixtures::logger
