
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

/**
 * @file test_logger.h
 * @brief Test logger implementation for functional tests.
 */

#pragma once

#include <catch2/catch_message.hpp>
#include <iostream>
#include <sstream>
#include <string>

#include "impl/logger/default_logger.h"
#include "vsag/logger.h"
#include "vsag/vsag.h"

namespace fixtures::logger {

/**
 * @class TestLogger
 * @brief Logger for tests that integrates with Catch2's UNSCOPED_INFO for test output.
 * Can output directly to stdout or through Catch2's reporting mechanism.
 */
class TestLogger : public vsag::Logger {
public:
    /**
     * @brief Sets whether to output directly to stdout.
     * @param output_directly If true, outputs to stdout; otherwise uses Catch2.
     */
    inline void
    OutputDirectly(bool output_directly) {
        output_directly_ = output_directly;
    }

public:
    /**
     * @brief Logs a message at the specified level.
     * @param msg The message to log.
     * @param level The log level (TRACE, DEBUG, INFO, WARN, ERR, CRITICAL).
     */
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
#define OUTPUT_DIRECTLY_OR(test_logger_output, level)                          \
    if (output_directly_) {                                                    \
        std::cout << "[test-logger]::[" << #level << "] " << msg << std::endl; \
    } else {                                                                   \
        test_logger_output;                                                    \
    }

    inline void
    SetLevel(Level log_level) override {
        std::lock_guard<std::mutex> lock(mutex_);
        level_ = log_level - vsag::Logger::Level::kTRACE;
    }

    /**
     * @brief Logs a trace-level message.
     * @param msg The message to log.
     */
    inline void
    Trace(const std::string& msg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level_ <= 0) {
            OUTPUT_DIRECTLY_OR(UNSCOPED_INFO("[test-logger]::[trace] " + msg), trace);
        }
    }

    /**
     * @brief Logs a debug-level message.
     * @param msg The message to log.
     */
    inline void
    Debug(const std::string& msg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level_ <= 1) {
            OUTPUT_DIRECTLY_OR(UNSCOPED_INFO("[test-logger]::[debug] " + msg), debug);
        }
    }

    /**
     * @brief Logs an info-level message.
     * @param msg The message to log.
     */
    inline void
    Info(const std::string& msg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level_ <= 2) {
            OUTPUT_DIRECTLY_OR(UNSCOPED_INFO("[test-logger]::[info] " + msg), info);
        }
    }

    /**
     * @brief Logs a warning-level message.
     * @param msg The message to log.
     */
    inline void
    Warn(const std::string& msg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level_ <= 3) {
            OUTPUT_DIRECTLY_OR(UNSCOPED_INFO("[test-logger]::[warn] " + msg), warn);
        }
    }

    /**
     * @brief Logs an error-level message.
     * @param msg The message to log.
     */
    inline void
    Error(const std::string& msg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level_ <= 4) {
            OUTPUT_DIRECTLY_OR(UNSCOPED_INFO("[test-logger]::[error] " + msg), error);
        }
    }

    /**
     * @brief Logs a critical-level message.
     * @param msg The message to log.
     */
    void
    Critical(const std::string& msg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level_ <= 5) {
            OUTPUT_DIRECTLY_OR(UNSCOPED_INFO("[test-logger]::[critical] " + msg), critical);
        }
    }

private:
    int64_t level_ = 0;             // Current log level.
    std::mutex mutex_;              // Mutex for thread-safe logging.
    bool output_directly_ = false;  // Whether to output directly to stdout.
};

/**
 * @class LoggerStream
 * @brief Stream buffer for logging through TestLogger with configurable log level.
 */
class LoggerStream : public std::basic_streambuf<char> {
public:
    /**
     * @brief Constructs a LoggerStream with specified logger and level.
     * @param logger Pointer to the TestLogger instance.
     * @param level The log level for this stream.
     * @param buffer_size Size of the internal buffer.
     */
    explicit LoggerStream(TestLogger* logger,
                          vsag::Logger::Level level,
                          uint64_t buffer_size = 1024)
        : logger_(logger), level_(level), buffer_(buffer_size + 1) {
        auto base = &buffer_.front();
        this->setp(base, base + buffer_size);
    }

    /**
     * @brief Destructor that clears the logger pointer.
     */
    virtual ~LoggerStream() {
        logger_ = nullptr;
    }

public:
    /**
     * @brief Handles buffer overflow by flushing and adding the character.
     * @param ch The character to add.
     * @return The character added, or EOF.
     */
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

    /**
     * @brief Synchronizes the stream buffer by flushing.
     * @return 0 on success.
     */
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
    TestLogger* logger_ = nullptr;  // Pointer to the TestLogger instance.
    vsag::Logger::Level level_;     // Log level for this stream.
    std::mutex mutex_;              // Mutex for thread-safe operations.
    std::vector<char> buffer_;      // Internal buffer for stream data.
    uint64_t size_;                 // Size of the buffer.
};

extern TestLogger test_logger;             // Global test logger instance.
extern std::basic_ostream<char> trace;     // Stream for trace-level logging.
extern std::basic_ostream<char> debug;     // Stream for debug-level logging.
extern std::basic_ostream<char> info;      // Stream for info-level logging.
extern std::basic_ostream<char> warn;      // Stream for warn-level logging.
extern std::basic_ostream<char> error;     // Stream for error-level logging.
extern std::basic_ostream<char> critical;  // Stream for critical-level logging.

// catch2 logger is NOT supported to be used in multi-threading tests, so
//  we need to replace it at the start of all the test cases in this file
/**
 * @class LoggerReplacer
 * @brief RAII helper to temporarily replace the global logger for a test scope.
 */
class LoggerReplacer {
public:
    /**
     * @brief Constructs a LoggerReplacer, replacing the global logger.
     */
    LoggerReplacer() {
        origin_logger_ = vsag::Options::Instance().logger();
        vsag::Options::Instance().set_logger(&logger_);
    }

    /**
     * @brief Destructor that restores the original logger.
     */
    ~LoggerReplacer() {
        vsag::Options::Instance().set_logger(origin_logger_);
    }

private:
    vsag::Logger* origin_logger_;  // Pointer to the original logger.
    vsag::DefaultLogger logger_;   // Default logger instance to use.
};

}  // namespace fixtures::logger
