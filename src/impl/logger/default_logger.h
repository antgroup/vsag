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
 * @file default_logger.h
 * @brief Default logger implementation for the vsag library.
 */

#pragma once

#include <atomic>
#include <mutex>
#include <string_view>

#include "vsag/logger.h"
#include "vsag/options.h"

namespace vsag {

/**
 * @brief Default logger implementation that outputs to stdout.
 *
 * This logger provides thread-safe logging with configurable log levels.
 * Messages below the current level are filtered out.
 */
class DefaultLogger : public Logger {
public:
    /**
     * @brief Sets the log level.
     *
     * @param log_level The log level to set.
     */
    void
    SetLevel(Logger::Level log_level) override;

    /**
     * @brief Logs a trace level message.
     *
     * @param msg The message to log.
     */
    void
    Trace(const std::string& msg) override;

    /**
     * @brief Logs a debug level message.
     *
     * @param msg The message to log.
     */
    void
    Debug(const std::string& msg) override;

    /**
     * @brief Logs an info level message.
     *
     * @param msg The message to log.
     */
    void
    Info(const std::string& msg) override;

    /**
     * @brief Logs a warning level message.
     *
     * @param msg The message to log.
     */
    void
    Warn(const std::string& msg) override;

    /**
     * @brief Logs an error level message.
     *
     * @param msg The message to log.
     */
    void
    Error(const std::string& msg) override;

    /**
     * @brief Logs a critical level message.
     *
     * @param msg The message to log.
     */
    void
    Critical(const std::string& msg) override;

private:
    /**
     * @brief Checks if a message at the given level should be logged.
     *
     * @param log_level The level to check.
     * @return true if the message should be logged, false otherwise.
     */
    [[nodiscard]] bool
    should_log(Logger::Level log_level) const;

    /**
     * @brief Logs a message with the specified level.
     *
     * @param log_level The log level.
     * @param msg The message to log.
     */
    void
    log_message(Logger::Level log_level, std::string_view msg);

    /// Current log level (atomic for thread-safety).
    std::atomic<int> level_{Logger::Level::kINFO};
    /// Mutex for synchronized output.
    std::mutex mutex_;
};

}  // namespace vsag