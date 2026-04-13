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
 * @file logger.h
 * @brief Logger interface and utility functions for the vsag library.
 */

#pragma once

#include <fmt/format.h>

#include "vsag/logger.h"
#include "vsag/options.h"

namespace vsag {
namespace logger {

/**
 * @brief Enumeration of log levels.
 */
enum class level {
    /// Trace level for detailed debugging information.
    trace = Logger::Level::kTRACE,
    /// Debug level for debugging information.
    debug = Logger::Level::kDEBUG,
    /// Info level for informational messages.
    info = Logger::Level::kINFO,
    /// Warning level for warning messages.
    warn = Logger::Level::kWARN,
    /// Error level for error messages.
    err = Logger::Level::kERR,
    /// Critical level for critical error messages.
    critical = Logger::Level::kCRITICAL,
    /// Off level to disable logging.
    off = Logger::Level::kOFF
};

/**
 * @brief Sets the global log level.
 *
 * @param log_level The log level to set.
 */
void
set_level(level log_level);

/**
 * @brief Logs a trace level message.
 *
 * @param msg The message to log.
 */
void
trace(const std::string& msg);

/**
 * @brief Logs a debug level message.
 *
 * @param msg The message to log.
 */
void
debug(const std::string& msg);

/**
 * @brief Logs an info level message.
 *
 * @param msg The message to log.
 */
void
info(const std::string& msg);

/**
 * @brief Logs a warning level message.
 *
 * @param msg The message to log.
 */
void
warn(const std::string& msg);

/**
 * @brief Logs an error level message.
 *
 * @param msg The message to log.
 */
void
error(const std::string& msg);

/**
 * @brief Logs a critical level message.
 *
 * @param msg The message to log.
 */
void
critical(const std::string& msg);

/**
 * @brief Logs a trace level message with format.
 *
 * @tparam Args Format argument types.
 * @param fmt Format string.
 * @param args Format arguments.
 */
template <typename... Args>
void
trace(fmt::format_string<Args...> fmt, Args&&... args) {
    trace(fmt::format(fmt, std::forward<Args>(args)...));
}

/**
 * @brief Logs a debug level message with format.
 *
 * @tparam Args Format argument types.
 * @param fmt Format string.
 * @param args Format arguments.
 */
template <typename... Args>
void
debug(fmt::format_string<Args...> fmt, Args&&... args) {
    debug(fmt::format(fmt, std::forward<Args>(args)...));
}

/**
 * @brief Logs an info level message with format.
 *
 * @tparam Args Format argument types.
 * @param fmt Format string.
 * @param args Format arguments.
 */
template <typename... Args>
void
info(fmt::format_string<Args...> fmt, Args&&... args) {
    info(fmt::format(fmt, std::forward<Args>(args)...));
}

/**
 * @brief Logs a warning level message with format.
 *
 * @tparam Args Format argument types.
 * @param fmt Format string.
 * @param args Format arguments.
 */
template <typename... Args>
void
warn(fmt::format_string<Args...> fmt, Args&&... args) {
    warn(fmt::format(fmt, std::forward<Args>(args)...));
}

/**
 * @brief Logs an error level message with format.
 *
 * @tparam Args Format argument types.
 * @param fmt Format string.
 * @param args Format arguments.
 */
template <typename... Args>
void
error(fmt::format_string<Args...> fmt, Args&&... args) {
    error(fmt::format(fmt, std::forward<Args>(args)...));
}

/**
 * @brief Logs a critical level message with format.
 *
 * @tparam Args Format argument types.
 * @param fmt Format string.
 * @param args Format arguments.
 */
template <typename... Args>
void
critical(fmt::format_string<Args...> fmt, Args&&... args) {
    critical(fmt::format(fmt, std::forward<Args>(args)...));
}

}  // namespace logger
}  // namespace vsag