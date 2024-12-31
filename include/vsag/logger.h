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

#include <memory>
#include <string>

namespace vsag {
/**
 * @class Logger
 * @brief An abstract base class for logging messages with different severity levels.
 *
 * The `Logger` class provides methods to log messages at various severity levels,
 * such as trace, debug, info, warning, error, and critical levels.
 */
class Logger {
public:
    /**
     * @enum Level
     * @brief Enumeration of logging levels.
     *
     * The logging levels define the severity of log messages.
     */
    enum Level : int {
        kTRACE = 0,     ///< Trace level for fine-grained debug information.
        kDEBUG = 1,     ///< Debug level for general debugging information.
        kINFO = 2,      ///< Info level for informational messages.
        kWARN = 3,      ///< Warn level for warning messages.
        kERR = 4,       ///< Error level for error messages.
        kCRITICAL = 5,  ///< Critical level for critical messages indicating severe failures.
        kOFF = 6,       ///< Off level to turn off logging.
        kN_LEVELS       ///< Number of log levels.
    };

    /**
     * @brief Sets the logging level.
     *
     * This function sets the current logging level to filter messages. Only messages
     * with a severity equal to or higher than the specified level will be logged.
     *
     * @param log_level The logging level to be set.
     */
    virtual void
    SetLevel(Level log_level) = 0;

    /**
     * @brief Logs a trace level message.
     *
     * This function logs a message at the trace level, which is typically used for
     * fine-grained, highly detailed, and low-level debugging information.
     *
     * @param msg The message to be logged.
     */
    virtual void
    Trace(const std::string& msg) = 0;

    /**
     * @brief Logs a debug level message.
     *
     * This function logs a message at the debug level, which is typically used for
     * general debugging information.
     *
     * @param msg The message to be logged.
     */
    virtual void
    Debug(const std::string& msg) = 0;

    /**
     * @brief Logs an info level message.
     *
     * This function logs a message at the info level, which is typically used for
     * informational messages about application operation.
     *
     * @param msg The message to be logged.
     */
    virtual void
    Info(const std::string& msg) = 0;

    /**
     * @brief Logs a warning level message.
     *
     * This function logs a message at the warning level, which is typically used for
     * potentially harmful situations or important precautions.
     *
     * @param msg The message to be logged.
     */
    virtual void
    Warn(const std::string& msg) = 0;

    /**
     * @brief Logs an error level message.
     *
     * This function logs a message at the error level, which is typically used for
     * error events that might allow the application to continue running.
     *
     * @param msg The message to be logged.
     */
    virtual void
    Error(const std::string& msg) = 0;

    /**
     * @brief Logs a critical level message.
     *
     * This function logs a message at the critical level, which is typically used for
     * very severe error events that might lead the application to abort.
     *
     * @param msg The message to be logged.
     */
    virtual void
    Critical(const std::string& msg) = 0;

public:
    virtual ~Logger() = default;
};

}  // namespace vsag
