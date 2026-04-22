
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

#ifdef _WIN32
#include <io.h>
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
#define STDERR_FILENO 2
#endif
#define isatty _isatty
#else
#include <unistd.h>
#endif

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace vsag {

namespace {

[[nodiscard]] std::string
format_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::system_clock::to_time_t(now);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm time_info{};
#ifdef _WIN32
    localtime_s(&time_info, &seconds);
#else
    localtime_r(&seconds, &time_info);
#endif

    std::ostringstream stream;
    stream << std::put_time(&time_info, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0')
           << std::setw(3) << milliseconds.count();
    return stream.str();
}

[[nodiscard]] std::ostream&
get_stream(Logger::Level log_level) {
    if (log_level >= Logger::Level::kWARN) {
        return std::cerr;
    }
    return std::cout;
}

[[nodiscard]] std::string_view
get_level_name(Logger::Level log_level) {
    switch (log_level) {
        case Logger::Level::kTRACE:
            return "trace";
        case Logger::Level::kDEBUG:
            return "debug";
        case Logger::Level::kINFO:
            return "info";
        case Logger::Level::kWARN:
            return "warn";
        case Logger::Level::kERR:
            return "error";
        case Logger::Level::kCRITICAL:
            return "critical";
        case Logger::Level::kOFF:
        case Logger::Level::kN_LEVELS:
            return "off";
    }

    return "unknown";
}

[[nodiscard]] int
get_stream_fd(Logger::Level log_level) {
    if (log_level >= Logger::Level::kWARN) {
        return STDERR_FILENO;
    }
    return STDOUT_FILENO;
}

[[nodiscard]] bool
supports_color(Logger::Level log_level) {
    static const auto check = [](int fd) {
        if (isatty(fd) == 0) {
            return false;
        }

        if (std::getenv("NO_COLOR") != nullptr) {
            return false;
        }

        const auto* term = std::getenv("TERM");
        return term != nullptr and std::string_view(term) != "dumb";
    };

    static const bool cout_supports_color = check(STDOUT_FILENO);
    static const bool cerr_supports_color = check(STDERR_FILENO);

    if (get_stream_fd(log_level) == STDERR_FILENO) {
        return cerr_supports_color;
    }
    return cout_supports_color;
}

[[nodiscard]] std::string_view
get_level_color(Logger::Level log_level) {
    switch (log_level) {
        case Logger::Level::kTRACE:
            return "\033[37m";
        case Logger::Level::kDEBUG:
            return "\033[36m";
        case Logger::Level::kINFO:
            return "\033[32m";
        case Logger::Level::kWARN:
            return "\033[33m";
        case Logger::Level::kERR:
            return "\033[31m";
        case Logger::Level::kCRITICAL:
            return "\033[1;31m";
        case Logger::Level::kOFF:
        case Logger::Level::kN_LEVELS:
            return "";
    }

    return "";
}

}  // namespace

void
DefaultLogger::SetLevel(Logger::Level log_level) {
    level_.store(static_cast<int>(log_level), std::memory_order_release);
}

bool
DefaultLogger::should_log(Logger::Level log_level) const {
    const auto current_level = static_cast<Logger::Level>(level_.load(std::memory_order_acquire));
    return current_level != Logger::Level::kOFF && log_level >= current_level;
}

void
DefaultLogger::log_message(Logger::Level log_level, std::string_view msg) {
    if (not this->should_log(log_level)) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto& stream = get_stream(log_level);
    stream << '[' << format_timestamp() << "] [";
    if (supports_color(log_level)) {
        stream << get_level_color(log_level) << get_level_name(log_level) << "\033[0m";
    } else {
        stream << get_level_name(log_level);
    }
    stream << "] " << msg << '\n';
}

void
DefaultLogger::Trace(const std::string& msg) {
    this->log_message(Logger::Level::kTRACE, msg);
}

void
DefaultLogger::Debug(const std::string& msg) {
    this->log_message(Logger::Level::kDEBUG, msg);
}

void
DefaultLogger::Info(const std::string& msg) {
    this->log_message(Logger::Level::kINFO, msg);
}

void
DefaultLogger::Warn(const std::string& msg) {
    this->log_message(Logger::Level::kWARN, msg);
}

void
DefaultLogger::Error(const std::string& msg) {
    this->log_message(Logger::Level::kERR, msg);
}

void
DefaultLogger::Critical(const std::string& msg) {
    this->log_message(Logger::Level::kCRITICAL, msg);
}

}  // namespace vsag
