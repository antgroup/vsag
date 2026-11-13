// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <catch2/generators/catch_generators.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace fixtures::daily {

// Only an explicit, process-local Daily plan enables parameter selection.
// PR/Branch/release invocations retain Catch2's original Cartesian product.
template <typename... Args>
auto
Choices(uint64_t line, Args&&... args) {
    auto generator = Catch::Generators::makeGenerators(std::forward<Args>(args)...);
    const auto* text = std::getenv("VSAG_DAILY_CHOICES");
    using T = typename decltype(generator)::type;
    if (text == nullptr) {
        return Catch::Generators::GeneratorWrapper<T>(
            Catch::Detail::make_unique<Catch::Generators::Generators<T>>(std::move(generator)));
    }
    const auto choices = nlohmann::json::parse(text);
    const auto key = std::to_string(line);
    if (!choices.is_object() || !choices.contains(key) || !choices[key].is_number_unsigned()) {
        throw std::invalid_argument("Daily plan is missing a generator choice");
    }
    const auto index = choices[key].get<uint64_t>();
    for (uint64_t i = 0; i < index; ++i) {
        if (!generator.next()) {
            throw std::invalid_argument("Daily generator choice is out of range");
        }
    }
    std::cerr << "DAILY_CHOICE " << key << " " << index << std::endl;
    return Catch::Generators::value(generator.get());
}

// Catch2 captures stdout/stderr until a case finishes. Flush a separate trace so a
// timed-out or killed process still leaves its last entered operation (#3225).
inline void
WritePhaseEvent(nlohmann::json event) {
    event["unix_seconds"] =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    if (const auto* path = std::getenv("VSAG_DAILY_PHASE_FILE")) {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        std::ofstream output(path, std::ios::app);
        output << event.dump() << std::endl;
        if (!output) {
            throw std::runtime_error("Cannot write Daily phase trace");
        }
    }
    std::cerr << "DAILY_PHASE " << event << std::endl;
}

class Phase {
public:
    explicit Phase(const std::string& name, nlohmann::json details = nlohmann::json::object())
        : name_(name),
          details_(std::move(details)),
          start_(std::chrono::steady_clock::now()),
          enabled_(std::getenv("VSAG_DAILY_EVENTS") != nullptr) {
        if (enabled_) {
            WritePhaseEvent({{"phase", name_}, {"event", "start"}, {"details", details_}});
        }
    }
    ~Phase() noexcept {
        if (enabled_) {
            const auto seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
            try {
                WritePhaseEvent({{"phase", name_}, {"event", "end"}, {"seconds", seconds}});
            } catch (const std::exception& error) {
                // Do not terminate while unwinding a real assertion failure. The trace
                // retains its unmatched start and the runner reports the incomplete phase.
                std::cerr << "Daily phase trace error: " << error.what() << std::endl;
            }
        }
    }

private:
    std::string name_;
    nlohmann::json details_;
    std::chrono::steady_clock::time_point start_;
    bool enabled_;
};

}  // namespace fixtures::daily

#define DAILY_GENERATE(...) GENERATE(fixtures::daily::Choices(__LINE__, __VA_ARGS__))
