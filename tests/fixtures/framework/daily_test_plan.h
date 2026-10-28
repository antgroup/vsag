// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <catch2/generators/catch_generators.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
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

class Phase {
public:
    explicit Phase(const std::string& name)
        : name_(name),
          start_(std::chrono::steady_clock::now()),
          enabled_(std::getenv("VSAG_DAILY_EVENTS") != nullptr) {
        if (enabled_) {
            std::cerr << "DAILY_PHASE " << nlohmann::json({{"phase", name_}, {"event", "start"}})
                      << std::endl;
        }
    }
    ~Phase() {
        if (enabled_) {
            const auto seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
            std::cerr << "DAILY_PHASE "
                      << nlohmann::json({{"phase", name_}, {"event", "end"}, {"seconds", seconds}})
                      << std::endl;
        }
    }

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
    bool enabled_;
};

}  // namespace fixtures::daily

#define DAILY_GENERATE(...) GENERATE(fixtures::daily::Choices(__LINE__, __VA_ARGS__))
