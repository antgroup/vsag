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

#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/internal/catch_context.hpp>
#include <stdexcept>

#include "core/test_budget.h"

namespace fixtures::budget {

inline CaseIdentity
CurrentCaseIdentity(const std::string& source_file) {
    auto name = Catch::getResultCapture().getCurrentTestName();
    if (name.empty() || source_file.empty()) {
        throw std::invalid_argument("test budget requires an active testcase and its source file");
    }
    // Catch's active capture exposes the name, not the registration line or tags. Do not invent
    // a helper-call line: reporting resolves this only against a unique binary/name/source match.
    return {name, "", source_file, 0};
}

inline Policy
CurrentPolicy(bool daily = false) {
    const auto* config = Catch::getCurrentContext().getConfig();
    if (config == nullptr) {
        throw std::invalid_argument("test budget requires an active Catch configuration");
    }
    auto policy = daily ? Policy::Daily() : Policy::PullRequest();
    policy.seed = config->rngSeed();
    return policy;
}

}  // namespace fixtures::budget
