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

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "json_types.h"

namespace vsag {

inline bool
IsJsonParameterCacheable(const std::string& parameters) {
    constexpr uint64_t kMaxCachedJsonParameterBytes = 4096;
    return static_cast<uint64_t>(parameters.size()) <= kMaxCachedJsonParameterBytes;
}

inline const JsonType*
GetCachedJsonParameter(const std::string& parameters) {
    if (not IsJsonParameterCacheable(parameters)) {
        return nullptr;
    }

    struct JsonParameterCache {
        std::string parameters;
        std::optional<JsonType> json;
    };
    thread_local JsonParameterCache cache;

    if (cache.json.has_value() && cache.parameters == parameters) {
        return &cache.json.value();
    }

    auto parsed = JsonType::Parse(parameters);
    cache.parameters = parameters;
    cache.json.emplace(std::move(parsed));
    return &cache.json.value();
}

inline const JsonType&
GetOrParseJsonParameter(const std::string& parameters, std::optional<JsonType>& uncached) {
    if (const auto* cached = GetCachedJsonParameter(parameters); cached != nullptr) {
        return *cached;
    }

    // Oversized parameters belong to the caller, not the lifetime of a worker thread.
    uncached.emplace(JsonType::Parse(parameters));
    return *uncached;
}

}  // namespace vsag
