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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <optional>
#include <string>

#include "common.h"
#include "json_types.h"
#include "vsag/dataset.h"

namespace vsag {

inline constexpr const char* SEARCH_THRESHOLD = "threshold";

inline const JsonType*
GetCachedSearchParameters(const std::string& parameters) {
    constexpr uint64_t kMaxCachedParameterBytes = 4096;
    if (static_cast<uint64_t>(parameters.size()) > kMaxCachedParameterBytes) {
        return nullptr;
    }

    struct SearchParameterCache {
        std::string parameters;
        std::optional<JsonType> json;
        bool valid{false};
    };
    thread_local SearchParameterCache cache;

    if (cache.valid && cache.parameters == parameters) {
        return &cache.json.value();
    }

    cache.valid = false;
    cache.parameters = parameters;
    cache.json.emplace(JsonType::Parse(parameters));
    cache.valid = true;
    return &cache.json.value();
}

inline std::optional<float>
ParseSearchThreshold(const std::string& parameters) {
    if (parameters.empty()) {
        return std::nullopt;
    }
    const auto* cached_json = GetCachedSearchParameters(parameters);
    std::optional<JsonType> uncached_json;
    if (cached_json == nullptr) {
        uncached_json.emplace(JsonType::Parse(parameters));
        cached_json = &uncached_json.value();
    }
    const auto& json = *cached_json;
    if (not json.Contains(SEARCH_THRESHOLD)) {
        return std::nullopt;
    }
    CHECK_ARGUMENT(json[SEARCH_THRESHOLD].IsNumber(), "search threshold must be a number");
    const auto threshold = json[SEARCH_THRESHOLD].GetFloat();
    CHECK_ARGUMENT(std::isfinite(threshold), "search threshold must be finite");
    return threshold;
}

inline void
ValidateSearchThreshold(const std::optional<float>& threshold) {
    if (threshold.has_value()) {
        CHECK_ARGUMENT(std::isfinite(threshold.value()), "search threshold must be finite");
    }
}

template <typename T>
inline T*
AllocateThresholdArray(uint64_t count, Allocator* allocator) {
    if (count == 0) {
        return nullptr;
    }
    if (allocator != nullptr) {
        auto* result = static_cast<T*>(allocator->Allocate(sizeof(T) * count));
        if (result == nullptr) {
            throw std::bad_alloc();
        }
        return result;
    }
    return new T[count];
}

inline DatasetPtr
FilterDatasetByThreshold(const DatasetPtr& input,
                         const std::optional<float>& threshold,
                         Allocator* allocator = nullptr,
                         int64_t max_results = -1) {
    if (not threshold.has_value()) {
        return input;
    }
    // Count first so allocator-owned output arrays are exact-sized while preserving result order
    // and extra-info alignment without a temporary owning container.
    int64_t result_count = 0;
    for (int64_t i = 0; i < input->GetDim(); ++i) {
        if (std::isfinite(input->GetDistances()[i]) and
            input->GetDistances()[i] <= threshold.value()) {
            ++result_count;
            if (max_results > 0 and result_count == max_results) {
                break;
            }
        }
    }
    auto result = Dataset::Make();
    result->NumElements(1)->Owner(true, allocator);
    auto* result_ids = AllocateThresholdArray<int64_t>(result_count, allocator);
    result->Dim(result_count)->Ids(result_ids);
    auto* result_distances = AllocateThresholdArray<float>(result_count, allocator);
    result->Distances(result_distances);
    const auto extra_size = input->GetExtraInfoSize();
    const auto* input_extra_infos = input->GetExtraInfos();
    char* extra_infos = nullptr;
    if (result_count > 0 and extra_size > 0 and input_extra_infos != nullptr) {
        extra_infos = AllocateThresholdArray<char>(static_cast<uint64_t>(result_count) * extra_size,
                                                   allocator);
        result->ExtraInfos(extra_infos)->ExtraInfoSize(extra_size);
    }
    int64_t result_index = 0;
    for (int64_t i = 0; i < input->GetDim() and result_index < result_count; ++i) {
        if (std::isfinite(input->GetDistances()[i]) and
            input->GetDistances()[i] <= threshold.value()) {
            result_ids[result_index] = input->GetIds()[i];
            result_distances[result_index] = input->GetDistances()[i];
            if (extra_infos != nullptr) {
                std::memcpy(extra_infos + result_index * extra_size,
                            input_extra_infos + i * extra_size,
                            extra_size);
            }
            ++result_index;
        }
    }
    if (result_count == 0) {
        result->ExtraInfos(nullptr)->ExtraInfoSize(0);
    }
    result->Statistics(input->GetStatistics())->Reasoning(input->GetReasoning());
    return result;
}

}  // namespace vsag
