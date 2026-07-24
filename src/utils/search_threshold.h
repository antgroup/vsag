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
#include <cstring>
#include <optional>
#include <string>

#include "common.h"
#include "json_types.h"
#include "vsag/dataset.h"

namespace vsag {

inline constexpr const char* SEARCH_THRESHOLD = "threshold";

inline std::optional<float>
ParseSearchThreshold(const std::string& parameters) {
    if (parameters.empty()) {
        return std::nullopt;
    }
    const auto json = JsonType::Parse(parameters);
    if (not json.Contains(SEARCH_THRESHOLD)) {
        return std::nullopt;
    }
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
        return static_cast<T*>(allocator->Allocate(sizeof(T) * count));
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
    std::vector<int64_t> ids;
    std::vector<float> distances;
    for (int64_t i = 0; i < input->GetDim(); ++i) {
        if (input->GetDistances()[i] <= threshold.value()) {
            ids.push_back(input->GetIds()[i]);
            distances.push_back(input->GetDistances()[i]);
            if (max_results > 0 and static_cast<int64_t>(ids.size()) == max_results) {
                break;
            }
        }
    }
    auto result = Dataset::Make();
    auto* result_ids = AllocateThresholdArray<int64_t>(ids.size(), allocator);
    auto* result_distances = AllocateThresholdArray<float>(distances.size(), allocator);
    if (not ids.empty()) {
        std::copy(ids.begin(), ids.end(), result_ids);
        std::copy(distances.begin(), distances.end(), result_distances);
    }
    result->NumElements(1)
        ->Dim(static_cast<int64_t>(ids.size()))
        ->Ids(result_ids)
        ->Distances(result_distances)
        ->Owner(true, allocator);
    if (input->GetExtraInfoSize() > 0 and input->GetExtraInfos() != nullptr) {
        auto extra_size = input->GetExtraInfoSize();
        auto* extra_infos =
            AllocateThresholdArray<char>(static_cast<uint64_t>(ids.size()) * extra_size, allocator);
        int64_t result_index = 0;
        for (int64_t i = 0; i < input->GetDim(); ++i) {
            if (input->GetDistances()[i] <= threshold.value() and
                result_index < static_cast<int64_t>(ids.size())) {
                std::memcpy(extra_infos + result_index * extra_size,
                            input->GetExtraInfos() + i * extra_size,
                            extra_size);
                ++result_index;
            }
        }
        result->ExtraInfos(extra_infos)->ExtraInfoSize(extra_size);
    }
    return result;
}

}  // namespace vsag
