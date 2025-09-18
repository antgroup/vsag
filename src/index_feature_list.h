
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
#include <memory>
#include <vector>

#include "utils/pointer_define.h"
#include "vsag/index_features.h"

namespace vsag {

DEFINE_POINTER(IndexFeatureList);

class IndexFeatureList {
public:
    explicit IndexFeatureList();

    ~IndexFeatureList() = default;

    [[nodiscard]] bool
    CheckFeature(IndexFeature feature) const;

    void
    SetFeature(IndexFeature feature, bool val = true);

    void
    SetFeatures(const std::vector<IndexFeature>& features, bool val = true);

private:
    std::vector<uint8_t> data_{};

    const uint32_t feature_count_{0};
};
}  // namespace vsag
