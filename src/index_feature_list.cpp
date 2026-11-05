
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

#include "index_feature_list.h"

#include <fmt/format.h>

#include <stdexcept>

#include "vsag_exception.h"

namespace vsag {

static std::pair<uint32_t, uint32_t>
get_pos(const uint32_t val) {
    return {val / 8, val % 8};
}

IndexFeatureList::IndexFeatureList()
    : feature_count_(static_cast<uint32_t>(IndexFeature::INDEX_FEATURE_COUNT)) {
    uint32_t size = (feature_count_ + 7) / 8;
    data_.resize(size, 0);
}

bool
IndexFeatureList::CheckFeature(IndexFeature feature) const {
    auto val = static_cast<uint32_t>(feature);
    if (val >= feature_count_ or val == 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("wrong feature value: {}", val));
    }

    auto [x, y] = get_pos(val);
    return (data_[x] & (1U << y)) != 0U;
}

void
IndexFeatureList::SetFeature(vsag::IndexFeature feature, bool val) {
    if (static_cast<uint32_t>(feature) >= feature_count_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "wrong feature");
    }

    auto [x, y] = get_pos(static_cast<uint32_t>(feature));
    if (val) {
        data_[x] |= (1U << y);
    } else {
        data_[x] &= ~(1U << y);
    }
}

void
IndexFeatureList::SetFeatures(const std::vector<IndexFeature>& features, bool val) {
    for (auto feature : features) {
        this->SetFeature(feature, val);
    }
}

}  // namespace vsag
