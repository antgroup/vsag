
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
#include <string>

#include "dataset.h"
#include "filter.h"

namespace vsag {
enum class SearchMode {
    KNN_SEARCH = 1,
    RANGE_SEARCH = 2,
};

class SearchRequest {
public:
    DatasetPtr query_{nullptr};
    SearchMode mode_{SearchMode::KNN_SEARCH};
    int64_t topk_;
    std::string params_str_;

    bool enable_attribute_filter_{false};
    std::string attribute_filter_str_;
    FilterPtr filter_{nullptr};
};

}  // namespace vsag
