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

#include <optional>
#include <utility>

#include "vsag/dataset.h"
#include "vsag/search_metrics.h"

namespace vsag {

class SearchMetricsDatasetAccessor {
public:
    virtual ~SearchMetricsDatasetAccessor() = default;

    virtual void
    SetSearchMetricsInternal(SearchResultMetrics metrics) = 0;

    [[nodiscard]] virtual std::optional<SearchResultMetrics>
    GetSearchMetricsInternal() const = 0;
};

[[nodiscard]] inline bool
AttachSearchMetrics(const DatasetPtr& dataset, SearchResultMetrics metrics) {
    // dataset keeps the object alive for the full accessor call. A mutable cast is required because
    // attachment updates the internal result state; the public getter uses the const accessor path.
    auto* accessor = dynamic_cast<SearchMetricsDatasetAccessor*>(dataset.get());
    if (accessor == nullptr) {
        return false;
    }
    accessor->SetSearchMetricsInternal(std::move(metrics));
    return true;
}

}  // namespace vsag
