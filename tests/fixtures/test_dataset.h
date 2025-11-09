
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

#include <functional>
#include <vector>

#include "vsag/dataset.h"
namespace fixtures {

class TestDataset {
public:
    using DatasetPtr = vsag::DatasetPtr;

    int64_t id_shift = 16;

    static std::shared_ptr<TestDataset>
    CreateTestDataset(uint64_t dim,
                      uint64_t count,
                      std::string metric_str = "l2",
                      bool with_path = false,
                      float valid_ratio = 0.8,
                      std::string vector_type = "dense",
                      uint64_t extra_info_size = 0,
                      bool has_duplicate = false,
                      int64_t id_shift = 16);

    static std::shared_ptr<TestDataset>
    CreateNanDataset(const std::string& metric_str);

    DatasetPtr base_{nullptr};

    DatasetPtr query_{nullptr};
    DatasetPtr ground_truth_{nullptr};
    int64_t top_k{10};

    DatasetPtr range_query_{nullptr};
    DatasetPtr range_ground_truth_{nullptr};
    std::vector<float> range_radius_{0.0f};

    DatasetPtr filter_query_{nullptr};
    DatasetPtr filter_ground_truth_{nullptr};
    DatasetPtr ex_filter_ground_truth_{nullptr};
    std::function<bool(int64_t)> filter_function_{nullptr};
    std::function<bool(const char*)> ex_filter_function_{nullptr};

    uint64_t dim_{0};
    uint64_t count_{0};

    float valid_ratio_{1.0F};

private:
    TestDataset() = default;
};

using TestDatasetPtr = std::shared_ptr<TestDataset>;
}  // namespace fixtures
