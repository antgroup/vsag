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

/**
 * @file test_dataset.h
 * @brief Test dataset wrapper for functional tests.
 */

#pragma once

#include <functional>
#include <vector>

#include "vsag/dataset.h"
namespace fixtures {

/**
 * @class TestDataset
 * @brief Container for test datasets including base, query, and ground truth data.
 * Supports various dataset types: dense, sparse, with extra info, with attributes.
 */
class TestDataset {
public:
    using DatasetPtr = vsag::DatasetPtr;

    int64_t id_shift = 16;  // Offset added to generated IDs.

    /**
     * @brief Creates a test dataset with specified parameters.
     * @param dim Dimension of vectors.
     * @param count Number of vectors.
     * @param metric_str Distance metric type ("l2", "ip", etc.).
     * @param with_path Whether to include path information.
     * @param valid_ratio Ratio of valid vectors.
     * @param vector_type Type of vectors ("dense", "sparse", etc.).
     * @param extra_info_size Size of extra info per vector.
     * @param has_duplicate Whether to include duplicate vectors.
     * @param id_shift Offset for generated IDs.
     * @param use_fixed_seed Whether to use a fixed seed for reproducible data generation.
     * @return Shared pointer to the created TestDataset.
     */
    static std::shared_ptr<TestDataset>
    CreateTestDataset(uint64_t dim,
                      uint64_t count,
                      std::string metric_str = "l2",
                      bool with_path = false,
                      float valid_ratio = 0.8,
                      std::string vector_type = "dense",
                      uint64_t extra_info_size = 0,
                      bool has_duplicate = false,
                      int64_t id_shift = 16,
                      bool use_fixed_seed = true);

    /**
     * @brief Creates a test dataset with NaN values.
     * @param metric_str Distance metric type.
     * @return Shared pointer to the created TestDataset.
     */
    static std::shared_ptr<TestDataset>
    CreateNanDataset(const std::string& metric_str);

    DatasetPtr base_{nullptr};          // Base dataset for indexing.
    DatasetPtr query_{nullptr};         // Query dataset for search.
    DatasetPtr ground_truth_{nullptr};  // Ground truth results for knn search.
    int64_t top_k{10};                  // Number of results for knn search.

    DatasetPtr range_query_{nullptr};         // Query dataset for range search.
    DatasetPtr range_ground_truth_{nullptr};  // Ground truth for range search.
    std::vector<float> range_radius_{0.0f};   // Search radius values for range search.

    DatasetPtr filter_query_{nullptr};            // Query dataset for filtered search.
    DatasetPtr filter_ground_truth_{nullptr};     // Ground truth for filtered search.
    DatasetPtr ex_filter_ground_truth_{nullptr};  // Ground truth for extra filtered search.
    std::function<bool(int64_t)> filter_function_{nullptr};         // ID filter function.
    std::function<bool(const char*)> ex_filter_function_{nullptr};  // Extra info filter function.

    uint64_t dim_{0};    // Dimension of vectors.
    uint64_t count_{0};  // Number of vectors.

    float valid_ratio_{1.0F};  // Ratio of valid vectors.

private:
    TestDataset() = default;
};

using TestDatasetPtr = std::shared_ptr<TestDataset>;
}  // namespace fixtures
