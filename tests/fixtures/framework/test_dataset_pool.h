
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
 * @file test_dataset_pool.h
 * @brief Pool for caching and reusing test datasets.
 */

#pragma once

#include <unordered_map>
#include <vector>

#include "test_dataset.h"

namespace fixtures {

static const std::string NAN_DATASET = "nan_dataset";

/**
 * @class TestDatasetPool
 * @brief Cache for TestDataset objects to avoid regenerating datasets across tests.
 * Uses keyed lookup based on dim, count, metric type, and other parameters.
 */
class TestDatasetPool {
public:
    /**
     * @brief Gets or creates a test dataset with specified parameters.
     * @param dim Dimension of vectors.
     * @param count Number of vectors.
     * @param metric_str Distance metric type.
     * @param with_path Whether to include path information.
     * @param valid_ratio Ratio of valid vectors.
     * @param extra_info_size Size of extra info per vector.
     * @param id_shift Offset for generated IDs.
     * @return Shared pointer to the TestDataset.
     */
    TestDatasetPtr
    GetDatasetAndCreate(uint64_t dim,
                        uint64_t count,
                        const std::string& metric_str = "l2",
                        bool with_path = false,
                        float valid_ratio = 0.8,
                        uint64_t extra_info_size = 0,
                        int64_t id_shift = 16);

    /**
     * @brief Gets a test dataset with duplicate vectors.
     * @param dim Dimension of vectors.
     * @param count Number of vectors.
     * @param metric_str Distance metric type.
     * @param with_path Whether to include path information.
     * @param valid_ratio Ratio of valid vectors.
     * @param extra_info_size Size of extra info per vector.
     * @return Shared pointer to the TestDataset.
     */
    TestDatasetPtr
    GetDuplicateDataset(uint64_t dim,
                        uint64_t count,
                        const std::string& metric_str = "l2",
                        bool with_path = false,
                        float valid_ratio = 0.8,
                        uint64_t extra_info_size = 0);

    /**
     * @brief Gets the NaN test dataset for specified metric.
     * @param metric_str Distance metric type.
     * @return Shared pointer to the NaN TestDataset.
     */
    TestDatasetPtr
    GetNanDataset(const std::string& metric_str);

    /**
     * @brief Gets or creates a sparse test dataset.
     * @param count Number of vectors.
     * @param dim Dimension of vectors.
     * @param valid_ratio Ratio of valid vectors.
     * @return Shared pointer to the TestDataset.
     */
    TestDatasetPtr
    GetSparseDatasetAndCreate(uint64_t count, uint64_t dim, float valid_ratio = 0.8);

private:
    /**
     * @brief Generates a unique key for dataset lookup.
     * @param dim Dimension of vectors.
     * @param count Number of vectors.
     * @param metric_str Distance metric type.
     * @param with_path Whether to include path information.
     * @param filter_ratio Ratio of valid vectors.
     * @param extra_info_size Size of extra info per vector.
     * @param id_shift Offset for generated IDs.
     * @return Unique string key for the dataset.
     */
    static std::string
    key_gen(int64_t dim,
            uint64_t count,
            const std::string& metric_str = "l2",
            bool with_path = false,
            float filter_ratio = 0.8,
            uint64_t extra_info_size = 0,
            int64_t id_shift = 16);

private:
    std::unordered_map<std::string, TestDatasetPtr> pool_;   // Cache of TestDataset objects.
    std::vector<std::pair<uint64_t, uint64_t>> dim_counts_;  // List of (dim, count) pairs used.
};
}  // namespace fixtures
