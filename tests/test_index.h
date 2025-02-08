
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

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <unordered_set>
#include <utility>

#include "fixtures/fixtures.h"
#include "fixtures/random_allocator.h"
#include "fixtures/test_dataset.h"
#include "vsag/dataset.h"
#include "vsag/errors.h"
#include "vsag/logger.h"
#include "vsag/options.h"
#include "vsag/vsag.h"

namespace fixtures {
class TestIndex {
public:
    using IndexPtr = vsag::IndexPtr;
    using DatasetPtr = vsag::DatasetPtr;
    TestIndex() {
        vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);
    }

protected:
    static IndexPtr
    TestFactory(const std::string& name,
                const std::string& build_param,
                bool expect_success = true) {
        auto new_index = vsag::Factory::CreateIndex(name, build_param);
        REQUIRE(new_index.has_value() == expect_success);
        return new_index.value();
    }

    static void
    TestBuildIndex(const IndexPtr& index,
                   const TestDatasetPtr& dataset,
                   bool expected_success = true);

    static void
    TestAddIndex(const IndexPtr& index,
                 const TestDatasetPtr& dataset,
                 bool expected_success = true);

    static void
    TestUpdateId(const IndexPtr& index,
                 const TestDatasetPtr& dataset,
                 const std::string& search_param,
                 bool expected_success = true);

    static void
    TestUpdateVector(const IndexPtr& index,
                     const TestDatasetPtr& dataset,
                     const std::string& search_param,
                     bool expected_success = true);

    static void
    TestContinueAdd(const IndexPtr& index,
                    const TestDatasetPtr& dataset,
                    bool expected_success = true);

    static void
    TestContinueAddIgnoreRequire(const IndexPtr& index, const TestDatasetPtr& dataset);

    static void
    TestKnnSearch(const IndexPtr& index,
                  const TestDatasetPtr& dataset,
                  const std::string& search_param,
                  float expected_recall = 0.99,
                  bool expected_success = true);

    static void
    TestSearchWithDirtyVector(const IndexPtr& index,
                              const TestDatasetPtr& dataset,
                              const std::string& search_param,
                              bool expected_success = true);

    static void
    TestRangeSearch(const IndexPtr& index,
                    const TestDatasetPtr& dataset,
                    const std::string& search_param,
                    float expected_recall = 0.99,
                    int64_t limited_size = -1,
                    bool expected_success = true);

    static void
    TestFilterSearch(const IndexPtr& index,
                     const TestDatasetPtr& dataset,
                     const std::string& search_param,
                     float expected_recall = 0.99,
                     bool expected_success = true,
                     bool support_filter_obj = false);

    static void
    TestCalcDistanceById(const IndexPtr& index, const TestDatasetPtr& dataset, float error = 1e-5);

    static void
    TestBatchCalcDistanceById(const IndexPtr& index,
                              const TestDatasetPtr& dataset,
                              float error = 1e-5);

    static void
    TestSerializeFile(const IndexPtr& index_from,
                      const IndexPtr& index_to,
                      const TestDatasetPtr& dataset,
                      const std::string& search_param,
                      bool expected_success = true);

    static void
    TestSerializeBinarySet(const IndexPtr& index_from,
                           const IndexPtr& index_to,
                           const TestDatasetPtr& dataset,
                           const std::string& search_param,
                           bool expected_success = true);

    static void
    TestSerializeReaderSet(const IndexPtr& index_from,
                           const IndexPtr& index_to,
                           const TestDatasetPtr& dataset,
                           const std::string& search_param,
                           const std::string& index_name,
                           bool expected_success = true);

    static void
    TestConcurrentKnnSearch(const IndexPtr& index,
                            const TestDatasetPtr& dataset,
                            const std::string& search_param,
                            float expected_recall = 0.99,
                            bool expected_success = true);

    static IndexPtr
    TestMergeIndex(const std::string& name,
                   const std::string& build_param,
                   const TestDatasetPtr& dataset,
                   int32_t split_num = 1,
                   bool expect_success = true);

    static void
    TestConcurrentAdd(const IndexPtr& index,
                      const TestDatasetPtr& dataset,
                      bool expected_success = true);
    static void
    TestDuplicateAdd(const IndexPtr& index, const TestDatasetPtr& dataset);

    static void
    TestEstimateMemory(const std::string& index_name,
                       const std::string& build_param,
                       const TestDatasetPtr& dataset);

    static void
    TestCheckIdExist(const IndexPtr& index, const TestDatasetPtr& dataset);

    constexpr static float RECALL_THRESHOLD = 0.95;
};

}  // namespace fixtures
