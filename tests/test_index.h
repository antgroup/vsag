
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

#include "fixtures/fixtures.h"
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

    enum class IndexStatus {
        Init = 0,
        Factory = 1,
        Build = 2,
        DeSerialize = 3,
    };

    TestIndex() {
        vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);
    }

protected:
    IndexPtr
    FastGeneralTest(const std::string& name,
                    const std::string& build_param,
                    const std::string& search_parameters,
                    const std::string& metric_type,
                    int64_t dim,
                    IndexStatus end_status = IndexStatus::DeSerialize) const;

    static IndexPtr
    TestFactory(const std::string& name,
                const std::string& build_param,
                bool expect_success = true);

    void
    TestBuildIndex(IndexPtr index, int64_t dim, bool expected_success = true) const;

    static void
    TestBuildIndex(IndexPtr index, DatasetPtr dataset, bool expected_success = true);

    void
    TestAddIndex(IndexPtr index, int64_t dim, bool expected_success = true) const;

    static void
    TestAddIndex(IndexPtr index, DatasetPtr dataset, bool expected_success = true);

    static void
    TestContinueAdd(IndexPtr index, int64_t dim, int64_t count = 100, bool expected_success = true);

    static void
    TestKnnSearch(IndexPtr index,
                  std::shared_ptr<fixtures::TestDataset> dataset,
                  const std::string& search_param,
                  int topk = 10,
                  float recall = 0.99,
                  bool expected_success = true);

    static void
    TestRangeSearch(IndexPtr index,
                    std::shared_ptr<fixtures::TestDataset> dataset,
                    const std::string& search_param,
                    float radius = 0.01,
                    float recall = 0.99,
                    int64_t limited_size = -1,
                    bool expected_success = true);
    static void
    TestCalcDistanceById(IndexPtr index,
                         std::shared_ptr<fixtures::TestDataset> dataset,
                         const std::string& metric_str,
                         float error = 1e-5);

    static void
    TestSerializeFile(IndexPtr index, const std::string& path, bool expected_success = true);

    static IndexPtr
    TestDeserializeFile(const std::string& path,
                        const std::string& name,
                        const std::string& param,
                        bool expected_success = true);

    bool
    SetDataset(const std::string& key, TestDatasetPtr value) const {
        if (datasets.find(key) != datasets.end()) {
            return false;
        }
        datasets[key] = value;
        return true;
    }

    TestDatasetPtr
    GetDataset(const std::string& key) const {
        auto iter = datasets.find(key);
        if (iter == datasets.end()) {
            return nullptr;
        }
        return iter->second;
    }

    void
    DeleteDataset(const std::string& key) const {
        auto iter = datasets.find(key);
        if (iter != datasets.end()) {
            datasets.erase(iter);
        }
    }

    template <typename T>
    TestDatasetPtr
    GenerateAndSetDataset(int64_t dim, uint64_t count) const {
        std::string datatype = "float";
        if constexpr (std::is_same_v<T, int8_t>) {
            datatype = "int8";
        } else if constexpr (std::is_same_v<T, float>) {
            datatype = "float";
        } else {
            // TODO throw;
        }
        auto key = KeyGen(dim, count, datatype);
        if (datasets.find(key) == datasets.end()) {
            datasets[key] = GenerateDataset<T>(dim, count);
        }

        return datasets[key];
    }

    template <typename T>
    TestDatasetPtr
    GenerateDataset(int64_t dim, uint64_t count) const {
        if constexpr (std::is_same_v<T, int8_t>) {
            return GenerateDatasetInt8(dim, count);
        } else if constexpr (std::is_same_v<T, float>) {
            return GenerateDatasetFloat(dim, count);
        } else {
            // TODO throw;
            return nullptr;
        }
    }

    static TestDatasetPtr
    GenerateDatasetFloat(int64_t dim, uint64_t count);

    TestDatasetPtr
    GenerateDatasetInt8(int64_t dim, uint64_t count) const {
        return nullptr;  // TODO
    };

    std::string
    KeyGen(int64_t dim,
           uint64_t count,
           std::string datatype = "float",
           std::string name = "classic") const {
        return fmt::format(
            "[dim={}][count={}][type={}][dataset_name={}]", dim, count, datatype, name);
    }

    std::string
    KeyGenIndex(int64_t dim,
                uint64_t count,
                std::string index_name,
                std::string datatype = "float",
                std::string dataset_name = "classic") const {
        auto str = KeyGen(dim, count, datatype, dataset_name);
        return str + fmt::format("[index_name={}]", index_name);
    }

    std::pair<IndexPtr, IndexStatus>
    LoadIndex(std::string key) const {
        if (indexs.find(key) == indexs.end()) {
            return {nullptr, IndexStatus::Init};
        }
        return indexs[key];
    }

    void
    SaveIndex(const std::string& key, IndexPtr index, IndexStatus status) const {
        indexs[key] = {index, status};
    }

    mutable int dataset_base_count{1000};

private:
    mutable std::unordered_map<std::string, std::shared_ptr<TestDataset>> datasets;

    mutable std::unordered_map<std::string, std::pair<IndexPtr, IndexStatus>> indexs;
};

}  // namespace fixtures
