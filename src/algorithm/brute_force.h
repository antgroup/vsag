
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

#include "algorithm/inner_index_interface.h"
#include "base_filter_functor.h"
#include "brute_force_parameter.h"
#include "common.h"
#include "data_cell/flatten_datacell.h"
#include "index_feature_list.h"
#include "label_table.h"
#include "typing.h"

namespace vsag {
class BruteForce : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    explicit BruteForce(const BruteForceParameterPtr& param, const IndexCommonParam& common_param);

    explicit BruteForce(const ParamPtr& param, const IndexCommonParam& common_param)
        : BruteForce(std::dynamic_pointer_cast<BruteForceParameter>(param), common_param){};

    ~BruteForce() override = default;

    [[nodiscard]] std::string
    GetName() const override {
        return INDEX_BRUTE_FORCE;
    }

    void
    InitFeatures() override;

    std::vector<int64_t>
    Build(const DatasetPtr& data) override;

    std::vector<int64_t>
    Add(const DatasetPtr& data) override;

    DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    float
    CalcDistanceById(const float* vector, int64_t id) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    [[nodiscard]] int64_t
    GetNumElements() const override {
        return this->total_count_;
    }

    [[nodiscard]] int64_t
    GetMemoryUsage() const override;

    [[nodiscard]] uint64_t
    EstimateMemory(uint64_t num_elements) const override;

private:
    Vector<DatasetPtr>
    split_dataset_by_duplicate_label(const DatasetPtr& dataset,
                                     std::vector<LabelType>& failed_ids) const;

private:
    FlattenInterfacePtr inner_codes_{nullptr};

    Allocator* const allocator_{nullptr};

    int64_t dim_{0};

    uint64_t total_count_{0};
};
}  // namespace vsag
