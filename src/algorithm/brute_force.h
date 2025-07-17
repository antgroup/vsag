
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
#include "brute_force_parameter.h"
#include "data_cell/flatten_interface.h"
#include "label_table.h"
#include "typing.h"
#include "vsag/filter.h"

namespace vsag {

// BruteForce index was introduced since v0.13
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

    [[nodiscard]] InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return std::make_shared<BruteForce>(this->create_param_ptr_, param);
    }

    void
    InitFeatures() override;

    std::vector<int64_t>
    Build(const DatasetPtr& data) override;

    IndexType
    GetIndexType() override {
        return IndexType::BRUTEFORCE;
    }

    void
    Train(const DatasetPtr& data) override;

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

    void
    GetVectorByInnerId(InnerIdType inner_id, float* data) const override;

private:
    void
    resize(uint64_t new_size);

    void
    add_one(const float* data, InnerIdType inner_id);

private:
    FlattenInterfacePtr inner_codes_{nullptr};

    uint64_t total_count_{0};

    uint64_t resize_increase_count_bit_{DEFAULT_RESIZE_BIT};

    std::shared_ptr<SafeThreadPool> build_pool_{nullptr};

    mutable std::shared_mutex global_mutex_;
    mutable std::shared_mutex add_mutex_;

    std::atomic<InnerIdType> max_capacity_{0};

    static constexpr uint64_t DEFAULT_RESIZE_BIT = 10;
};
}  // namespace vsag
