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

#include <atomic>
#include <memory>
#include <shared_mutex>

#include "algorithm/bruteforce/bruteforce.h"
#include "algorithm/hgraph/hgraph.h"
#include "algorithm/inner_index_interface.h"
#include "index_common_param.h"
#include "lazy_hgraph_parameter.h"

namespace vsag {

class LazyHGraph : public InnerIndexInterface {
public:
    enum class Phase : uint8_t { FLAT = 0, GRAPH = 1 };

    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    LazyHGraph(const LazyHGraphParameterPtr& param, const IndexCommonParam& common_param);

    LazyHGraph(const ParamPtr& param, const IndexCommonParam& common_param);

    std::vector<int64_t>
    Add(const DatasetPtr& data, AddMode mode = AddMode::DEFAULT) override;

    std::vector<int64_t>
    Build(const DatasetPtr& data) override;

    float
    CalcDistanceById(const float* query,
                     int64_t id,
                     bool calculate_precise_distance = true) const override;

    void
    Deserialize(StreamReader& reader) override;

    uint64_t
    EstimateMemory(uint64_t num_elements) const override;

    DatasetPtr
    ExportIDs() const override;

    [[nodiscard]] InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return std::make_shared<LazyHGraph>(this->create_param_ptr_, param);
    }

    [[nodiscard]] bool
    CheckIdExist(int64_t id) const override;

    [[nodiscard]] IndexType
    GetIndexType() const override {
        return IndexType::LAZY_HGRAPH;
    }

    [[nodiscard]] std::string
    GetName() const override {
        return INDEX_LAZY_HGRAPH;
    }

    [[nodiscard]] int64_t
    GetNumElements() const override;

    [[nodiscard]] int64_t
    GetNumberRemoved() const override;

    [[nodiscard]] DatasetPtr
    GetDataByIds(const int64_t* ids, int64_t count) const override;

    [[nodiscard]] int64_t
    GetMemoryUsage() const override;

    void
    GetVectorByInnerId(InnerIdType inner_id, float* data) const override;

    DatasetPtr
    GetVectorByIds(const int64_t* ids,
                   int64_t count,
                   Allocator* specified_allocator) const override;

    void
    InitFeatures() override;

    [[nodiscard]] DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    [[nodiscard]] DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    uint32_t
    Remove(const std::vector<int64_t>& ids, RemoveMode mode = RemoveMode::MARK_REMOVE) override;

    void
    Serialize(StreamWriter& writer) const override;

    [[nodiscard]] Phase
    GetPhase() const {
        return phase_.load(std::memory_order_acquire);
    }

private:
    void
    TransitionToGraph();

    InnerIndexInterface*
    ActiveIndex() const;

private:
    std::atomic<Phase> phase_{Phase::FLAT};
    uint64_t transition_threshold_{1000};
    BruteForceParameterPtr flat_param_{nullptr};
    HGraphParameterPtr graph_param_{nullptr};
    IndexCommonParam common_param_;
    std::shared_ptr<BruteForce> flat_index_{nullptr};
    std::shared_ptr<HGraph> graph_index_{nullptr};
    mutable std::shared_mutex phase_mutex_;
};

}  // namespace vsag
