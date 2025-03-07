
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

#include <nlohmann/json.hpp>
#include <random>
#include <shared_mutex>

#include "ThreadPool.h"
#include "algorithm/hnswlib/algorithm_interface.h"
#include "algorithm/hnswlib/visited_list_pool.h"
#include "common.h"
#include "data_cell/flatten_interface.h"
#include "data_cell/graph_interface.h"
#include "hgraph_parameter.h"
#include "impl/basic_searcher.h"
#include "index/index_common_param.h"
#include "index_feature_list.h"
#include "inner_index_interface.h"
#include "lock_strategy.h"
#include "typing.h"
#include "vsag/index.h"
#include "vsag/index_features.h"

namespace vsag {
class HGraph : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    HGraph(const HGraphParameterPtr& param, const IndexCommonParam& common_param);

    HGraph(const ParamPtr& param, const IndexCommonParam& common_param)
        : HGraph(std::dynamic_pointer_cast<HGraphParameter>(param), common_param){};

    ~HGraph() override = default;

    [[nodiscard]] std::string
    GetName() const override {
        return INDEX_TYPE_HGRAPH;
    }

    void
    InitFeatures() override {
        return this->init_features();
    }

    std::vector<int64_t>
    Build(const DatasetPtr& data) override;

    std::vector<int64_t>
    Add(const DatasetPtr& data) override;

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

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    int64_t
    GetNumElements() const override {
        return this->basic_flatten_codes_->TotalCount();
    }

    uint64_t
    EstimateMemory(uint64_t num_elements) const override;

    // TODO(LHT): implement
    inline int64_t
    GetMemoryUsage() const override {
        return 0;
    }

    float
    CalcDistanceById(const float* query, int64_t id) const override;

    DatasetPtr
    CalDistanceById(const float* query, const int64_t* ids, int64_t count) const override;

    inline void
    SetBuildThreadsCount(uint64_t count) {
        this->build_thread_count_ = count;
        this->build_pool_->set_pool_size(count);
    }

private:
    inline int
    get_random_level() {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * mult_;
        return (int)r;
    }

    void
    hnsw_add(const DatasetPtr& data);

    void
    resize(uint64_t new_size);

    GraphInterfacePtr
    generate_one_route_graph();

    template <InnerSearchMode mode = InnerSearchMode::KNN_SEARCH>
    MaxHeap
    search_one_graph(const float* query,
                     const GraphInterfacePtr& graph,
                     const FlattenInterfacePtr& flatten,
                     InnerSearchParam& inner_search_param) const;
    void
    serialize_basic_info(StreamWriter& writer) const;

    void
    deserialize_basic_info(StreamReader& reader);

    inline LabelType
    get_label_by_id(InnerIdType inner_id) const {
        std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
        // the inner_id is guarantee in label_lookup
        return this->label_table_->GetLabelById(inner_id);
    }

    void
    add_one_point(const float* data, int level, InnerIdType id);

    void
    init_features();

    Vector<DatasetPtr>
    split_dataset_by_duplicate_label(const DatasetPtr& dataset,
                                     std::vector<LabelType>& failed_ids) const;

    void
    reorder(const float* query,
            const FlattenInterfacePtr& flatten_interface,
            MaxHeap& candidate_heap,
            int64_t k) const;

private:
    FlattenInterfacePtr basic_flatten_codes_{nullptr};
    FlattenInterfacePtr high_precise_codes_{nullptr};
    Vector<GraphInterfacePtr> route_graphs_;
    GraphInterfacePtr bottom_graph_{nullptr};

    bool use_reorder_{false};

    int64_t dim_{0};
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};

    std::default_random_engine level_generator_{2021};
    double mult_{1.0};

    InnerIdType entry_point_id_{std::numeric_limits<InnerIdType>::max()};
    uint64_t max_level_{0};

    uint64_t ef_construct_{400};
    mutable std::shared_mutex global_mutex_;

    std::shared_ptr<hnswlib::VisitedListPool> pool_{nullptr};

    mutable MutexArrayPtr neighbors_mutex_;

    std::unique_ptr<progschj::ThreadPool> build_pool_{nullptr};
    uint64_t build_thread_count_{100};

    InnerIdType max_capacity_{0};

    const uint64_t resize_increase_count_bit_{10};  // 2^resize_increase_count_bit_ for resize count
};
}  // namespace vsag
