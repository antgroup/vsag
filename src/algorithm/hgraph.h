
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

#include "algorithm/hnswlib/algorithm_interface.h"
#include "algorithm/hnswlib/visited_list_pool.h"
#include "common.h"
#include "data_cell/extra_info_interface.h"
#include "data_cell/flatten_interface.h"
#include "data_cell/graph_interface.h"
#include "data_cell/sparse_graph_datacell_parameter.h"
#include "default_thread_pool.h"
#include "hgraph_parameter.h"
#include "impl/basic_searcher.h"
#include "index/index_common_param.h"
#include "index/iterator_filter.h"
#include "index_feature_list.h"
#include "inner_index_interface.h"
#include "lock_strategy.h"
#include "typing.h"
#include "utils/distance_heap.h"
#include "utils/visited_list.h"
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
    InitFeatures() override;

    [[nodiscard]] InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return std::make_shared<HGraph>(this->create_param_ptr_, param);
    }

    void
    Train(const DatasetPtr& base) override;

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
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter,
              IteratorContext*& iter_ctx,
              bool is_last_filter) const override;

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
        return this->total_count_;
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

    std::pair<int64_t, int64_t>
    GetMinAndMaxId() const override;

    void
    GetExtraInfoByIds(const int64_t* ids, int64_t count, char* extra_infos) const override;

    InnerIndexPtr
    ExportModel(const IndexCommonParam& param) const override;

    inline void
    SetBuildThreadsCount(uint64_t count) {
        this->build_thread_count_ = count;
        this->build_pool_->SetPoolSize(count);
    }

private:
    const void*
    get_data(const DatasetPtr& dataset, uint32_t index = 0) const {
        if (data_type_ == DataTypes::DATA_TYPE_FLOAT) {
            return dataset->GetFloat32Vectors() + index * dim_;
        } else if (data_type_ == DataTypes::DATA_TYPE_SPARSE) {
            return dataset->GetSparseVectors() + index;
        }
        throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid data_type in HGraph");
    }

    int
    get_random_level() {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * mult_;
        return static_cast<int>(r);
    }

    Vector<InnerIdType>
    get_unique_inner_ids(InnerIdType count) {
        auto start = static_cast<InnerIdType>(this->total_count_);
        Vector<InnerIdType> ret(count, this->allocator_);
        if (ret.size() != count) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "allocate memory failed");
        }
        std::iota(ret.begin(), ret.end(), start);
        this->total_count_ += count;
        return ret;
    }

    std::vector<int64_t>
    build_by_odescent(const DatasetPtr& data);

    void
    add_one_point(const void* data, int level, InnerIdType id);

    void
    graph_add_one(const void* data, int level, InnerIdType inner_id);

    void
    resize(uint64_t new_size);

    GraphInterfacePtr
    generate_one_route_graph();

    template <InnerSearchMode mode = InnerSearchMode::KNN_SEARCH>
    DistHeapPtr
    search_one_graph(const void* query,
                     const GraphInterfacePtr& graph,
                     const FlattenInterfacePtr& flatten,
                     InnerSearchParam& inner_search_param) const;

    template <InnerSearchMode mode = InnerSearchMode::KNN_SEARCH>
    DistHeapPtr
    search_one_graph(const void* query,
                     const GraphInterfacePtr& graph,
                     const FlattenInterfacePtr& flatten,
                     InnerSearchParam& inner_search_param,
                     IteratorFilterContext* iter_ctx) const;

    void
    serialize_basic_info(StreamWriter& writer) const;

    void
    deserialize_basic_info(StreamReader& reader);

    void
    reorder(const void* query,
            const FlattenInterfacePtr& flatten_interface,
            const DistHeapPtr& candidate_heap,
            int64_t k) const;

    void
    elp_optimize();

private:
    FlattenInterfacePtr basic_flatten_codes_{nullptr};
    FlattenInterfacePtr high_precise_codes_{nullptr};
    Vector<GraphInterfacePtr> route_graphs_;
    GraphInterfacePtr bottom_graph_{nullptr};
    SparseGraphDatacellParamPtr sparse_datacell_param_{nullptr};

    mutable bool use_reorder_{false};
    bool use_elp_optimizer_{false};
    bool ignore_reorder_{false};

    BasicSearcherPtr searcher_;

    std::default_random_engine level_generator_{2021};
    double mult_{1.0};

    InnerIdType entry_point_id_{std::numeric_limits<InnerIdType>::max()};

    ODescentParameterPtr odescent_param_{nullptr};
    std::string graph_type_{GRAPH_TYPE_NSW};

    uint64_t ef_construct_{400};

    uint64_t total_count_{0};

    std::shared_ptr<VisitedListPool> pool_{nullptr};

    mutable std::shared_mutex global_mutex_;
    mutable MutexArrayPtr neighbors_mutex_;
    mutable std::shared_mutex add_mutex_;

    std::shared_ptr<SafeThreadPool> build_pool_{nullptr};
    uint64_t build_thread_count_{100};

    std::atomic<InnerIdType> max_capacity_{0};

    uint64_t resize_increase_count_bit_{
        DEFAULT_RESIZE_BIT};  // 2^resize_increase_count_bit_ for resize count

    ExtraInfoInterfacePtr extra_infos_{nullptr};
    uint64_t extra_info_size_{0};

    static constexpr uint64_t DEFAULT_RESIZE_BIT = 10;

    std::shared_ptr<Optimizer<BasicSearcher>> optimizer_;
};
}  // namespace vsag
