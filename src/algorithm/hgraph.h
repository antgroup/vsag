
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

#include <random>
#include <shared_mutex>
#include <string>

#include "common.h"
#include "datacell/attribute_inverted_interface.h"
#include "datacell/flatten_interface.h"
#include "datacell/graph_interface.h"
#include "datacell/sparse_graph_datacell_parameter.h"
#include "hgraph_parameter.h"
#include "impl/basic_optimizer.h"
#include "impl/heap/distance_heap.h"
#include "impl/reorder/flatten_reorder.h"
#include "impl/searcher/basic_searcher.h"
#include "impl/searcher/parallel_searcher.h"
#include "impl/thread_pool/default_thread_pool.h"
#include "index/iterator_filter.h"
#include "index_common_param.h"
#include "index_feature_list.h"
#include "inner_index_interface.h"
#include "typing.h"
#include "utils/lock_strategy.h"
#include "utils/util_functions.h"
#include "utils/visited_list.h"
#include "vsag/index.h"
#include "vsag/index_features.h"

namespace vsag {

// HGraph index was introduced since v0.12
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

    std::vector<int64_t>
    Add(const DatasetPtr& data) override;

    std::string
    AnalyzeIndexBySearch(const SearchRequest& request) override;

    std::vector<int64_t>
    Build(const DatasetPtr& data) override;

    float
    CalcDistanceById(const float* query, int64_t id) const override;

    DatasetPtr
    CalDistanceById(const float* query, const int64_t* ids, int64_t count) const override;

    void
    Deserialize(StreamReader& reader) override;

    InnerIndexPtr
    ExportModel(const IndexCommonParam& param) const override;

    uint64_t
    EstimateMemory(uint64_t num_elements) const override;

    void
    GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const override;

    void
    GetCodeByInnerId(InnerIdType inner_id, uint8_t* data) const override;

    int64_t
    GetMemoryUsage() const override {
        return static_cast<int64_t>(this->CalSerializeSize());
    }

    std::string
    GetMemoryUsageDetail() const override;

    std::pair<int64_t, int64_t>
    GetMinAndMaxId() const override;

    [[nodiscard]] std::string
    GetName() const override {
        return INDEX_TYPE_HGRAPH;
    }

    int64_t
    GetNumElements() const override {
        return static_cast<int64_t>(this->total_count_) - delete_count_;
    }

    int64_t
    GetNumberRemoved() const override {
        return delete_count_;
    }

    std::string
    GetStats() const override;

    void
    GetVectorByInnerId(InnerIdType inner_id, float* data) const override;

    IndexType
    GetIndexType() const override {
        return IndexType::HGRAPH;
    }

    void
    InitFeatures() override;

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
              Allocator* allocator) const override;

    [[nodiscard]] DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter,
              Allocator* allocator,
              IteratorContext*& iter_ctx,
              bool is_last_filter) const override;

    [[nodiscard]] InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return std::make_shared<HGraph>(this->create_param_ptr_, param);
    }

    void
    Merge(const std::vector<MergeUnit>& merge_units) override;

    [[nodiscard]] DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    [[nodiscard]] DatasetPtr
    SearchWithRequest(const SearchRequest& request) const override;

    bool
    Remove(int64_t id) override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    SetBuildThreadsCount(uint64_t count) {
        this->build_thread_count_ = count;
        this->build_pool_->SetPoolSize(count);
    }

    void
    SetImmutable() override;

    void
    SetIO(const std::shared_ptr<Reader> reader) override;

    void
    Train(const DatasetPtr& base) override;

    bool
    UpdateId(int64_t old_id, int64_t new_id) override;

    bool
    UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update = false) override;

    void
    UpdateAttribute(int64_t id, const AttributeSet& new_attrs) override;

    void
    UpdateAttribute(int64_t id,
                    const AttributeSet& new_attrs,
                    const AttributeSet& origin_attrs) override;

private:
    const void*
    get_data(const DatasetPtr& dataset, uint32_t index = 0) const {
        if (data_type_ == DataTypes::DATA_TYPE_FLOAT) {
            return dataset->GetFloat32Vectors() + index * dim_;
        } else if (data_type_ == DataTypes::DATA_TYPE_INT8) {
            return dataset->GetInt8Vectors() + index * dim_;
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

    bool
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
                     InnerSearchParam& inner_search_param,
                     const VisitedListPtr& vt,
                     Statistics& stats) const;

    template <InnerSearchMode mode = InnerSearchMode::KNN_SEARCH>
    DistHeapPtr
    search_one_graph(const void* query,
                     const GraphInterfacePtr& graph,
                     const FlattenInterfacePtr& flatten,
                     InnerSearchParam& inner_search_param,
                     IteratorFilterContext* iter_ctx,
                     Statistics& stats) const;

private:
    // since v0.15
    JsonType
    serialize_basic_info() const;

    void
    deserialize_basic_info(const JsonType& jsonify_basic_info);

    void
    serialize_label_info(StreamWriter& writer) const;

    void
    deserialize_label_info(StreamReader& reader) const;

    // used in version [0.12.*, 0.14.*]
    void
    serialize_basic_info_v0_14(StreamWriter& writer) const;

    void
    deserialize_basic_info_v0_14(StreamReader& reader);

private:
    void
    reorder(const void* query,
            const FlattenInterfacePtr& flatten,
            DistHeapPtr& candidate_heap,
            int64_t k) const;

    void
    elp_optimize();

    void
    recover_remove(int64_t id);

    bool
    try_recover_tombstone(const DatasetPtr& data, std::vector<int64_t>& failed_ids);

    DatasetPtr
    get_single_dataset(const DatasetPtr& data, uint32_t j);

private:
    void
    analyze_graph_recall(JsonType& stats,
                         Vector<float>& data,
                         uint64_t sample_data_size,
                         int64_t topk,
                         const std::string& search_param) const;

    void
    analyze_graph_connection(JsonType& stats) const;

    void
    check_and_init_raw_vector(const FlattenInterfaceParamPtr& raw_vector_param,
                              const IndexCommonParam& common_param);

private:
    FlattenInterfacePtr basic_flatten_codes_{nullptr};
    FlattenInterfacePtr high_precise_codes_{nullptr};

    Vector<GraphInterfacePtr> route_graphs_;
    GraphInterfacePtr bottom_graph_{nullptr};
    SparseGraphDatacellParamPtr hierarchical_datacell_param_{nullptr};

    bool use_elp_optimizer_{false};
    bool ignore_reorder_{false};
    bool build_by_base_{false};

    BasicSearcherPtr searcher_;
    ParallelSearcherPtr parallel_searcher_;

    std::default_random_engine level_generator_{2021};
    double mult_{1.0};

    InnerIdType entry_point_id_{std::numeric_limits<InnerIdType>::max()};

    ODescentParameterPtr odescent_param_{nullptr};
    std::string graph_type_{GRAPH_TYPE_VALUE_NSW};

    uint64_t ef_construct_{400};
    float alpha_{1.0};

    std::atomic<uint64_t> total_count_{0};

    std::shared_ptr<VisitedListPool> pool_{nullptr};

    mutable std::shared_mutex global_mutex_;
    mutable MutexArrayPtr neighbors_mutex_;
    mutable std::shared_mutex add_mutex_;

    std::atomic<InnerIdType> max_capacity_{0};

    uint64_t resize_increase_count_bit_{
        DEFAULT_RESIZE_BIT};  // 2^resize_increase_count_bit_ for resize count

    static constexpr uint64_t DEFAULT_RESIZE_BIT = 10;

    std::atomic<int64_t> delete_count_{0};

    std::shared_ptr<Optimizer<BasicSearcher>> optimizer_;

    bool create_new_raw_vector_{false};
    FlattenInterfacePtr raw_vector_{nullptr};

    ReorderInterfacePtr reorder_{nullptr};

    bool use_old_serial_format_{false};
};
}  // namespace vsag
