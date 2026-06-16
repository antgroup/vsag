
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
#include <string_view>
#include <unordered_map>

#include "../inner_index_interface.h"
#include "common.h"
#include "datacell/attribute_inverted_interface.h"
#include "datacell/flatten_interface.h"
#include "datacell/graph_interface.h"
#include "datacell/sparse_graph_datacell_parameter.h"
#include "hgraph_cache.h"
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

    friend class HGraphAnalyzer;

public:
    HGraph(const HGraphParameterPtr& param, const IndexCommonParam& common_param);

    HGraph(const ParamPtr& param, const IndexCommonParam& common_param)
        : HGraph(std::dynamic_pointer_cast<HGraphParameter>(param), common_param){};

    ~HGraph() override = default;

    std::vector<int64_t>
    Add(const DatasetPtr& data, AddMode mode = AddMode::DEFAULT) override;

    std::string
    AnalyzeIndexBySearch(const SearchRequest& request) override;

    std::vector<int64_t>
    Build(const DatasetPtr& data) override;

    bool
    Tune(const std::string& parameters, bool disable_future_tuning) override;

    float
    CalcDistanceById(const float* query,
                     int64_t id,
                     bool calculate_precise_distance = true) const override;

    DatasetPtr
    CalDistanceById(const float* query,
                    const int64_t* ids,
                    int64_t count,
                    bool calculate_precise_distance = true) const override;

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

    uint32_t
    Remove(const std::vector<int64_t>& ids, RemoveMode mode = RemoveMode::MARK_REMOVE) override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    SetBuildThreadsCount(uint64_t count) {
        this->build_thread_count_ = count;
        this->thread_pool_->SetPoolSize(count);
    }

    void
    SetImmutable() override;

    void
    ExportCache(std::ostream& out_stream) const override;

    void
    ImportCache(std::istream& in_stream) override;

    void
    SetIO(const std::shared_ptr<Reader> reader) override;

    void
    Train(const DatasetPtr& base) override;

    bool
    UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update = false) override;

    void
    UpdateAttribute(int64_t id, const AttributeSet& new_attrs) override;

    void
    UpdateAttribute(int64_t id,
                    const AttributeSet& new_attrs,
                    const AttributeSet& origin_attrs) override;

    static JsonType
    map_hgraph_param(const JsonType& hgraph_json);

    const void*
    get_data(const DatasetPtr& dataset, uint32_t index = 0) const {
        if (data_type_ == DataTypes::DATA_TYPE_FLOAT) {
            auto* ptr = dataset->GetFloat32Vectors();
            return ptr ? ptr + static_cast<int64_t>(index) * dim_ : nullptr;
        } else if (data_type_ == DataTypes::DATA_TYPE_INT8) {
            auto* ptr = dataset->GetInt8Vectors();
            return ptr ? ptr + static_cast<int64_t>(index) * dim_ : nullptr;
        } else if (data_type_ == DataTypes::DATA_TYPE_FP16 ||
                   data_type_ == DataTypes::DATA_TYPE_BF16) {
            auto* ptr = dataset->GetFloat16Vectors();
            return ptr ? ptr + static_cast<int64_t>(index) * dim_ : nullptr;
        } else if (data_type_ == DataTypes::DATA_TYPE_SPARSE) {
            auto* ptr = dataset->GetSparseVectors();
            return ptr ? ptr + index : nullptr;
        }
        throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid data_type in HGraph");
    }

    int
    get_random_level() {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * mult_;
        return static_cast<int>(r);
    }

    /**
     * @brief Allocate a contiguous block of fresh internal IDs.
     *
     * IDs are derived from "total_count_" but "total_count_" is intentionally
     * NOT advanced inside this helper. The caller is responsible for:
     *   1. Ensuring storage capacity covers the returned range (via resize()).
     *   2. Publishing the new range by incrementing total_count_ once insertion
     *      and capacity expansion have both succeeded.
     *
     * Thread-safety: the caller must hold add_mutex_ for the entire span of
     * "call get_unique_inner_ids -> resize -> increment total_count_". Concurrent
     * search threads read total_count_ under global_mutex_ and assume the
     * underlying storage is sized to at least total_count_; never publish IDs
     * before the resize completes.
     */
    Vector<InnerIdType>
    get_unique_inner_ids(InnerIdType count) {
        Vector<InnerIdType> ret(count, this->allocator_);
        if (ret.size() != count) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "allocate memory failed");
        }
        auto next_id = static_cast<InnerIdType>(this->total_count_.load());
        for (InnerIdType i = 0; i < count; ++i) {
            ret[i] = next_id++;
        }
        return ret;
    }

    std::vector<int64_t>
    build_by_odescent(const DatasetPtr& data);

    void
    add_one_point(const void* data, int level, InnerIdType id);

    void
    insert_persistent_codes(const void* data, InnerIdType inner_id);

    void
    add_one_point(const void* data, int level, InnerIdType id, bool insert_codes);

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
                     // ctx can be nullptr in adding scenario
                     QueryContext* ctx,
                     DistanceRecordVector* rabitq_lower_bound_candidates = nullptr) const;

    template <InnerSearchMode mode = InnerSearchMode::KNN_SEARCH>
    DistHeapPtr
    search_one_graph(const void* query,
                     const GraphInterfacePtr& graph,
                     const FlattenInterfacePtr& flatten,
                     InnerSearchParam& inner_search_param,
                     IteratorFilterContext* iter_ctx,
                     // ctx can be nullptr in adding scenario
                     QueryContext* ctx,
                     DistanceRecordVector* rabitq_lower_bound_candidates = nullptr) const;

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

    uint32_t
    force_remove_one(int64_t label);

    void
    find_new_entry_point();

    void
    graph_force_remove_one(const InnerIdType& inner_id,
                           const FlattenInterfacePtr& flatten,
                           const GraphInterfacePtr& graph);

    void
    move_id(InnerIdType from, InnerIdType to);

    void
    shrink_to_fit();

    template <InnerSearchMode mode = InnerSearchMode::KNN_SEARCH>
    DistHeapPtr
    brute_force_search(const void* query,
                       const FilterPtr& filter,
                       int64_t topk,
                       float radius,
                       QueryContext* ctx) const;

private:
    void
    reorder(const void* query,
            const FlattenInterfacePtr& flatten,
            DistHeapPtr& candidate_heap,
            int64_t k,
            IteratorFilterContext* iter_ctx,
            QueryContext& ctx,
            const DistanceRecordVector* rabitq_lower_bound_candidates = nullptr) const;

    void
    elp_optimize();

    void
    check_and_init_raw_vector(const FlattenInterfaceParamPtr& raw_vector_param,
                              const IndexCommonParam& common_param,
                              bool is_create_new = true);

    void
    init_resize_bit_and_reorder();

    void
    cal_memory_usage();

    [[nodiscard]] bool
    has_precise_reorder() const {
        return use_reorder_ and not reorder_by_base_;
    }

    [[nodiscard]] bool
    support_force_remove() const {
        return support_force_remove_;
    }

    [[nodiscard]] FlattenInterfacePtr
    get_reorder_codes() const {
        return reorder_by_base_ ? basic_flatten_codes_ : high_precise_codes_;
    }

    void
    fullfill_cache() const;

    // ---- Build-with-cache acceleration path ----
    // The build flow is automatically taken by Build() when ImportCache() has
    // populated cache_ before. Steps:
    //   (1) warm_start: seed each new node's neighbors from the cached
    //       neighbors keyed by source_id, classify nodes into hit/missed.
    //   (2) refine hit nodes (use_self_as_entry=true), then missed nodes
    //       (use_self_as_entry=false). Each refine round is two-phase:
    //       parallel search+select then serial writeback, plus a sharded
    //       reverse-edge install with distance-reuse and O(M) dedup.
    //   (3) build route graphs via ODescent over the sampled level ids.
    bool
    has_loaded_cache() const {
        return this->cache_ != nullptr && not this->cache_->neighbors_.empty();
    }

    std::vector<int64_t>
    build_with_cache(const DatasetPtr& data);

    // Internal scratch state shared across the 6 phases of build_with_cache.
    // Each phase reads/writes a well-defined subset; threading this through
    // a single struct keeps phase signatures small (1 ref instead of 8 outs)
    // and makes the data flow explicit at a glance.
    struct BuildCachePlan {
        Vector<int64_t> valid_indices;
        Vector<InnerIdType> inner_ids;
        Vector<Vector<InnerIdType>> route_graph_ids;
        std::vector<InnerIdType> inserted_inner_ids;
        std::unordered_map<InnerIdType, uint32_t> inner_id_to_input_idx;
        std::unordered_map<std::string, InnerIdType> source_id_to_new_inner;
        std::vector<int64_t> failed_ids;
        std::vector<InnerIdType> hit_ids;
        std::vector<InnerIdType> missed_ids;

        BuildCachePlan(Allocator* allocator)
            : valid_indices(allocator), inner_ids(allocator), route_graph_ids(allocator) {
        }
    };

    void
    cache_collect_valid_indices(const DatasetPtr& data, BuildCachePlan& plan);

    void
    cache_setup_metadata_serial(const DatasetPtr& data, BuildCachePlan& plan);

    void
    cache_encode_codes_parallel(const DatasetPtr& data, BuildCachePlan& plan);

    void
    cache_warm_start_and_classify(BuildCachePlan& plan);

    void
    cache_run_refine_two_phase(const DatasetPtr& data, BuildCachePlan& plan);

    void
    cache_rebuild_route_graphs(BuildCachePlan& plan);

    DistHeapPtr
    collect_refine_candidates(const DatasetPtr& data,
                              InnerIdType inner_id,
                              uint32_t input_idx,
                              const FlattenInterfacePtr& flatten_codes,
                              uint32_t refine_ef,
                              bool use_self_as_entry) const;

    void
    select_refine_neighbors_with_distances(const DatasetPtr& data,
                                           InnerIdType inner_id,
                                           uint32_t input_idx,
                                           const FlattenInterfacePtr& flatten_codes,
                                           uint32_t refine_ef,
                                           bool use_self_as_entry,
                                           Vector<InnerIdType>& out_neighbors,
                                           Vector<float>& out_distances) const;

    void
    refine_nodes_two_phase(const DatasetPtr& data,
                           const std::vector<InnerIdType>& ids_to_refine,
                           std::string_view phase_name,
                           uint32_t rounds,
                           uint32_t refine_ef,
                           bool use_self_as_entry,
                           const FlattenInterfacePtr& flatten_codes,
                           const std::unordered_map<InnerIdType, uint32_t>& inner_id_to_input_idx);

private:
    FlattenInterfacePtr basic_flatten_codes_{nullptr};
    FlattenInterfacePtr high_precise_codes_{nullptr};

    Vector<GraphInterfacePtr> route_graphs_;
    GraphInterfacePtr bottom_graph_{nullptr};
    SparseGraphDatacellParamPtr hierarchical_datacell_param_{nullptr};

    bool use_elp_optimizer_{false};
    bool ignore_reorder_{false};
    bool build_by_base_{false};
    bool reorder_by_base_{false};

    BasicSearcherPtr searcher_;
    ParallelSearcherPtr parallel_searcher_;

    std::default_random_engine level_generator_{2021};
    double mult_{1.0};

    InnerIdType entry_point_id_{INVALID_ENTRY_POINT};

    ODescentParameterPtr odescent_param_{nullptr};
    std::string graph_type_{GRAPH_TYPE_VALUE_NSW};

    uint64_t ef_construct_{400};
    float alpha_{1.0};

    std::shared_ptr<VisitedListPool> pool_{nullptr};

    mutable std::shared_mutex global_mutex_;
    mutable MutexArrayPtr neighbors_mutex_;
    mutable std::shared_mutex add_mutex_;
    mutable std::shared_mutex force_remove_mutex_;

    std::atomic<InnerIdType> max_capacity_{0};

    uint64_t resize_increase_count_bit_{
        DEFAULT_RESIZE_BIT};  // 2^resize_increase_count_bit_ for resize count

    static constexpr uint64_t DEFAULT_RESIZE_BIT = 10;

    std::atomic<int64_t> delete_count_{0};

    std::shared_ptr<Optimizer<BasicSearcher>> optimizer_;

    bool create_new_raw_vector_{false};
    FlattenInterfacePtr temporary_build_flatten_codes_{nullptr};
    FlattenInterfacePtr raw_vector_{nullptr};

    ReorderInterfacePtr reorder_{nullptr};

    bool use_old_serial_format_{false};

    bool support_duplicate_{false};
    bool support_force_remove_{false};
    float duplicate_distance_threshold_{0.0F};

    bool persist_source_id_{false};

    std::unique_ptr<HGraphCache> cache_{nullptr};

    // Build-time warm-start cache hit statistics, populated by
    // cache_warm_start_and_classify() when Build() takes the
    // build_with_cache() path (i.e. after ImportCache()). A negative
    // hit-rate marks "this index was not built from an imported cache",
    // in which case GetStats() emits a skipped_reason instead of values.
    float build_cache_hit_rate_{-1.0F};
    uint64_t build_cache_hit_nodes_{0};
    uint64_t build_cache_missed_nodes_{0};
};
}  // namespace vsag
