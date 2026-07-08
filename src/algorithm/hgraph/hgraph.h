
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
#include "index_common_param.h"
#include "index_feature_list.h"
#include "typing.h"
#include "utils/lock_strategy.h"
#include "utils/util_functions.h"
#include "utils/visited_list.h"
#include "vsag/index.h"
#include "vsag/index_features.h"

namespace vsag {
class IteratorFilterContext;

/**
 * @brief HGraph: hierarchical navigable graph index.
 *
 * Multi-layer NSW graph with bottom-level graph + optional upper route graphs.
 * Supports quantized codes, reorder, attribute filtering, cache warm-start,
 * force remove, and iterative search. Introduced since v0.12.
 */
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
    Add(const DatasetPtr& data) override;

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

    std::unordered_map<std::string, uint64_t>
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

    /// Set the number of threads used during Build().
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
    SetPreciseCodesIO(const std::shared_ptr<Reader>& reader);

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

    /// Map hgraph JSON parameters (legacy helper for external param mapping).
    static JsonType
    map_hgraph_param(const JsonType& hgraph_json);

    /**
     * @brief Extract a pointer to the data vector at the given index.
     *
     * Dispatches on data_type_ to the correct typed getter on the dataset
     * and offsets by `index * dim_`. Returns nullptr if the underlying
     * pointer is absent.
     */
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

    /// Sample a random level from the exponential distribution for a new node.
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

    /// Build all graphs (bottom + route) via ODescent in batch mode.
    std::vector<int64_t>
    build_by_odescent(const DatasetPtr& data);

    /// Insert a single point into the graph(s) at the given level.
    void
    add_one_point(const void* data, int level, InnerIdType id);

    /// Write codes for inner_id into the persistent flatten storage.
    void
    insert_persistent_codes(const void* data, InnerIdType inner_id);

    /// Insert a single point, optionally also inserting codes.
    void
    add_one_point(const void* data, int level, InnerIdType id, bool insert_codes);

    /// Core graph insertion: connect the new node and update neighbors.
    /// Returns true if the entry point was updated.
    bool
    graph_add_one(const void* data, int level, InnerIdType inner_id);

    /// Grow internal storage to at least new_size capacity.
    void
    resize(uint64_t new_size);

    /// Create a single route (upper-layer) graph from the hierarchical params.
    GraphInterfacePtr
    generate_one_route_graph();

    /// Search a single graph layer, returning a candidate distance heap.
    /// @param ctx  may be nullptr during add (non-query) scenarios.
    template <InnerSearchMode mode = InnerSearchMode::KNN_SEARCH>
    DistHeapPtr
    search_one_graph(const void* query,
                     const GraphInterfacePtr& graph,
                     const FlattenInterfacePtr& flatten,
                     InnerSearchParam& inner_search_param,
                     const VisitedListPtr& vt,
                     QueryContext* ctx,
                     DistanceRecordVector* rabitq_lower_bound_candidates = nullptr) const;

    /// Overload that accepts an IteratorFilterContext for iterative search.
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
    MetadataPtr
    collect_streaming_header() const override;

    void
    serialize_streaming_body(StreamWriter& writer) const override;

    void
    deserialize_streaming_body(StreamReader& reader, const MetadataPtr& metadata) override;

    void
    load_streaming_body(StreamReader& reader,
                        const MetadataPtr& metadata,
                        const LoadParameters& parameters) override;

    void
    read_streaming_body(StreamReader& reader,
                        const MetadataPtr& metadata,
                        const LoadParameters* load_parameters = nullptr);

    void
    deserialize_label_info_streaming(StreamReader& reader) const;

    // since v0.15: serialize basic index metadata to JSON.
    JsonType
    serialize_basic_info() const;

    /// Restore basic index metadata from JSON.
    void
    deserialize_basic_info(const JsonType& jsonify_basic_info);

    /// Write label (external id) mappings to stream.
    void
    serialize_label_info(StreamWriter& writer) const;

    /// Read label (external id) mappings from stream.
    void
    deserialize_label_info(StreamReader& reader) const;

    /// Legacy serialize for version [0.12.*, 0.14.*].
    void
    serialize_basic_info_v0_14(StreamWriter& writer) const;

    /// Legacy deserialize for version [0.12.*, 0.14.*].
    void
    deserialize_basic_info_v0_14(StreamReader& reader);

    /// Force-remove a single label: physically delete from all graphs.
    uint32_t
    force_remove_one(int64_t label);

    /// Scan all graphs to find a new entry point after the current one is removed.
    void
    find_new_entry_point();

    /// Force-remove a single inner_id from a specific graph layer.
    void
    graph_force_remove_one(const InnerIdType& inner_id,
                           const FlattenInterfacePtr& flatten,
                           const GraphInterfacePtr& graph);

    /// Move data at inner_id `from` to `to`, updating all references.
    void
    move_id(InnerIdType from, InnerIdType to);

    /// Compact internal storage after deletions.
    void
    shrink_to_fit();

    /// Flat brute-force search used when the index is too small or graph is unavailable.
    template <InnerSearchMode mode = InnerSearchMode::KNN_SEARCH>
    DistHeapPtr
    brute_force_search(const void* query,
                       const FilterPtr& filter,
                       int64_t topk,
                       float radius,
                       QueryContext* ctx) const;

private:
    /// Reorder the candidate heap using precise codes, updating in-place.
    void
    reorder(const void* query,
            const FlattenInterfacePtr& flatten,
            DistHeapPtr& candidate_heap,
            int64_t k,
            IteratorFilterContext* iter_ctx,
            QueryContext& ctx,
            const DistanceRecordVector* rabitq_lower_bound_candidates = nullptr) const;

    /// Run ELP (Edge-Link Pruning) optimizer on the bottom graph.
    void
    elp_optimize();

    /// Initialize raw_vector_ from the given parameter if needed.
    void
    check_and_init_raw_vector(const FlattenInterfaceParamPtr& raw_vector_param,
                              const IndexCommonParam& common_param,
                              bool is_create_new = true);

    /// Compute resize_increase_count_bit_ and initialize reorder if enabled.
    void
    init_resize_bit_and_reorder();

    /// Compute memory usage breakdown and cache the result.
    void
    cal_memory_usage();

    /// True when reorder uses a separate high-precision flatten (not base codes).
    [[nodiscard]] bool
    has_precise_reorder() const {
        return use_reorder_ and not reorder_by_base_;
    }

    /// True when force-remove is enabled.
    [[nodiscard]] bool
    support_force_remove() const {
        return support_force_remove_;
    }

    /// Return the flatten used for reorder (base or high-precision).
    [[nodiscard]] FlattenInterfacePtr
    get_reorder_codes() const {
        return reorder_by_base_ ? basic_flatten_codes_ : high_precise_codes_;
    }

    /// Populate the neighbor cache from the index state.
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

    /// Build from imported cache: warm-start, refine, and rebuild route graphs.
    std::vector<int64_t>
    build_with_cache(const DatasetPtr& data);

    /// Internal scratch state shared across the 6 phases of build_with_cache.
    /// Each phase reads/writes a well-defined subset; threading this through
    /// a single struct keeps phase signatures small (1 ref instead of 8 outs)
    /// and makes the data flow explicit at a glance.
    struct BuildCachePlan {
        Vector<int64_t> valid_indices;                // input indices that pass filter
        Vector<InnerIdType> inner_ids;                // allocated inner ids per valid input
        Vector<Vector<InnerIdType>> route_graph_ids;  // ids assigned to each route graph level
        std::vector<InnerIdType> inserted_inner_ids;  // all successfully inserted ids
        std::unordered_map<InnerIdType, uint32_t> inner_id_to_input_idx;  // inner_id -> input index
        std::unordered_map<std::string, InnerIdType>
            source_id_to_new_inner;           // source_id -> inner_id
        std::vector<int64_t> failed_ids;      // ids that failed insertion
        std::vector<InnerIdType> hit_ids;     // nodes with cache hit (warm-started)
        std::vector<InnerIdType> missed_ids;  // nodes without cache hit

        BuildCachePlan(Allocator* allocator)
            : valid_indices(allocator), inner_ids(allocator), route_graph_ids(allocator) {
        }
    };

    /// Phase 1: collect valid input indices (serial).
    void
    cache_collect_valid_indices(const DatasetPtr& data, BuildCachePlan& plan);

    /// Phase 2: set up metadata (ids, route graph assignment) (serial).
    void
    cache_setup_metadata_serial(const DatasetPtr& data, BuildCachePlan& plan);

    /// Phase 3: encode all codes in parallel.
    void
    cache_encode_codes_parallel(const DatasetPtr& data, BuildCachePlan& plan);

    /// Phase 4: warm-start neighbors from cache, classify nodes as hit/missed.
    void
    cache_warm_start_and_classify(BuildCachePlan& plan);

    /// Phase 5: refine hit and missed nodes via two-phase parallel refine.
    void
    cache_run_refine_two_phase(const DatasetPtr& data, BuildCachePlan& plan);

    /// Phase 6: rebuild route (upper-layer) graphs via ODescent.
    void
    cache_rebuild_route_graphs(BuildCachePlan& plan);

    /// Collect refine candidates for a single node via graph search.
    DistHeapPtr
    collect_refine_candidates(const DatasetPtr& data,
                              InnerIdType inner_id,
                              uint32_t input_idx,
                              const FlattenInterfacePtr& flatten_codes,
                              uint32_t refine_ef,
                              bool use_self_as_entry) const;

    /// Select refined neighbors and their distances for a single node.
    void
    select_refine_neighbors_with_distances(const DatasetPtr& data,
                                           InnerIdType inner_id,
                                           uint32_t input_idx,
                                           const FlattenInterfacePtr& flatten_codes,
                                           uint32_t refine_ef,
                                           bool use_self_as_entry,
                                           Vector<InnerIdType>& out_neighbors,
                                           Vector<float>& out_distances) const;

    /// Refine a batch of nodes in two phases: parallel search+select, then serial writeback.
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
    FlattenInterfacePtr basic_flatten_codes_{nullptr};  // coarse/quantized codes for graph search
    FlattenInterfacePtr high_precise_codes_{nullptr};   // precise codes for reorder (optional)

    Vector<GraphInterfacePtr> route_graphs_;   // upper-layer route graphs
    GraphInterfacePtr bottom_graph_{nullptr};  // base-level graph (all vectors)
    SparseGraphDatacellParamPtr hierarchical_datacell_param_{nullptr};  // params for route graphs

    bool use_elp_optimizer_{false};  // enable ELP edge-link pruning
    bool ignore_reorder_{false};     // skip reorder even if configured
    bool build_by_base_{false};      // build graph using base (not quantized) codes
    bool reorder_by_base_{false};    // use base codes for reorder (no separate precise)

    BasicSearcherPtr searcher_;              // single-thread graph searcher
    ParallelSearcherPtr parallel_searcher_;  // multi-thread graph searcher

    std::default_random_engine level_generator_{
        2021};          // random number generator for level sampling
    double mult_{1.0};  // level multiplier (1/ln(max_degree))

    InnerIdType entry_point_id_{INVALID_ENTRY_POINT};  // top-level entry point

    ODescentParameterPtr odescent_param_{nullptr};  // ODescent build parameters
    std::string graph_type_{GRAPH_TYPE_VALUE_NSW};  // graph algorithm type

    uint64_t ef_construct_{400};  // expansion factor during graph construction
    float alpha_{1.0};            // Relative Neighborhood Graph pruning coefficient

    std::shared_ptr<VisitedListPool> pool_{nullptr};  // pool of visited-lists for search

    mutable std::shared_mutex global_mutex_;        // guards total_count_, entry_point_id_
    mutable MutexArrayPtr neighbors_mutex_;         // per-node locks for neighbor lists
    mutable std::shared_mutex add_mutex_;           // serializes Add() operations
    mutable std::shared_mutex force_remove_mutex_;  // serializes force-remove operations

    std::atomic<InnerIdType> max_capacity_{0};  // allocated storage capacity

    uint64_t resize_increase_count_bit_{DEFAULT_RESIZE_BIT};  // log2(resize batch size)

    static constexpr uint64_t DEFAULT_RESIZE_BIT = 10;  // default resize batch = 1024

    std::atomic<int64_t> delete_count_{0};  // number of force-removed vectors

    std::shared_ptr<Optimizer<BasicSearcher>> optimizer_;  // search parameter optimizer

    bool create_new_raw_vector_{false};  // whether a separate raw vector exists
    FlattenInterfacePtr temporary_build_flatten_codes_{nullptr};  // temp flatten during build
    FlattenInterfacePtr raw_vector_{nullptr};  // raw float vectors (for distance calc)

    ReorderInterfacePtr reorder_{nullptr};  // reorder helper

    bool use_old_serial_format_{false};  // true when deserialized from legacy format

    bool support_duplicate_{false};             // allow duplicate external ids
    bool support_force_remove_{false};          // enable physical deletion
    float duplicate_distance_threshold_{0.0F};  // distance threshold for duplicate detection

    bool persist_source_id_{false};  // whether to persist source_id in serialization

    std::unique_ptr<HGraphCache> cache_{nullptr};  // neighbor cache for warm-start build

    float build_cache_hit_rate_{-1.0F};     // cache hit rate from last cache-based build
    uint64_t build_cache_hit_nodes_{0};     // number of nodes with cache hit
    uint64_t build_cache_missed_nodes_{0};  // number of nodes without cache hit
};
}  // namespace vsag
