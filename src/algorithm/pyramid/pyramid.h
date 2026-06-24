
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

#include <memory>
#include <optional>
#include <utility>

#include "algorithm/inner_index_interface.h"
#include "algorithm/pyramid/pyramid_build_cache.h"
#include "datacell/graph_interface.h"
#include "datacell/sparse_graph_datacell_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/filter/filter_headers.h"
#include "impl/heap/distance_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/reorder/flatten_reorder.h"
#include "impl/searcher/basic_searcher.h"
#include "index_feature_list.h"
#include "io/memory_io/memory_io_parameter.h"
#include "pyramid_index_node.h"
#include "pyramid_zparameters.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "query_context.h"
#include "utils/lock_strategy.h"

namespace vsag {

std::vector<std::string>
split(const std::string& str, char delimiter);

/**
 * @brief Pyramid: hierarchical graph index for path-labeled vectors.
 *
 * Organizes vectors into a tree of IndexNode graphs keyed by hierarchical
 * path labels (e.g. "country/city"). Search traverses matching branches.
 * Introduced since v0.14.
 */
class Pyramid : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    Pyramid(const PyramidParamPtr& pyramid_param, const IndexCommonParam& common_param)
        : InnerIndexInterface(pyramid_param, common_param),
          hierarchies_(common_param.allocator_.get()),
          odescent_param_(pyramid_param->odescent_param),
          index_min_size_(pyramid_param->index_min_size),
          graph_type_(pyramid_param->graph_type),
          default_rabitq_one_bit_search_(pyramid_param->use_reorder and
                                         pyramid_param->base_codes_param->name ==
                                             RABITQ_SPLIT_DATA_CELL),
          support_duplicate_(pyramid_param->support_duplicate),
          persist_source_id_(pyramid_param->persist_source_id),
          cache_(std::make_unique<PyramidBuildCache>(common_param.allocator_.get())) {
        base_codes_ = FlattenInterface::MakeInstance(pyramid_param->base_codes_param, common_param);
        if (pyramid_param->has_hierarchies) {
            for (const auto& h_param : pyramid_param->hierarchies) {
                auto graph_param = pyramid_param->graph_param;
                if (h_param.max_degree != pyramid_param->max_degree) {
                    auto new_gp = std::make_shared<SparseGraphDatacellParameter>();
                    new_gp->FromJson(graph_param->ToJson());
                    new_gp->max_degree_ = h_param.max_degree;
                    graph_param = new_gp;
                }
                auto root =
                    std::make_unique<IndexNode>(allocator_, graph_param, h_param.index_min_size);
                auto h = std::make_unique<Hierarchy>(h_param.name, std::move(root), allocator_);
                h->no_build_levels.assign(h_param.no_build_levels.begin(),
                                          h_param.no_build_levels.end());
                h->ef_construction = h_param.ef_construction;
                h->alpha = h_param.alpha;
                hierarchies_.insert({h_param.name, std::move(h)});
            }
        } else {
            auto root = std::make_unique<IndexNode>(
                allocator_, pyramid_param->graph_param, index_min_size_);
            auto h = std::make_unique<Hierarchy>("", std::move(root), allocator_);
            h->no_build_levels.assign(pyramid_param->no_build_levels.begin(),
                                      pyramid_param->no_build_levels.end());
            h->ef_construction = pyramid_param->ef_construction;
            h->alpha = pyramid_param->alpha;
            hierarchies_.insert({"", std::move(h)});
        }
        points_mutex_ = std::make_shared<PointsMutex>(max_capacity_, allocator_);
        searcher_ = std::make_unique<BasicSearcher>(common_param, points_mutex_);
        if (has_precise_reorder()) {
            precise_codes_ =
                FlattenInterface::MakeInstance(pyramid_param->precise_codes_param, common_param);
        }
        if (use_reorder_) {
            reorder_ = std::make_shared<FlattenReorder>(get_reorder_codes(), allocator_);
        }
        if (pyramid_param->store_raw_vector) {
            raw_vector_ =
                FlattenInterface::MakeInstance(pyramid_param->raw_vector_param, common_param);
            has_raw_vector_ = true;
        }
    }

    explicit Pyramid(const ParamPtr& param, const IndexCommonParam& common_param)
        : Pyramid(std::dynamic_pointer_cast<PyramidParameters>(param), common_param){};

    ~Pyramid() override = default;

    std::vector<int64_t>
    Add(const DatasetPtr& base) override;

    std::vector<int64_t>
    Build(const DatasetPtr& base) override;

    float
    CalcDistanceById(const float* query,
                     int64_t id,
                     bool calculate_precise_distance = true) const override;

    DatasetPtr
    CalcDistancesById(const float* query,
                      const int64_t* ids,
                      int64_t count,
                      bool calculate_precise_distance = true) const override;

    DatasetPtr
    CalDistanceById(const float* query,
                    const int64_t* ids,
                    int64_t count,
                    bool calculate_precise_distance = true,
                    int64_t topk = -1) const override;

    void
    Deserialize(StreamReader& reader) override;

    [[nodiscard]] InnerIndexPtr
    ExportModel(const IndexCommonParam& param) const override;

    [[nodiscard]] InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return std::make_shared<Pyramid>(this->create_param_ptr_, param);
    }

    IndexType
    GetIndexType() const override {
        return IndexType::PYRAMID;
    }

    std::string
    GetName() const override {
        return INDEX_PYRAMID;
    }

    int64_t
    GetNumElements() const override;

    int64_t
    GetNumberRemoved() const override;

    uint32_t
    Remove(const std::vector<int64_t>& ids, RemoveMode mode) override;

    std::string
    GetStats() const override;

    std::string
    AnalyzeIndexBySearch(const SearchRequest& request) override;

    void
    InitFeatures() override;

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

    DatasetPtr
    SearchWithRequest(const SearchRequest& request) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    SetImmutable() override;

    void
    Train(const vsag::DatasetPtr& base) override;

    void
    ExportCache(std::ostream& out_stream) const override;

    void
    ImportCache(std::istream& in_stream) override;

    void
    GetVectorByInnerId(InnerIdType inner_id, float* data) const override;

    friend class PyramidAnalyzer;

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
    read_streaming_body(StreamReader& reader, const MetadataPtr& metadata);

    void
    serialize_hierarchies(StreamWriter& writer) const;

    void
    deserialize_hierarchies(StreamReader& reader, const JsonType& basic_info);

    // RAII guard that returns the VisitedList to the pool on scope exit,
    // ensuring no leak if the search throws.
    class VisitedListGuard {
    public:
        explicit VisitedListGuard(VisitedListPool* pool) : pool_(pool), vl_(pool->TakeOne()) {
        }
        ~VisitedListGuard() {
            if (vl_ != nullptr) {
                pool_->ReturnOne(vl_);
            }
        }
        VisitedListGuard(const VisitedListGuard&) = delete;
        VisitedListGuard&
        operator=(const VisitedListGuard&) = delete;
        [[nodiscard]] const VisitedListPtr&
        get() const {
            return vl_;
        }

    private:
        VisitedListPool* pool_;
        VisitedListPtr vl_;
    };
    void
    serialize_source_id_table(StreamWriter& writer) const;

    void
    deserialize_source_id_table(StreamReader& reader);

    /// One named hierarchy with its own root IndexNode and build parameters.
    struct Hierarchy {
        std::string name;                          // hierarchy name (empty = default)
        std::unique_ptr<IndexNode> root{nullptr};  // root node of the tree
        Vector<int32_t> no_build_levels;           // depths where graph build is skipped
        uint64_t ef_construction{400};             // expansion factor during graph build
        float alpha{1.2F};  // Relative Neighborhood Graph pruning coefficient

        Hierarchy(const std::string& n, std::unique_ptr<IndexNode> r, Allocator* alloc)
            : name(n), root(std::move(r)), no_build_levels(alloc) {
        }
    };

    /// Pre-create the IndexNode tree structure from the path labels.
    static void
    populate_path_tree(Hierarchy& h, const std::string* paths, int64_t count);

    /// Insert vectors and their path labels into the hierarchy tree.
    void
    add_to_hierarchy(Hierarchy& h,
                     const float* data_vectors,
                     const std::string* paths,
                     const Vector<int64_t>& data_biases,
                     int64_t local_cur_element_count);

    /// Search a single hierarchy along a path prefix, accumulating candidates.
    void
    search_hierarchy(const Hierarchy& h,
                     const SearchFunc& search_func,
                     const VisitedListPtr& vl,
                     DistHeapPtr& search_result,
                     const std::string& path,
                     const InnerSearchParam& search_param,
                     ReasoningContext* reasoning_ctx) const;

    /// Grow internal storage to accommodate new_max_capacity vectors.
    void
    resize(int64_t new_max_capacity);

    /// Execute search across hierarchies and return results as a Dataset.
    DatasetPtr
    search_impl(const DatasetPtr& query,
                const SearchFunc& search_func,
                InnerSearchParam& search_param,
                QueryContext& ctx,
                const std::string& hierarchy_name,
                const DistanceRecordVector* rabitq_lower_bound_candidates = nullptr) const;

    InnerSearchParam
    create_knn_search_param(const PyramidSearchParameters& parsed_param,
                            int64_t k,
                            const FilterPtr& filter,
                            const std::optional<float>& threshold = std::nullopt) const;

    /// Probabilistic check: should total_count trigger a new entry-point update?
    bool
    is_update_entry_point(uint64_t total_count) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double rand_value = distribution(level_generator_);
        return static_cast<double>(total_count) * rand_value < 1.0;
    }

    /// Build all hierarchy graphs via ODescent in batch mode.
    std::vector<int64_t>
    build_by_odescent(const DatasetPtr& base);

    /// Recursively insert a single vector into the hierarchy tree.
    void
    add_one_point(const Hierarchy& h,
                  IndexNode* node,
                  InnerIdType inner_id,
                  const float* vector,
                  uint64_t ef_construction = 0,
                  bool use_self_as_entry = false);

    /// Split a path string into its hierarchical segments.
    static std::vector<std::vector<std::string>>
    parse_path(const std::string& path);

    /// Search a single IndexNode's graph, returning candidate heap.
    DistHeapPtr
    search_node(const IndexNode* node,
                const VisitedListPtr& vl,
                const InnerSearchParam& search_param,
                const DatasetPtr& query,
                const FlattenInterfacePtr& codes,
                QueryContext& ctx,
                uint64_t subindex_ef_search,
                DistanceRecordVector* rabitq_lower_bound_candidates = nullptr) const;

    [[nodiscard]] bool
    has_precise_reorder() const {
        return use_reorder_ and not base_codes_->SupportSplitCodeStorage();
    }

    [[nodiscard]] FlattenInterfacePtr
    get_reorder_codes() const {
        return base_codes_->SupportSplitCodeStorage() ? base_codes_ : precise_codes_;
    }

    [[nodiscard]] FlattenInterfacePtr
    decodable_codes() const {
        return raw_vector_ != nullptr ? raw_vector_
                                      : (has_precise_reorder() ? precise_codes_ : base_codes_);
    }

    bool
    has_loaded_cache() const {
        return this->cache_ != nullptr && not this->cache_->Empty();
    }

    void
    fulfill_cache(PyramidBuildCache& cache_snapshot) const;

    std::vector<int64_t>
    build_with_cache(const DatasetPtr& base);

    static void
    collect_graph_nodes(IndexNode* node,
                        const std::string& node_path,
                        std::vector<std::pair<std::string, IndexNode*>>& out);

    void
    init_index_nodes_with_ids(IndexNode* node) const;

private:
    ODescentParameterPtr odescent_param_{nullptr};  // ODescent build parameters
    UnorderedMap<std::string, std::unique_ptr<Hierarchy>> hierarchies_;  // named hierarchies
    FlattenInterfacePtr base_codes_{nullptr};          // coarse codes for graph build/search
    FlattenInterfacePtr precise_codes_{nullptr};       // precise codes for reorder (if enabled)
    FlattenInterfacePtr raw_vector_{nullptr};          // original vectors for decode-only paths
    std::unique_ptr<VisitedListPool> pool_ = nullptr;  // pool of visited-lists for search

    MutexArrayPtr points_mutex_{nullptr};                // per-point locks for concurrent access
    std::unique_ptr<BasicSearcher> searcher_ = nullptr;  // graph traversal engine
    int64_t max_capacity_{0};                            // allocated capacity
    int64_t cur_element_count_{0};                       // number of vectors currently stored
    std::atomic<int64_t> delete_count_{0};               // number of deleted vectors
    bool support_duplicate_{false};                      // whether to allow duplicate ids

    mutable std::shared_mutex resize_mutex_;        // guards resize operations
    mutable std::mutex cur_element_count_mutex_;    // guards cur_element_count_ updates
    std::string graph_type_{GRAPH_TYPE_VALUE_NSW};  // graph algorithm type
    bool default_rabitq_one_bit_search_{false};     // default split lower-bound search

    std::mutex entry_point_mutex_;  // guards entry-point selection
    std::default_random_engine level_generator_{
        2021};                              // random number generator for level promotion
    ReorderInterfacePtr reorder_{nullptr};  // reorder helper (if use_reorder_)

    uint32_t index_min_size_{0};  // min node size before graph is built

    bool persist_source_id_{false};  // whether to persist source_id in serialization

    std::unique_ptr<PyramidBuildCache> cache_{nullptr};  // per-graph caches for warm-start build

    float build_cache_hit_rate_{-1.0F};     // cache hit rate from last cache-based build
    uint64_t build_cache_hit_nodes_{0};     // number of nodes with cache hit
    uint64_t build_cache_missed_nodes_{0};  // number of nodes without cache hit
};

}  // namespace vsag
