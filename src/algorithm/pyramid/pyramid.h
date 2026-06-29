
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
#include <utility>

#include "algorithm/inner_index_interface.h"
#include "datacell/graph_interface.h"
#include "datacell/sparse_graph_datacell_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/filter/filter_headers.h"
#include "impl/heap/distance_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/reorder/flatten_reorder.h"
#include "impl/searcher/basic_searcher.h"
#include "index_feature_list.h"
#include "io/memory_io_parameter.h"
#include "pyramid_zparameters.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "query_context.h"
#include "utils/lock_strategy.h"

namespace vsag {

class IndexNode;
using SearchFunc = std::function<DistHeapPtr(const IndexNode* node, const VisitedListPtr& vl)>;

std::vector<std::string>
split(const std::string& str, char delimiter);

/**
 * @brief IndexNode: a tree node in the Pyramid hierarchy.
 *
 * Each IndexNode optionally holds a small graph (when the number of ids
 * exceeds index_min_size_) and a map of child nodes keyed by path segment.
 * The tree structure mirrors the hierarchical path labels (e.g. "a/b/c")
 * assigned to vectors at insertion time.
 */
class IndexNode {
public:
    enum class Status { NO_INDEX = 0, GRAPH = 1, FLAT = 2 };

public:
    IndexNode(Allocator* allocator_, GraphInterfaceParamPtr graph_param, uint32_t index_min_size);

    /// Build the internal graph using ODescent over the stored ids.
    void
    Build(ODescent& odescent);

    /// Allocate the graph storage if not yet done.
    void
    Init();

    /**
     * @brief Recursively search this node and its matching children.
     *
     * @param search_func  functor that searches a single node's graph;
     *                     typically bound to the caller's query and ef.
     * @param vl           visited-list for dedup across the recursion.
     * @param search_result  output heap accumulating candidates.
     * @param ef_search    expansion factor passed to the graph search.
     */
    void
    Search(const SearchFunc& search_func,
           const VisitedListPtr& vl,
           const DistHeapPtr& search_result,
           uint64_t ef_search) const;

    void
    AddChild(const std::string& key);

    IndexNode*
    GetChild(const std::string& key, bool need_init = false);

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader);

    friend class PyramidAnalyzer;

public:
    GraphInterfacePtr graph_{nullptr};  // graph over the ids in this node
    InnerIdType entry_point_{0};        // entry point for graph search
    uint32_t level_{0};                 // depth in the tree (root = 0)
    mutable std::shared_mutex mutex_;   // per-node lock for concurrent add/search

    Vector<InnerIdType> ids_;          // internal ids stored at this node
    uint32_t index_min_size_{0};       // threshold to trigger graph build
    Status status_{Status::NO_INDEX};  // current build state

private:
    UnorderedMap<std::string, std::unique_ptr<IndexNode>> children_;  // keyed by path segment
    Allocator* allocator_{nullptr};
    GraphInterfaceParamPtr graph_param_{nullptr};
};

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
          support_duplicate_(pyramid_param->support_duplicate) {
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
        if (use_reorder_) {
            precise_codes_ =
                FlattenInterface::MakeInstance(pyramid_param->precise_codes_param, common_param);
            reorder_ = std::make_shared<FlattenReorder>(precise_codes_, allocator_);
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
    CalDistanceById(const float* query,
                    const int64_t* ids,
                    int64_t count,
                    bool calculate_precise_distance = true) const override;

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

    void
    Serialize(StreamWriter& writer) const override;

    void
    SetImmutable() override;

    void
    Train(const vsag::DatasetPtr& base) override;

    void
    GetVectorByInnerId(InnerIdType inner_id, float* data) const override;

    friend class PyramidAnalyzer;

private:
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
                     const InnerSearchParam& search_param) const;

    /// Grow internal storage to accommodate new_max_capacity vectors.
    void
    resize(int64_t new_max_capacity);

    /// Execute search across hierarchies and return results as a Dataset.
    DatasetPtr
    search_impl(const DatasetPtr& query,
                const SearchFunc& search_func,
                InnerSearchParam& search_param,
                const std::string& hierarchy_name = "") const;

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
    add_one_point(const Hierarchy& h, IndexNode* node, InnerIdType inner_id, const float* vector);

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
                uint64_t subindex_ef_search) const;

private:
    ODescentParameterPtr odescent_param_{nullptr};  // ODescent build parameters
    UnorderedMap<std::string, std::unique_ptr<Hierarchy>> hierarchies_;  // named hierarchies
    FlattenInterfacePtr base_codes_{nullptr};          // coarse codes for graph build/search
    FlattenInterfacePtr precise_codes_{nullptr};       // precise codes for reorder (if enabled)
    std::unique_ptr<VisitedListPool> pool_ = nullptr;  // pool of visited-lists for search

    MutexArrayPtr points_mutex_{nullptr};                // per-point locks for concurrent access
    std::unique_ptr<BasicSearcher> searcher_ = nullptr;  // graph traversal engine
    int64_t max_capacity_{0};                            // allocated capacity
    int64_t cur_element_count_{0};                       // number of vectors currently stored
    std::atomic<int64_t> delete_count_{0};               // number of deleted vectors
    bool support_duplicate_{false};                      // whether to allow duplicate ids

    mutable std::shared_mutex resize_mutex_;        // guards resize operations
    std::mutex cur_element_count_mutex_;            // guards cur_element_count_ updates
    std::string graph_type_{GRAPH_TYPE_VALUE_NSW};  // graph algorithm type

    std::mutex entry_point_mutex_;  // guards entry-point selection
    std::default_random_engine level_generator_{
        2021};                              // random number generator for level promotion
    ReorderInterfacePtr reorder_{nullptr};  // reorder helper (if use_reorder_)

    uint32_t index_min_size_{0};  // min node size before graph is built
    bool immutable_{false};       // true after SetImmutable()
};

}  // namespace vsag
