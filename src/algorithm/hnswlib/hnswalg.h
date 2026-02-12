
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
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "algorithm_interface.h"
#include "block_manager.h"
#include "datacell/flatten_interface.h"
#include "datacell/graph_interface.h"
#include "impl/allocator/default_allocator.h"
#include "index/iterator_filter.h"
#include "simd/simd.h"
#include "utils/lock_strategy.h"
#include "utils/prefetch.h"
#include "visited_list_pool.h"
#include "vsag/dataset.h"
#include "vsag/iterator_context.h"

namespace hnswlib {
using linklistsizeint = unsigned int;
using reverselinklist = vsag::UnorderedSet<uint32_t>;
struct CompareByFirst {
    constexpr bool
    operator()(std::pair<float, InnerIdType> const& a,
               std::pair<float, InnerIdType> const& b) const noexcept {
        return a.first < b.first;
    }
};
using MaxHeap = std::priority_queue<std::pair<float, InnerIdType>,
                                    vsag::Vector<std::pair<float, InnerIdType>>,
                                    CompareByFirst>;

class HierarchicalNSW : public AlgorithmInterface<float> {
private:
    static const unsigned char DELETE_MARK = 0x01;

    uint64_t max_elements_ = 0;
    mutable std::atomic<uint64_t> cur_element_count_{0};  // current number of elements
    uint64_t size_data_per_element_{0};
    uint64_t size_links_per_element_{0};
    mutable std::atomic<uint64_t> num_deleted_{0};  // number of deleted elements
    uint64_t M_{0};
    uint64_t maxM_{0};
    uint64_t maxM0_{0};
    uint64_t ef_construction_{0};
    uint64_t dim_{0};

    double mult_{0.0}, rev_size_{0.0};
    int max_level_{0};

    VisitedListPool* visited_list_pool_{nullptr};

    mutable std::shared_mutex
        resize_mutex_{};  // Ensures safety during the resize process; is the largest lock.
    mutable std::shared_mutex
        max_level_mutex_{};  // Ensures access safety for global max_level and entry point.
    mutable vsag::MutexArrayPtr
        points_locks_;  // Ensures access safety for the link list and label of a specific point.
    mutable std::shared_mutex
        label_lookup_lock_{};  // Ensures access safety for the global label lookup table.

    InnerIdType enterpoint_node_{0};

    uint64_t size_links_level0_{0};
    uint64_t offset_data_{0};
    uint64_t offsetLevel0_{0};
    uint64_t label_offset_{0};

    bool normalize_{false};
    float* molds_{nullptr};

    std::shared_ptr<BlockManager> data_level0_memory_{nullptr};
    char** link_lists_{nullptr};
    int* element_levels_{nullptr};  // keeps level of each element

    bool use_reversed_edges_{false};
    reverselinklist** reversed_level0_link_list_{nullptr};
    vsag::UnorderedMap<int, reverselinklist>** reversed_link_lists_{nullptr};

    uint64_t data_size_{0};
    uint64_t prefetch_jump_code_size_{1};

    uint64_t data_element_per_block_{0};

    DISTFUNC fstdistfunc_{nullptr};
    void* dist_func_param_{nullptr};

    vsag::PGUnorderedMap<LabelType, InnerIdType> label_lookup_;

    std::default_random_engine level_generator_{2021};
    mutable std::default_random_engine update_probability_generator_;

    vsag::Allocator* allocator_{nullptr};

    mutable std::atomic<uint64_t> metric_distance_computations_{0};
    mutable std::atomic<uint64_t> metric_hops_{0};

    vsag::DistanceFuncType ip_func_{nullptr};

    // flag to replace deleted elements (marked as deleted) during insertion
    bool allow_replace_deleted_{false};

    std::mutex deleted_elements_lock_{};  // lock for deleted_elements_
    vsag::PGUnorderedMap<LabelType, InnerIdType>
        deleted_elements_;  // contains labels and internal ids of deleted elements

    bool immutable_{false};

public:
    HierarchicalNSW(SpaceInterface* s,
                    uint64_t max_elements,
                    vsag::Allocator* allocator,
                    uint64_t M = 16,
                    uint64_t ef_construction = 200,
                    bool use_reversed_edges = false,
                    bool normalize = false,
                    uint64_t block_size_limit = 128 * 1024 * 1024,
                    uint64_t random_seed = 100,
                    bool allow_replace_deleted = true);

    ~HierarchicalNSW() override;

    void
    normalizeVector(const void*& data_point, std::shared_ptr<float[]>& normalize_data) const;

    float
    getDistanceByLabel(LabelType label, const void* data_point) override;

    float
    getDistanceByInternalId(uint32_t internal_id, const void* data_point) override;

    float
    getSelfDistanceByInternalId(uint32_t internal_id) override;

    tl::expected<vsag::DatasetPtr, vsag::Error>
    getBatchDistanceByLabel(const int64_t* ids, const void* data_point, int64_t count) override;
    std::pair<int64_t, int64_t>
    getMinAndMaxId() override;
    bool
    isValidLabel(LabelType label) override;

    bool
    isTombLabel(LabelType label) override;

    virtual uint32_t
    getInternalId(LabelType label) override;

    virtual void
    getNeighborsInternalId(uint32_t internal_id, vsag::Vector<InnerIdType>& neighbor_ids) override;

    uint64_t
    getMaxDegree() {
        return maxM0_;
    };

    linklistsizeint*
    get_linklist0(InnerIdType internal_id) const {
        // only for test now
        return (linklistsizeint*)(data_level0_memory_->GetElementPtr(internal_id, offsetLevel0_));
    }

    inline LabelType
    getExternalLabel(InnerIdType internal_id) const {
        vsag::SharedLock lock(points_locks_, internal_id);
        LabelType value;
        std::memcpy(&value,
                    data_level0_memory_->GetElementPtr(internal_id, label_offset_),
                    sizeof(LabelType));
        return value;
    }

    inline void
    setExternalLabel(InnerIdType internal_id, LabelType label) const {
        vsag::LockGuard lock(points_locks_, internal_id);
        std::memcpy(data_level0_memory_->GetElementPtr(internal_id, label_offset_),
                    &label,
                    sizeof(LabelType));
    }

    inline reverselinklist&
    getEdges(InnerIdType internal_id, int level = 0) {
        if (level != 0) {
            auto& edge_map_ptr = reversed_link_lists_[internal_id];
            if (edge_map_ptr == nullptr) {
                edge_map_ptr = new vsag::UnorderedMap<int, reverselinklist>(allocator_);
            }
            auto& edge_map = *edge_map_ptr;
            if (edge_map.find(level) == edge_map.end()) {
                edge_map.insert(std::make_pair(level, reverselinklist(allocator_)));
            }
            return edge_map.at(level);
        } else {
            auto& edge_ptr = reversed_level0_link_list_[internal_id];
            if (edge_ptr == nullptr) {
                edge_ptr = new reverselinklist(allocator_);
            }
            return *edge_ptr;
        }
    }

    void
    updateConnections(InnerIdType internal_id,
                      const vsag::Vector<InnerIdType>& cand_neighbors,
                      int level,
                      bool is_update);

    bool
    checkReverseConnection();

    inline char*
    getDataByInternalId(InnerIdType internal_id) const {
        return (data_level0_memory_->GetElementPtr(internal_id, offset_data_));
    }

    std::priority_queue<std::pair<float, LabelType>>
    bruteForce(const void* data_point,
               int64_t k,
               const vsag::FilterPtr is_id_allowed = nullptr) const override;

    int
    getRandomLevel(double reverse_size);

    uint64_t
    getMaxElements() override {
        return max_elements_;
    }

    uint64_t
    getCurrentElementCount() override {
        return cur_element_count_;
    }

    uint64_t
    getDeletedCount() override {
        return num_deleted_;
    }

    vsag::PGUnorderedMap<LabelType, InnerIdType>
    getDeletedElements() override {
        return deleted_elements_;
    }

    MaxHeap
    searchBaseLayer(InnerIdType ep_id, const void* data_point, int layer) const;

    template <bool has_deletions, bool collect_metrics = false>
    MaxHeap
    searchBaseLayerST(InnerIdType ep_id,
                      const void* data_point,
                      uint64_t ef,
                      const vsag::FilterPtr is_id_allowed = nullptr,
                      const float skip_ratio = 0.9f,
                      vsag::Allocator* allocator = nullptr,
                      vsag::IteratorFilterContext* iter_ctx = nullptr) const;

    template <bool has_deletions, bool collect_metrics = false>
    MaxHeap
    searchBaseLayerST(InnerIdType ep_id,
                      const void* data_point,
                      float radius,
                      int64_t ef,
                      const vsag::FilterPtr is_id_allowed = nullptr) const;

    void
    getNeighborsByHeuristic2(MaxHeap& top_candidates, uint64_t M);

    void
    setBatchNeigohbors(InnerIdType internal_id,
                       int level,
                       const InnerIdType* neighbors,
                       uint64_t neigbor_count);

    void
    appendNeigohbor(InnerIdType internal_id, int level, InnerIdType neighbor, uint64_t max_degree);

    linklistsizeint*
    getLinklist0(InnerIdType internal_id) const {
        return (linklistsizeint*)(data_level0_memory_->GetElementPtr(internal_id, offsetLevel0_));
    }

    linklistsizeint*
    getLinklist(InnerIdType internal_id, int level) const {
        return (linklistsizeint*)(link_lists_[internal_id] + (level - 1) * size_links_per_element_);
    }

    linklistsizeint*
    getLinklistAtLevel(InnerIdType internal_id, int level) const {
        return level == 0 ? getLinklist0(internal_id) : getLinklist(internal_id, level);
    }

    inline void
    getLinklistAtLevel(InnerIdType internal_id, int level, void* neighbors) const {
        if (level == 0) {
            vsag::SharedLock lock(points_locks_, internal_id);
            auto src = data_level0_memory_->GetElementPtr(internal_id, offsetLevel0_);
            std::memcpy(neighbors, src, size_links_level0_);
        } else {
            vsag::SharedLock lock(points_locks_, internal_id);
            std::memcpy(neighbors,
                        link_lists_[internal_id] + (level - 1) * size_links_per_element_,
                        size_links_per_element_);
        }
    }

    InnerIdType
    mutuallyConnectNewElement(InnerIdType cur_c, MaxHeap& top_candidates, int level, bool isUpdate);

    void
    resizeIndex(uint64_t new_max_elements) override;

    void
    setDataAndGraph(vsag::FlattenInterfacePtr& data,
                    vsag::GraphInterfacePtr& graph,
                    vsag::Vector<LabelType>& ids);

    uint64_t
    calcSerializeSize() override;

    void
    saveIndex(StreamWriter& writer) override;

    void
    SerializeImpl(StreamWriter& writer);

    void
    loadIndex(StreamReader& buffer_reader, SpaceInterface* s, uint64_t max_elements_i = 0) override;

    void
    DeserializeImpl(StreamReader& reader, SpaceInterface* s, uint64_t max_elements_i = 0);

    const float*
    getDataByLabel(LabelType label) const override;

    void
    copyDataByLabel(LabelType label, void* data_point) override;

    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void
    markDelete(LabelType label);

    /*
    * Remove mark on an element that deleted, does NOT really change the current graph.
    */
    void
    recoverMarkDelete(LabelType label);

    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void
    markDeletedInternal(InnerIdType internal_id);

    /*
     * Recover the procee
     */
    void
    recoveryMarkDeletedInternal(InnerIdType internal_id);

    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool
    isMarkedDeleted(InnerIdType internal_id) const {
        // no need to use fine-grained lock
        auto src = data_level0_memory_->GetElementPtr(internal_id, offsetLevel0_);
        unsigned char* ll_cur = ((unsigned char*)src) + 2;
        return *ll_cur & DELETE_MARK;
    }

    static inline unsigned short int
    getListCount(const linklistsizeint* ptr) {
        return *((unsigned short int*)ptr);
    }

    static inline void
    setListCount(linklistsizeint* ptr, unsigned short int size) {
        *((unsigned short int*)(ptr)) = size;
    }

    /*
    * Adds point.
    */
    bool
    addPoint(const void* data_point, LabelType label) override;

    void
    modifyOutEdge(InnerIdType old_internal_id, InnerIdType new_internal_id);

    void
    modifyInEdges(InnerIdType right_internal_id, InnerIdType wrong_internal_id, bool is_erase);

    bool
    swapConnections(InnerIdType pre_internal_id, InnerIdType post_internal_id);

    void
    dealNoInEdge(InnerIdType id, int level, int m_curmax, int skip_c);

    void
    updateLabel(LabelType old_label, LabelType new_label);

    void
    updateVector(LabelType label, const void* data_point);

    void
    removePoint(LabelType label);

    InnerIdType
    addPoint(const void* data_point, LabelType label, int level);

    std::priority_queue<std::pair<float, LabelType>>
    searchKnn(const void* query_data,
              uint64_t k,
              uint64_t ef,
              const vsag::FilterPtr is_id_allowed = nullptr,
              const float skip_ratio = 0.9f,
              vsag::Allocator* allocator = nullptr,
              vsag::IteratorFilterContext* iter_ctx = nullptr,
              bool is_last_filter = false) const override;

    std::priority_queue<std::pair<float, LabelType>>
    searchRange(const void* query_data,
                float radius,
                uint64_t ef,
                const vsag::FilterPtr is_id_allowed = nullptr) const override;

    void
    reset();

    bool
    init_memory_space() override;

    uint64_t
    estimateMemory(uint64_t num_elements) override;

    void
    setImmutable() override;
};
}  // namespace hnswlib
