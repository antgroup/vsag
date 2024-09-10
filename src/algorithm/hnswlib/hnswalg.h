
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
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "../../default_allocator.h"
#include "../../simd/simd.h"
#include "algorithm_interface.h"
#include "block_manager.h"
#include "visited_list_pool.h"

namespace hnswlib {
using tableint = unsigned int;
using linklistsizeint = unsigned int;
using reverselinklist = std::
    unordered_set<tableint, std::hash<tableint>, std::equal_to<>, vsag::AllocatorWrapper<tableint>>;
struct CompareByFirst {
    constexpr bool
    operator()(std::pair<float, tableint> const& a,
               std::pair<float, tableint> const& b) const noexcept {
        return a.first < b.first;
    }
};
using MaxHeap = std::priority_queue<std::pair<float, tableint>,
                                    std::vector<std::pair<float, tableint>>,
                                    CompareByFirst>;
const static float THRESHOLD_ERROR = 1e-6;

class HierarchicalNSW : public AlgorithmInterface<float> {
private:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_ = 0;
    mutable std::atomic<size_t> cur_element_count_{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t dim_{0};

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    VisitedListPool* visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_{};

    std::mutex global_{};
    std::vector<std::recursive_mutex> link_list_locks_{};

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0};
    size_t offsetLevel0_{0};
    size_t label_offset_{0};

    bool normalize_{false};
    float* molds_{nullptr};

    BlockManager* data_level0_memory_{nullptr};
    char** link_lists_{nullptr};
    int* element_levels_{nullptr};  // keeps level of each element

    bool use_reversed_edges_{false};
    reverselinklist** reversed_level0_link_list_{nullptr};
    std::unordered_map<int, reverselinklist>** reversed_link_lists_{nullptr};

    size_t data_size_{0};

    size_t data_element_per_block_{0};

    DISTFUNC fstdistfunc_{nullptr};
    void* dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock_{};  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_{};

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    vsag::Allocator* allocator_{nullptr};
    std::shared_ptr<vsag::AllocatorWrapper<tableint>> reverse_link_list_allocator_{nullptr};

    mutable std::atomic<uint64_t> metric_distance_computations_{0};
    mutable std::atomic<uint64_t> metric_hops_{0};

    vsag::DistanceFunc ip_func_{nullptr};

    // flag to replace deleted elements (marked as deleted) during insertion
    bool allow_replace_deleted_{false};

    std::mutex deleted_elements_lock_{};               // lock for deleted_elements_
    std::unordered_set<tableint> deleted_elements_{};  // contains internal ids of deleted elements

public:
    HierarchicalNSW(SpaceInterface* s) {
    }

    HierarchicalNSW(SpaceInterface* s,
                    const std::string& location,
                    bool nmslib = false,
                    size_t max_elements = 0,
                    bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }

    HierarchicalNSW(SpaceInterface* s,
                    size_t max_elements,
                    vsag::Allocator* allocator,
                    size_t M = 16,
                    size_t ef_construction = 200,
                    bool use_reversed_edges = false,
                    bool normalize = false,
                    size_t block_size_limit = 128 * 1024 * 1024,
                    size_t random_seed = 100,
                    bool allow_replace_deleted = false);

    ~HierarchicalNSW() override;

    void
    normalizeVector(const void*& data_point, std::shared_ptr<float[]>& normalize_data) const;

    float
    getDistanceByLabel(labeltype label, const void* data_point) override;

    bool
    isValidLabel(labeltype label) override;

    inline std::mutex&
    getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }

    inline labeltype
    getExternalLabel(tableint internal_id) const {
        labeltype value;
        std::memcpy(&value,
                    data_level0_memory_->GetElementPtr(internal_id, label_offset_),
                    sizeof(labeltype));
        return value;
    }

    inline void
    setExternalLabel(tableint internal_id, labeltype label) const {
        *(labeltype*)(data_level0_memory_->GetElementPtr(internal_id, label_offset_)) = label;
    }

    inline labeltype*
    getExternalLabeLp(tableint internal_id) const {
        return (labeltype*)(data_level0_memory_->GetElementPtr(internal_id, label_offset_));
    }

    inline reverselinklist&
    getEdges(tableint internal_id, int level = 0) {
        if (level != 0) {
            auto& edge_map_ptr = reversed_link_lists_[internal_id];
            if (edge_map_ptr == nullptr) {
                edge_map_ptr = new std::unordered_map<int, reverselinklist>();
            }
            auto& edge_map = *edge_map_ptr;
            if (edge_map.find(level) == edge_map.end()) {
                edge_map.insert(
                    std::make_pair(level, reverselinklist(*reverse_link_list_allocator_)));
            }
            return edge_map.at(level);
        } else {
            auto& edge_ptr = reversed_level0_link_list_[internal_id];
            if (edge_ptr == nullptr) {
                edge_ptr = new reverselinklist(*reverse_link_list_allocator_);
            }
            return *edge_ptr;
        }
    }

    void
    updateConnections(tableint internal_id,
                      const std::vector<tableint>& cand_neighbors,
                      int level,
                      bool is_update);

    bool
    checkReverseConnection();

    inline char*
    getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_->GetElementPtr(internal_id, offsetData_));
    }

    std::priority_queue<std::pair<float, labeltype>>
    bruteForce(const void* data_point, int64_t k) override;

    int
    getRandomLevel(double reverse_size);

    size_t
    getMaxElements() override {
        return max_elements_;
    }

    size_t
    getCurrentElementCount() override {
        return cur_element_count_;
    }

    size_t
    getDeletedCount() override {
        return num_deleted_;
    }

    MaxHeap
    searchBaseLayer(tableint ep_id, const void* data_point, int layer);

    template <bool has_deletions, bool collect_metrics = false>
    MaxHeap
    searchBaseLayerST(tableint ep_id,
                      const void* data_point,
                      size_t ef,
                      BaseFilterFunctor* isIdAllowed = nullptr) const;

    template <bool has_deletions, bool collect_metrics = false>
    MaxHeap
    searchBaseLayerST(tableint ep_id,
                      const void* data_point,
                      float radius,
                      int64_t ef,
                      BaseFilterFunctor* isIdAllowed = nullptr) const;

    void
    getNeighborsByHeuristic2(MaxHeap& top_candidates, size_t M);

    linklistsizeint*
    getLinklist0(tableint internal_id) const {
        return (linklistsizeint*)(data_level0_memory_->GetElementPtr(internal_id, offsetLevel0_));
    }

    linklistsizeint*
    getLinklist(tableint internal_id, int level) const {
        return (linklistsizeint*)(link_lists_[internal_id] + (level - 1) * size_links_per_element_);
    }

    linklistsizeint*
    getLinklistAtLevel(tableint internal_id, int level) const {
        return level == 0 ? getLinklist0(internal_id) : getLinklist(internal_id, level);
    }

    tableint
    mutuallyConnectNewElement(tableint cur_c, MaxHeap& top_candidates, int level, bool isUpdate);

    void
    resizeIndex(size_t new_max_elements) override;

    size_t
    calcSerializeSize() override;

    void
    saveIndex(void* d) override;
    // save index to a file stream
    void
    saveIndex(std::ostream& out_stream) override;

    void
    saveIndex(const std::string& location) override;

    void
    SerializeImpl(StreamWriter& writer);

    // load using reader
    void
    loadIndex(std::function<void(uint64_t, uint64_t, void*)> read_func,
              SpaceInterface* s,
              size_t max_elements_i) override;

    // load index from a file stream
    void
    loadIndex(std::istream& in_stream, SpaceInterface* s, size_t max_elements_i) override;

    // origin load function
    void
    loadIndex(const std::string& location, SpaceInterface* s, size_t max_elements_i = 0);

    void
    DeserializeImpl(StreamReader& reader, SpaceInterface* s, size_t max_elements_i = 0);

    const float*
    getDataByLabel(labeltype label) const override;
    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void
    markDelete(labeltype label);

    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void
    markDeletedInternal(tableint internalId);

    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    *
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void
    unmarkDelete(labeltype label);
    /*
    * Remove the deleted mark of the node.
    */
    void
    unmarkDeletedInternal(tableint internalId);

    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool
    isMarkedDeleted(tableint internalId) const {
        unsigned char* ll_cur = ((unsigned char*)getLinklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }

    static inline unsigned short int
    getListCount(const linklistsizeint* ptr) {
        return *((unsigned short int*)ptr);
    }

    static inline void
    setListCount(linklistsizeint* ptr, unsigned short int size) {
        *((unsigned short int*)(ptr)) = *((unsigned short int*)&size);
    }

    /*
    * Adds point.
    */
    bool
    addPoint(const void* data_point, labeltype label) override;

    void
    modifyOutEdge(tableint old_internal_id, tableint new_internal_id);

    void
    modifyInEdges(tableint right_internal_id, tableint wrong_internal_id, bool is_erase);

    bool
    swapConnections(tableint pre_internal_id, tableint post_internal_id);

    void
    dealNoInEdge(tableint id, int level, int m_curmax, int skip_c);

    void
    removePoint(labeltype label);

    tableint
    addPoint(const void* data_point, labeltype label, int level);

    std::priority_queue<std::pair<float, labeltype>>
    searchKnn(const void* query_data,
              size_t k,
              uint64_t ef,
              BaseFilterFunctor* isIdAllowed = nullptr) const override;

    std::priority_queue<std::pair<float, labeltype>>
    searchRange(const void* query_data,
                float radius,
                uint64_t ef,
                BaseFilterFunctor* isIdAllowed = nullptr) const override;

    void
    checkIntegrity();
};
}  // namespace hnswlib
