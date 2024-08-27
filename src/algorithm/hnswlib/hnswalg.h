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

#include <assert.h>
#include <stdlib.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "../../default_allocator.h"
#include "../../simd/simd.h"
#include "hnswlib.h"
#include "visited_list_pool.h"

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;
typedef std::
    unordered_set<tableint, std::hash<tableint>, std::equal_to<>, vsag::AllocatorWrapper<tableint>>
        reverselinklist;

const static float THRESHOLD_ERROR = 1e-6;

class HierarchicalNSW : public AlgorithmInterface<float> {
private:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
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
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::recursive_mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

    bool normalize_ = false;
    float* molds_ = nullptr;

    BlockManager* data_level0_memory_;
    char** link_lists_{nullptr};
    int* element_levels_;  // keeps level of each element

    bool use_reversed_edges_ = false;
    reverselinklist** reversed_level0_link_list_{nullptr};
    std::map<int, reverselinklist>** reversed_link_lists_{nullptr};

    size_t data_size_{0};

    size_t data_element_per_block_{0};

    DISTFUNC fstdistfunc_;
    void* dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    vsag::Allocator* allocator_;
    std::shared_ptr<vsag::AllocatorWrapper<tableint>> reverse_link_list_allocator_;

    mutable std::atomic<long> metric_distance_computations_{0};
    mutable std::atomic<long> metric_hops_{0};

    vsag::DistanceFunc ip_func_;

    bool allow_replace_deleted_ =
        false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;               // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

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
                    bool allow_replace_deleted = false)
        : allocator_(allocator),
          link_list_locks_(max_elements),
          label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
          allow_replace_deleted_(allow_replace_deleted),
          use_reversed_edges_(use_reversed_edges),
          normalize_(normalize) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        dim_ = data_size_ / sizeof(float);
        M_ = M;
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);

        element_levels_ = (int*)allocator->Allocate(max_elements * sizeof(int));

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        reverse_link_list_allocator_.reset(new vsag::AllocatorWrapper<tableint>(allocator_));

        if (use_reversed_edges_) {
            reversed_level0_link_list_ =
                (reverselinklist**)allocator->Allocate(max_elements_ * sizeof(reverselinklist*));
            memset(reversed_level0_link_list_, 0, max_elements_ * sizeof(reverselinklist*));
            reversed_link_lists_ = (std::map<int, reverselinklist>**)allocator->Allocate(
                max_elements_ * sizeof(std::map<int, reverselinklist>*));
            memset(
                reversed_link_lists_, 0, max_elements_ * sizeof(std::map<int, reverselinklist>*));
        }

        if (normalize) {
            ip_func_ = vsag::InnerProduct;
            molds_ = (float*)allocator->Allocate(max_elements_ * sizeof(float));
        }

        data_level0_memory_ =
            new BlockManager(max_elements_, size_data_per_element_, block_size_limit, allocator_);
        data_element_per_block_ = block_size_limit / size_data_per_element_;

        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count_ = 0;

        visited_list_pool_ = new VisitedListPool(1, max_elements, allocator_);

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        link_lists_ = (char**)allocator->Allocate(sizeof(void*) * max_elements_);
        if (link_lists_ == nullptr)
            throw std::runtime_error(
                "Not enough memory: HierarchicalNSW failed to allocate linklists");
        memset(link_lists_, 0, sizeof(void*) * max_elements_);
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }

    ~HierarchicalNSW() {
        delete data_level0_memory_;
        for (tableint i = 0; i < max_elements_; i++) {
            if (element_levels_[i] > 0 || link_lists_[i] != nullptr)
                allocator_->Deallocate(link_lists_[i]);
        }

        if (use_reversed_edges_) {
            for (tableint i = 0; i < max_elements_; i++) {
                auto& in_edges_level0 = *(reversed_level0_link_list_ + i);
                if (in_edges_level0) {
                    delete in_edges_level0;
                }
                auto& in_edges = *(reversed_link_lists_ + i);
                if (in_edges) {
                    delete in_edges;
                }
            }
            allocator_->Deallocate(reversed_link_lists_);
            allocator_->Deallocate(reversed_level0_link_list_);
        }
        if (normalize_) {
            allocator_->Deallocate(molds_);
        }
        allocator_->Deallocate(element_levels_);
        allocator_->Deallocate(link_lists_);
        delete visited_list_pool_;
    }

    void
    normalize_vector(const void*& data_point, std::shared_ptr<float[]>& normalize_data) const {
        if (normalize_) {
            float query_mold = std::sqrt(ip_func_(data_point, data_point, dist_func_param_));
            normalize_data.reset(new float[dim_]);
            for (int i = 0; i < dim_; ++i) {
                normalize_data[i] = ((float*)data_point)[i] / query_mold;
            }
            data_point = normalize_data.get();
        }
    }

    float
    getDistanceByLabel(labeltype label, const void* data_point) override {
        std::unique_lock<std::mutex> lock_table(label_lookup_lock);

        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internal_id = search->second;
        lock_table.unlock();
        std::shared_ptr<float[]> normalize_query;
        normalize_vector(data_point, normalize_query);
        float dist = fstdistfunc_(data_point, getDataByInternalId(internal_id), dist_func_param_);
        return dist;
    }

    bool
    isValidLabel(labeltype label) override {
        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        bool is_valid = (label_lookup_.find(label) != label_lookup_.end());
        lock_table.unlock();
        return is_valid;
    }

    struct CompareByFirst {
        constexpr bool
        operator()(std::pair<float, tableint> const& a,
                   std::pair<float, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

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
                    data_level0_memory_->getElementPtr(internal_id, label_offset_),
                    sizeof(labeltype));
        return value;
    }

    inline void
    setExternalLabel(tableint internal_id, labeltype label) const {
        *(labeltype*)(data_level0_memory_->getElementPtr(internal_id, label_offset_)) = label;
    }

    inline labeltype*
    getExternalLabeLp(tableint internal_id) const {
        return (labeltype*)(data_level0_memory_->getElementPtr(internal_id, label_offset_));
    }

    inline reverselinklist&
    getEdges(tableint internal_id, int level = 0) {
        if (level != 0) {
            auto& edge_map_ptr = reversed_link_lists_[internal_id];
            if (edge_map_ptr == nullptr) {
                edge_map_ptr = new std::map<int, reverselinklist>();
            }
            auto& edge_map = *edge_map_ptr;
            if (edge_map.find(level) == edge_map.end()) {
                edge_map.insert(
                    std::make_pair(level, reverselinklist(*reverse_link_list_allocator_.get())));
            }
            return edge_map.at(level);
        } else {
            auto& edge_ptr = reversed_level0_link_list_[internal_id];
            if (edge_ptr == nullptr) {
                edge_ptr = new reverselinklist(*reverse_link_list_allocator_.get());
            }
            return *edge_ptr;
        }
    }

    void
    updateConnections(tableint internal_id,
                      const std::vector<tableint>& cand_neighbors,
                      int level,
                      bool is_update) {
        linklistsizeint* ll_cur;
        if (level == 0)
            ll_cur = get_linklist0(internal_id);
        else
            ll_cur = get_linklist(internal_id, level);

        auto cur_size = getListCount(ll_cur);
        tableint* data = (tableint*)(ll_cur + 1);

        if (is_update && use_reversed_edges_) {
            for (int i = 0; i < cur_size; ++i) {
                auto id = data[i];
                auto& in_edges = getEdges(id, level);
                // remove the node that point to the current node
                std::unique_lock<std::recursive_mutex> lock(link_list_locks_[i]);
                in_edges.erase(internal_id);
            }
        }

        setListCount(ll_cur, cand_neighbors.size());
        for (size_t i = 0; i < cand_neighbors.size(); i++) {
            auto id = cand_neighbors[i];
            data[i] = cand_neighbors[i];
            if (not use_reversed_edges_) {
                continue;
            }
            std::unique_lock<std::recursive_mutex> lock(link_list_locks_[id]);
            auto& in_edges = getEdges(id, level);
            in_edges.insert(internal_id);
        }
    }

    bool
    checkReverseConnection() {
        int edge_count = 0;
        int reversed_edge_count = 0;
        for (int internal_id = 0; internal_id < cur_element_count_; ++internal_id) {
            for (int level = 0; level <= element_levels_[internal_id]; ++level) {
                unsigned int* data;
                if (level == 0) {
                    data = get_linklist0(internal_id);
                } else {
                    data = get_linklist(internal_id, level);
                }
                auto link_list = data + 1;
                auto size = getListCount(data);
                edge_count += size;
                reversed_edge_count += getEdges(internal_id, level).size();
                for (int j = 0; j < size; ++j) {
                    auto id = link_list[j];
                    const auto& in_edges = getEdges(id, level);
                    if (in_edges.find(internal_id) == in_edges.end()) {
                        std::cout << "can not find internal_id (" << internal_id
                                  << ") in its neighbor (" << id << ")" << std::endl;
                        return false;
                    }
                }
            }
        }

        if (edge_count != reversed_edge_count) {
            std::cout << "mismatch: edge_count (" << edge_count << ") != reversed_edge_count("
                      << reversed_edge_count << ")" << std::endl;
            return false;
        }

        return true;
    }

    inline char*
    getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_->getElementPtr(internal_id, offsetData_));
    }

    std::priority_queue<std::pair<float, labeltype>>
    bruteForce(const void* data_point, int64_t k) override {
        std::priority_queue<std::pair<float, labeltype>> results;
        for (uint32_t i = 0; i < cur_element_count_; i++) {
            float dist = fstdistfunc_(data_point, getDataByInternalId(i), dist_func_param_);
            if (results.size() < k) {
                results.push({dist, *getExternalLabeLp(i)});
            } else {
                float current_max_dist = results.top().first;
                if (dist < current_max_dist) {
                    results.pop();
                    results.push({dist, *getExternalLabeLp(i)});
                }
            }
        }
        return results;
    }

    int
    getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
    }

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

    std::priority_queue<std::pair<float, tableint>,
                        std::vector<std::pair<float, tableint>>,
                        CompareByFirst>
    searchBaseLayer(tableint ep_id, const void* data_point, int layer) {
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<float, tableint>,
                            std::vector<std::pair<float, tableint>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<float, tableint>,
                            std::vector<std::pair<float, tableint>>,
                            CompareByFirst>
            candidateSet;

        float lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<float>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<float, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock<std::recursive_mutex> lock(link_list_locks_[curNodeNum]);

            int* data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
                //                    data = (int *) (link_lists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint* datal = (tableint*)(data + 1);
#ifdef USE_SSE
            _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
#ifdef USE_SSE
                size_t pre_l = std::min(j, size - 2);
                _mm_prefetch((char*)(visited_array + *(datal + pre_l + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + pre_l + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                char* currObj1 = (getDataByInternalId(candidate_id));

                float dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }

    template <bool has_deletions, bool collect_metrics = false>
    std::priority_queue<std::pair<float, tableint>,
                        std::vector<std::pair<float, tableint>>,
                        CompareByFirst>
    searchBaseLayerST(tableint ep_id,
                      const void* data_point,
                      size_t ef,
                      BaseFilterFunctor* isIdAllowed = nullptr) const {
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<float, tableint>,
                            std::vector<std::pair<float, tableint>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<float, tableint>,
                            std::vector<std::pair<float, tableint>>,
                            CompareByFirst>
            candidate_set;

        float lowerBound;
        if ((!has_deletions || !isMarkedDeleted(ep_id)) &&
            ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))) {
            float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<float>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<float, tableint> current_node_pair = candidate_set.top();

            if ((-current_node_pair.first) > lowerBound &&
                (top_candidates.size() == ef || (!isIdAllowed && !has_deletions))) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int* data = (int*)get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
            //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops_++;
                metric_distance_computations_ += size;
            }

            auto vector_data_ptr = data_level0_memory_->getElementPtr((*(data + 1)), offsetData_);
#ifdef USE_SSE
            _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(vector_data_ptr, _MM_HINT_T0);
            _mm_prefetch((char*)(data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                size_t pre_l = std::min(j, size - 2);
                auto vector_data_ptr =
                    data_level0_memory_->getElementPtr((*(data + pre_l + 1)), offsetData_);
#ifdef USE_SSE
                _mm_prefetch((char*)(visited_array + *(data + pre_l + 1)), _MM_HINT_T0);
                _mm_prefetch(vector_data_ptr, _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char* currObj1 = (getDataByInternalId(candidate_id));
                    float dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef || lowerBound > dist) {
                        candidate_set.emplace(-dist, candidate_id);
                        auto vector_data_ptr = data_level0_memory_->getElementPtr(
                            candidate_set.top().second, offsetLevel0_);
#ifdef USE_SSE
                        _mm_prefetch(vector_data_ptr, _MM_HINT_T0);
#endif

                        if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
                            ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))
                            top_candidates.emplace(dist, candidate_id);

                        if (top_candidates.size() > ef)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    template <bool has_deletions, bool collect_metrics = false>
    std::priority_queue<std::pair<float, tableint>,
                        std::vector<std::pair<float, tableint>>,
                        CompareByFirst>
    searchBaseLayerST(tableint ep_id,
                      const void* data_point,
                      float radius,
                      int64_t ef,
                      BaseFilterFunctor* isIdAllowed = nullptr) const {
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<float, tableint>,
                            std::vector<std::pair<float, tableint>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<float, tableint>,
                            std::vector<std::pair<float, tableint>>,
                            CompareByFirst>
            candidate_set;

        float lowerBound;
        if ((!has_deletions || !isMarkedDeleted(ep_id)) &&
            ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))) {
            float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            lowerBound = dist;
            if (dist <= radius + THRESHOLD_ERROR)
                top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<float>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;
        uint64_t visited_count = 0;

        while (!candidate_set.empty()) {
            std::pair<float, tableint> current_node_pair = candidate_set.top();

            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int* data = (int*)get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
            //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops_++;
                metric_distance_computations_ += size;
            }

            auto vector_data_ptr = data_level0_memory_->getElementPtr((*(data + 1)), offsetData_);
#ifdef USE_SSE
            _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(vector_data_ptr, _MM_HINT_T0);
            _mm_prefetch((char*)(data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                size_t pre_l = std::min(j, size - 2);
                auto vector_data_ptr =
                    data_level0_memory_->getElementPtr((*(data + pre_l + 1)), offsetData_);
#ifdef USE_SSE
                _mm_prefetch((char*)(visited_array + *(data + pre_l + 1)), _MM_HINT_T0);
                _mm_prefetch(vector_data_ptr, _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;
                    ++visited_count;

                    char* currObj1 = (getDataByInternalId(candidate_id));
                    float dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    if (visited_count < ef || dist <= radius + THRESHOLD_ERROR ||
                        lowerBound > dist) {
                        candidate_set.emplace(-dist, candidate_id);
                        auto vector_data_ptr = data_level0_memory_->getElementPtr(
                            candidate_set.top().second, offsetLevel0_);
#ifdef USE_SSE
                        _mm_prefetch(vector_data_ptr, _MM_HINT_T0);  ////////////////////////
#endif

                        if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
                            ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))
                            top_candidates.emplace(dist, candidate_id);

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }
        while (not top_candidates.empty() &&
               top_candidates.top().first > radius + THRESHOLD_ERROR) {
            top_candidates.pop();
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    void
    getNeighborsByHeuristic2(std::priority_queue<std::pair<float, tableint>,
                                                 std::vector<std::pair<float, tableint>>,
                                                 CompareByFirst>& top_candidates,
                             const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<float, tableint>> queue_closest;
        std::vector<std::pair<float, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<float, tableint> curent_pair = queue_closest.top();
            float floato_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<float, tableint> second_pair : return_list) {
                float curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                                             getDataByInternalId(curent_pair.second),
                                             dist_func_param_);
                if (curdist < floato_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<float, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }

    linklistsizeint*
    get_linklist0(tableint internal_id) const {
        return (linklistsizeint*)(data_level0_memory_->getElementPtr(internal_id, offsetLevel0_));
    }

    linklistsizeint*
    get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint*)(link_lists_[internal_id] + (level - 1) * size_links_per_element_);
    }

    linklistsizeint*
    get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }

    tableint
    mutuallyConnectNewElement(const void* data_point,
                              tableint cur_c,
                              std::priority_queue<std::pair<float, tableint>,
                                                  std::vector<std::pair<float, tableint>>,
                                                  CompareByFirst>& top_candidates,
                              int level,
                              bool isUpdate) {
        size_t m_curmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error(
                "Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock<std::recursive_mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            updateConnections(cur_c, selectedNeighbors, level, isUpdate);
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock<std::recursive_mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint* ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > m_curmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint* data = (tableint*)(ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < m_curmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                    if (use_reversed_edges_) {
                        auto& cur_in_edges = getEdges(cur_c, level);
                        cur_in_edges.insert(selectedNeighbors[idx]);
                    }
                } else {
                    // finding the "weakest" element to replace it with the new one
                    float d_max = fstdistfunc_(getDataByInternalId(cur_c),
                                               getDataByInternalId(selectedNeighbors[idx]),
                                               dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<float, tableint>,
                                        std::vector<std::pair<float, tableint>>,
                                        CompareByFirst>
                        candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(fstdistfunc_(getDataByInternalId(data[j]),
                                                        getDataByInternalId(selectedNeighbors[idx]),
                                                        dist_func_param_),
                                           data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, m_curmax);

                    std::vector<tableint> cand_neighbors;
                    while (candidates.size() > 0) {
                        cand_neighbors.push_back(candidates.top().second);
                        candidates.pop();
                    }
                    updateConnections(selectedNeighbors[idx], cand_neighbors, level, true);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        float d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }

    void
    resizeIndex(size_t new_max_elements) override {
        if (new_max_elements < cur_element_count_)
            throw std::runtime_error(
                "Cannot resize, max element is less than the current number of elements");

        if (visited_list_pool_ != nullptr) {
            delete visited_list_pool_;
        }
        visited_list_pool_ = new VisitedListPool(1, new_max_elements, allocator_);

        auto element_levels_new =
            (int*)allocator_->Reallocate(element_levels_, new_max_elements * sizeof(int));
        if (element_levels_new == nullptr) {
            throw std::runtime_error(
                "Not enough memory: resizeIndex failed to allocate element_levels_");
        }
        element_levels_ = element_levels_new;
        std::vector<std::recursive_mutex>(new_max_elements).swap(link_list_locks_);

        if (normalize_) {
            auto new_molds =
                (float*)allocator_->Reallocate(molds_, new_max_elements * sizeof(float));
            if (new_molds == nullptr) {
                throw std::runtime_error(
                    "Not enough memory: resizeIndex failed to allocate molds_");
            }
            molds_ = new_molds;
        }

        // Reallocate base layer
        if (not data_level0_memory_->resize(new_max_elements))
            throw std::runtime_error(
                "Not enough memory: resizeIndex failed to allocate base layer");

        if (use_reversed_edges_) {
            auto reversed_level0_link_list_new = (reverselinklist**)allocator_->Reallocate(
                reversed_level0_link_list_, new_max_elements * sizeof(reverselinklist*));
            if (reversed_level0_link_list_new == nullptr) {
                throw std::runtime_error(
                    "Not enough memory: resizeIndex failed to allocate reversed_level0_link_list_");
            }
            reversed_level0_link_list_ = reversed_level0_link_list_new;

            memset(reversed_level0_link_list_ + max_elements_,
                   0,
                   (new_max_elements - max_elements_) * sizeof(reverselinklist*));

            auto reversed_link_lists_new = (std::map<int, reverselinklist>**)allocator_->Reallocate(
                reversed_link_lists_, new_max_elements * sizeof(std::map<int, reverselinklist>*));
            if (reversed_link_lists_new == nullptr) {
                throw std::runtime_error(
                    "Not enough memory: resizeIndex failed to allocate reversed_link_lists_");
            }
            reversed_link_lists_ = reversed_link_lists_new;
            memset(reversed_link_lists_ + max_elements_,
                   0,
                   (new_max_elements - max_elements_) * sizeof(std::map<int, reverselinklist>*));
        }

        // Reallocate all other layers
        char** linkLists_new =
            (char**)allocator_->Reallocate(link_lists_, sizeof(void*) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error(
                "Not enough memory: resizeIndex failed to allocate other layers");
        link_lists_ = linkLists_new;
        memset(link_lists_ + max_elements_, 0, (new_max_elements - max_elements_) * sizeof(void*));
        max_elements_ = new_max_elements;
    }

    template <typename T>
    static void
    writeVarToMem(char*& dest, const T& ref) {
        std::memcpy(dest, (char*)&ref, sizeof(T));
        dest += sizeof(T);
    }

    static void
    writeBinaryToMem(char*& dest, const char* src, size_t len) {
        std::memcpy(dest, src, len);
        dest += len;
    }

    void
    saveIndex(void* d) override {
        // std::ofstream output(location, std::ios::binary);
        // std::streampos position;
        char* dest = (char*)d;

        // writeBinaryPOD(output, offsetLevel0_);
        writeVarToMem(dest, offsetLevel0_);
        // writeBinaryPOD(output, max_elements_);
        writeVarToMem(dest, max_elements_);
        // writeBinaryPOD(output, cur_element_count_);
        writeVarToMem(dest, cur_element_count_);
        // writeBinaryPOD(output, size_data_per_element_);
        writeVarToMem(dest, size_data_per_element_);
        // writeBinaryPOD(output, label_offset_);
        writeVarToMem(dest, label_offset_);
        // writeBinaryPOD(output, offsetData_);
        writeVarToMem(dest, offsetData_);
        // writeBinaryPOD(output, maxlevel_);
        writeVarToMem(dest, maxlevel_);
        // writeBinaryPOD(output, enterpoint_node_);
        writeVarToMem(dest, enterpoint_node_);
        // writeBinaryPOD(output, maxM_);
        writeVarToMem(dest, maxM_);

        // writeBinaryPOD(output, maxM0_);
        writeVarToMem(dest, maxM0_);
        // writeBinaryPOD(output, M_);
        writeVarToMem(dest, M_);
        // writeBinaryPOD(output, mult_);
        writeVarToMem(dest, mult_);
        // writeBinaryPOD(output, ef_construction_);
        writeVarToMem(dest, ef_construction_);

        data_level0_memory_->serialize(dest, cur_element_count_);

        for (size_t i = 0; i < cur_element_count_; i++) {
            unsigned int link_list_size =
                element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            // writeBinaryPOD(output, link_list_size);
            writeVarToMem(dest, link_list_size);
            if (link_list_size) {
                // output.write(link_lists_[i], link_list_size);
                writeBinaryToMem(dest, link_lists_[i], link_list_size);
            }
        }

        if (normalize_) {
            writeBinaryToMem(dest, (char*)molds_, max_elements_ * sizeof(float));
        }
        // output.close();
    }

    size_t
    calcSerializeSize() override {
        // std::ofstream output(location, std::ios::binary);
        // std::streampos position;
        size_t size = 0;

        // writeBinaryPOD(output, offsetLevel0_);
        size += sizeof(offsetLevel0_);
        // writeBinaryPOD(output, max_elements_);
        size += sizeof(max_elements_);
        // writeBinaryPOD(output, cur_element_count_);
        size += sizeof(cur_element_count_);
        // writeBinaryPOD(output, size_data_per_element_);
        size += sizeof(size_data_per_element_);
        // writeBinaryPOD(output, label_offset_);
        size += sizeof(label_offset_);
        // writeBinaryPOD(output, offsetData_);
        size += sizeof(offsetData_);
        // writeBinaryPOD(output, maxlevel_);
        size += sizeof(maxlevel_);
        // writeBinaryPOD(output, enterpoint_node_);
        size += sizeof(enterpoint_node_);
        // writeBinaryPOD(output, maxM_);
        size += sizeof(maxM_);

        // writeBinaryPOD(output, maxM0_);
        size += sizeof(maxM0_);
        // writeBinaryPOD(output, M_);
        size += sizeof(M_);
        // writeBinaryPOD(output, mult_);
        size += sizeof(mult_);
        // writeBinaryPOD(output, ef_construction_);
        size += sizeof(ef_construction_);

        // output.write(data_level0_memory_, cur_element_count_ * size_data_per_element_);
        size += data_level0_memory_->getSize();
        size += maxM0_ * sizeof(uint32_t) * max_elements_;
        for (size_t i = 0; i < cur_element_count_; i++) {
            unsigned int link_list_size =
                element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            // writeBinaryPOD(output, link_list_size);
            size += sizeof(link_list_size);
            if (link_list_size) {
                // output.write(link_lists_[i], link_list_size);
                size += link_list_size * 2;  // record the size of reversed link list
            }
        }

        if (normalize_) {
            size += max_elements_ * sizeof(float);
        }

        // output.close();
        return size;
    }

    // save index to a file stream
    void
    saveIndex(std::ostream& out_stream) override {
        writeBinaryPOD(out_stream, offsetLevel0_);
        writeBinaryPOD(out_stream, max_elements_);
        writeBinaryPOD(out_stream, cur_element_count_);
        writeBinaryPOD(out_stream, size_data_per_element_);
        writeBinaryPOD(out_stream, label_offset_);
        writeBinaryPOD(out_stream, offsetData_);
        writeBinaryPOD(out_stream, maxlevel_);
        writeBinaryPOD(out_stream, enterpoint_node_);
        writeBinaryPOD(out_stream, maxM_);

        writeBinaryPOD(out_stream, maxM0_);
        writeBinaryPOD(out_stream, M_);
        writeBinaryPOD(out_stream, mult_);
        writeBinaryPOD(out_stream, ef_construction_);

        data_level0_memory_->serialize(out_stream, cur_element_count_);

        for (size_t i = 0; i < cur_element_count_; i++) {
            unsigned int link_list_size =
                element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(out_stream, link_list_size);
            if (link_list_size) {
                out_stream.write(link_lists_[i], link_list_size);
            }
        }
        if (normalize_) {
            out_stream.write((char*)molds_, max_elements_ * sizeof(float));
        }
    }

    void
    saveIndex(const std::string& location) override {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count_);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        data_level0_memory_->serialize(output, cur_element_count_);

        for (size_t i = 0; i < cur_element_count_; i++) {
            unsigned int link_list_size =
                element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, link_list_size);
            if (link_list_size)
                output.write(link_lists_[i], link_list_size);
        }
        output.close();
    }

    template <typename T>
    void
    readFromReader(std::function<void(uint64_t, uint64_t, void*)> read_func,
                   uint64_t& cursor,
                   T& var) {
        read_func(cursor, sizeof(T), &var);
        cursor += sizeof(T);
    }

    // load using reader
    void
    loadIndex(std::function<void(uint64_t, uint64_t, void*)> read_func,
              SpaceInterface* s,
              size_t max_elements_i = 0) override {
        // std::ifstream input(location, std::ios::binary);

        // if (!input.is_open())
        //     throw std::runtime_error("Cannot open file");

        // get file size:
        // input.seekg(0, input.end);
        // std::streampos total_filesize = input.tellg();
        // input.seekg(0, input.beg);

        uint64_t cursor = 0;

        // readBinaryPOD(input, offsetLevel0_);
        readFromReader(read_func, cursor, offsetLevel0_);
        // readBinaryPOD(input, max_elements_);
        size_t max_elements;
        readFromReader(read_func, cursor, max_elements);
        max_elements = std::max(max_elements, max_elements_i);
        max_elements = std::max(max_elements, max_elements_);

        // readBinaryPOD(input, cur_element_count_);
        readFromReader(read_func, cursor, cur_element_count_);
        // readBinaryPOD(input, size_data_per_element_);
        readFromReader(read_func, cursor, size_data_per_element_);
        // readBinaryPOD(input, label_offset_);
        readFromReader(read_func, cursor, label_offset_);
        // readBinaryPOD(input, offsetData_);
        readFromReader(read_func, cursor, offsetData_);
        // readBinaryPOD(input, maxlevel_);
        readFromReader(read_func, cursor, maxlevel_);
        // readBinaryPOD(input, enterpoint_node_);
        readFromReader(read_func, cursor, enterpoint_node_);

        // readBinaryPOD(input, maxM_);
        readFromReader(read_func, cursor, maxM_);
        // readBinaryPOD(input, maxM0_);
        readFromReader(read_func, cursor, maxM0_);
        // readBinaryPOD(input, M_);
        readFromReader(read_func, cursor, M_);
        // readBinaryPOD(input, mult_);
        readFromReader(read_func, cursor, mult_);
        // readBinaryPOD(input, ef_construction_);
        readFromReader(read_func, cursor, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        // auto pos = input.tellg();

        /// Optional - check if index is ok:
        // input.seekg(cur_element_count_ * size_data_per_element_, input.cur);
        // for (size_t i = 0; i < cur_element_count_; i++) {
        //     if (input.tellg() < 0 || input.tellg() >= total_filesize) {
        //         throw std::runtime_error("Index seems to be corrupted or unsupported");
        //     }

        //     unsigned int link_list_size;
        //     readBinaryPOD(input, link_list_size);
        //     if (link_list_size != 0) {
        //         input.seekg(link_list_size, input.cur);
        //     }
        // }

        // throw exception if it either corrupted or old index
        // if (input.tellg() != total_filesize)
        //     throw std::runtime_error("Index seems to be corrupted or unsupported");

        // input.clear();
        /// Optional check end

        // input.seekg(pos, input.beg);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        // input.read(data_level0_memory_, cur_element_count_ * size_data_per_element_);
        resizeIndex(max_elements);
        data_level0_memory_->deserialize(read_func, cursor, cur_element_count_);
        cursor += cur_element_count_ * size_data_per_element_;

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::recursive_mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        if (link_lists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");

        revSize_ = 1.0 / mult_;
        for (size_t i = 0; i < cur_element_count_; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int link_list_size;
            // readBinaryPOD(input, link_list_size);
            readFromReader(read_func, cursor, link_list_size);
            if (link_list_size == 0) {
                element_levels_[i] = 0;
                link_lists_[i] = nullptr;
            } else {
                element_levels_[i] = link_list_size / size_links_per_element_;
                link_lists_[i] = (char*)allocator_->Allocate(link_list_size);
                if (link_lists_[i] == nullptr)
                    throw std::runtime_error(
                        "Not enough memory: loadIndex failed to allocate linklist");
                // input.read(link_lists_[i], link_list_size);
                read_func(cursor, link_list_size, link_lists_[i]);
                cursor += link_list_size;
            }
        }

        if (normalize_) {
            read_func(cursor, max_elements_ * sizeof(float), molds_);
        }

        if (use_reversed_edges_) {
            for (int internal_id = 0; internal_id < cur_element_count_; ++internal_id) {
                for (int level = 0; level <= element_levels_[internal_id]; ++level) {
                    unsigned int* data = get_linklist_at_level(internal_id, level);
                    auto link_list = data + 1;
                    auto size = getListCount(data);
                    for (int j = 0; j < size; ++j) {
                        auto id = link_list[j];
                        auto& in_edges = getEdges(id, level);
                        in_edges.insert(internal_id);
                    }
                }
            }
        }

        for (size_t i = 0; i < cur_element_count_; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_)
                    deleted_elements.insert(i);
            }
        }

        // input.close();

        return;
    }

    // load index from a file stream
    void
    loadIndex(std::istream& in_stream, SpaceInterface* s, size_t max_elements_i = 0) override {
        auto beg_pos = in_stream.tellg();

        readBinaryPOD(in_stream, offsetLevel0_);

        size_t max_elements;
        readBinaryPOD(in_stream, max_elements);
        max_elements = std::max(max_elements, max_elements_i);
        max_elements = std::max(max_elements, max_elements_);

        readBinaryPOD(in_stream, cur_element_count_);
        readBinaryPOD(in_stream, size_data_per_element_);
        readBinaryPOD(in_stream, label_offset_);
        readBinaryPOD(in_stream, offsetData_);
        readBinaryPOD(in_stream, maxlevel_);
        readBinaryPOD(in_stream, enterpoint_node_);

        readBinaryPOD(in_stream, maxM_);
        readBinaryPOD(in_stream, maxM0_);
        readBinaryPOD(in_stream, M_);
        readBinaryPOD(in_stream, mult_);
        readBinaryPOD(in_stream, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = in_stream.tellg();

        in_stream.seekg(pos, in_stream.beg);

        resizeIndex(max_elements);
        data_level0_memory_->deserialize(in_stream, cur_element_count_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::recursive_mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        revSize_ = 1.0 / mult_;
        for (size_t i = 0; i < cur_element_count_; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int link_list_size;
            readBinaryPOD(in_stream, link_list_size);
            if (link_list_size == 0) {
                element_levels_[i] = 0;
                link_lists_[i] = nullptr;
            } else {
                element_levels_[i] = link_list_size / size_links_per_element_;
                link_lists_[i] = (char*)allocator_->Allocate(link_list_size);
                if (link_lists_[i] == nullptr)
                    throw std::runtime_error(
                        "Not enough memory: loadIndex failed to allocate linklist");
                in_stream.read(link_lists_[i], link_list_size);
            }
        }
        if (normalize_) {
            in_stream.read((char*)molds_, sizeof(float) * max_elements_);
        }

        if (use_reversed_edges_) {
            for (int internal_id = 0; internal_id < cur_element_count_; ++internal_id) {
                for (int level = 0; level <= element_levels_[internal_id]; ++level) {
                    unsigned int* data = get_linklist_at_level(internal_id, level);
                    auto link_list = data + 1;
                    auto size = getListCount(data);
                    for (int j = 0; j < size; ++j) {
                        auto id = link_list[j];
                        auto& in_edges = getEdges(id, level);
                        in_edges.insert(internal_id);
                    }
                }
            }
        }

        for (size_t i = 0; i < cur_element_count_; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_)
                    deleted_elements.insert(i);
            }
        }

        return;
    }

    // origin load function
    void
    loadIndex(const std::string& location, SpaceInterface* s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count_);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count_)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count_ * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count_; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int link_list_size;
            readBinaryPOD(input, link_list_size);
            if (link_list_size != 0) {
                input.seekg(link_list_size, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_->deserialize(input, cur_element_count_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::recursive_mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        delete visited_list_pool_;
        visited_list_pool_ = new VisitedListPool(1, max_elements, allocator_);

        link_lists_ = (char**)malloc(sizeof(void*) * max_elements);
        if (link_lists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");

        revSize_ = 1.0 / mult_;
        for (size_t i = 0; i < cur_element_count_; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int link_list_size;
            readBinaryPOD(input, link_list_size);
            if (link_list_size == 0) {
                element_levels_[i] = 0;
                link_lists_[i] = nullptr;
            } else {
                element_levels_[i] = link_list_size / size_links_per_element_;
                link_lists_[i] = (char*)malloc(link_list_size);
                if (link_lists_[i] == nullptr)
                    throw std::runtime_error(
                        "Not enough memory: loadIndex failed to allocate linklist");
                input.read(link_lists_[i], link_list_size);
            }
        }

        for (size_t i = 0; i < cur_element_count_; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_)
                    deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    }

    const float*
    getDataByLabel(labeltype label) const override {
        std::lock_guard<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        float* data_ptr = (float*)data_ptrv;

        return data_ptr;
    }

    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void
    markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        label_lookup_.erase(search);
        lock_table.unlock();
        markDeletedInternal(internalId);
    }

    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void
    markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count_);
        if (!isMarkedDeleted(internalId)) {
            unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }

    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    *
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void
    unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }

    /*
    * Remove the deleted mark of the node.
    */
    void
    unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count_);
        if (isMarkedDeleted(internalId)) {
            unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }

    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool
    isMarkedDeleted(tableint internalId) const {
        unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }

    unsigned short int
    getListCount(linklistsizeint* ptr) const {
        return *((unsigned short int*)ptr);
    }

    void
    setListCount(linklistsizeint* ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr)) = *((unsigned short int*)&size);
    }

    /*
    * Adds point.
    */
    bool
    addPoint(const void* data_point, labeltype label) override {
        std::lock_guard<std::mutex> lock_label(getLabelOpMutex(label));
        if (addPoint(data_point, label, -1) == -1) {
            return false;
        }
        return true;
    }

    inline void
    modify_out_edge(tableint old_internal_id, tableint new_internal_id) {
        for (int level = 0; level <= element_levels_[old_internal_id]; ++level) {
            auto& edges = getEdges(old_internal_id, level);
            for (const auto& in_node : edges) {
                auto data = get_linklist_at_level(in_node, level);
                size_t link_size = getListCount(data);
                tableint* links = (tableint*)(data + 1);
                for (int i = 0; i < link_size; ++i) {
                    if (links[i] == old_internal_id) {
                        links[i] = new_internal_id;
                        break;
                    }
                }
            }
        }
    }

    inline void
    modify_in_edges(tableint right_internal_id, tableint wrong_internal_id, bool is_erase) {
        for (int level = 0; level <= element_levels_[right_internal_id]; ++level) {
            auto data = get_linklist_at_level(right_internal_id, level);
            size_t link_size = getListCount(data);
            tableint* links = (tableint*)(data + 1);
            for (int i = 0; i < link_size; ++i) {
                auto& in_egdes = getEdges(links[i], level);
                if (is_erase) {
                    in_egdes.erase(wrong_internal_id);
                } else {
                    in_egdes.insert(right_internal_id);
                }
            }
        }
    };

    bool
    swapConnections(tableint pre_internal_id, tableint post_internal_id) {
        {
            // modify the connectivity relationships in the graph.
            // Through the reverse edges, change the edges pointing to pre_internal_id to point to
            // post_internal_id.
            modify_out_edge(pre_internal_id, post_internal_id);
            modify_out_edge(post_internal_id, pre_internal_id);

            // Swap the data and the adjacency lists of the graph.
            auto tmp_data_element = std::shared_ptr<char[]>(new char[size_data_per_element_]);
            memcpy(tmp_data_element.get(), get_linklist0(pre_internal_id), size_data_per_element_);
            memcpy(get_linklist0(pre_internal_id),
                   get_linklist0(post_internal_id),
                   size_data_per_element_);
            memcpy(get_linklist0(post_internal_id), tmp_data_element.get(), size_data_per_element_);

            if (normalize_) {
                std::swap(molds_[pre_internal_id], molds_[post_internal_id]);
            }
            std::swap(link_lists_[pre_internal_id], link_lists_[post_internal_id]);
            std::swap(element_levels_[pre_internal_id], element_levels_[post_internal_id]);
        }

        {
            // Repair the incorrect reverse edges caused by swapping two points.
            std::swap(reversed_level0_link_list_[pre_internal_id],
                      reversed_level0_link_list_[post_internal_id]);
            std::swap(reversed_link_lists_[pre_internal_id],
                      reversed_link_lists_[post_internal_id]);

            // First, remove the incorrect connectivity relationships in the reverse edges and then
            // proceed with the insertion. This avoids losing edges when a point simultaneously
            // has edges pointing to both pre_internal_id and post_internal_id.

            modify_in_edges(pre_internal_id, post_internal_id, true);
            modify_in_edges(post_internal_id, pre_internal_id, true);
            modify_in_edges(pre_internal_id, post_internal_id, false);
            modify_in_edges(post_internal_id, pre_internal_id, false);
        }

        if (enterpoint_node_ == post_internal_id) {
            enterpoint_node_ = pre_internal_id;
        } else if (enterpoint_node_ == pre_internal_id) {
            enterpoint_node_ = post_internal_id;
        }

        return true;
    }

    void
    dealNoInEdge(tableint id, int level, int m_curmax, int skip_c) {
        // Establish edges from the neighbors of the id pointing to the id.
        auto alone_data = get_linklist_at_level(id, level);
        int alone_size = getListCount(alone_data);
        auto alone_link = (unsigned int*)(alone_data + 1);
        auto& in_edges = getEdges(id, level);
        for (int j = 0; j < alone_size; ++j) {
            if (alone_link[j] == skip_c) {
                continue;
            }
            auto to_edge_data_cur = (unsigned int*)get_linklist_at_level(alone_link[j], level);
            int to_edge_size_cur = getListCount(to_edge_data_cur);
            auto to_edge_data_link_cur = (unsigned int*)(to_edge_data_cur + 1);
            if (to_edge_size_cur < m_curmax) {
                to_edge_data_link_cur[to_edge_size_cur] = id;
                setListCount(to_edge_data_cur, to_edge_size_cur + 1);
                in_edges.insert(alone_link[j]);
            }
        }
    }

    void
    removePoint(labeltype label) {
        tableint cur_c = 0;
        tableint internal_id = 0;
        std::lock_guard<std::mutex> lock(global);
        {
            // Swap the connection relationship corresponding to the label to be deleted with the
            // last element, and modify the information in label_lookup_. By swapping the two points,
            // fill the void left by the deletion.
            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto iter = label_lookup_.find(label);
            if (iter == label_lookup_.end()) {
                throw std::runtime_error("no label in FreshHnsw");
            } else {
                internal_id = iter->second;
                label_lookup_.erase(iter);
            }

            cur_element_count_--;
            cur_c = cur_element_count_;

            if (cur_c == 0) {
                for (int level = 0; level < element_levels_[cur_c]; ++level) {
                    getEdges(cur_c, level).clear();
                }
                enterpoint_node_ = -1;
                maxlevel_ = -1;
                return;
            } else if (cur_c != internal_id) {
                label_lookup_[getExternalLabel(cur_c)] = internal_id;
                swapConnections(cur_c, internal_id);
            }
        }

        // If the node to be deleted is an entry node, find another top-level node.
        if (cur_c == enterpoint_node_) {
            for (int level = maxlevel_; level >= 0; level--) {
                auto data = (unsigned int*)get_linklist_at_level(enterpoint_node_, level);
                int size = getListCount(data);
                if (size != 0) {
                    maxlevel_ = level;
                    enterpoint_node_ = *(data + 1);
                    break;
                }
            }
        }

        // Repair the connection relationship between the indegree and outdegree nodes at each
        // level. We connect each indegree node with each outdegree node, and then prune the
        // indegree nodes.
        for (int level = 0; level <= element_levels_[cur_c]; ++level) {
            const auto in_edges_cur = getEdges(cur_c, level);
            auto data_cur = get_linklist_at_level(cur_c, level);
            int size_cur = getListCount(data_cur);
            auto data_link_cur = (unsigned int*)(data_cur + 1);

            for (const auto in_edge : in_edges_cur) {
                std::priority_queue<std::pair<float, tableint>,
                                    std::vector<std::pair<float, tableint>>,
                                    CompareByFirst>
                    candidates;
                std::unordered_set<tableint> unique_ids;

                // Add the original neighbors of the indegree node to the candidate queue.
                for (int i = 0; i < size_cur; ++i) {
                    if (data_link_cur[i] == cur_c || data_link_cur[i] == in_edge) {
                        continue;
                    }
                    unique_ids.insert(data_link_cur[i]);
                    candidates.emplace(fstdistfunc_(getDataByInternalId(data_link_cur[i]),
                                                    getDataByInternalId(in_edge),
                                                    dist_func_param_),
                                       data_link_cur[i]);
                }

                // Add the neighbors of the node to be deleted to the candidate queue.
                auto in_edge_data_cur = (unsigned int*)get_linklist_at_level(in_edge, level);
                int in_edge_size_cur = getListCount(in_edge_data_cur);
                auto in_edge_data_link_cur = (unsigned int*)(in_edge_data_cur + 1);
                for (int i = 0; i < in_edge_size_cur; ++i) {
                    if (in_edge_data_link_cur[i] == cur_c ||
                        unique_ids.find(in_edge_data_link_cur[i]) != unique_ids.end()) {
                        continue;
                    }
                    unique_ids.insert(in_edge_data_link_cur[i]);
                    candidates.emplace(fstdistfunc_(getDataByInternalId(in_edge_data_link_cur[i]),
                                                    getDataByInternalId(in_edge),
                                                    dist_func_param_),
                                       in_edge_data_link_cur[i]);
                }

                if (candidates.size() == 0) {
                    setListCount(in_edge_data_cur, 0);
                    getEdges(cur_c, level).erase(in_edge);
                    continue;
                }
                mutuallyConnectNewElement(
                    getDataByInternalId(in_edge), in_edge, candidates, level, true);

                // Handle the operations of the deletion point which result in some nodes having no
                // indegree nodes, and carry out repairs.
                size_t m_curmax = level ? maxM_ : maxM0_;
                for (auto id : unique_ids) {
                    if (getEdges(id, level).size() == 0) {
                        dealNoInEdge(id, level, m_curmax, cur_c);
                    }
                }
            }

            for (int i = 0; i < size_cur; ++i) {
                getEdges(data_link_cur[i], level).erase(cur_c);
            }
        }
    }

    tableint
    addPoint(const void* data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                return -1;
            }

            if (cur_element_count_ >= max_elements_) {
                resizeIndex(max_elements_ + data_element_per_block_);
            }

            cur_c = cur_element_count_;
            cur_element_count_++;
            label_lookup_[label] = cur_c;
        }

        std::shared_ptr<float[]> normalize_data;
        normalize_vector(data_point, normalize_data);

        std::unique_lock<std::recursive_mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;
        std::unique_lock<std::mutex> lock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            lock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_->getElementPtr(cur_c, offsetLevel0_), 0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);
        if (curlevel) {
            auto new_link_lists = (char*)allocator_->Reallocate(
                link_lists_[cur_c], size_links_per_element_ * curlevel + 1);
            if (new_link_lists == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            link_lists_[cur_c] = new_link_lists;
            memset(link_lists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                float curdist =
                    fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int* data;
                        std::unique_lock<std::recursive_mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint* datal = (tableint*)(data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            float d = fstdistfunc_(
                                data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<float, tableint>,
                                    std::vector<std::pair<float, tableint>>,
                                    CompareByFirst>
                    top_candidates = searchBaseLayer(currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(
                        fstdistfunc_(
                            data_point, getDataByInternalId(enterpoint_copy), dist_func_param_),
                        enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj =
                    mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }

    std::priority_queue<std::pair<float, labeltype>>
    searchKnn(const void* query_data,
              size_t k,
              uint64_t ef,
              BaseFilterFunctor* isIdAllowed = nullptr) const override {
        std::priority_queue<std::pair<float, labeltype>> result;
        if (cur_element_count_ == 0)
            return result;

        std::shared_ptr<float[]> normalize_query;
        normalize_vector(query_data, normalize_query);
        tableint currObj = enterpoint_node_;
        float curdist =
            fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int* data;

                data = (unsigned int*)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops_++;
                metric_distance_computations_ += size;

                tableint* datal = (tableint*)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    float d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<float, tableint>,
                            std::vector<std::pair<float, tableint>>,
                            CompareByFirst>
            top_candidates;
        if (num_deleted_) {
            top_candidates =
                searchBaseLayerST<true, true>(currObj, query_data, std::max(ef, k), isIdAllowed);
        } else {
            top_candidates =
                searchBaseLayerST<false, true>(currObj, query_data, std::max(ef, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<float, tableint> rez = top_candidates.top();
            result.push(std::pair<float, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    std::priority_queue<std::pair<float, labeltype>>
    searchRange(const void* query_data,
                float radius,
                uint64_t ef,
                BaseFilterFunctor* isIdAllowed = nullptr) const override {
        std::priority_queue<std::pair<float, labeltype>> result;
        if (cur_element_count_ == 0)
            return result;

        std::shared_ptr<float[]> normalize_query;
        normalize_vector(query_data, normalize_query);
        tableint currObj = enterpoint_node_;
        float curdist =
            fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int* data;

                data = (unsigned int*)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops_++;
                metric_distance_computations_ += size;

                tableint* datal = (tableint*)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    float d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<float, tableint>,
                            std::vector<std::pair<float, tableint>>,
                            CompareByFirst>
            top_candidates;
        if (num_deleted_) {
            throw std::runtime_error(
                "not support perform range search on a index that deleted some vectors");
        } else {
            top_candidates =
                searchBaseLayerST<false, true>(currObj, query_data, radius, ef, isIdAllowed);
            // std::cout << "top_candidates.size(): " << top_candidates.size() << std::endl;
        }

        while (top_candidates.size() > 0) {
            std::pair<float, tableint> rez = top_candidates.top();
            result.push(std::pair<float, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }

        // std::cout << "hnswalg::result.size(): " << result.size() << std::endl;
        return result;
    }

    void
    checkIntegrity() {
        int connections_checked = 0;
        std::vector<int> inbound_connections_num(cur_element_count_, 0);
        for (int i = 0; i < cur_element_count_; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint* ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint* data = (tableint*)(ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] > 0);
                    assert(data[j] < cur_element_count_);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count_ > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i = 0; i < cur_element_count_; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }
};
}  // namespace hnswlib
