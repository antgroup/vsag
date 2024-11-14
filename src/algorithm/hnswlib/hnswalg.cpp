
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

#include "hnswalg.h"

#include <memory>
namespace hnswlib {
HierarchicalNSW::HierarchicalNSW(SpaceInterface* s,
                                 size_t max_elements,
                                 vsag::Allocator* allocator,
                                 size_t M,
                                 size_t ef_construction,
                                 bool use_reversed_edges,
                                 bool normalize,
                                 size_t block_size_limit,
                                 size_t random_seed,
                                 bool allow_replace_deleted)
    : allocator_(allocator),
      link_list_locks_(max_elements, allocator),
      label_op_locks_(MAX_LABEL_OPERATION_LOCKS, allocator),
      allow_replace_deleted_(allow_replace_deleted),
      use_reversed_edges_(use_reversed_edges),
      normalize_(normalize),
      label_lookup_(allocator),
      deleted_elements_(allocator) {
    max_elements_ = max_elements;
    num_deleted_ = 0;
    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    dim_ = *((size_t*)dist_func_param_);
    M_ = M;
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);

    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    size_links_level0_ = maxM0_ * sizeof(InnerIdType) + sizeof(linklistsizeint);
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(LabelType);
    offset_data_ = size_links_level0_;
    label_offset_ = size_links_level0_ + data_size_;
    offsetLevel0_ = 0;

    data_level0_memory_ =
        std::make_shared<BlockManager>(size_data_per_element_, block_size_limit, allocator_);
    data_element_per_block_ = block_size_limit / size_data_per_element_;

    cur_element_count_ = 0;

    visited_list_pool_ = std::make_shared<VisitedListPool>(1, max_elements, allocator_);

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    max_level_ = -1;
    size_links_per_element_ = maxM_ * sizeof(InnerIdType) + sizeof(linklistsizeint);
    mult_ = 1 / log(1.0 * static_cast<double>(M_));
    rev_size_ = 1.0 / mult_;
}

void
HierarchicalNSW::reset() {
    allocator_->Deallocate(element_levels_);
    element_levels_ = nullptr;
    allocator_->Deallocate(reversed_level0_link_list_);
    reversed_level0_link_list_ = nullptr;
    allocator_->Deallocate(reversed_link_lists_);
    reversed_link_lists_ = nullptr;
    allocator_->Deallocate(molds_);
    molds_ = nullptr;
    allocator_->Deallocate(link_lists_);
    link_lists_ = nullptr;
}

bool
HierarchicalNSW::init_memory_space() {
    // release the memory allocated by the init_memory_space function that was called earlier
    reset();
    element_levels_ = (int*)allocator_->Allocate(max_elements_ * sizeof(int));
    if (not data_level0_memory_->Resize(max_elements_)) {
        throw std::runtime_error("allocate data_level0_memory_ error");
    }
    if (use_reversed_edges_) {
        reversed_level0_link_list_ =
            (reverselinklist**)allocator_->Allocate(max_elements_ * sizeof(reverselinklist*));
        if (reversed_level0_link_list_ == nullptr) {
            throw std::runtime_error("allocate reversed_level0_link_list_ fail");
        }
        memset(reversed_level0_link_list_, 0, max_elements_ * sizeof(reverselinklist*));
        reversed_link_lists_ = (vsag::UnorderedMap<int, reverselinklist>**)allocator_->Allocate(
            max_elements_ * sizeof(vsag::UnorderedMap<int, reverselinklist>*));
        if (reversed_link_lists_ == nullptr) {
            throw std::runtime_error("allocate reversed_link_lists_ fail");
        }
        memset(reversed_link_lists_,
               0,
               max_elements_ * sizeof(vsag::UnorderedMap<int, reverselinklist>*));
    }

    if (normalize_) {
        ip_func_ = vsag::InnerProduct;
        molds_ = (float*)allocator_->Allocate(max_elements_ * sizeof(float));
    }

    link_lists_ = (char**)allocator_->Allocate(sizeof(void*) * max_elements_);
    if (link_lists_ == nullptr)
        throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
    memset(link_lists_, 0, sizeof(void*) * max_elements_);
    return true;
}

HierarchicalNSW::~HierarchicalNSW() {
    if (link_lists_ != nullptr) {
        for (InnerIdType i = 0; i < max_elements_; i++) {
            if (element_levels_[i] > 0 || link_lists_[i] != nullptr)
                allocator_->Deallocate(link_lists_[i]);
        }
    }

    if (use_reversed_edges_) {
        for (InnerIdType i = 0; i < max_elements_; i++) {
            auto& in_edges_level0 = *(reversed_level0_link_list_ + i);
            delete in_edges_level0;
            auto& in_edges = *(reversed_link_lists_ + i);
            delete in_edges;
        }
    }
    reset();
}

void
HierarchicalNSW::normalizeVector(const void*& data_point,
                                 std::shared_ptr<float[]>& normalize_data) const {
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
HierarchicalNSW::getDistanceByLabel(LabelType label, const void* data_point) {
    std::unique_lock<std::mutex> lock_table(label_lookup_lock_);

    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end()) {
        throw std::runtime_error("Label not found");
    }
    InnerIdType internal_id = search->second;
    lock_table.unlock();
    std::shared_ptr<float[]> normalize_query;
    normalizeVector(data_point, normalize_query);
    float dist = fstdistfunc_(data_point, getDataByInternalId(internal_id), dist_func_param_);
    return dist;
}

bool
HierarchicalNSW::isValidLabel(LabelType label) {
    std::unique_lock<std::mutex> lock_table(label_lookup_lock_);
    bool is_valid = (label_lookup_.find(label) != label_lookup_.end());
    lock_table.unlock();
    return is_valid;
}

void
HierarchicalNSW::updateConnections(InnerIdType internal_id,
                                   const vsag::Vector<InnerIdType>& cand_neighbors,
                                   int level,
                                   bool is_update) {
    linklistsizeint* ll_cur;
    if (level == 0)
        ll_cur = getLinklist0(internal_id);
    else
        ll_cur = getLinklist(internal_id, level);

    auto cur_size = getListCount(ll_cur);
    auto* data = (InnerIdType*)(ll_cur + 1);

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
HierarchicalNSW::checkReverseConnection() {
    int edge_count = 0;
    uint64_t reversed_edge_count = 0;
    for (int internal_id = 0; internal_id < cur_element_count_; ++internal_id) {
        for (int level = 0; level <= element_levels_[internal_id]; ++level) {
            unsigned int* data;
            if (level == 0) {
                data = getLinklist0(internal_id);
            } else {
                data = getLinklist(internal_id, level);
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

std::priority_queue<std::pair<float, LabelType>>
HierarchicalNSW::bruteForce(const void* data_point, int64_t k) {
    std::priority_queue<std::pair<float, LabelType>> results;
    for (uint32_t i = 0; i < cur_element_count_; i++) {
        float dist = fstdistfunc_(data_point, getDataByInternalId(i), dist_func_param_);
        if (results.size() < k) {
            results.emplace(dist, *getExternalLabeLp(i));
        } else {
            float current_max_dist = results.top().first;
            if (dist < current_max_dist) {
                results.pop();
                results.emplace(dist, *getExternalLabeLp(i));
            }
        }
    }
    return results;
}

int
HierarchicalNSW::getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int)r;
}

MaxHeap
HierarchicalNSW::searchBaseLayer(InnerIdType ep_id, const void* data_point, int layer) {
    VisitedListPtr vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    MaxHeap top_candidates(allocator_);
    MaxHeap candidateSet(allocator_);

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

    while (not candidateSet.empty()) {
        std::pair<float, InnerIdType> curr_el_pair = candidateSet.top();
        if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
            break;
        }
        candidateSet.pop();

        InnerIdType curNodeNum = curr_el_pair.second;

        std::unique_lock<std::recursive_mutex> lock(link_list_locks_[curNodeNum]);

        int* data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
        if (layer == 0) {
            data = (int*)getLinklist0(curNodeNum);
        } else {
            data = (int*)getLinklist(curNodeNum, layer);
            //                    data = (int *) (link_lists_[curNodeNum] + (layer - 1) * size_links_per_element_);
        }
        size_t size = getListCount((linklistsizeint*)data);
        auto* datal = (InnerIdType*)(data + 1);
#ifdef USE_SSE
        _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

        for (size_t j = 0; j < size; j++) {
            InnerIdType candidate_id = *(datal + j);
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

                if (not isMarkedDeleted(candidate_id))
                    top_candidates.emplace(dist1, candidate_id);

                if (top_candidates.size() > ef_construction_)
                    top_candidates.pop();

                if (not top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);

    return top_candidates;
}

template <bool has_deletions, bool collect_metrics>
MaxHeap
HierarchicalNSW::searchBaseLayerST(InnerIdType ep_id,
                                   const void* data_point,
                                   size_t ef,
                                   vsag::BaseFilterFunctor* isIdAllowed) const {
    VisitedListPtr vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    MaxHeap top_candidates(allocator_);
    MaxHeap candidate_set(allocator_);

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

    while (not candidate_set.empty()) {
        std::pair<float, InnerIdType> current_node_pair = candidate_set.top();

        if ((-current_node_pair.first) > lowerBound &&
            (top_candidates.size() == ef || (!isIdAllowed && !has_deletions))) {
            break;
        }
        candidate_set.pop();

        InnerIdType current_node_id = current_node_pair.second;
        int* data = (int*)getLinklist0(current_node_id);
        size_t size = getListCount((linklistsizeint*)data);
        //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
        if (collect_metrics) {
            metric_hops_++;
            metric_distance_computations_ += size;
        }

        auto vector_data_ptr = data_level0_memory_->GetElementPtr((*(data + 1)), offset_data_);
#ifdef USE_SSE
        _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(vector_data_ptr, _MM_HINT_T0);
        _mm_prefetch((char*)(data + 2), _MM_HINT_T0);
#endif

        for (size_t j = 1; j <= size; j++) {
            int candidate_id = *(data + j);
            size_t pre_l = std::min(j, size - 2);
            vector_data_ptr =
                data_level0_memory_->GetElementPtr((*(data + pre_l + 1)), offset_data_);
#ifdef USE_SSE
            _mm_prefetch((char*)(visited_array + *(data + pre_l + 1)), _MM_HINT_T0);
            _mm_prefetch(vector_data_ptr, _MM_HINT_T0);  ////////////
#endif
            if (visited_array[candidate_id] != visited_array_tag) {
                visited_array[candidate_id] = visited_array_tag;

                char* currObj1 = (getDataByInternalId(candidate_id));
                float dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef || lowerBound > dist) {
                    candidate_set.emplace(-dist, candidate_id);
                    vector_data_ptr = data_level0_memory_->GetElementPtr(candidate_set.top().second,
                                                                         offsetLevel0_);
#ifdef USE_SSE
                    _mm_prefetch(vector_data_ptr, _MM_HINT_T0);
#endif

                    if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
                        ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))
                        top_candidates.emplace(dist, candidate_id);

                    if (top_candidates.size() > ef)
                        top_candidates.pop();

                    if (not top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
    }

    visited_list_pool_->releaseVisitedList(vl);
    return top_candidates;
}

template <bool has_deletions, bool collect_metrics>
MaxHeap
HierarchicalNSW::searchBaseLayerST(InnerIdType ep_id,
                                   const void* data_point,
                                   float radius,
                                   int64_t ef,
                                   vsag::BaseFilterFunctor* isIdAllowed) const {
    VisitedListPtr vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    MaxHeap top_candidates(allocator_);
    MaxHeap candidate_set(allocator_);

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

    while (not candidate_set.empty()) {
        std::pair<float, InnerIdType> current_node_pair = candidate_set.top();

        candidate_set.pop();

        InnerIdType current_node_id = current_node_pair.second;
        int* data = (int*)getLinklist0(current_node_id);
        size_t size = getListCount((linklistsizeint*)data);
        //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
        if (collect_metrics) {
            metric_hops_++;
            metric_distance_computations_ += size;
        }

        auto vector_data_ptr = data_level0_memory_->GetElementPtr((*(data + 1)), offset_data_);
#ifdef USE_SSE
        _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(vector_data_ptr, _MM_HINT_T0);
        _mm_prefetch((char*)(data + 2), _MM_HINT_T0);
#endif

        for (size_t j = 1; j <= size; j++) {
            int candidate_id = *(data + j);
            size_t pre_l = std::min(j, size - 2);
            vector_data_ptr =
                data_level0_memory_->GetElementPtr((*(data + pre_l + 1)), offset_data_);
#ifdef USE_SSE
            _mm_prefetch((char*)(visited_array + *(data + pre_l + 1)), _MM_HINT_T0);
            _mm_prefetch(vector_data_ptr, _MM_HINT_T0);  ////////////
#endif
            if (visited_array[candidate_id] != visited_array_tag) {
                visited_array[candidate_id] = visited_array_tag;
                ++visited_count;

                char* currObj1 = (getDataByInternalId(candidate_id));
                float dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                if (visited_count < ef || dist <= radius + THRESHOLD_ERROR || lowerBound > dist) {
                    candidate_set.emplace(-dist, candidate_id);
                    vector_data_ptr = data_level0_memory_->GetElementPtr(candidate_set.top().second,
                                                                         offsetLevel0_);
#ifdef USE_SSE
                    _mm_prefetch(vector_data_ptr, _MM_HINT_T0);  ////////////////////////
#endif

                    if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
                        ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))
                        top_candidates.emplace(dist, candidate_id);

                    if (not top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
    }
    while (not top_candidates.empty() && top_candidates.top().first > radius + THRESHOLD_ERROR) {
        top_candidates.pop();
    }

    visited_list_pool_->releaseVisitedList(vl);
    return top_candidates;
}

void
HierarchicalNSW::getNeighborsByHeuristic2(MaxHeap& top_candidates, size_t M) {
    if (top_candidates.size() < M) {
        return;
    }

    std::priority_queue<std::pair<float, InnerIdType>> queue_closest;
    vsag::Vector<std::pair<float, InnerIdType>> return_list(allocator_);
    while (not top_candidates.empty()) {
        queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
        top_candidates.pop();
    }

    while (not queue_closest.empty()) {
        if (return_list.size() >= M)
            break;
        std::pair<float, InnerIdType> curent_pair = queue_closest.top();
        float float_query = -curent_pair.first;
        queue_closest.pop();
        bool good = true;

        for (std::pair<float, InnerIdType> second_pair : return_list) {
            float curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                                         getDataByInternalId(curent_pair.second),
                                         dist_func_param_);
            if (curdist < float_query) {
                good = false;
                break;
            }
        }
        if (good) {
            return_list.emplace_back(curent_pair);
        }
    }

    for (std::pair<float, InnerIdType> curent_pair : return_list) {
        top_candidates.emplace(-curent_pair.first, curent_pair.second);
    }
}

InnerIdType
HierarchicalNSW::mutuallyConnectNewElement(InnerIdType cur_c,
                                           MaxHeap& top_candidates,
                                           int level,
                                           bool isUpdate) {
    size_t m_curmax = level ? maxM_ : maxM0_;
    getNeighborsByHeuristic2(top_candidates, M_);
    if (top_candidates.size() > M_)
        throw std::runtime_error(
            "Should be not be more than M_ candidates returned by the heuristic");

    vsag::Vector<InnerIdType> selectedNeighbors(allocator_);
    selectedNeighbors.reserve(M_);
    while (not top_candidates.empty()) {
        selectedNeighbors.push_back(top_candidates.top().second);
        top_candidates.pop();
    }

    InnerIdType next_closest_entry_point = selectedNeighbors.back();

    {
        // lock only during the update
        // because during the addition the lock for cur_c is already acquired
        std::unique_lock<std::recursive_mutex> lock(link_list_locks_[cur_c], std::defer_lock);
        if (isUpdate) {
            lock.lock();
        }
        updateConnections(cur_c, selectedNeighbors, level, isUpdate);
    }

    for (unsigned int selectedNeighbor : selectedNeighbors) {
        std::unique_lock<std::recursive_mutex> lock(link_list_locks_[selectedNeighbor]);

        linklistsizeint* ll_other;
        if (level == 0)
            ll_other = getLinklist0(selectedNeighbor);
        else
            ll_other = getLinklist(selectedNeighbor, level);

        size_t sz_link_list_other = getListCount(ll_other);

        if (sz_link_list_other > m_curmax)
            throw std::runtime_error("Bad value of sz_link_list_other");
        if (selectedNeighbor == cur_c)
            throw std::runtime_error("Trying to connect an element to itself");
        if (level > element_levels_[selectedNeighbor])
            throw std::runtime_error("Trying to make a link on a non-existent level");

        auto* data = (InnerIdType*)(ll_other + 1);

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
                    cur_in_edges.insert(selectedNeighbor);
                }
            } else {
                // finding the "weakest" element to replace it with the new one
                float d_max = fstdistfunc_(getDataByInternalId(cur_c),
                                           getDataByInternalId(selectedNeighbor),
                                           dist_func_param_);
                // Heuristic:
                MaxHeap candidates(allocator_);
                candidates.emplace(d_max, cur_c);

                for (size_t j = 0; j < sz_link_list_other; j++) {
                    candidates.emplace(fstdistfunc_(getDataByInternalId(data[j]),
                                                    getDataByInternalId(selectedNeighbor),
                                                    dist_func_param_),
                                       data[j]);
                }

                getNeighborsByHeuristic2(candidates, m_curmax);

                vsag::Vector<InnerIdType> cand_neighbors(allocator_);
                while (not candidates.empty()) {
                    cand_neighbors.push_back(candidates.top().second);
                    candidates.pop();
                }
                updateConnections(selectedNeighbor, cand_neighbors, level, true);
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
HierarchicalNSW::resizeIndex(size_t new_max_elements) {
    if (new_max_elements < cur_element_count_)
        throw std::runtime_error(
            "Cannot Resize, max element is less than the current number of elements");

    visited_list_pool_.reset(new VisitedListPool(1, new_max_elements, allocator_));

    auto element_levels_new =
        (int*)allocator_->Reallocate(element_levels_, new_max_elements * sizeof(int));
    if (element_levels_new == nullptr) {
        throw std::runtime_error(
            "Not enough memory: resizeIndex failed to allocate element_levels_");
    }
    element_levels_ = element_levels_new;
    vsag::Vector<std::recursive_mutex>(new_max_elements, allocator_).swap(link_list_locks_);

    if (normalize_) {
        auto new_molds = (float*)allocator_->Reallocate(molds_, new_max_elements * sizeof(float));
        if (new_molds == nullptr) {
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate molds_");
        }
        molds_ = new_molds;
    }

    // Reallocate base layer
    if (not data_level0_memory_->Resize(new_max_elements))
        throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");

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

        auto reversed_link_lists_new =
            (vsag::UnorderedMap<int, reverselinklist>**)allocator_->Reallocate(
                reversed_link_lists_,
                new_max_elements * sizeof(vsag::UnorderedMap<int, reverselinklist>*));
        if (reversed_link_lists_new == nullptr) {
            throw std::runtime_error(
                "Not enough memory: resizeIndex failed to allocate reversed_link_lists_");
        }
        reversed_link_lists_ = reversed_link_lists_new;
        memset(
            reversed_link_lists_ + max_elements_,
            0,
            (new_max_elements - max_elements_) * sizeof(vsag::UnorderedMap<int, reverselinklist>*));
    }

    // Reallocate all other layers
    char** linkLists_new =
        (char**)allocator_->Reallocate(link_lists_, sizeof(void*) * new_max_elements);
    if (linkLists_new == nullptr)
        throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
    link_lists_ = linkLists_new;
    memset(link_lists_ + max_elements_, 0, (new_max_elements - max_elements_) * sizeof(void*));
    max_elements_ = new_max_elements;
}

size_t
HierarchicalNSW::calcSerializeSize() {
    auto calSizeFunc = [](uint64_t cursor, uint64_t size, void* buf) { return; };
    WriteFuncStreamWriter writer(calSizeFunc, 0);
    this->SerializeImpl(writer);
    return writer.cursor_;
}

void
HierarchicalNSW::saveIndex(void* d) {
    char* dest = (char*)d;
    BufferStreamWriter writer(dest);
    SerializeImpl(writer);
}
// save index to a file stream
void
HierarchicalNSW::saveIndex(std::ostream& out_stream) {
    IOStreamWriter writer(out_stream);
    SerializeImpl(writer);
}

void
HierarchicalNSW::saveIndex(const std::string& location) {
    std::ofstream output(location, std::ios::binary);
    IOStreamWriter writer(output);
    SerializeImpl(writer);
    output.close();
}

template <typename T>
static void
WriteOne(StreamWriter& writer, T& value) {
    writer.Write(reinterpret_cast<char*>(&value), sizeof(value));
}

void
HierarchicalNSW::SerializeImpl(StreamWriter& writer) {
    WriteOne(writer, offsetLevel0_);
    WriteOne(writer, max_elements_);
    WriteOne(writer, cur_element_count_);
    WriteOne(writer, size_data_per_element_);
    WriteOne(writer, label_offset_);
    WriteOne(writer, offset_data_);
    WriteOne(writer, max_level_);
    WriteOne(writer, enterpoint_node_);
    WriteOne(writer, maxM_);

    WriteOne(writer, maxM0_);
    WriteOne(writer, M_);
    WriteOne(writer, mult_);
    WriteOne(writer, ef_construction_);

    data_level0_memory_->SerializeImpl(writer, cur_element_count_);

    for (size_t i = 0; i < cur_element_count_; i++) {
        unsigned int link_list_size =
            element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
        WriteOne(writer, link_list_size);
        if (link_list_size) {
            writer.Write(link_lists_[i], link_list_size);
        }
    }
    if (normalize_) {
        writer.Write(reinterpret_cast<char*>(molds_), max_elements_ * sizeof(float));
    }
}

// load using reader
void
HierarchicalNSW::loadIndex(std::function<void(uint64_t, uint64_t, void*)> read_func,
                           SpaceInterface* s,
                           size_t max_elements_i) {
    int64_t cursor = 0;
    ReadFuncStreamReader reader(read_func, cursor);
    DeserializeImpl(reader, s, max_elements_i);
}

// load index from a file stream
void
HierarchicalNSW::loadIndex(std::istream& in_stream, SpaceInterface* s, size_t max_elements_i) {
    IOStreamReader reader(in_stream);
    this->DeserializeImpl(reader, s, max_elements_i);
}

// origin load function
void
HierarchicalNSW::loadIndex(const std::string& location, SpaceInterface* s, size_t max_elements_i) {
    std::ifstream input(location, std::ios::binary);
    IOStreamReader reader(input);
    this->DeserializeImpl(reader, s, max_elements_i);
    input.close();
}

template <typename T>
static void
ReadOne(StreamReader& reader, T& value) {
    reader.Read(reinterpret_cast<char*>(&value), sizeof(value));
}

void
HierarchicalNSW::DeserializeImpl(StreamReader& reader, SpaceInterface* s, size_t max_elements_i) {
    ReadOne(reader, offsetLevel0_);

    size_t max_elements;
    ReadOne(reader, max_elements);
    max_elements = std::max(max_elements, max_elements_i);
    max_elements = std::max(max_elements, max_elements_);

    ReadOne(reader, cur_element_count_);
    ReadOne(reader, size_data_per_element_);
    ReadOne(reader, label_offset_);
    ReadOne(reader, offset_data_);
    ReadOne(reader, max_level_);
    ReadOne(reader, enterpoint_node_);

    ReadOne(reader, maxM_);
    ReadOne(reader, maxM0_);
    ReadOne(reader, M_);
    ReadOne(reader, mult_);
    ReadOne(reader, ef_construction_);

    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();

    resizeIndex(max_elements);
    data_level0_memory_->DeserializeImpl(reader, cur_element_count_);

    size_links_per_element_ = maxM_ * sizeof(InnerIdType) + sizeof(linklistsizeint);

    size_links_level0_ = maxM0_ * sizeof(InnerIdType) + sizeof(linklistsizeint);
    vsag::Vector<std::recursive_mutex>(max_elements, allocator_).swap(link_list_locks_);
    vsag::Vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS, allocator_).swap(label_op_locks_);

    rev_size_ = 1.0 / mult_;
    for (size_t i = 0; i < cur_element_count_; i++) {
        label_lookup_[getExternalLabel(i)] = i;
        unsigned int link_list_size;
        ReadOne(reader, link_list_size);
        if (link_list_size == 0) {
            element_levels_[i] = 0;
            link_lists_[i] = nullptr;
        } else {
            element_levels_[i] = link_list_size / size_links_per_element_;
            link_lists_[i] = (char*)allocator_->Allocate(link_list_size);
            if (link_lists_[i] == nullptr)
                throw std::runtime_error(
                    "Not enough memory: loadIndex failed to allocate linklist");
            reader.Read(link_lists_[i], link_list_size);
        }
    }
    if (normalize_) {
        reader.Read(reinterpret_cast<char*>(molds_), max_elements_ * sizeof(float));
    }

    if (use_reversed_edges_) {
        for (int internal_id = 0; internal_id < cur_element_count_; ++internal_id) {
            for (int level = 0; level <= element_levels_[internal_id]; ++level) {
                unsigned int* data = getLinklistAtLevel(internal_id, level);
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
                deleted_elements_.insert(i);
        }
    }
}

const float*
HierarchicalNSW::getDataByLabel(LabelType label) const {
    std::lock_guard<std::mutex> lock_label(getLabelOpMutex(label));

    std::unique_lock<std::mutex> lock_table(label_lookup_lock_);
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
        throw std::runtime_error("Label not found");
    }
    InnerIdType internalId = search->second;
    lock_table.unlock();

    char* data_ptrv = getDataByInternalId(internalId);
    auto* data_ptr = (float*)data_ptrv;

    return data_ptr;
}

/*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
void
HierarchicalNSW::markDelete(LabelType label) {
    // lock all operations with element by label
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

    std::unique_lock<std::mutex> lock_table(label_lookup_lock_);
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end()) {
        throw std::runtime_error("Label not found");
    }
    InnerIdType internalId = search->second;
    label_lookup_.erase(search);
    lock_table.unlock();
    markDeletedInternal(internalId);
}

/*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
void
HierarchicalNSW::markDeletedInternal(InnerIdType internalId) {
    assert(internalId < cur_element_count_);
    if (!isMarkedDeleted(internalId)) {
        unsigned char* ll_cur = ((unsigned char*)getLinklist0(internalId)) + 2;
        *ll_cur |= DELETE_MARK;
        num_deleted_ += 1;
        if (allow_replace_deleted_) {
            std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock_);
            deleted_elements_.insert(internalId);
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
HierarchicalNSW::unmarkDelete(LabelType label) {
    // lock all operations with element by label
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

    std::unique_lock<std::mutex> lock_table(label_lookup_lock_);
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end()) {
        throw std::runtime_error("Label not found");
    }
    InnerIdType internalId = search->second;
    lock_table.unlock();

    unmarkDeletedInternal(internalId);
}

/*
    * Remove the deleted mark of the node.
    */
void
HierarchicalNSW::unmarkDeletedInternal(InnerIdType internalId) {
    assert(internalId < cur_element_count_);
    if (isMarkedDeleted(internalId)) {
        unsigned char* ll_cur = ((unsigned char*)getLinklist0(internalId)) + 2;
        *ll_cur &= ~DELETE_MARK;
        num_deleted_ -= 1;
        if (allow_replace_deleted_) {
            std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock_);
            deleted_elements_.erase(internalId);
        }
    } else {
        throw std::runtime_error("The requested to undelete element is not deleted");
    }
}

/*
    * Adds point.
    */
bool
HierarchicalNSW::addPoint(const void* data_point, LabelType label) {
    std::lock_guard<std::mutex> lock_label(getLabelOpMutex(label));
    if (addPoint(data_point, label, -1) == -1) {
        return false;
    }
    return true;
}

void
HierarchicalNSW::modifyOutEdge(InnerIdType old_internal_id, InnerIdType new_internal_id) {
    for (int level = 0; level <= element_levels_[old_internal_id]; ++level) {
        auto& edges = getEdges(old_internal_id, level);
        for (const auto& in_node : edges) {
            auto data = getLinklistAtLevel(in_node, level);
            size_t link_size = getListCount(data);
            auto* links = (InnerIdType*)(data + 1);
            for (int i = 0; i < link_size; ++i) {
                if (links[i] == old_internal_id) {
                    links[i] = new_internal_id;
                    break;
                }
            }
        }
    }
}

void
HierarchicalNSW::modifyInEdges(InnerIdType right_internal_id,
                               InnerIdType wrong_internal_id,
                               bool is_erase) {
    for (int level = 0; level <= element_levels_[right_internal_id]; ++level) {
        auto data = getLinklistAtLevel(right_internal_id, level);
        size_t link_size = getListCount(data);
        auto* links = (InnerIdType*)(data + 1);
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
HierarchicalNSW::swapConnections(InnerIdType pre_internal_id, InnerIdType post_internal_id) {
    {
        // modify the connectivity relationships in the graph.
        // Through the reverse edges, change the edges pointing to pre_internal_id to point to
        // post_internal_id.
        modifyOutEdge(pre_internal_id, post_internal_id);
        modifyOutEdge(post_internal_id, pre_internal_id);

        // Swap the data and the adjacency lists of the graph.
        auto tmp_data_element = std::shared_ptr<char[]>(new char[size_data_per_element_]);
        memcpy(tmp_data_element.get(), getLinklist0(pre_internal_id), size_data_per_element_);
        memcpy(
            getLinklist0(pre_internal_id), getLinklist0(post_internal_id), size_data_per_element_);
        memcpy(getLinklist0(post_internal_id), tmp_data_element.get(), size_data_per_element_);

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
        std::swap(reversed_link_lists_[pre_internal_id], reversed_link_lists_[post_internal_id]);

        // First, remove the incorrect connectivity relationships in the reverse edges and then
        // proceed with the insertion. This avoids losing edges when a point simultaneously
        // has edges pointing to both pre_internal_id and post_internal_id.

        modifyInEdges(pre_internal_id, post_internal_id, true);
        modifyInEdges(post_internal_id, pre_internal_id, true);
        modifyInEdges(pre_internal_id, post_internal_id, false);
        modifyInEdges(post_internal_id, pre_internal_id, false);
    }

    if (enterpoint_node_ == post_internal_id) {
        enterpoint_node_ = pre_internal_id;
    } else if (enterpoint_node_ == pre_internal_id) {
        enterpoint_node_ = post_internal_id;
    }

    return true;
}

void
HierarchicalNSW::dealNoInEdge(InnerIdType id, int level, int m_curmax, int skip_c) {
    // Establish edges from the neighbors of the id pointing to the id.
    auto alone_data = getLinklistAtLevel(id, level);
    int alone_size = getListCount(alone_data);
    auto alone_link = (unsigned int*)(alone_data + 1);
    auto& in_edges = getEdges(id, level);
    for (int j = 0; j < alone_size; ++j) {
        if (alone_link[j] == skip_c) {
            continue;
        }
        auto to_edge_data_cur = (unsigned int*)getLinklistAtLevel(alone_link[j], level);
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
HierarchicalNSW::removePoint(LabelType label) {
    InnerIdType cur_c = 0;
    InnerIdType internal_id = 0;
    std::lock_guard<std::mutex> lock(global_);
    {
        // Swap the connection relationship corresponding to the label to be deleted with the
        // last element, and modify the information in label_lookup_. By swapping the two points,
        // fill the void left by the deletion.
        std::unique_lock<std::mutex> lock_table(label_lookup_lock_);
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
            max_level_ = -1;
            return;
        } else if (cur_c != internal_id) {
            label_lookup_[getExternalLabel(cur_c)] = internal_id;
            swapConnections(cur_c, internal_id);
        }
    }

    // If the node to be deleted is an entry node, find another top-level node.
    if (cur_c == enterpoint_node_) {
        for (int level = max_level_; level >= 0; level--) {
            auto data = (unsigned int*)getLinklistAtLevel(enterpoint_node_, level);
            int size = getListCount(data);
            if (size != 0) {
                max_level_ = level;
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
        auto data_cur = getLinklistAtLevel(cur_c, level);
        int size_cur = getListCount(data_cur);
        auto data_link_cur = (unsigned int*)(data_cur + 1);

        for (const auto in_edge : in_edges_cur) {
            MaxHeap candidates(allocator_);
            vsag::UnorderedSet<InnerIdType> unique_ids(allocator_);

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
            auto in_edge_data_cur = (unsigned int*)getLinklistAtLevel(in_edge, level);
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

            if (candidates.empty()) {
                setListCount(in_edge_data_cur, 0);
                getEdges(cur_c, level).erase(in_edge);
                continue;
            }
            mutuallyConnectNewElement(in_edge, candidates, level, true);

            // Handle the operations of the deletion point which result in some nodes having no
            // indegree nodes, and carry out repairs.
            size_t m_curmax = level ? maxM_ : maxM0_;
            for (auto id : unique_ids) {
                if (getEdges(id, level).empty()) {
                    dealNoInEdge(id, level, m_curmax, cur_c);
                }
            }
        }

        for (int i = 0; i < size_cur; ++i) {
            getEdges(data_link_cur[i], level).erase(cur_c);
        }
    }
}

InnerIdType
HierarchicalNSW::addPoint(const void* data_point, LabelType label, int level) {
    InnerIdType cur_c = 0;
    {
        // Checking if the element with the same label already exists
        // if so, updating it *instead* of creating a new element.
        std::unique_lock<std::mutex> lock_table(label_lookup_lock_);
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
    normalizeVector(data_point, normalize_data);

    std::unique_lock<std::recursive_mutex> lock_el(link_list_locks_[cur_c]);
    int curlevel = getRandomLevel(mult_);
    if (level > 0)
        curlevel = level;

    element_levels_[cur_c] = curlevel;
    std::unique_lock<std::mutex> lock(global_);
    int maxlevelcopy = max_level_;
    if (curlevel <= maxlevelcopy)
        lock.unlock();
    InnerIdType currObj = enterpoint_node_;
    InnerIdType enterpoint_copy = enterpoint_node_;

    memset(data_level0_memory_->GetElementPtr(cur_c, offsetLevel0_), 0, size_data_per_element_);

    // Initialisation of the data and label
    memcpy(getExternalLabeLp(cur_c), &label, sizeof(LabelType));
    memcpy(getDataByInternalId(cur_c), data_point, data_size_);
    if (curlevel) {
        auto new_link_lists = (char*)allocator_->Reallocate(link_lists_[cur_c],
                                                            size_links_per_element_ * curlevel + 1);
        if (new_link_lists == nullptr)
            throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
        link_lists_[cur_c] = new_link_lists;
        memset(link_lists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
    }

    if ((signed)currObj != -1) {
        if (curlevel < maxlevelcopy) {
            float curdist =
                fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
            for (int lev = maxlevelcopy; lev > curlevel; lev--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int* data;
                    std::unique_lock<std::recursive_mutex> link_lock(link_list_locks_[currObj]);
                    data = getLinklist(currObj, lev);
                    int size = getListCount(data);

                    auto* datal = (InnerIdType*)(data + 1);
                    for (int i = 0; i < size; i++) {
                        InnerIdType cand = datal[i];
                        if (cand > max_elements_)
                            throw std::runtime_error("cand error");
                        float d =
                            fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
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
        for (int lev = std::min(curlevel, maxlevelcopy); lev >= 0; lev--) {
            if (lev > maxlevelcopy)  // possible?
                throw std::runtime_error("Level error");

            MaxHeap top_candidates = searchBaseLayer(currObj, data_point, lev);
            if (epDeleted) {
                top_candidates.emplace(
                    fstdistfunc_(
                        data_point, getDataByInternalId(enterpoint_copy), dist_func_param_),
                    enterpoint_copy);
                if (top_candidates.size() > ef_construction_)
                    top_candidates.pop();
            }
            currObj = mutuallyConnectNewElement(cur_c, top_candidates, lev, false);
        }
    } else {
        // Do nothing for the first element
        enterpoint_node_ = 0;
        max_level_ = curlevel;
    }

    // Releasing lock for the maximum level
    if (curlevel > maxlevelcopy) {
        enterpoint_node_ = cur_c;
        max_level_ = curlevel;
    }
    return cur_c;
}

std::priority_queue<std::pair<float, LabelType>>
HierarchicalNSW::searchKnn(const void* query_data,
                           size_t k,
                           uint64_t ef,
                           vsag::BaseFilterFunctor* isIdAllowed) const {
    std::priority_queue<std::pair<float, LabelType>> result;
    if (cur_element_count_ == 0)
        return result;

    std::shared_ptr<float[]> normalize_query;
    normalizeVector(query_data, normalize_query);
    InnerIdType currObj = enterpoint_node_;
    float curdist =
        fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
    for (int level = max_level_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int* data;

            data = (unsigned int*)getLinklist(currObj, level);
            int size = getListCount(data);
            metric_hops_++;
            metric_distance_computations_ += size;

            auto* datal = (InnerIdType*)(data + 1);
            for (int i = 0; i < size; i++) {
                InnerIdType cand = datal[i];
                if (cand > max_elements_)
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

    MaxHeap top_candidates(allocator_);

    top_candidates =
        searchBaseLayerST<false, true>(currObj, query_data, std::max(ef, k), isIdAllowed);

    while (top_candidates.size() > k) {
        top_candidates.pop();
    }
    while (not top_candidates.empty()) {
        std::pair<float, InnerIdType> rez = top_candidates.top();
        result.emplace(rez.first, getExternalLabel(rez.second));
        top_candidates.pop();
    }
    return result;
}

std::priority_queue<std::pair<float, LabelType>>
HierarchicalNSW::searchRange(const void* query_data,
                             float radius,
                             uint64_t ef,
                             vsag::BaseFilterFunctor* isIdAllowed) const {
    std::priority_queue<std::pair<float, LabelType>> result;
    if (cur_element_count_ == 0)
        return result;

    std::shared_ptr<float[]> normalize_query;
    normalizeVector(query_data, normalize_query);
    InnerIdType currObj = enterpoint_node_;
    float curdist =
        fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

    for (int level = max_level_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int* data;

            data = (unsigned int*)getLinklist(currObj, level);
            int size = getListCount(data);
            metric_hops_++;
            metric_distance_computations_ += size;

            auto* datal = (InnerIdType*)(data + 1);
            for (int i = 0; i < size; i++) {
                InnerIdType cand = datal[i];
                if (cand > max_elements_)
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

    MaxHeap top_candidates(allocator_);

    top_candidates = searchBaseLayerST<false, true>(currObj, query_data, radius, ef, isIdAllowed);

    while (not top_candidates.empty()) {
        std::pair<float, InnerIdType> rez = top_candidates.top();
        result.emplace(rez.first, getExternalLabel(rez.second));
        top_candidates.pop();
    }

    // std::cout << "hnswalg::result.size(): " << result.size() << std::endl;
    return result;
}

void
HierarchicalNSW::checkIntegrity() {
    int connections_checked = 0;
    vsag::Vector<int> inbound_connections_num(cur_element_count_, 0, allocator_);
    for (int i = 0; i < cur_element_count_; i++) {
        for (int l = 0; l <= element_levels_[i]; l++) {
            linklistsizeint* ll_cur = getLinklistAtLevel(i, l);
            int size = getListCount(ll_cur);
            auto* data = (InnerIdType*)(ll_cur + 1);
            vsag::UnorderedSet<InnerIdType> s(allocator_);
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

}  // namespace hnswlib
