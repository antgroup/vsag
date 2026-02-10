
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

#include "datacell/graph_interface.h"
#include "impl/searcher/basic_searcher.h"
#include "utils/linear_congruential_generator.h"
#include "utils/prefetch.h"

namespace hnswlib {

const static InnerIdType UNUSED_ENTRY_POINT_NODE = 0;
HierarchicalNSW::HierarchicalNSW(SpaceInterface* s,
                                 uint64_t max_elements,
                                 vsag::Allocator* allocator,
                                 uint64_t M,
                                 uint64_t ef_construction,
                                 bool use_reversed_edges,
                                 bool normalize,
                                 uint64_t block_size_limit,
                                 uint64_t random_seed,
                                 bool allow_replace_deleted)
    : allocator_(allocator),
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
    dim_ = *((uint64_t*)dist_func_param_);
    prefetch_jump_code_size_ = std::max(1, static_cast<int32_t>(data_size_ / (64 * 2)) - 1);
    M_ = M;
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);

    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    points_locks_ = std::make_shared<vsag::PointsMutex>(max_elements, allocator);

    size_links_level0_ = maxM0_ * sizeof(InnerIdType) + sizeof(linklistsizeint);
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(LabelType);
    offset_data_ = size_links_level0_;
    label_offset_ = size_links_level0_ + data_size_;
    offsetLevel0_ = 0;

    data_level0_memory_ =
        std::make_shared<BlockManager>(size_data_per_element_, block_size_limit, allocator_);
    data_element_per_block_ = block_size_limit / size_data_per_element_;

    cur_element_count_ = 0;

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    max_level_ = -1;
    size_links_per_element_ = maxM_ * sizeof(InnerIdType) + sizeof(linklistsizeint);
    mult_ = 1 / log(1.0 * static_cast<double>(M_));
    rev_size_ = 1.0 / mult_;
}

void
HierarchicalNSW::reset() {
    if (visited_list_pool_) {
        allocator_->Delete(visited_list_pool_);
        visited_list_pool_ = nullptr;
    }
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
    visited_list_pool_ = allocator_->New<VisitedListPool>(max_elements_, allocator_);
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

uint64_t
HierarchicalNSW::estimateMemory(uint64_t num_elements) {
    uint64_t size = 0;
    size += sizeof(unsigned short int) * num_elements;  // visited_list_pool_
    size += sizeof(int) * num_elements;                 // element_levels_
    size += num_elements * size_data_per_element_;      // data_level0_memory_
    if (use_reversed_edges_) {
        size += sizeof(reverselinklist*) * num_elements;  // reversed_level0_link_list_
        size += sizeof(vsag::UnorderedMap<int, reverselinklist>*) *
                num_elements;  // reversed_link_lists_
    }
    if (normalize_) {
        size += sizeof(float) * num_elements;  // molds_
    }
    size += sizeof(void*) * num_elements;              // link_lists_
    size += sizeof(std::shared_mutex) * num_elements;  // points_locks_
    return size;
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
    InnerIdType internal_id = getInternalId(label);
    return getDistanceByInternalId(internal_id, data_point);
}

float
HierarchicalNSW::getDistanceByInternalId(uint32_t internal_id, const void* data_point) {
    std::shared_lock resize_lock(resize_mutex_);

    std::shared_ptr<float[]> normalize_query;
    normalizeVector(data_point, normalize_query);
    float dist = fstdistfunc_(data_point, getDataByInternalId(internal_id), dist_func_param_);
    return dist;
}

float
HierarchicalNSW::getSelfDistanceByInternalId(uint32_t internal_id) {
    return getDistanceByInternalId(internal_id, getDataByInternalId(internal_id));
}

uint32_t
HierarchicalNSW::getInternalId(LabelType label) {
    std::shared_lock lock_table(label_lookup_lock_);

    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end()) {
        throw std::runtime_error("Label not found");
    }

    return search->second;
}

void
HierarchicalNSW::getNeighborsInternalId(uint32_t internal_id,
                                        vsag::Vector<InnerIdType>& neighbor_ids) {
    vsag::Vector<uint32_t> neighbors_id(allocator_);

    int* data = (int*)get_linklist0(internal_id);
    uint32_t size = getListCount((hnswlib::linklistsizeint*)data);
    neighbor_ids.resize(size);
    for (uint32_t i = 0; i < size; i++) {
        neighbor_ids[i] = *(data + i + 1);
    }
}

tl::expected<vsag::DatasetPtr, vsag::Error>
HierarchicalNSW::getBatchDistanceByLabel(const int64_t* ids,
                                         const void* data_point,
                                         int64_t count) {
    std::shared_lock lock_table(label_lookup_lock_);
    int64_t valid_cnt = 0;
    auto result = vsag::Dataset::Make();
    result->Owner(true, allocator_);
    auto* distances = (float*)allocator_->Allocate(sizeof(float) * count);
    result->Distances(distances);
    std::shared_ptr<float[]> normalize_query;
    normalizeVector(data_point, normalize_query);
    for (int i = 0; i < count; i++) {
        auto search = label_lookup_.find(ids[i]);
        if (search == label_lookup_.end()) {
            distances[i] = -1;
        } else {
            InnerIdType internal_id = search->second;
            float dist =
                fstdistfunc_(data_point, getDataByInternalId(internal_id), dist_func_param_);
            distances[i] = dist;
            valid_cnt++;
        }
    }
    result->NumElements(count);
    return std::move(result);
}

std::pair<int64_t, int64_t>
HierarchicalNSW::getMinAndMaxId() {
    int64_t min_id = INT64_MAX;
    int64_t max_id = INT64_MIN;
    std::shared_lock lock_table(label_lookup_lock_);
    if (label_lookup_.size() == 0) {
        throw std::runtime_error("Label map size is zero");
    }
    for (auto& it : label_lookup_) {
        max_id = it.first > max_id ? it.first : max_id;
        min_id = it.first < min_id ? it.first : min_id;
    }
    return {min_id, max_id};
}

bool
HierarchicalNSW::isValidLabel(LabelType label) {
    std::shared_lock lock_table(label_lookup_lock_);
    bool is_valid = (label_lookup_.find(label) != label_lookup_.end());
    return is_valid;
}

bool
HierarchicalNSW::isTombLabel(LabelType label) {
    std::shared_lock lock_table(label_lookup_lock_);

    if (not allow_replace_deleted_) {
        return false;
    }
    auto is_tomb = (deleted_elements_.find(label) != deleted_elements_.end());
    return is_tomb;
}

void
HierarchicalNSW::setBatchNeigohbors(InnerIdType internal_id,
                                    int level,
                                    const InnerIdType* neighbors,
                                    uint64_t neigbor_count) {
    vsag::LockGuard lock(points_locks_, internal_id);
    linklistsizeint* ll_cur = getLinklistAtLevel(internal_id, level);
    for (int i = 1; i <= neigbor_count; ++i) {
        ll_cur[i] = neighbors[i - 1];
    }

    setListCount(ll_cur, neigbor_count);
}

void
HierarchicalNSW::appendNeigohbor(InnerIdType internal_id,
                                 int level,
                                 InnerIdType neighbor,
                                 uint64_t max_degree) {
    vsag::LockGuard lock(points_locks_, internal_id);
    linklistsizeint* ll_cur = getLinklistAtLevel(internal_id, level);
    uint64_t neigbor_count = getListCount(ll_cur) + 1;
    if (neigbor_count <= max_degree) {
        ll_cur[neigbor_count] = neighbor;
        setListCount(ll_cur, neigbor_count);
    }
}

void
HierarchicalNSW::updateConnections(InnerIdType internal_id,
                                   const vsag::Vector<InnerIdType>& cand_neighbors,
                                   int level,
                                   bool is_update) {
    std::shared_ptr<char[]> link_data = std::shared_ptr<char[]>(new char[size_links_level0_]);
    getLinklistAtLevel(internal_id, level, link_data.get());
    linklistsizeint* ll_cur = (linklistsizeint*)link_data.get();

    auto cur_size = getListCount(ll_cur);
    auto* data = (InnerIdType*)(ll_cur + 1);

    if (use_reversed_edges_) {
        if (is_update) {
            for (int i = 0; i < cur_size; ++i) {
                auto id = data[i];
                auto& in_edges = getEdges(id, level);
                // remove the node that point to the current node
                in_edges.erase(internal_id);
            }
        }
        for (uint64_t i = 0; i < cand_neighbors.size(); i++) {
            auto id = cand_neighbors[i];
            auto& in_edges = getEdges(id, level);
            in_edges.insert(internal_id);
        }
    }
    setBatchNeigohbors(internal_id, level, cand_neighbors.data(), cand_neighbors.size());
}

bool
HierarchicalNSW::checkReverseConnection() {
    int edge_count = 0;
    uint64_t reversed_edge_count = 0;
    std::shared_ptr<char[]> link_data = std::shared_ptr<char[]>(new char[size_links_level0_]);
    for (int internal_id = 0; internal_id < cur_element_count_; ++internal_id) {
        for (int level = 0; level <= element_levels_[internal_id]; ++level) {
            getLinklistAtLevel(internal_id, level, link_data.get());
            unsigned int* data = (unsigned int*)link_data.get();
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
HierarchicalNSW::bruteForce(const void* data_point,
                            int64_t k,
                            const vsag::FilterPtr is_id_allowed) const {
    std::shared_lock resize_lock(resize_mutex_);
    std::priority_queue<std::pair<float, LabelType>> results;
    for (uint32_t i = 0; i < cur_element_count_; i++) {
        if (is_id_allowed && not is_id_allowed->CheckValid(getExternalLabel(i))) {
            continue;
        }
        float dist = fstdistfunc_(data_point, getDataByInternalId(i), dist_func_param_);
        if (results.size() < k) {
            results.emplace(dist, this->getExternalLabel(i));
        } else {
            float current_max_dist = results.top().first;
            if (dist < current_max_dist) {
                results.pop();
                results.emplace(dist, this->getExternalLabel(i));
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
HierarchicalNSW::searchBaseLayer(InnerIdType ep_id, const void* data_point, int layer) const {
    VisitedListPtr vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    MaxHeap top_candidates(allocator_);
    MaxHeap candidateSet(allocator_);

    float lower_bound;
    if (!isMarkedDeleted(ep_id)) {
        float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        top_candidates.emplace(dist, ep_id);
        lower_bound = dist;
        candidateSet.emplace(-dist, ep_id);
    } else {
        lower_bound = std::numeric_limits<float>::max();
        candidateSet.emplace(-lower_bound, ep_id);
    }
    visited_array[ep_id] = visited_array_tag;

    std::shared_ptr<char[]> link_data = std::shared_ptr<char[]>(new char[size_links_level0_]);
    while (not candidateSet.empty()) {
        std::pair<float, InnerIdType> curr_el_pair = candidateSet.top();
        if ((-curr_el_pair.first) > lower_bound && top_candidates.size() == ef_construction_) {
            break;
        }
        candidateSet.pop();

        InnerIdType curNodeNum = curr_el_pair.second;

        getLinklistAtLevel(curNodeNum, layer, link_data.get());
        int* data =
            (int*)
                link_data.get();  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
        uint64_t size = getListCount((linklistsizeint*)data);
        auto* datal = (InnerIdType*)(data + 1);
        vsag::PrefetchLines((char*)(visited_array + *(data + 1)), 64);
        vsag::PrefetchLines((char*)(visited_array + *(data + 1) + 64), 64);
        vsag::PrefetchLines(getDataByInternalId(*datal), 64);
        vsag::PrefetchLines(getDataByInternalId(*(datal + 1)), 64);

        for (uint64_t j = 0; j < size; j++) {
            InnerIdType candidate_id = *(datal + j);
            uint64_t pre_l = std::min(j, size - 2);
            vsag::PrefetchLines((char*)(visited_array + *(datal + pre_l + 1)), 64);
            vsag::PrefetchLines(getDataByInternalId(*(datal + pre_l + 1)), 64);
            if (visited_array[candidate_id] == visited_array_tag)
                continue;
            visited_array[candidate_id] = visited_array_tag;
            char* currObj1 = (getDataByInternalId(candidate_id));

            float dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
            if (top_candidates.size() < ef_construction_ || lower_bound > dist1) {
                candidateSet.emplace(-dist1, candidate_id);
                vsag::PrefetchLines(getDataByInternalId(candidateSet.top().second), 64);

                if (not isMarkedDeleted(candidate_id))
                    top_candidates.emplace(dist1, candidate_id);

                if (top_candidates.size() > ef_construction_)
                    top_candidates.pop();

                if (not top_candidates.empty())
                    lower_bound = top_candidates.top().first;
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
                                   uint64_t ef,
                                   const vsag::FilterPtr is_id_allowed,
                                   const float skip_ratio,
                                   vsag::Allocator* allocator,
                                   vsag::IteratorFilterContext* iter_ctx) const {
    vsag::LinearCongruentialGenerator generator;
    VisitedListPtr vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    vsag::Allocator* search_allocator = allocator == nullptr ? allocator_ : allocator;

    MaxHeap top_candidates(search_allocator);
    MaxHeap candidate_set(search_allocator);

    float valid_ratio = is_id_allowed ? is_id_allowed->ValidRatio() : 1.0F;
    float skip_threshold = valid_ratio == 1.0F ? 0 : (1 - ((1 - valid_ratio) * skip_ratio));

    float lower_bound;
    if (iter_ctx != nullptr && !iter_ctx->IsFirstUsed()) {
        lower_bound = 0.0;
        while (!iter_ctx->Empty()) {
            uint32_t cur_inner_id = iter_ctx->GetTopID();
            float cur_dist = iter_ctx->GetTopDist();
            visited_array[cur_inner_id] = visited_array_tag;
            if (iter_ctx->CheckPoint(cur_inner_id)) {
                top_candidates.emplace(cur_dist, cur_inner_id);
                candidate_set.emplace(-cur_dist, cur_inner_id);
                lower_bound = std::max(lower_bound, cur_dist);
            }
            iter_ctx->PopDiscard();
        }
    } else {
        if ((!has_deletions || !isMarkedDeleted(ep_id)) &&
            ((!is_id_allowed) || is_id_allowed->CheckValid(getExternalLabel(ep_id)))) {
            float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            lower_bound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);
        } else {
            lower_bound = std::numeric_limits<float>::max();
            candidate_set.emplace(-lower_bound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;
    }

    std::shared_ptr<char[]> link_data = std::shared_ptr<char[]>(new char[size_links_level0_]);
    while (not candidate_set.empty()) {
        std::pair<float, InnerIdType> current_node_pair = candidate_set.top();

        if ((-current_node_pair.first) > lower_bound &&
            (top_candidates.size() == ef || (!is_id_allowed && !has_deletions))) {
            break;
        }
        candidate_set.pop();

        InnerIdType current_node_id = current_node_pair.second;
        getLinklistAtLevel(current_node_id, 0, link_data.get());
        int* data = (int*)link_data.get();
        uint64_t size = getListCount((linklistsizeint*)data);
        //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
        if (collect_metrics) {
            metric_hops_++;
            metric_distance_computations_ += size;
        }

        vsag::PrefetchLines((char*)(visited_array + *(data + 1)), 64);
        vsag::PrefetchLines((char*)(visited_array + *(data + 1) + 64), 64);
        char* vector_data_ptr = nullptr;
        vector_data_ptr = data_level0_memory_->GetElementPtr((*(data + 1)), offset_data_);
        vsag::PrefetchLines(vector_data_ptr, data_size_);
        vsag::PrefetchLines((char*)(data + 2), 64);

        for (uint64_t j = 1; j <= size; j++) {
            int candidate_id = *(data + j);
            uint64_t pre_l = std::min(j, size - 2);
            if (pre_l + prefetch_jump_code_size_ <= size) {
                vector_data_ptr = data_level0_memory_->GetElementPtr(
                    (*(data + pre_l + prefetch_jump_code_size_)), offset_data_);
                vsag::PrefetchLines(
                    (char*)(visited_array + *(data + pre_l + prefetch_jump_code_size_)), 64);
                vsag::PrefetchLines(vector_data_ptr, data_size_);
            }
            if (visited_array[candidate_id] != visited_array_tag) {
                visited_array[candidate_id] = visited_array_tag;
                if (is_id_allowed && not candidate_set.empty() &&
                    generator.NextFloat() < skip_threshold &&
                    not is_id_allowed->CheckValid(getExternalLabel(candidate_id))) {
                    continue;
                }
                float dist = 0;
                char* currObj1 = getDataByInternalId(candidate_id);
                dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef || lower_bound > dist) {
                    candidate_set.emplace(-dist, candidate_id);
                    vector_data_ptr = data_level0_memory_->GetElementPtr(candidate_set.top().second,
                                                                         offsetLevel0_);
                    vsag::PrefetchLines(vector_data_ptr, 64);

                    if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
                        ((!is_id_allowed) ||
                         is_id_allowed->CheckValid(getExternalLabel(candidate_id)))) {
                        if (iter_ctx != nullptr && !iter_ctx->CheckPoint(candidate_id)) {
                            continue;
                        }
                        top_candidates.emplace(dist, candidate_id);
                    }

                    if (top_candidates.size() > ef) {
                        auto cur_node_pair = top_candidates.top();
                        if (iter_ctx != nullptr && iter_ctx->CheckPoint(cur_node_pair.second)) {
                            iter_ctx->AddDiscardNode(cur_node_pair.first, cur_node_pair.second);
                        }
                        top_candidates.pop();
                    }

                    if (not top_candidates.empty())
                        lower_bound = top_candidates.top().first;
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
                                   const vsag::FilterPtr is_id_allowed) const {
    VisitedListPtr vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    MaxHeap top_candidates(allocator_);
    MaxHeap candidate_set(allocator_);

    float lower_bound;
    if ((!has_deletions || !isMarkedDeleted(ep_id)) &&
        ((!is_id_allowed) || is_id_allowed->CheckValid(getExternalLabel(ep_id)))) {
        float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        lower_bound = dist;
        if (dist <= radius + vsag::THRESHOLD_ERROR)
            top_candidates.emplace(dist, ep_id);
        candidate_set.emplace(-dist, ep_id);
    } else {
        lower_bound = std::numeric_limits<float>::max();
        candidate_set.emplace(-lower_bound, ep_id);
    }

    visited_array[ep_id] = visited_array_tag;
    uint64_t visited_count = 0;

    std::shared_ptr<char[]> link_data = std::shared_ptr<char[]>(new char[size_links_level0_]);
    while (not candidate_set.empty()) {
        std::pair<float, InnerIdType> current_node_pair = candidate_set.top();

        candidate_set.pop();

        InnerIdType current_node_id = current_node_pair.second;
        getLinklistAtLevel(current_node_id, 0, link_data.get());
        int* data = (int*)link_data.get();
        uint64_t size = getListCount((linklistsizeint*)data);
        //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
        if (collect_metrics) {
            metric_hops_++;
            metric_distance_computations_ += size;
        }

        vsag::PrefetchLines((char*)(visited_array + *(data + 1)), 64);
        vsag::PrefetchLines((char*)(visited_array + *(data + 1) + 64), 64);
        char* vector_data_ptr = nullptr;
        vector_data_ptr = data_level0_memory_->GetElementPtr((*(data + 1)), offset_data_);
        vsag::PrefetchLines(vector_data_ptr, 64);
        vsag::PrefetchLines((char*)(data + 2), 64);

        for (uint64_t j = 1; j <= size; j++) {
            int candidate_id = *(data + j);
            uint64_t pre_l = std::min(j, size - 2);
            if (pre_l + prefetch_jump_code_size_ <= size) {
                vector_data_ptr = data_level0_memory_->GetElementPtr(
                    (*(data + pre_l + prefetch_jump_code_size_)), offset_data_);
                vsag::PrefetchLines(
                    (char*)(visited_array + *(data + pre_l + prefetch_jump_code_size_)), 64);
                vsag::PrefetchLines(vector_data_ptr, data_size_);
            }
            if (visited_array[candidate_id] != visited_array_tag) {
                visited_array[candidate_id] = visited_array_tag;
                ++visited_count;

                char* currObj1 = (getDataByInternalId(candidate_id));
                float dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                if (visited_count < ef || dist <= radius + vsag::THRESHOLD_ERROR ||
                    lower_bound > dist) {
                    candidate_set.emplace(-dist, candidate_id);
                    vector_data_ptr = data_level0_memory_->GetElementPtr(candidate_set.top().second,
                                                                         offsetLevel0_);
                    vsag::PrefetchLines(vector_data_ptr, 64);

                    if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
                        ((!is_id_allowed) ||
                         is_id_allowed->CheckValid(getExternalLabel(candidate_id))))
                        top_candidates.emplace(dist, candidate_id);

                    if (not top_candidates.empty())
                        lower_bound = top_candidates.top().first;
                }
            }
        }
    }
    while (not top_candidates.empty() &&
           top_candidates.top().first > radius + vsag::THRESHOLD_ERROR) {
        top_candidates.pop();
    }

    visited_list_pool_->releaseVisitedList(vl);
    return top_candidates;
}

void
HierarchicalNSW::getNeighborsByHeuristic2(MaxHeap& top_candidates, uint64_t M) {
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
        std::pair<float, InnerIdType> current_pair = queue_closest.top();
        float float_query = -current_pair.first;
        queue_closest.pop();
        bool good = true;

        for (std::pair<float, InnerIdType> second_pair : return_list) {
            float curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                                         getDataByInternalId(current_pair.second),
                                         dist_func_param_);
            if (curdist < float_query) {
                good = false;
                break;
            }
        }
        if (good) {
            return_list.emplace_back(current_pair);
        }
    }

    for (std::pair<float, InnerIdType> current_pair : return_list) {
        top_candidates.emplace(-current_pair.first, current_pair.second);
    }
}

InnerIdType
HierarchicalNSW::mutuallyConnectNewElement(InnerIdType cur_c,
                                           MaxHeap& top_candidates,
                                           int level,
                                           bool isUpdate) {
    uint64_t m_curmax = level ? maxM_ : maxM0_;
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

    updateConnections(cur_c, selectedNeighbors, level, isUpdate);

    std::shared_ptr<char[]> ll_other_data = std::shared_ptr<char[]>(new char[size_links_level0_]);
    for (unsigned int selectedNeighbor : selectedNeighbors) {
        getLinklistAtLevel(selectedNeighbor, level, ll_other_data.get());
        linklistsizeint* ll_other = (linklistsizeint*)ll_other_data.get();

        uint64_t sz_link_list_other = getListCount(ll_other);

        if (sz_link_list_other > m_curmax)
            throw std::runtime_error("Bad value of sz_link_list_other");
        if (selectedNeighbor == cur_c)
            throw std::runtime_error("Trying to connect an element to itself");
        if (level > element_levels_[selectedNeighbor])
            throw std::runtime_error("Trying to make a link on a non-existent level");

        auto* data = (InnerIdType*)(ll_other + 1);

        bool is_cur_c_present = false;
        if (isUpdate) {
            for (uint64_t j = 0; j < sz_link_list_other; j++) {
                if (data[j] == cur_c) {
                    is_cur_c_present = true;
                    break;
                }
            }
        }

        // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
        if (!is_cur_c_present) {
            if (sz_link_list_other < m_curmax) {
                appendNeigohbor(selectedNeighbor, level, cur_c, m_curmax);
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

                for (uint64_t j = 0; j < sz_link_list_other; j++) {
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
HierarchicalNSW::resizeIndex(uint64_t new_max_elements) {
    std::unique_lock resize_lock(resize_mutex_);
    if (new_max_elements < cur_element_count_)
        throw std::runtime_error(
            "Cannot Resize, max element is less than the current number of elements");

    auto new_visited_list_pool = allocator_->New<VisitedListPool>(new_max_elements, allocator_);
    allocator_->Delete(visited_list_pool_);
    visited_list_pool_ = new_visited_list_pool;

    auto element_levels_new =
        (int*)allocator_->Reallocate(element_levels_, new_max_elements * sizeof(int));
    if (element_levels_new == nullptr) {
        throw std::runtime_error(
            "Not enough memory: resizeIndex failed to allocate element_levels_");
    }
    element_levels_ = element_levels_new;
    this->points_locks_->Resize(new_max_elements);

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
    char** link_lists_new =
        (char**)allocator_->Reallocate(link_lists_, sizeof(void*) * new_max_elements);
    if (link_lists_new == nullptr)
        throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
    link_lists_ = link_lists_new;
    memset(link_lists_ + max_elements_, 0, (new_max_elements - max_elements_) * sizeof(void*));
    max_elements_ = new_max_elements;
}

uint64_t
HierarchicalNSW::calcSerializeSize() {
    auto calSizeFunc = [](uint64_t cursor, uint64_t size, void* buf) { return; };
    WriteFuncStreamWriter writer(calSizeFunc, 0);
    this->SerializeImpl(writer);
    return writer.cursor_;
}

void
HierarchicalNSW::saveIndex(StreamWriter& writer) {
    SerializeImpl(writer);
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

    for (uint64_t i = 0; i < cur_element_count_; i++) {
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

// load index from a file stream
void
HierarchicalNSW::loadIndex(StreamReader& buffer_reader,
                           SpaceInterface* s,
                           uint64_t max_elements_i) {
    this->DeserializeImpl(buffer_reader, s, max_elements_i);
}

template <typename T>
static void
ReadOne(StreamReader& reader, T& value) {
    reader.Read(reinterpret_cast<char*>(&value), sizeof(value));
}

void
HierarchicalNSW::DeserializeImpl(StreamReader& reader, SpaceInterface* s, uint64_t max_elements_i) {
    ReadOne(reader, offsetLevel0_);

    uint64_t max_elements;
    ReadOne(reader, max_elements);
    max_elements = std::max(max_elements, max_elements_i);
    max_elements = std::max(max_elements, max_elements_);

    ReadOne(reader, cur_element_count_);
    ReadOne(reader, size_data_per_element_);
    ReadOne(reader, label_offset_);
    ReadOne(reader, offset_data_);
    ReadOne(reader, max_level_);

    // Fixes #623: Unified entrypoint_node type during index loading (old: int64_t → new: InnerIdType (i.e., uint32))
    /*
     * Header Format Diagram for enterpoint_node_ and maxM_
     *
     * This diagram illustrates the serialization format differences between old and new versions.
     * The code handles backward compatibility by trying both formats during index loading.
     */
    // New Format (v2) Header Layout
    // -----------------------------
    // | Field              | Type       | Size (bytes) | Description                  |
    // |--------------------|------------|--------------|------------------------------|
    // | enterpoint_node_   | InnerIdType| 4            | 32-bit unsigned integer      |
    // | maxM_              | uint64_t   | 8            | Maximum connections count    |
    // ----------------------------- 4 + 8 = 12 bytes total -----------------------------------

    // Old Format (v1) Header Layout
    // -----------------------------
    // | Field              | Type       | Size (bytes) | Description                  |
    // |--------------------|------------|--------------|------------------------------|
    // | enterpoint_node_   | int64_t    | 8            | 64-bit signed integer        |
    // | maxM_              | uint64_t   | 8            | Maximum connections count    |
    // ----------------------------- 8 + 8 = 16 bytes total -----------------------------------

    /*
     * Compatibility Logic Flow:
     * 1. Try reading newer format (12 bytes)
     * 2. If validation fails (M_ != maxM_),
     * 3. Read older format (16 bytes) as fallback
     */

    // to resolve compatibility issues
    auto buffer_size = sizeof(int64_t) + sizeof(uint64_t);
    auto newer_format_size = sizeof(InnerIdType) + sizeof(uint64_t);
    vsag::Vector<char> buffer(buffer_size, allocator_);
    char* raw_buffer = buffer.data();

    // step 1, try to parse/read with the newer serial format
    reader.Read(raw_buffer, newer_format_size);
    enterpoint_node_ = *(InnerIdType*)(raw_buffer);
    maxM_ = *(uint64_t*)(raw_buffer + sizeof(InnerIdType));
    bool is_newer_format = (M_ == maxM_);

    // step 2, try to read with the older serial format
    if (not is_newer_format) {
        reader.Read(raw_buffer + newer_format_size, buffer_size - newer_format_size);
        enterpoint_node_ = *(int64_t*)(raw_buffer);
        maxM_ = *(uint64_t*)(raw_buffer + sizeof(int64_t));
        if (M_ != maxM_) {
            // this condition will be true only when the parameter used in create_index is not equal
            // to the parameter of the serialized index
            throw vsag::VsagException(
                vsag::ErrorType::INTERNAL_ERROR,
                "The index was saved with different M_ value, please use the same M_ value");
        }
    }

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
    this->points_locks_->Resize(max_elements);

    rev_size_ = 1.0 / mult_;
    for (uint64_t i = 0; i < cur_element_count_; i++) {
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
                std::shared_ptr<char[]> link_data =
                    std::shared_ptr<char[]>(new char[size_links_level0_]);
                getLinklistAtLevel(internal_id, level, link_data.get());
                unsigned int* data = (unsigned int*)link_data.get();
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

    for (uint64_t i = 0; i < cur_element_count_; i++) {
        if (isMarkedDeleted(i)) {
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                deleted_elements_.insert({getExternalLabel(i), i});
            }
        }
    }
}

const float*
HierarchicalNSW::getDataByLabel(LabelType label) const {
    std::unique_lock lock_table(label_lookup_lock_);
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
        throw std::runtime_error("Label not found");
    }
    InnerIdType internalId = search->second;

    char* data_ptrv = getDataByInternalId(internalId);
    auto* data_ptr = (float*)data_ptrv;

    return data_ptr;
}

void
HierarchicalNSW::copyDataByLabel(LabelType label, void* data_point) {
    std::unique_lock lock_table(label_lookup_lock_);

    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
        throw std::runtime_error("Label not found");
    }
    InnerIdType internal_id = search->second;

    memcpy(data_point, getDataByInternalId(internal_id), data_size_);
}

/*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
void
HierarchicalNSW::markDelete(LabelType label) {
    // no need to use lock since we use global rw lock in hnsw.cpp
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end()) {
        throw std::runtime_error("Label not found");
    }
    InnerIdType internalId = search->second;
    markDeletedInternal(internalId);
    label_lookup_.erase(search);
}

void
HierarchicalNSW::recoverMarkDelete(LabelType label) {
    if (not allow_replace_deleted_) {
        return;
    }

    // lock all operations with element by label
    std::scoped_lock lock_table(label_lookup_lock_);
    auto search = deleted_elements_.find(label);
    if (search == deleted_elements_.end()) {
        throw std::runtime_error("Label not found");
    }
    InnerIdType internalId = search->second;
    recoveryMarkDeletedInternal(internalId);
    label_lookup_[label] = internalId;
}

/*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
void
HierarchicalNSW::markDeletedInternal(InnerIdType internalId) {
    assert(internalId < cur_element_count_);
    if (!isMarkedDeleted(internalId)) {
        unsigned char* ll_cur =
            (unsigned char*)data_level0_memory_->GetElementPtr(internalId, offsetLevel0_) + 2;
        *ll_cur |= DELETE_MARK;
        num_deleted_ += 1;
        if (allow_replace_deleted_) {
            std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock_);
            deleted_elements_.insert({getExternalLabel(internalId), internalId});
        }
    } else {
        throw std::runtime_error("The requested to delete element is already deleted");
    }
}

void
HierarchicalNSW::recoveryMarkDeletedInternal(InnerIdType internalId) {
    if (isMarkedDeleted(internalId)) {
        unsigned char* ll_cur =
            (unsigned char*)data_level0_memory_->GetElementPtr(internalId, offsetLevel0_) + 2;
        *ll_cur &= ~DELETE_MARK;
        num_deleted_ -= 1;
        if (allow_replace_deleted_) {
            std::scoped_lock lock_deleted_elements(deleted_elements_lock_);
            deleted_elements_.erase(getExternalLabel(internalId));
        }
    } else {
        throw std::runtime_error("The requested to delete element is not deleted");
    }
}

/*
    * Adds point.
    */
bool
HierarchicalNSW::addPoint(const void* data_point, LabelType label) {
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
            uint64_t link_size = getListCount(data);
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
        uint64_t link_size = getListCount(data);
        auto* links = (InnerIdType*)(data + 1);
        for (int i = 0; i < link_size; ++i) {
            auto& in_edges = getEdges(links[i], level);
            if (is_erase) {
                in_edges.erase(wrong_internal_id);
            } else {
                in_edges.insert(right_internal_id);
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
HierarchicalNSW::updateVector(LabelType label, const void* data_point) {
    std::unique_lock lock(label_lookup_lock_);
    auto iter = label_lookup_.find(label);
    if (iter == label_lookup_.end()) {
        throw std::runtime_error(fmt::format("no label {} in HNSW", label));
    } else {
        InnerIdType internal_id = iter->second;

        // reset data
        std::shared_ptr<float[]> normalize_data;
        normalizeVector(data_point, normalize_data);
        std::unique_lock resize_lock(resize_mutex_);
        memcpy(getDataByInternalId(internal_id), data_point, data_size_);
    }
}

void
HierarchicalNSW::updateLabel(LabelType old_label, LabelType new_label) {
    std::unique_lock lock(label_lookup_lock_);

    // 1. check whether new_label is occupied
    auto iter_new = label_lookup_.find(new_label);
    if (iter_new != label_lookup_.end()) {
        throw std::runtime_error(fmt::format("new label {} has been in HNSW", new_label));
    }

    // 2. check whether old_label exists
    InnerIdType internal_id = 0;
    auto iter_old = label_lookup_.find(old_label);
    if (iter_old == label_lookup_.end()) {
        // 3. deal the situation of mark delete
        auto iter_mark_delete = deleted_elements_.find(old_label);
        if (iter_mark_delete == deleted_elements_.end()) {
            throw std::runtime_error(fmt::format("no old label {} in HNSW", old_label));
        }

        // 4. update label to id
        internal_id = iter_mark_delete->second;
        deleted_elements_.erase(iter_mark_delete);
        deleted_elements_.insert({new_label, internal_id});
    } else {
        // 4. update label to id
        internal_id = iter_old->second;
        label_lookup_.erase(iter_old);
        label_lookup_[new_label] = internal_id;
    }

    // 5. reset id to label
    std::unique_lock resize_lock(resize_mutex_);
    setExternalLabel(internal_id, new_label);
}

void
HierarchicalNSW::removePoint(LabelType label) {
    InnerIdType cur_c = 0;
    InnerIdType internal_id = 0;
    std::unique_lock lock(max_level_mutex_);
    {
        // Swap the connection relationship corresponding to the label to be deleted with the
        // last element, and modify the information in label_lookup_. By swapping the two points,
        // fill the void left by the deletion.
        std::unique_lock lock_table(label_lookup_lock_);
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
            uint64_t m_curmax = level ? maxM_ : maxM0_;
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
    int curlevel;
    std::shared_ptr<float[]> normalize_data;
    normalizeVector(data_point, normalize_data);
    {
        // Checking if the element with the same label already exists
        // if so, updating it *instead* of creating a new element.
        std::unique_lock lock_table(label_lookup_lock_);
        auto search = label_lookup_.find(label);
        if (search != label_lookup_.end()) {
            return -1;
        }

        if (cur_element_count_ >= max_elements_) {
            uint64_t extend_size = std::min(max_elements_, data_element_per_block_);
            resizeIndex(max_elements_ + extend_size);
        }

        cur_c = cur_element_count_;
        label_lookup_[label] = cur_c;

        curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;
        element_levels_[cur_c] = curlevel;
        memset(data_level0_memory_->GetElementPtr(cur_c, offsetLevel0_), 0, size_data_per_element_);

        // Initialisation of the data and label
        setExternalLabel(cur_c, label);
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);
        cur_element_count_++;
    }

    std::shared_lock resize_lock(resize_mutex_);
    std::unique_lock lock(max_level_mutex_);
    int maxlevelcopy = max_level_;
    if (curlevel <= maxlevelcopy)
        lock.unlock();
    int64_t currObj = enterpoint_node_;
    int64_t enterpoint_copy = enterpoint_node_;

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
            std::shared_ptr<char[]> link_data =
                std::shared_ptr<char[]>(new char[size_links_level0_]);
            for (int lev = maxlevelcopy; lev > curlevel; lev--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    getLinklistAtLevel(currObj, lev, link_data.get());
                    auto* data = (unsigned int*)link_data.get();
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
                           uint64_t k,
                           uint64_t ef,
                           const vsag::FilterPtr is_id_allowed,
                           const float skip_ratio,
                           vsag::Allocator* allocator,
                           vsag::IteratorFilterContext* iter_ctx,
                           bool is_last_filter) const {
    std::shared_lock resize_lock(resize_mutex_);
    std::priority_queue<std::pair<float, LabelType>> result;
    if (cur_element_count_ == 0)
        return result;

    vsag::Allocator* search_allocator = allocator == nullptr ? allocator_ : allocator;
    std::shared_ptr<float[]> normalize_query;
    normalizeVector(query_data, normalize_query);
    MaxHeap top_candidates(search_allocator);
    if (iter_ctx != nullptr && !iter_ctx->IsFirstUsed()) {
        if (iter_ctx->Empty())
            return result;
        if (is_last_filter) {
            while (!iter_ctx->Empty()) {
                uint32_t cur_inner_id = iter_ctx->GetTopID();
                float cur_dist = iter_ctx->GetTopDist();
                result.emplace(cur_dist, getExternalLabel(cur_inner_id));
                iter_ctx->PopDiscard();
            }
            return result;
        }
        top_candidates = searchBaseLayerST<false, true>(UNUSED_ENTRY_POINT_NODE,
                                                        query_data,
                                                        std::max(ef, k),
                                                        is_id_allowed,
                                                        skip_ratio,
                                                        allocator,
                                                        iter_ctx);
    } else {
        int64_t currObj;
        int max_level_copy;
        {
            std::shared_lock data_loc(max_level_mutex_);
            currObj = enterpoint_node_;
            max_level_copy = max_level_;
        }
        if (currObj > cur_element_count_) {
            return result;
        }

        float curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
        std::shared_ptr<char[]> link_data = std::shared_ptr<char[]>(new char[size_links_level0_]);
        for (int level = max_level_copy; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                getLinklistAtLevel(currObj, level, link_data.get());
                auto* data = (unsigned int*)link_data.get();
                int size = getListCount(data);
                //            metric_hops_++;
                //            metric_distance_computations_ += size;

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

        if (num_deleted_ == 0) {
            top_candidates = searchBaseLayerST<false, true>(currObj,
                                                            query_data,
                                                            std::max(ef, k),
                                                            is_id_allowed,
                                                            skip_ratio,
                                                            allocator,
                                                            iter_ctx);
        } else {
            top_candidates = searchBaseLayerST<true, true>(currObj,
                                                           query_data,
                                                           std::max(ef, k),
                                                           is_id_allowed,
                                                           skip_ratio,
                                                           allocator,
                                                           iter_ctx);
        }
    }

    while (top_candidates.size() > k) {
        if (iter_ctx != nullptr) {
            std::pair<float, InnerIdType> curr = top_candidates.top();
            iter_ctx->AddDiscardNode(curr.first, curr.second);
        }
        top_candidates.pop();
    }
    while (not top_candidates.empty()) {
        std::pair<float, InnerIdType> rez = top_candidates.top();
        result.emplace(rez.first, getExternalLabel(rez.second));
        if (iter_ctx != nullptr) {
            iter_ctx->SetPoint(rez.second);
        }
        top_candidates.pop();
    }
    if (iter_ctx != nullptr) {
        iter_ctx->SetOFFFirstUsed();
    }
    return result;
}

std::priority_queue<std::pair<float, LabelType>>
HierarchicalNSW::searchRange(const void* query_data,
                             float radius,
                             uint64_t ef,
                             const vsag::FilterPtr is_id_allowed) const {
    std::shared_lock resize_lock(resize_mutex_);
    std::priority_queue<std::pair<float, LabelType>> result;
    if (cur_element_count_ == 0)
        return result;

    std::shared_ptr<float[]> normalize_query;
    normalizeVector(query_data, normalize_query);
    int64_t currObj;
    int max_level_copy;
    {
        std::shared_lock data_loc(max_level_mutex_);
        currObj = enterpoint_node_;
        max_level_copy = max_level_;
    }
    float curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);

    std::shared_ptr<char[]> link_data = std::shared_ptr<char[]>(new char[size_links_level0_]);
    for (int level = max_level_copy; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            getLinklistAtLevel(currObj, level, link_data.get());
            auto* data = (unsigned int*)link_data.get();
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
    if (num_deleted_ == 0) {
        top_candidates =
            searchBaseLayerST<false, true>(currObj, query_data, radius, ef, is_id_allowed);
    } else {
        top_candidates =
            searchBaseLayerST<true, true>(currObj, query_data, radius, ef, is_id_allowed);
    }

    while (not top_candidates.empty()) {
        std::pair<float, InnerIdType> rez = top_candidates.top();
        result.emplace(rez.first, getExternalLabel(rez.second));
        top_candidates.pop();
    }

    // std::cout << "hnswalg::result.size(): " << result.size() << std::endl;
    return result;
}

void
HierarchicalNSW::setDataAndGraph(vsag::FlattenInterfacePtr& data,
                                 vsag::GraphInterfacePtr& graph,
                                 vsag::Vector<LabelType>& ids) {
    resizeIndex(data->total_count_);
    std::shared_ptr<uint8_t[]> temp_vector =
        std::shared_ptr<uint8_t[]>(new uint8_t[data->code_size_]);
    for (int i = 0; i < data->total_count_; ++i) {
        data->GetCodesById(i, temp_vector.get());
        std::memcpy(getDataByInternalId(i),
                    reinterpret_cast<const char*>(temp_vector.get()),
                    data->code_size_);
        vsag::Vector<InnerIdType> edges(allocator_);
        graph->GetNeighbors(i, edges);
        setBatchNeigohbors(i, 0, edges.data(), edges.size());
        setExternalLabel(i, ids[i]);
        label_lookup_[ids[i]] = i;
        element_levels_[i] = 0;
    }
    cur_element_count_ = data->total_count_;
    enterpoint_node_ = 0;
    max_level_ = 0;
}

template MaxHeap
HierarchicalNSW::searchBaseLayerST<false, false>(
    InnerIdType ep_id,
    const void* data_point,
    uint64_t ef,
    const vsag::FilterPtr is_id_allowed,
    const float skip_ratio,
    vsag::Allocator* allocator,
    vsag::IteratorFilterContext* iter_ctx = nullptr) const;

template MaxHeap
HierarchicalNSW::searchBaseLayerST<false, false>(InnerIdType ep_id,
                                                 const void* data_point,
                                                 float radius,
                                                 int64_t ef,
                                                 const vsag::FilterPtr is_id_allowed) const;

void
HierarchicalNSW::setImmutable() {
    if (this->immutable_) {
        return;
    }
    this->points_locks_.reset();
    this->points_locks_ = std::make_shared<vsag::EmptyMutex>();
    this->immutable_ = true;
}

}  // namespace hnswlib
