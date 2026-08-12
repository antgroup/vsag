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

#include "hgraph_rabitq_searcher.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>

#include "attr/executor/executor.h"
#include "datacell/rabitq_split_datacell.h"
#include "impl/heap/standard_heap.h"
#include "impl/reasoning/search_reasoning.h"
#include "index_common_param.h"
#include "simd/rabitq_simd.h"
#include "vsag/allocator.h"

namespace vsag {
namespace {

class MaybeSharedLock {
public:
    MaybeSharedLock(const MutexArrayPtr& mutexes, InnerIdType id) : mutexes_(mutexes), id_(id) {
        if (mutexes_ != nullptr) {
            mutexes_->SharedLock(id_);
        }
    }

    ~MaybeSharedLock() {
        if (mutexes_ != nullptr) {
            mutexes_->SharedUnlock(id_);
        }
    }

private:
    // Search owns the shared mutex array for longer than every per-node guard; retaining only a
    // reference avoids shared_ptr atomic traffic in the traversal hot loop.
    const MutexArrayPtr& mutexes_;
    InnerIdType id_;
};

constexpr uint32_t K_FUSED_CLUSTER_COUNT = 16;
constexpr uint32_t K_FULL_DISTANCE_AVAILABLE = 1U << 31U;
constexpr uint32_t K_CANDIDATE_INDEX_MASK = K_FULL_DISTANCE_AVAILABLE - 1U;
constexpr uint32_t K_INVALID_CANDIDATE_INDEX = K_CANDIDATE_INDEX_MASK;

struct search_buffer_record {
    float distance;
    InnerIdType id;
    bool checked;
};

struct bounded_result_record {
    float distance;
    float filter_inner_product;
    InnerIdType id;
    uint32_t state;

    [[nodiscard]] bool
    FullDistanceAvailable() const {
        return (state & K_FULL_DISTANCE_AVAILABLE) != 0;
    }

    [[nodiscard]] uint32_t
    CandidateIndex() const {
        return state & K_CANDIDATE_INDEX_MASK;
    }
};

static_assert(sizeof(bounded_result_record) == 16);

/**
 * Fixed-capacity sorted result buffer matching RaBitQ-Library's BoundedKNN.
 *
 * Keeping this concrete avoids virtual heap operations in the per-neighbor hot loop.  The exact
 * x-bit inner product travels with a deferred candidate, so 2/3/4+y rerank never rescans x.
 */
class BoundedResults {
public:
    BoundedResults(uint64_t capacity, Allocator* allocator)
        : data_(capacity + 1, allocator), capacity_(capacity) {
    }

    void
    Insert(InnerIdType id,
           float distance,
           float filter_inner_product,
           bool full_distance_available = false,
           uint32_t candidate_index = K_INVALID_CANDIDATE_INDEX) {
        if (capacity_ == 0 or (size_ == capacity_ and distance > data_[size_ - 1].distance)) {
            return;
        }
        const auto position = binary_search(distance);
        std::memmove(data_.data() + position + 1,
                     data_.data() + position,
                     (size_ - position) * sizeof(bounded_result_record));
        data_[position] = {
            distance,
            filter_inner_product,
            id,
            candidate_index | (full_distance_available ? K_FULL_DISTANCE_AVAILABLE : 0U)};
        size_ += static_cast<uint64_t>(size_ < capacity_);
    }

    [[nodiscard]] uint64_t
    Size() const {
        return size_;
    }

    [[nodiscard]] float
    WorstDistance() const {
        return size_ == capacity_ ? data_[size_ - 1].distance : std::numeric_limits<float>::max();
    }

    [[nodiscard]] const bounded_result_record&
    operator[](uint64_t index) const {
        return data_[index];
    }

    [[nodiscard]] InnerIdType
    WorstId() const {
        return data_[size_ - 1].id;
    }

private:
    [[nodiscard]] uint64_t
    binary_search(float distance) const {
        uint64_t lo = 0;
        uint64_t length = size_;
        // Advance by a conditionally selected half to keep this hot search branchless.
        while (length > 1) {
            const auto half = length >> 1U;
            length -= half;
            lo += static_cast<uint64_t>(data_[lo + half - 1].distance < distance) * half;
        }
        return lo < size_ and data_[lo].distance < distance ? lo + 1 : lo;
    }

private:
    Vector<bounded_result_record> data_;
    uint64_t size_{0};
    uint64_t capacity_{0};
};

inline float
read_float(const uint8_t* address) {
    float value = 0.0F;
    std::memcpy(&value, address, sizeof(value));
    return value;
}

template <uint32_t filter_bits>
struct AffineFilterIP;

template <>
struct AffineFilterIP<1> {
    static float
    Approximate(const RaBitQFusedTraversalQuery& query, const uint8_t* code) {
        const uint64_t packed = RaBitQSQ4UBinaryIPWithBaseSum(query.query_planes, code, query.dim);
        const auto raw_ip = static_cast<uint32_t>(packed);
        const auto base_sum = static_cast<uint32_t>(packed >> 32U);
        return query.query_delta * static_cast<float>(raw_ip) +
               query.query_vl * static_cast<float>(base_sum) - 0.5F * query.query_sum;
    }

    static float
    Exact(const RaBitQFusedTraversalQuery& query, const uint8_t* code) {
        return 0.5F * RaBitQFloatBinaryIP(query.transformed_query, code, query.dim, 1.0F);
    }

    static void
    ApproximateBatch4(const RaBitQFusedTraversalQuery& query,
                      const uint8_t* code0,
                      const uint8_t* code1,
                      const uint8_t* code2,
                      const uint8_t* code3,
                      float* results) {
        uint64_t packed[4];
        RaBitQSQ4UBinaryIPWithBaseSumBatch4(
            query.query_planes, code0, code1, code2, code3, query.dim, packed);
        for (uint32_t i = 0; i < 4; ++i) {
            const auto raw_ip = static_cast<uint32_t>(packed[i]);
            const auto base_sum = static_cast<uint32_t>(packed[i] >> 32U);
            results[i] = query.query_delta * static_cast<float>(raw_ip) +
                         query.query_vl * static_cast<float>(base_sum) - 0.5F * query.query_sum;
        }
    }
};

template <>
struct AffineFilterIP<2> {
    static float
    Exact(const RaBitQFusedTraversalQuery& query, const uint8_t* code) {
        return RaBitQFloatTwoBitCenteredIP(query.transformed_query, code, query.dim);
    }

    static void
    ExactBatch4(const RaBitQFusedTraversalQuery& query,
                const uint8_t* code0,
                const uint8_t* code1,
                const uint8_t* code2,
                const uint8_t* code3,
                float* results) {
        RaBitQFloatTwoBitCenteredIPBatch4(
            query.transformed_query, code0, code1, code2, code3, query.dim, results);
    }
};

template <>
struct AffineFilterIP<3> {
    static float
    Exact(const RaBitQFusedTraversalQuery& query, const uint8_t* code) {
        return RaBitQFloatThreeBitCenteredIP(query.transformed_query, code, query.dim);
    }

    static void
    ExactBatch4(const RaBitQFusedTraversalQuery& query,
                const uint8_t* code0,
                const uint8_t* code1,
                const uint8_t* code2,
                const uint8_t* code3,
                float* results) {
        RaBitQFloatThreeBitCenteredIPBatch4(
            query.transformed_query, code0, code1, code2, code3, query.dim, results);
    }
};

template <>
struct AffineFilterIP<4> {
    static float
    Exact(const RaBitQFusedTraversalQuery& query, const uint8_t* code) {
        return RaBitQFloatFourBitCenteredIP(query.transformed_query, code, query.dim);
    }

    static void
    ExactBatch4(const RaBitQFusedTraversalQuery& query,
                const uint8_t* code0,
                const uint8_t* code1,
                const uint8_t* code2,
                const uint8_t* code3,
                float* results) {
        RaBitQFloatFourBitCenteredIPBatch4(
            query.transformed_query, code0, code1, code2, code3, query.dim, results);
    }
};

template <uint32_t filter_bits>
class AffineScorer {
public:
    static constexpr bool K_EXACT_FILTER_HINT = filter_bits >= 2;

    AffineScorer(const RaBitQFusedTraversalQuery& query, float runtime_error_rate) : query_(query) {
        const float error_rate =
            IsFiniteRaBitQValue(runtime_error_rate) and runtime_error_rate > 0.0F
                ? runtime_error_rate
                : query.default_rabitq_error_rate;
        const auto count = std::min<uint32_t>(query.cluster_count, K_FUSED_CLUSTER_COUNT);
        for (uint32_t i = 0; i < count; ++i) {
            scaled_cluster_g_error_[i] = error_rate * query.cluster_g_error[i];
        }
    }

    [[nodiscard]] bool
    Estimate(const HGraphRaBitQFusedDataCell::CodeView& node,
             float* distance,
             float* lower_bound,
             float* filter_inner_product) const {
        if (node.cluster_id >= query_.cluster_count or
            node.cluster_id >= scaled_cluster_g_error_.size()) {
            return false;
        }
        if constexpr (filter_bits == 1) {
            *filter_inner_product = AffineFilterIP<1>::Approximate(query_, node.one_bit_code);
        } else {
            *filter_inner_product = AffineFilterIP<filter_bits>::Exact(query_, node.one_bit_code);
        }
        const auto* metadata = node.one_bit_code + query_.one_bit_metadata_offset;
        const float filter_add = read_float(metadata);
        const float filter_rescale = read_float(metadata + sizeof(float));
        const float filter_error_unit = read_float(metadata + 2U * sizeof(float));
        *distance = filter_add + query_.cluster_g_add[node.cluster_id] +
                    filter_rescale * *filter_inner_product;
        const float raw_lower_bound =
            *distance - filter_error_unit * scaled_cluster_g_error_[node.cluster_id];
        *lower_bound = raw_lower_bound - 1e-5F * std::max(1.0F, std::fabs(raw_lower_bound));
        return IsFiniteRaBitQValue(*distance) and IsFiniteRaBitQValue(*lower_bound) and
               IsFiniteRaBitQValue(*filter_inner_product);
    }

    void
    EstimateBatch4(const HGraphRaBitQFusedDataCell::CodeView* nodes,
                   float* distances,
                   float* lower_bounds,
                   float* filter_inner_products,
                   bool* valid) const {
        if constexpr (filter_bits == 1) {
            AffineFilterIP<1>::ApproximateBatch4(query_,
                                                 nodes[0].one_bit_code,
                                                 nodes[1].one_bit_code,
                                                 nodes[2].one_bit_code,
                                                 nodes[3].one_bit_code,
                                                 filter_inner_products);
        } else {
            AffineFilterIP<filter_bits>::ExactBatch4(query_,
                                                     nodes[0].one_bit_code,
                                                     nodes[1].one_bit_code,
                                                     nodes[2].one_bit_code,
                                                     nodes[3].one_bit_code,
                                                     filter_inner_products);
        }
        for (uint32_t i = 0; i < 4; ++i) {
            valid[i] = nodes[i].cluster_id < query_.cluster_count and
                       nodes[i].cluster_id < scaled_cluster_g_error_.size();
            if (not valid[i]) {
                continue;
            }
            const auto* metadata = nodes[i].one_bit_code + query_.one_bit_metadata_offset;
            const float filter_add = read_float(metadata);
            const float filter_rescale = read_float(metadata + sizeof(float));
            const float filter_error_unit = read_float(metadata + 2U * sizeof(float));
            distances[i] = filter_add + query_.cluster_g_add[nodes[i].cluster_id] +
                           filter_rescale * filter_inner_products[i];
            const float raw_lower_bound =
                distances[i] - filter_error_unit * scaled_cluster_g_error_[nodes[i].cluster_id];
            lower_bounds[i] = raw_lower_bound - 1e-5F * std::max(1.0F, std::fabs(raw_lower_bound));
            valid[i] = IsFiniteRaBitQValue(distances[i]) and
                       IsFiniteRaBitQValue(lower_bounds[i]) and
                       IsFiniteRaBitQValue(filter_inner_products[i]);
        }
    }

    [[nodiscard]] bool
    FullWithHint(const HGraphRaBitQFusedDataCell::CodeView& node,
                 float exact_filter_inner_product,
                 float* distance) const {
        if constexpr (filter_bits == 1) {
            (void)node;
            (void)exact_filter_inner_product;
            (void)distance;
            return false;
        } else {
            return compute_full(node, exact_filter_inner_product, distance);
        }
    }

    [[nodiscard]] bool
    FullDirect(const HGraphRaBitQFusedDataCell::CodeView& node, float* distance) const {
        const float exact_filter_inner_product =
            AffineFilterIP<filter_bits>::Exact(query_, node.one_bit_code);
        return compute_full(node, exact_filter_inner_product, distance);
    }

    [[nodiscard]] bool
    HasValidFilterDistance(const HGraphRaBitQFusedDataCell::CodeView& node, float distance) const {
        return node.cluster_id < query_.cluster_count and
               node.cluster_id < scaled_cluster_g_error_.size() and
               IsFiniteRaBitQValue(distance) and distance < std::numeric_limits<float>::max();
    }

private:
    [[nodiscard]] bool
    compute_full(const HGraphRaBitQFusedDataCell::CodeView& node,
                 float exact_filter_inner_product,
                 float* distance) const {
        if (node.cluster_id >= query_.cluster_count or
            not IsFiniteRaBitQValue(exact_filter_inner_product)) {
            return false;
        }
        const float supplement_ip = RaBitQFloatSupplementCodeIP(
            query_.transformed_query, node.supplement_code, query_.dim, query_.supplement_bits);
        const auto* metadata = node.supplement_code + query_.supplement_metadata_offset;
        const float full_add = read_float(metadata);
        const float full_rescale = read_float(metadata + sizeof(float));
        const float supplement_center =
            0.5F * static_cast<float>((1U << query_.supplement_bits) - 1U);
        const float full_inner_product =
            static_cast<float>(1U << query_.supplement_bits) * exact_filter_inner_product +
            supplement_ip - supplement_center * query_.query_sum;
        *distance =
            full_add + query_.cluster_g_add[node.cluster_id] + full_rescale * full_inner_product;
        return IsFiniteRaBitQValue(*distance);
    }

private:
    const RaBitQFusedTraversalQuery& query_;
    std::array<float, K_FUSED_CLUSTER_COUNT> scaled_cluster_g_error_{};
};

class LegacyOneBitScorer {
public:
    static constexpr bool K_EXACT_FILTER_HINT = false;

    LegacyOneBitScorer(const RaBitQFusedTraversalQuery& query, float runtime_error_rate)
        : query_(query) {
        const float error_rate =
            IsFiniteRaBitQValue(runtime_error_rate) and runtime_error_rate > 0.0F
                ? runtime_error_rate
                : query.default_rabitq_error_rate;
        const float error_rate_scale =
            error_rate / RaBitQuantizerParameter::DEFAULT_RABITQ_ERROR_RATE;
        const auto count = std::min<uint32_t>(query.cluster_count, K_FUSED_CLUSTER_COUNT);
        for (uint32_t i = 0; i < count; ++i) {
            scaled_cluster_g_error_[i] = error_rate_scale * query.cluster_g_error[i];
        }
    }

    [[nodiscard]] bool
    Estimate(const HGraphRaBitQFusedDataCell::CodeView& node,
             float* distance,
             float* lower_bound,
             float* filter_inner_product) const {
        if (node.cluster_id >= query_.cluster_count or
            node.cluster_id >= scaled_cluster_g_error_.size()) {
            return false;
        }
        const uint64_t packed =
            RaBitQSQ4UBinaryIPWithBaseSum(query_.query_planes, node.one_bit_code, query_.dim);
        const auto raw_ip = static_cast<uint32_t>(packed);
        const auto base_sum = static_cast<uint32_t>(packed >> 32U);
        const float uncentered_ip = query_.query_delta * static_cast<float>(raw_ip) +
                                    query_.query_vl * static_cast<float>(base_sum);
        *filter_inner_product = uncentered_ip - 0.5F * query_.query_sum;
        const auto* metadata = node.one_bit_code + query_.one_bit_metadata_offset;
        const float filter_add = read_float(metadata);
        const float filter_rescale = read_float(metadata + sizeof(float));
        const float filter_error = read_float(metadata + 2U * sizeof(float));
        *distance = filter_add + query_.cluster_g_add[node.cluster_id] +
                    filter_rescale * *filter_inner_product;
        *lower_bound = *distance - filter_error * scaled_cluster_g_error_[node.cluster_id];
        return IsFiniteRaBitQValue(*distance) and IsFiniteRaBitQValue(*lower_bound) and
               IsFiniteRaBitQValue(*filter_inner_product);
    }

    void
    EstimateBatch4(const HGraphRaBitQFusedDataCell::CodeView* nodes,
                   float* distances,
                   float* lower_bounds,
                   float* filter_inner_products,
                   bool* valid) const {
        AffineFilterIP<1>::ApproximateBatch4(query_,
                                             nodes[0].one_bit_code,
                                             nodes[1].one_bit_code,
                                             nodes[2].one_bit_code,
                                             nodes[3].one_bit_code,
                                             filter_inner_products);
        for (uint32_t i = 0; i < 4; ++i) {
            valid[i] = nodes[i].cluster_id < query_.cluster_count and
                       nodes[i].cluster_id < scaled_cluster_g_error_.size();
            if (not valid[i]) {
                continue;
            }
            const auto* metadata = nodes[i].one_bit_code + query_.one_bit_metadata_offset;
            const float filter_add = read_float(metadata);
            const float filter_rescale = read_float(metadata + sizeof(float));
            const float filter_error = read_float(metadata + 2U * sizeof(float));
            distances[i] = filter_add + query_.cluster_g_add[nodes[i].cluster_id] +
                           filter_rescale * filter_inner_products[i];
            lower_bounds[i] =
                distances[i] - filter_error * scaled_cluster_g_error_[nodes[i].cluster_id];
            valid[i] = IsFiniteRaBitQValue(distances[i]) and
                       IsFiniteRaBitQValue(lower_bounds[i]) and
                       IsFiniteRaBitQValue(filter_inner_products[i]);
        }
    }

    [[nodiscard]] static bool
    FullWithHint(const HGraphRaBitQFusedDataCell::CodeView& /*node*/,
                 float /*filter_inner_product*/,
                 float* /*distance*/) {
        return false;
    }

    [[nodiscard]] bool
    FullDirect(const HGraphRaBitQFusedDataCell::CodeView& node, float* distance) const {
        if (node.cluster_id >= query_.cluster_count) {
            return false;
        }
        const float exact_centered_ip = AffineFilterIP<1>::Exact(query_, node.one_bit_code);
        const float supplement_ip =
            query_.supplement_bits == 7 and (query_.dim & 63U) == 0U
                ? RaBitQFloatExCode7IP(query_.transformed_query, node.supplement_code, query_.dim)
                : RaBitQFloatSupplementCodeIP(query_.transformed_query,
                                              node.supplement_code,
                                              query_.dim,
                                              query_.supplement_bits);
        const auto* metadata = node.supplement_code + query_.supplement_metadata_offset;
        const float full_add = read_float(metadata);
        const float full_rescale = read_float(metadata + sizeof(float));
        const auto supplement_scale = static_cast<float>(1U << query_.supplement_bits);
        const float supplement_center = 0.5F * (supplement_scale - 1.0F);
        *distance = full_add + query_.cluster_g_add[node.cluster_id] +
                    full_rescale * (supplement_scale * exact_centered_ip + supplement_ip -
                                    supplement_center * query_.query_sum);
        return IsFiniteRaBitQValue(*distance);
    }

    [[nodiscard]] bool
    HasValidFilterDistance(const HGraphRaBitQFusedDataCell::CodeView& node, float distance) const {
        return node.cluster_id < query_.cluster_count and
               node.cluster_id < scaled_cluster_g_error_.size() and
               IsFiniteRaBitQValue(distance) and distance < std::numeric_limits<float>::max();
    }

private:
    const RaBitQFusedTraversalQuery& query_;
    std::array<float, K_FUSED_CLUSTER_COUNT> scaled_cluster_g_error_{};
};

/**
 * Sorted linear beam buffer ported from RaBitQ-Library's SearchBuffer
 * (Apache-2.0). VSAG keeps the checked flag separate because InnerIdType is
 * also used by remove-version encoding.
 */
class SearchBuffer {
public:
    SearchBuffer(uint64_t capacity, Allocator* allocator)
        : data_(capacity + 1, allocator), capacity_(capacity) {
    }

    void
    Insert(InnerIdType id, float distance) {
        if (IsFull(distance)) {
            return;
        }
        const auto pos = binary_search(distance);
        std::memmove(data_.data() + pos + 1,
                     data_.data() + pos,
                     (size_ - pos) * sizeof(search_buffer_record));
        data_[pos] = {distance, id, false};
        size_ += static_cast<uint64_t>(size_ < capacity_);
        current_ = std::min(current_, pos);
    }

    [[nodiscard]] bool
    HasNext() const {
        return current_ < size_;
    }

    InnerIdType
    Pop() {
        const auto id = data_[current_].id;
        data_[current_].checked = true;
        ++current_;
        while (current_ < size_ and data_[current_].checked) {
            ++current_;
        }
        return id;
    }

    [[nodiscard]] InnerIdType
    NextId() const {
        return data_[current_].id;
    }

    [[nodiscard]] bool
    IsFull(float distance) const {
        return size_ == capacity_ and distance > data_[size_ - 1].distance;
    }

private:
    [[nodiscard]] uint64_t
    binary_search(float distance) const {
        uint64_t lo = 0;
        uint64_t length = size_;
        // Advance by a conditionally selected half to keep this hot search branchless.
        while (length > 1) {
            const auto half = length >> 1U;
            length -= half;
            lo += static_cast<uint64_t>(data_[lo + half - 1].distance < distance) * half;
        }
        return lo < size_ and data_[lo].distance < distance ? lo + 1 : lo;
    }

private:
    Vector<search_buffer_record> data_;
    uint64_t size_{0};
    uint64_t current_{0};
    uint64_t capacity_{0};
};

template <typename Scorer>
DistHeapPtr
search_direct_fused(const HGraphRaBitQFusedDataCellPtr& graph,
                    const VisitedListPtr& visited_list,
                    const InnerSearchParam& search_param,
                    QueryContext* ctx,
                    RaBitQCandidateVector* lower_bound_candidates,
                    Allocator* allocator,
                    const MutexArrayPtr& neighbors_mutex,
                    const Scorer& scorer) {
    if (search_param.ef == 0 or not graph->CheckIdExists(search_param.ep)) {
        return std::make_shared<StandardHeap<true, false>>(allocator, -1);
    }
    if (lower_bound_candidates != nullptr) {
        lower_bound_candidates->clear();
        const uint64_t reserve_hint =
            (static_cast<uint64_t>(search_param.ef) + 4U) * graph->MaximumDegree() + 1U;
        lower_bound_candidates->reserve(std::min<uint64_t>(graph->TotalCount(), reserve_hint));
    }

    const uint64_t rerank_topk = static_cast<uint64_t>(std::max<int64_t>(
        1, search_param.rerank_topk > 0 ? search_param.rerank_topk : search_param.topk));
    const bool range_search = search_param.search_mode == RANGE_SEARCH;
    const uint64_t result_limit =
        range_search ? (search_param.range_search_limit_size > 0
                            ? static_cast<uint64_t>(search_param.range_search_limit_size)
                            : static_cast<uint64_t>(graph->Capacity()))
                     : rerank_topk;
    const bool should_rerank =
        search_param.enable_rabitq_one_bit_search and search_param.enable_reorder;
    const bool deferred_rerank =
        not range_search and HGraphRaBitQSearcher::ShouldDeferRerank(search_param);
    const uint64_t deferred_rerank_count = std::max<uint64_t>(search_param.ef, 2 * rerank_topk);
    BoundedResults results(deferred_rerank ? deferred_rerank_count : result_limit, allocator);
    SearchBuffer candidate_set(search_param.ef, allocator);

    uint32_t rabitq_filter_count = 0;
    uint32_t rabitq_full_count = 0;
    uint32_t rabitq_filter_fallback_full_count = 0;
    uint32_t rabitq_hint_full_count = 0;
    uint32_t rabitq_reorder_fallback_full_count = 0;
    uint32_t deferred_finalize_full_count = 0;
    Filter* attribute_filter = nullptr;
    if (not search_param.executors.empty() and search_param.executors[0] != nullptr) {
        search_param.executors[0]->Clear();
        attribute_filter = search_param.executors[0]->Run();
    }
    const auto is_allowed = [&search_param, attribute_filter](InnerIdType id) {
        return (search_param.is_inner_id_allowed == nullptr or
                search_param.is_inner_id_allowed->CheckValid(id)) and
               (attribute_filter == nullptr or attribute_filter->CheckValid(id));
    };

    const auto score_node = [&](const HGraphRaBitQFusedDataCell::CodeView& node,
                                float* distance,
                                float* lower_bound,
                                float* filter_inner_product,
                                bool* full_distance_available) {
        *full_distance_available = false;
        if (search_param.enable_rabitq_one_bit_search) {
            ++rabitq_filter_count;
            if (scorer.Estimate(node, distance, lower_bound, filter_inner_product)) {
                return true;
            }
            if (not should_rerank) {
                if (scorer.HasValidFilterDistance(node, *distance)) {
                    *lower_bound = *distance;
                    return true;
                }
            }
            ++rabitq_filter_fallback_full_count;
        }
        ++rabitq_full_count;
        *filter_inner_product = std::numeric_limits<float>::quiet_NaN();
        *full_distance_available = scorer.FullDirect(node, distance);
        if (not *full_distance_available) {
            return false;
        }
        *lower_bound = *distance;
        return true;
    };
    const auto refine_node = [&](const HGraphRaBitQFusedDataCell::CodeView& node,
                                 float filter_inner_product,
                                 float* distance) {
        ++rabitq_full_count;
        if constexpr (Scorer::K_EXACT_FILTER_HINT) {
            if (IsFiniteRaBitQValue(filter_inner_product)) {
                if (scorer.FullWithHint(node, filter_inner_product, distance)) {
                    ++rabitq_hint_full_count;
                    return true;
                }
            }
        }
        ++rabitq_reorder_fallback_full_count;
        return scorer.FullDirect(node, distance);
    };
    const auto record_lower_bound = [&](InnerIdType id,
                                        float lower_bound,
                                        float filter_inner_product,
                                        bool allowed,
                                        float full_distance) {
        if (lower_bound_candidates == nullptr or not allowed) {
            return K_INVALID_CANDIDATE_INDEX;
        }
        const auto candidate_index = lower_bound_candidates->size();
        if (candidate_index >= K_INVALID_CANDIDATE_INDEX) {
            return K_INVALID_CANDIDATE_INDEX;
        }
        lower_bound_candidates->push_back({lower_bound,
                                           Scorer::K_EXACT_FILTER_HINT
                                               ? filter_inner_product
                                               : std::numeric_limits<float>::quiet_NaN(),
                                           id,
                                           full_distance});
        return static_cast<uint32_t>(candidate_index);
    };
    const auto result_eligible = [&](float distance) {
        return not range_search or distance <= search_param.radius;
    };
    const auto insert_duplicates = [&](InnerIdType id) {
        if (not search_param.consider_duplicate) {
            return;
        }
        const auto group_id = graph->GetGroupId(id);
        auto duplicate_ids = graph->GetDuplicateIds(group_id);
        duplicate_ids.push_back(group_id);
        for (const auto duplicate : duplicate_ids) {
            if (duplicate == id) {
                continue;
            }
            if (not is_allowed(duplicate)) {
                continue;
            }
            float duplicate_distance = 0.0F;
            ++rabitq_full_count;
            if (not scorer.FullDirect(graph->GetCodeView(duplicate), &duplicate_distance) or
                not result_eligible(duplicate_distance)) {
                continue;
            }
            const auto candidate_index = record_lower_bound(duplicate,
                                                            duplicate_distance,
                                                            std::numeric_limits<float>::quiet_NaN(),
                                                            true,
                                                            duplicate_distance);
            results.Insert(duplicate,
                           duplicate_distance,
                           std::numeric_limits<float>::quiet_NaN(),
                           true,
                           candidate_index);
        }
    };

    const auto entry_node = graph->GetCodeView(search_param.ep);
    float entry_distance = 0.0F;
    float entry_lower_bound = 0.0F;
    float entry_filter_ip = std::numeric_limits<float>::quiet_NaN();
    bool entry_full_distance_available = false;
    if (not score_node(entry_node,
                       &entry_distance,
                       &entry_lower_bound,
                       &entry_filter_ip,
                       &entry_full_distance_available)) {
        return std::make_shared<StandardHeap<true, false>>(allocator, -1);
    }
    const bool entry_allowed = is_allowed(search_param.ep);
    // RaBitQ-Library inserts the bottom-layer entry point with its full distance.
    if (should_rerank and not entry_full_distance_available) {
        if (not refine_node(entry_node, entry_filter_ip, &entry_distance)) {
            return std::make_shared<StandardHeap<true, false>>(allocator, -1);
        }
        entry_full_distance_available = true;
    }
    const auto entry_candidate_index = record_lower_bound(
        search_param.ep,
        entry_lower_bound,
        entry_filter_ip,
        entry_allowed,
        entry_full_distance_available ? entry_distance : std::numeric_limits<float>::quiet_NaN());
    candidate_set.Insert(search_param.ep, entry_distance);
    visited_list->Set(search_param.ep);
    if (entry_allowed and result_eligible(entry_distance)) {
        results.Insert(
            search_param.ep,
            entry_distance,
            Scorer::K_EXACT_FILTER_HINT ? entry_filter_ip : std::numeric_limits<float>::quiet_NaN(),
            entry_full_distance_available,
            entry_candidate_index);
    }
    insert_duplicates(search_param.ep);

    uint32_t hops = 0;
    uint32_t distance_computations = 1;
    auto* reasoning = ctx == nullptr ? nullptr : ctx->reasoning_ctx;
    InnerIdType last_prefetched_candidate = std::numeric_limits<InnerIdType>::max();
    const auto prefetch_next_candidate = [&]() {
        if (not candidate_set.HasNext()) {
            return;
        }
        const auto next_id = candidate_set.NextId();
        if (next_id != last_prefetched_candidate) {
            graph->PrefetchNodeHeader(next_id);
            last_prefetched_candidate = next_id;
        }
    };
    const auto process_scored_node = [&](InnerIdType neighbor,
                                         const HGraphRaBitQFusedDataCell::CodeView& node,
                                         float distance,
                                         float lower_bound,
                                         float filter_inner_product,
                                         bool supplement_prefetched,
                                         bool full_distance_available) {
        ++distance_computations;
        if (reasoning != nullptr) {
            reasoning->RecordVisit(neighbor, distance, hops);
        }

        const bool allowed = is_allowed(neighbor);
        if (deferred_rerank) {
            const auto candidate_index = record_lower_bound(
                neighbor,
                lower_bound,
                filter_inner_product,
                allowed,
                full_distance_available ? distance : std::numeric_limits<float>::quiet_NaN());
            if (not candidate_set.IsFull(distance)) {
                candidate_set.Insert(neighbor, distance);
            }
            if (allowed) {
                if (result_eligible(distance)) {
                    results.Insert(neighbor,
                                   distance,
                                   filter_inner_product,
                                   full_distance_available,
                                   candidate_index);
                }
            } else if (reasoning != nullptr) {
                reasoning->RecordFilterReject(neighbor);
            }
            insert_duplicates(neighbor);
            prefetch_next_candidate();
            return;
        }

        const bool lower_bound_promising =
            results.Size() < rerank_topk or
            (should_rerank ? lower_bound : distance) < results.WorstDistance();
        const bool promising =
            results.Size() < rerank_topk or (should_rerank ? lower_bound < results.WorstDistance()
                                                           : distance < results.WorstDistance());
        if (promising and should_rerank and not full_distance_available) {
            if (not supplement_prefetched) {
                graph->PrefetchFusedSupplement(neighbor);
            }
            if (not refine_node(node, filter_inner_product, &distance)) {
                return;
            }
            full_distance_available = true;
        }
        if (lower_bound_promising) {
            record_lower_bound(
                neighbor,
                lower_bound,
                filter_inner_product,
                allowed,
                full_distance_available ? distance : std::numeric_limits<float>::quiet_NaN());
        }
        if (not candidate_set.IsFull(distance)) {
            candidate_set.Insert(neighbor, distance);
        }
        if (promising and allowed and result_eligible(distance)) {
            results.Insert(neighbor, distance, filter_inner_product, full_distance_available);
        } else if (not allowed and reasoning != nullptr) {
            reasoning->RecordFilterReject(neighbor);
        }
        if (promising) {
            insert_duplicates(neighbor);
        }
        prefetch_next_candidate();
    };

    while (candidate_set.HasNext()) {
        ++hops;
        if (hops >= search_param.hops_limit) {
            if (reasoning != nullptr) {
                reasoning->SetTermination(ReasoningContext::kTerminationHopsLimitReached);
            }
            break;
        }
        if (search_param.time_cost != nullptr and search_param.time_cost->CheckOvertime()) {
            if (ctx != nullptr and ctx->stats != nullptr) {
                ctx->stats->is_timeout.store(true, std::memory_order_relaxed);
            }
            if (reasoning != nullptr) {
                reasoning->SetTermination(ReasoningContext::kTerminationTimeout);
            }
            break;
        }
        if (reasoning != nullptr) {
            reasoning->AddSearchHop();
        }

        const auto current_id = candidate_set.Pop();
        MaybeSharedLock node_lock(neighbors_mutex, current_id);
        const auto current_node = graph->GetNodeView(current_id);
        const auto neighbor_count = current_node.neighbor_count;
        CHECK_ARGUMENT(neighbor_count <= graph->MaximumDegree(),
                       "corrupt fused graph neighbor count");
        const auto* neighbors = current_node.neighbors;
        uint32_t cursor = 0;
        while (cursor < neighbor_count) {
            InnerIdType batch_ids[4]{};
            uint32_t batch_count = 0;
            while (cursor < neighbor_count and batch_count < 4) {
                InnerIdType neighbor = 0;
                const auto stored_neighbor = neighbors[cursor++];
                if (not graph->ResolveNeighbor(stored_neighbor, neighbor) or
                    visited_list->TestAndSet(neighbor)) {
                    continue;
                }
                batch_ids[batch_count] = neighbor;
                graph->PrefetchFusedFilter(neighbor);
                ++batch_count;
            }
            if (batch_count == 0) {
                continue;
            }
            HGraphRaBitQFusedDataCell::CodeView batch_codes[4]{};
            for (uint32_t lane = 0; lane < batch_count; ++lane) {
                batch_codes[lane] = graph->GetCodeView(batch_ids[lane]);
            }

            float distances[4]{};
            float lower_bounds[4]{};
            float filter_inner_products[4]{std::numeric_limits<float>::quiet_NaN(),
                                           std::numeric_limits<float>::quiet_NaN(),
                                           std::numeric_limits<float>::quiet_NaN(),
                                           std::numeric_limits<float>::quiet_NaN()};
            bool valid[4]{};
            bool full_distance_available[4]{};
            if (batch_count == 4 and search_param.enable_rabitq_one_bit_search) {
                rabitq_filter_count += 4;
                scorer.EstimateBatch4(
                    batch_codes, distances, lower_bounds, filter_inner_products, valid);
                for (uint32_t lane = 0; lane < 4; ++lane) {
                    if (not valid[lane]) {
                        if (not should_rerank) {
                            if (scorer.HasValidFilterDistance(batch_codes[lane], distances[lane])) {
                                valid[lane] = true;
                                lower_bounds[lane] = distances[lane];
                                continue;
                            }
                        }
                        ++rabitq_filter_fallback_full_count;
                        ++rabitq_full_count;
                        filter_inner_products[lane] = std::numeric_limits<float>::quiet_NaN();
                        valid[lane] = scorer.FullDirect(batch_codes[lane], distances + lane);
                        full_distance_available[lane] = valid[lane];
                        lower_bounds[lane] = distances[lane];
                    }
                }
            } else {
                for (uint32_t lane = 0; lane < batch_count; ++lane) {
                    valid[lane] = score_node(batch_codes[lane],
                                             distances + lane,
                                             lower_bounds + lane,
                                             filter_inner_products + lane,
                                             full_distance_available + lane);
                }
            }

            bool supplement_prefetched[4]{};
            if (not deferred_rerank and should_rerank) {
                const bool result_not_full = results.Size() < rerank_topk;
                const float bound_snapshot = results.WorstDistance();
                for (uint32_t lane = 0; lane < batch_count; ++lane) {
                    if (valid[lane] and not full_distance_available[lane] and
                        (result_not_full or lower_bounds[lane] < bound_snapshot)) {
                        graph->PrefetchFusedSupplement(batch_ids[lane]);
                        supplement_prefetched[lane] = true;
                    }
                }
            }
            for (uint32_t lane = 0; lane < batch_count; ++lane) {
                if (valid[lane]) {
                    process_scored_node(batch_ids[lane],
                                        batch_codes[lane],
                                        distances[lane],
                                        lower_bounds[lane],
                                        filter_inner_products[lane],
                                        supplement_prefetched[lane],
                                        full_distance_available[lane]);
                }
            }
        }
    }

    if (deferred_rerank) {
        BoundedResults refined(rerank_topk, allocator);
        const auto has_valid_full_distance = [](float distance) {
            return IsFiniteRaBitQValue(distance) and distance < std::numeric_limits<float>::max();
        };
        constexpr uint32_t k_deferred_prefetch = 4;
        const auto initial_prefetch = std::min<uint64_t>(k_deferred_prefetch, results.Size());
        for (uint64_t i = 0; i < initial_prefetch; ++i) {
            if (not results[i].FullDistanceAvailable()) {
                graph->PrefetchFusedSupplement(results[i].id);
            }
        }
        for (uint64_t i = 0; i < results.Size(); ++i) {
            const auto lookahead = i + k_deferred_prefetch;
            if (lookahead < results.Size() and not results[lookahead].FullDistanceAvailable()) {
                graph->PrefetchFusedSupplement(results[lookahead].id);
            }
            const auto& candidate = results[i];
            float full_distance = candidate.distance;
            bool full_distance_available = candidate.FullDistanceAvailable();
            if (not full_distance_available) {
                const auto node = graph->GetCodeView(candidate.id);
                full_distance_available =
                    refine_node(node, candidate.filter_inner_product, &full_distance);
            }
            if (candidate.CandidateIndex() <
                (lower_bound_candidates == nullptr ? 0 : lower_bound_candidates->size())) {
                (*lower_bound_candidates)[candidate.CandidateIndex()].full_distance =
                    full_distance_available ? full_distance : std::numeric_limits<float>::max();
            } else if (full_distance_available) {
                refined.Insert(candidate.id,
                               full_distance,
                               Scorer::K_EXACT_FILTER_HINT
                                   ? candidate.filter_inner_product
                                   : std::numeric_limits<float>::quiet_NaN(),
                               true);
            }
        }

        if (lower_bound_candidates != nullptr) {
            const auto insert_refined = [&](const RaBitQCandidateRecord& candidate,
                                            float full_distance) {
                if (reasoning != nullptr) {
                    reasoning->RecordReorder(candidate.id, candidate.lower_bound, full_distance);
                }
                const bool will_evict =
                    refined.Size() == rerank_topk and full_distance <= refined.WorstDistance();
                const auto evicted_id = will_evict ? refined.WorstId() : 0;
                refined.Insert(candidate.id, full_distance, candidate.filter_inner_product, true);
                if (will_evict and reasoning != nullptr) {
                    reasoning->RecordReorderEviction(evicted_id, 0);
                }
            };
            // Establish an exact kth threshold from every full distance already computed for the
            // shortlist or by a filter fallback before considering any missing supplement.
            for (const auto& candidate : *lower_bound_candidates) {
                if (has_valid_full_distance(candidate.full_distance)) {
                    insert_refined(candidate, candidate.full_distance);
                }
            }
            for (auto& candidate : *lower_bound_candidates) {
                if (has_valid_full_distance(candidate.full_distance) or
                    candidate.full_distance == std::numeric_limits<float>::max()) {
                    continue;
                }
                const bool promising = refined.Size() < rerank_topk or
                                       not IsFiniteRaBitQValue(candidate.lower_bound) or
                                       candidate.lower_bound < refined.WorstDistance();
                if (not promising) {
                    continue;
                }
                graph->PrefetchFusedSupplement(candidate.id);
                const auto node = graph->GetCodeView(candidate.id);
                float full_distance = 0.0F;
                ++deferred_finalize_full_count;
                if (not refine_node(node, candidate.filter_inner_product, &full_distance)) {
                    candidate.full_distance = std::numeric_limits<float>::max();
                    continue;
                }
                candidate.full_distance = full_distance;
                insert_refined(candidate, full_distance);
            }
        }
        results = std::move(refined);
    }

    auto output = std::make_shared<StandardHeap<true, false>>(allocator, -1);
    for (uint64_t i = 0; i < results.Size(); ++i) {
        output->Push(results[i].distance, results[i].id);
    }
    if (ctx != nullptr and ctx->stats != nullptr) {
        ctx->stats->dist_cmp.fetch_add(distance_computations, std::memory_order_relaxed);
        ctx->stats->hops.fetch_add(hops, std::memory_order_relaxed);
        ctx->stats->rabitq_filter_count.fetch_add(rabitq_filter_count, std::memory_order_relaxed);
        ctx->stats->rabitq_full_count.fetch_add(rabitq_full_count, std::memory_order_relaxed);
        ctx->stats->rabitq_filter_fallback_full_count.fetch_add(rabitq_filter_fallback_full_count,
                                                                std::memory_order_relaxed);
        ctx->stats->rabitq_reorder_hint_full_count.fetch_add(rabitq_hint_full_count,
                                                             std::memory_order_relaxed);
        ctx->stats->rabitq_reorder_fallback_full_count.fetch_add(rabitq_reorder_fallback_full_count,
                                                                 std::memory_order_relaxed);
        ctx->stats->reorder_distance_count.fetch_add(deferred_finalize_full_count,
                                                     std::memory_order_relaxed);
        if (ctx->track_distance_evaluations and deferred_finalize_full_count > 0) {
            ctx->stats->AddDistance(DistanceEvaluationPhase::RERANK,
                                    DistanceEvaluationBackend::RABITQ,
                                    deferred_finalize_full_count);
        }
    }
    return output;
}

template <typename Scorer>
InnerIdType
route_direct_fused(const GraphInterfacePtr& route_graph,
                   const HGraphRaBitQFusedDataCellPtr& fused_graph,
                   InnerIdType entry_point,
                   Allocator* allocator,
                   const MutexArrayPtr& neighbors_mutex,
                   const Scorer& scorer) {
    const auto score = [&fused_graph, &scorer](InnerIdType id, float* distance) {
        const auto node = fused_graph->GetCodeView(id);
        float lower_bound = 0.0F;
        float filter_inner_product = 0.0F;
        if (scorer.Estimate(node, distance, &lower_bound, &filter_inner_product) or
            scorer.HasValidFilterDistance(node, *distance)) {
            return true;
        }
        return scorer.FullDirect(node, distance);
    };
    float current_distance = 0.0F;
    if (not score(entry_point, &current_distance)) {
        return entry_point;
    }
    Vector<InnerIdType> neighbors(allocator);
    bool changed = true;
    while (changed) {
        changed = false;
        {
            MaybeSharedLock node_lock(neighbors_mutex, entry_point);
            route_graph->GetNeighbors(entry_point, neighbors);
        }
        for (const auto neighbor : neighbors) {
            float distance = 0.0F;
            if (score(neighbor, &distance) and distance < current_distance) {
                current_distance = distance;
                entry_point = neighbor;
                changed = true;
            }
        }
    }
    return entry_point;
}

}  // namespace

HGraphRaBitQSearcher::HGraphRaBitQSearcher(const IndexCommonParam& common_param,
                                           MutexArrayPtr neighbors_mutex)
    : allocator_(common_param.allocator_.get()), neighbors_mutex_(std::move(neighbors_mutex)) {
}

InnerIdType
HGraphRaBitQSearcher::Route(const GraphInterfacePtr& route_graph,
                            const HGraphRaBitQFusedDataCellPtr& fused_graph,
                            const FlattenInterfacePtr& flatten,
                            const ComputerInterfacePtr& computer,
                            InnerIdType entry_point,
                            bool enable_one_bit_search) const {
    auto* split_codes = dynamic_cast<RaBitQSplitDataCellInterface*>(flatten.get());
    if (route_graph == nullptr or fused_graph == nullptr or split_codes == nullptr or
        computer == nullptr or not fused_graph->CheckIdExists(entry_point)) {
        return entry_point;
    }
    RaBitQFusedTraversalQuery traversal_query;
    if (enable_one_bit_search and split_codes->GetFusedTraversalQuery(computer, &traversal_query)) {
        if (not traversal_query.affine) {
            LegacyOneBitScorer scorer(traversal_query, std::numeric_limits<float>::quiet_NaN());
            return route_direct_fused(
                route_graph, fused_graph, entry_point, allocator_, neighbors_mutex_, scorer);
        }
        const float no_runtime_override = std::numeric_limits<float>::quiet_NaN();
        switch (traversal_query.filter_bits) {
            case 1: {
                AffineScorer<1> scorer(traversal_query, no_runtime_override);
                return route_direct_fused(
                    route_graph, fused_graph, entry_point, allocator_, neighbors_mutex_, scorer);
            }
            case 2: {
                AffineScorer<2> scorer(traversal_query, no_runtime_override);
                return route_direct_fused(
                    route_graph, fused_graph, entry_point, allocator_, neighbors_mutex_, scorer);
            }
            case 3: {
                AffineScorer<3> scorer(traversal_query, no_runtime_override);
                return route_direct_fused(
                    route_graph, fused_graph, entry_point, allocator_, neighbors_mutex_, scorer);
            }
            case 4: {
                AffineScorer<4> scorer(traversal_query, no_runtime_override);
                return route_direct_fused(
                    route_graph, fused_graph, entry_point, allocator_, neighbors_mutex_, scorer);
            }
            default:
                break;
        }
    }
    const auto score = [&](InnerIdType id, float* distance) {
        const auto node = fused_graph->GetNodeView(id);
        *distance = std::numeric_limits<float>::max();
        if (not enable_one_bit_search) {
            return split_codes->ComputeFusedFull(computer,
                                                 node.cluster_id,
                                                 node.one_bit_code,
                                                 node.supplement_code,
                                                 distance,
                                                 nullptr);
        }
        float lower_bound = 0.0F;
        float filter_inner_product = 0.0F;
        if (split_codes->ComputeFusedOneBitWithFilterIP(computer,
                                                        node.cluster_id,
                                                        node.one_bit_code,
                                                        node.supplement_code,
                                                        distance,
                                                        &lower_bound,
                                                        &filter_inner_product,
                                                        nullptr) or
            (IsFiniteRaBitQValue(*distance) and *distance < std::numeric_limits<float>::max())) {
            return true;
        }
        return split_codes->ComputeFusedFull(
            computer, node.cluster_id, node.one_bit_code, node.supplement_code, distance, nullptr);
    };

    float current_distance = 0.0F;
    if (not score(entry_point, &current_distance)) {
        return entry_point;
    }
    Vector<InnerIdType> neighbors(allocator_);
    bool changed = true;
    while (changed) {
        changed = false;
        {
            MaybeSharedLock node_lock(neighbors_mutex_, entry_point);
            route_graph->GetNeighbors(entry_point, neighbors);
        }
        for (const auto neighbor : neighbors) {
            float distance = 0.0F;
            if (score(neighbor, &distance) and distance < current_distance) {
                current_distance = distance;
                entry_point = neighbor;
                changed = true;
            }
        }
    }
    return entry_point;
}

DistHeapPtr
HGraphRaBitQSearcher::Search(const HGraphRaBitQFusedDataCellPtr& graph,
                             const FlattenInterfacePtr& flatten,
                             const VisitedListPtr& visited_list,
                             const void* query,
                             const InnerSearchParam& search_param,
                             QueryContext* ctx,
                             RaBitQCandidateVector* lower_bound_candidates,
                             bool* search_finalized) const {
    if (search_finalized != nullptr) {
        *search_finalized = false;
    }
    auto* allocator = select_query_allocator(ctx, allocator_);
    auto* split_codes = dynamic_cast<RaBitQSplitDataCellInterface*>(flatten.get());
    if (graph == nullptr or split_codes == nullptr or search_param.find_duplicate) {
        return nullptr;
    }
    if (search_param.ef == 0 or not graph->CheckIdExists(search_param.ep)) {
        return std::make_shared<StandardHeap<true, false>>(allocator, -1);
    }
    if (query == nullptr) {
        return nullptr;
    }
    if (lower_bound_candidates != nullptr) {
        lower_bound_candidates->clear();
    }
    const uint64_t rerank_topk = static_cast<uint64_t>(std::max<int64_t>(
        1, search_param.rerank_topk > 0 ? search_param.rerank_topk : search_param.topk));
    const bool range_search = search_param.search_mode == RANGE_SEARCH;
    const uint64_t result_limit =
        range_search ? (search_param.range_search_limit_size > 0
                            ? static_cast<uint64_t>(search_param.range_search_limit_size)
                            : static_cast<uint64_t>(graph->Capacity()))
                     : rerank_topk;
    const bool should_rerank =
        search_param.enable_rabitq_one_bit_search and search_param.enable_reorder;
    const bool deferred_rerank =
        not range_search and HGraphRaBitQSearcher::ShouldDeferRerank(search_param);
    const bool exact_filter_ip_hint = split_codes->FusedFilterBits() >= 2;
    const uint64_t deferred_rerank_count = std::max<uint64_t>(search_param.ef, 2 * rerank_topk);
    auto computer = search_param.rabitq_fused_computer != nullptr
                        ? search_param.rabitq_fused_computer
                        : split_codes->FactoryFusedComputer(query);
    if (computer == nullptr) {
        return nullptr;
    }
    RaBitQFusedTraversalQuery traversal_query;
    const bool has_direct_traversal_query =
        split_codes->GetFusedTraversalQuery(computer, &traversal_query);
    if (has_direct_traversal_query) {
        const auto* float_query = static_cast<const float*>(query);
        for (uint64_t i = 0; i < traversal_query.dim; ++i) {
            if (not std::isfinite(float_query[i])) {
                return nullptr;
            }
        }
    }
    const float runtime_error_rate =
        ctx == nullptr ? std::numeric_limits<float>::quiet_NaN() : ctx->rabitq_error_rate;
    if (has_direct_traversal_query) {
        const auto run_direct_search = [&](const auto& scorer) {
            auto result = search_direct_fused(graph,
                                              visited_list,
                                              search_param,
                                              ctx,
                                              lower_bound_candidates,
                                              allocator,
                                              neighbors_mutex_,
                                              scorer);
            if (search_finalized != nullptr) {
                *search_finalized = not deferred_rerank or lower_bound_candidates != nullptr;
            }
            return result;
        };
        if (not traversal_query.affine) {
            LegacyOneBitScorer scorer(traversal_query, runtime_error_rate);
            return run_direct_search(scorer);
        }
        switch (traversal_query.filter_bits) {
            case 1: {
                AffineScorer<1> scorer(traversal_query, runtime_error_rate);
                return run_direct_search(scorer);
            }
            case 2: {
                AffineScorer<2> scorer(traversal_query, runtime_error_rate);
                return run_direct_search(scorer);
            }
            case 3: {
                AffineScorer<3> scorer(traversal_query, runtime_error_rate);
                return run_direct_search(scorer);
            }
            case 4: {
                AffineScorer<4> scorer(traversal_query, runtime_error_rate);
                return run_direct_search(scorer);
            }
            default:
                break;
        }
    }
    auto result = std::make_shared<StandardHeap<true, false>>(allocator, -1);
    SearchBuffer candidate_set(search_param.ef, allocator);
    uint32_t rabitq_filter_count = 0;
    uint32_t rabitq_full_count = 0;
    uint32_t rabitq_filter_fallback_full_count = 0;
    uint32_t rabitq_hint_full_count = 0;
    uint32_t rabitq_reorder_fallback_full_count = 0;
    QueryContext rate_context;
    QueryContext* rate_context_ptr = nullptr;
    if (ctx != nullptr) {
        rate_context.rabitq_error_rate = ctx->rabitq_error_rate;
        rate_context_ptr = &rate_context;
    }
    Filter* attribute_filter = nullptr;
    if (not search_param.executors.empty() and search_param.executors[0] != nullptr) {
        search_param.executors[0]->Clear();
        attribute_filter = search_param.executors[0]->Run();
    }
    const auto is_allowed = [&search_param, attribute_filter](InnerIdType id) {
        return (search_param.is_inner_id_allowed == nullptr or
                search_param.is_inner_id_allowed->CheckValid(id)) and
               (attribute_filter == nullptr or attribute_filter->CheckValid(id));
    };

    auto score_node = [&](InnerIdType id,
                          float* distance,
                          float* lower_bound,
                          float* filter_inner_product) {
        const auto node = graph->GetCodeView(id);
        *distance = std::numeric_limits<float>::max();
        *lower_bound = std::numeric_limits<float>::max();
        *filter_inner_product = std::numeric_limits<float>::quiet_NaN();
        if (search_param.enable_rabitq_one_bit_search) {
            ++rabitq_filter_count;
            if (has_direct_traversal_query and node.cluster_id < traversal_query.cluster_count) {
                const uint64_t packed_ip = RaBitQSQ4UBinaryIPWithBaseSum(
                    traversal_query.query_planes, node.one_bit_code, traversal_query.dim);
                const auto raw_ip = static_cast<uint32_t>(packed_ip);
                const auto base_sum = static_cast<uint32_t>(packed_ip >> 32U);
                *filter_inner_product = traversal_query.query_delta * static_cast<float>(raw_ip) +
                                        traversal_query.query_vl * static_cast<float>(base_sum);

                float f_add = 0.0F;
                float f_rescale = 0.0F;
                float f_error = 0.0F;
                const auto* metadata = node.one_bit_code + traversal_query.one_bit_metadata_offset;
                std::memcpy(&f_add, metadata, sizeof(float));
                std::memcpy(&f_rescale, metadata + sizeof(float), sizeof(float));
                std::memcpy(&f_error, metadata + 2 * sizeof(float), sizeof(float));
                *distance = f_add + traversal_query.cluster_g_add[node.cluster_id] +
                            f_rescale * (*filter_inner_product - 0.5F * traversal_query.query_sum);
                const float effective_error_rate =
                    IsFiniteRaBitQValue(runtime_error_rate) and runtime_error_rate > 0.0F
                        ? runtime_error_rate
                        : traversal_query.default_rabitq_error_rate;
                const float error_rate_scale =
                    effective_error_rate / RaBitQuantizerParameter::DEFAULT_RABITQ_ERROR_RATE;
                *lower_bound = *distance - error_rate_scale * f_error *
                                               traversal_query.cluster_g_error[node.cluster_id];
                if (IsFiniteRaBitQValue(*distance) and
                    (not should_rerank or IsFiniteRaBitQValue(*lower_bound))) {
                    if (not should_rerank) {
                        *lower_bound = *distance;
                    }
                    return true;
                }
            }
            if (split_codes->ComputeFusedOneBitWithFilterIP(computer,
                                                            node.cluster_id,
                                                            node.one_bit_code,
                                                            node.supplement_code,
                                                            distance,
                                                            lower_bound,
                                                            filter_inner_product,
                                                            rate_context_ptr)) {
                return true;
            }
            if (not should_rerank and IsFiniteRaBitQValue(*distance) and
                *distance < std::numeric_limits<float>::max()) {
                *lower_bound = *distance;
                return true;
            }
            ++rabitq_filter_fallback_full_count;
            ++rabitq_full_count;
            *filter_inner_product = std::numeric_limits<float>::quiet_NaN();
            const bool computed = split_codes->ComputeFusedFull(computer,
                                                                node.cluster_id,
                                                                node.one_bit_code,
                                                                node.supplement_code,
                                                                distance,
                                                                nullptr);
            *lower_bound = *distance;
            return computed;
        }
        ++rabitq_full_count;
        *lower_bound = 0.0F;
        *filter_inner_product = std::numeric_limits<float>::quiet_NaN();
        const bool computed = split_codes->ComputeFusedFull(
            computer, node.cluster_id, node.one_bit_code, node.supplement_code, distance, nullptr);
        *lower_bound = *distance;
        return computed;
    };

    const auto refine_node = [&](InnerIdType id, float, float* distance) {
        const auto node = graph->GetCodeView(id);
        ++rabitq_full_count;
        // RaBitQ-Library's HNSW adaptive-rerank path calls
        // split_single_fulldist_direct(), which recomputes the x-bit/query inner product
        // from the float query. Reusing the 4-bit traversal estimate here changes that
        // reference behavior and materially degrades recall on GIST1M.
        ++rabitq_reorder_fallback_full_count;
        return split_codes->ComputeFusedFull(
            computer, node.cluster_id, node.one_bit_code, node.supplement_code, distance, nullptr);
    };

    const auto refine_node_with_hint = [&](InnerIdType id,
                                           float filter_inner_product,
                                           float* distance) {
        const auto node = graph->GetCodeView(id);
        ++rabitq_full_count;
        if (exact_filter_ip_hint and IsFiniteRaBitQValue(filter_inner_product) and
            split_codes->ComputeFusedFullWithFilterIP(computer,
                                                      node.cluster_id,
                                                      node.one_bit_code,
                                                      node.supplement_code,
                                                      filter_inner_product,
                                                      distance,
                                                      nullptr)) {
            ++rabitq_hint_full_count;
            return true;
        }
        ++rabitq_reorder_fallback_full_count;
        return split_codes->ComputeFusedFull(
            computer, node.cluster_id, node.one_bit_code, node.supplement_code, distance, nullptr);
    };
    const auto result_eligible = [&](float distance) {
        return not range_search or distance <= search_param.radius;
    };
    const auto push_duplicates = [&](InnerIdType id) {
        if (not search_param.consider_duplicate) {
            return;
        }
        const auto group_id = graph->GetGroupId(id);
        auto duplicate_ids = graph->GetDuplicateIds(group_id);
        duplicate_ids.push_back(group_id);
        for (const auto duplicate : duplicate_ids) {
            if (duplicate == id or not is_allowed(duplicate)) {
                continue;
            }
            const auto node = graph->GetCodeView(duplicate);
            float duplicate_distance = 0.0F;
            ++rabitq_full_count;
            if (split_codes->ComputeFusedFull(computer,
                                              node.cluster_id,
                                              node.one_bit_code,
                                              node.supplement_code,
                                              &duplicate_distance,
                                              nullptr) and
                result_eligible(duplicate_distance)) {
                result->Push(duplicate_distance, duplicate);
            }
        }
    };

    float entry_distance = 0.0F;
    float entry_lower_bound = 0.0F;
    float entry_filter_ip = 0.0F;
    if (not score_node(search_param.ep, &entry_distance, &entry_lower_bound, &entry_filter_ip)) {
        return result;
    }
    if (should_rerank and not deferred_rerank) {
        const bool refined =
            exact_filter_ip_hint
                ? refine_node_with_hint(search_param.ep, entry_filter_ip, &entry_distance)
                : refine_node(search_param.ep, entry_filter_ip, &entry_distance);
        if (not refined) {
            return result;
        }
    }
    candidate_set.Insert(search_param.ep, entry_distance);
    visited_list->Set(search_param.ep);
    if (is_allowed(search_param.ep) and result_eligible(entry_distance)) {
        result->Push(entry_distance, search_param.ep);
    }
    push_duplicates(search_param.ep);

    uint32_t hops = 0;
    uint32_t distance_computations = 1;
    auto* reasoning = ctx == nullptr ? nullptr : ctx->reasoning_ctx;
    const uint32_t prefetch_lookahead = [split_codes]() {
        switch (split_codes->FusedFilterBits()) {
            case 1:
                return 8U;
            case 2:
                return 5U;
            case 3:
                return 4U;
            default:
                return 3U;
        }
    }();
    const auto prefetch_codes_l1 = [&graph](InnerIdType id) { graph->PrefetchFusedFilter(id); };
    const auto prefetch_graph_l2 = [&graph](InnerIdType id) { graph->PrefetchNodeHeader(id); };

    while (candidate_set.HasNext()) {
        ++hops;
        if (hops >= search_param.hops_limit) {
            if (reasoning != nullptr) {
                reasoning->SetTermination(ReasoningContext::kTerminationHopsLimitReached);
            }
            break;
        }
        if (search_param.time_cost != nullptr and search_param.time_cost->CheckOvertime()) {
            if (ctx != nullptr and ctx->stats != nullptr) {
                ctx->stats->is_timeout.store(true, std::memory_order_relaxed);
            }
            if (reasoning != nullptr) {
                reasoning->SetTermination(ReasoningContext::kTerminationTimeout);
            }
            break;
        }
        if (reasoning != nullptr) {
            reasoning->AddSearchHop();
        }

        const auto current_id = candidate_set.Pop();
        MaybeSharedLock node_lock(neighbors_mutex_, current_id);
        const auto node = graph->GetNodeView(current_id);
        const auto neighbor_count = node.neighbor_count;
        CHECK_ARGUMENT(neighbor_count <= graph->MaximumDegree(),
                       "corrupt fused graph neighbor count");
        const auto* neighbors = node.neighbors;
        const auto initial_prefetch = std::min(prefetch_lookahead, neighbor_count);
        for (uint32_t i = 0; i < initial_prefetch; ++i) {
            InnerIdType neighbor = 0;
            if (graph->ResolveNeighbor(neighbors[i], neighbor)) {
                prefetch_codes_l1(neighbor);
            }
        }

        for (uint32_t i = 0; i < neighbor_count; ++i) {
            if (i + prefetch_lookahead < neighbor_count) {
                InnerIdType lookahead_neighbor = 0;
                if (graph->ResolveNeighbor(neighbors[i + prefetch_lookahead], lookahead_neighbor)) {
                    prefetch_codes_l1(lookahead_neighbor);
                }
            }
            InnerIdType neighbor = 0;
            if (not graph->ResolveNeighbor(neighbors[i], neighbor)) {
                continue;
            }
            if (visited_list->TestAndSet(neighbor)) {
                continue;
            }

            float distance = 0.0F;
            float lower_bound = 0.0F;
            float filter_ip = 0.0F;
            if (not score_node(neighbor, &distance, &lower_bound, &filter_ip)) {
                continue;
            }
            ++distance_computations;
            if (reasoning != nullptr) {
                reasoning->RecordVisit(neighbor, distance, hops);
            }

            const bool allowed = is_allowed(neighbor);
            if (deferred_rerank) {
                if (not candidate_set.IsFull(distance)) {
                    candidate_set.Insert(neighbor, distance);
                }
                if (allowed) {
                    result->Push(distance, neighbor);
                    while (result->Size() > deferred_rerank_count) {
                        result->Pop();
                    }
                } else if (reasoning != nullptr) {
                    reasoning->RecordFilterReject(neighbor);
                }
                if (candidate_set.HasNext()) {
                    prefetch_graph_l2(candidate_set.NextId());
                }
                continue;
            }
            const auto result_bound = result->Size() < rerank_topk
                                          ? std::numeric_limits<float>::max()
                                          : result->Top().first;
            const bool promising =
                result->Size() < rerank_topk or
                (should_rerank ? lower_bound < result_bound : distance < result_bound);
            if (promising and should_rerank) {
                graph->PrefetchFusedSupplement(neighbor);
                const bool refined = exact_filter_ip_hint
                                         ? refine_node_with_hint(neighbor, filter_ip, &distance)
                                         : refine_node(neighbor, filter_ip, &distance);
                if (not refined) {
                    continue;
                }
            }

            if (not candidate_set.IsFull(distance)) {
                candidate_set.Insert(neighbor, distance);
            }
            if (promising and allowed and result_eligible(distance)) {
                result->Push(distance, neighbor);
                while (result->Size() > rerank_topk) {
                    result->Pop();
                }
            } else if (not allowed and reasoning != nullptr) {
                reasoning->RecordFilterReject(neighbor);
            }
            if (promising) {
                push_duplicates(neighbor);
            }
            if (candidate_set.HasNext()) {
                prefetch_graph_l2(candidate_set.NextId());
            }
        }
    }

    if (deferred_rerank) {
        auto refined = std::make_shared<StandardHeap<true, false>>(allocator, -1);
        while (not result->Empty()) {
            const auto candidate = result->Top();
            result->Pop();
            float full_distance = candidate.first;
            float lower_bound = 0.0F;
            float filter_inner_product = 0.0F;
            const bool scored =
                score_node(candidate.second, &full_distance, &lower_bound, &filter_inner_product);
            bool refined_ok = false;
            if (scored) {
                refined_ok =
                    exact_filter_ip_hint
                        ? refine_node_with_hint(
                              candidate.second, filter_inner_product, &full_distance)
                        : refine_node(candidate.second, filter_inner_product, &full_distance);
            }
            if (refined_ok) {
                refined->Push(full_distance, candidate.second);
                while (refined->Size() > rerank_topk) {
                    refined->Pop();
                }
            }
        }
        result = std::move(refined);
    }
    while (result->Size() > result_limit) {
        result->Pop();
    }
    if (ctx != nullptr and ctx->stats != nullptr) {
        ctx->stats->dist_cmp.fetch_add(distance_computations, std::memory_order_relaxed);
        ctx->stats->hops.fetch_add(hops, std::memory_order_relaxed);
        ctx->stats->rabitq_filter_count.fetch_add(rabitq_filter_count, std::memory_order_relaxed);
        ctx->stats->rabitq_full_count.fetch_add(rabitq_full_count, std::memory_order_relaxed);
        ctx->stats->rabitq_filter_fallback_full_count.fetch_add(rabitq_filter_fallback_full_count,
                                                                std::memory_order_relaxed);
        ctx->stats->rabitq_reorder_hint_full_count.fetch_add(rabitq_hint_full_count,
                                                             std::memory_order_relaxed);
        ctx->stats->rabitq_reorder_fallback_full_count.fetch_add(rabitq_reorder_fallback_full_count,
                                                                 std::memory_order_relaxed);
    }
    return result;
}

}  // namespace vsag
