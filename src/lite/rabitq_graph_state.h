// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <chrono>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "lite/query_heap.h"
#include "lite/query_visited.h"
#include "lite/rabitq_codec.h"
#include "lite/rabitq_filter_ip.h"

// Fixed-model mutable state from the validated standalone RaBitQ probe.
// This internal module has no benchmark, filesystem or getrusage dependency.
namespace vsag::lite::detail::rabitq {

using Clock = std::chrono::steady_clock;

inline double
microseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

inline float
filter_centered_ip(const std::vector<float>& query, const uint8_t* filter) {
    static const auto compute = select_rabitq_filter_ip();
    return compute(query.data(), filter, query.size());
}

inline void
filter_centered_ip_batch4(const std::vector<float>& query,
                          const std::array<EncodedView, 4>& codes,
                          float* results) {
    static const auto compute = select_rabitq_filter_ip_batch4();
    compute(query.data(),
            codes[0].filter,
            codes[1].filter,
            codes[2].filter,
            codes[3].filter,
            query.size(),
            results);
}

inline float
supplement_ip(const std::vector<float>& query, const uint8_t* supplement) {
    static const auto compute = select_rabitq_supplement_ip();
    return compute(query.data(), supplement, query.size());
}

inline float
l2_distance(float base_norm, float query_norm, float normalized_ip) {
    return base_norm * base_norm + query_norm * query_norm -
           2.0F * base_norm * query_norm * normalized_ip;
}

inline uint32_t
popcount_byte(uint8_t value) {
    value = static_cast<uint8_t>(value - ((value >> 1U) & 0x55U));
    value = static_cast<uint8_t>((value & 0x33U) + ((value >> 2U) & 0x33U));
    return static_cast<uint32_t>((value + (value >> 4U)) & 0x0FU);
}

inline float
one_bit_code_distance(const EncodedView& left, const EncodedView& right, uint64_t dim) {
    codec_require(dim > 0 and left.filter != nullptr and right.filter != nullptr,
                  "invalid one-bit code distance input");
    uint64_t mismatches = 0;
    const uint64_t full_bytes = dim / 8;
    for (uint64_t byte = 0; byte < full_bytes; ++byte) {
        mismatches += popcount_byte(left.filter[byte] ^ right.filter[byte]);
    }
    const uint32_t tail_bits = static_cast<uint32_t>(dim & 7U);
    if (tail_bits != 0) {
        const auto mask = static_cast<uint8_t>((1U << tail_bits) - 1U);
        mismatches += popcount_byte((left.filter[full_bytes] ^ right.filter[full_bytes]) & mask);
    }
    const float normalized_ip =
        1.0F - 2.0F * static_cast<float>(mismatches) / static_cast<float>(dim);
    return l2_distance(left.metadata.norm, right.metadata.norm, normalized_ip);
}

struct FilterEstimate {
    float distance;
    float lower_bound;
    float centered_ip;
};

template <bool need_lower_bound = true>
inline FilterEstimate
filter_estimate(const std::vector<float>& query, float query_norm, const EncodedView& code) {
    const float centered_ip = filter_centered_ip(query, code.filter);
    const float normalized_ip =
        centered_ip / code.metadata.filter_norm / code.metadata.filter_error;
    const float distance = l2_distance(code.metadata.norm, query_norm, normalized_ip);
    if constexpr (not need_lower_bound) {
        return {distance, 0.0F, centered_ip};
    }
    const float error = 2.0F * code.metadata.norm * query_norm * K_ERROR_RATE *
                        code.metadata.lower_bound_error / code.metadata.filter_error;
    const float estimate = distance - error;
    const float lower_bound = estimate - 1e-5F * std::max(1.0F, std::fabs(estimate));
    return {distance, lower_bound, centered_ip};
}

template <bool need_lower_bound = true>
inline std::array<FilterEstimate, 4>
filter_estimate_batch4(const std::vector<float>& query,
                       float query_norm,
                       const std::array<EncodedView, 4>& codes) {
    float centered_ips[4];
    filter_centered_ip_batch4(query, codes, centered_ips);
    std::array<FilterEstimate, 4> results;
    for (uint32_t lane = 0; lane < 4; ++lane) {
        const auto& metadata = codes[lane].metadata;
        const float normalized_ip =
            centered_ips[lane] / metadata.filter_norm / metadata.filter_error;
        const float distance = l2_distance(metadata.norm, query_norm, normalized_ip);
        if constexpr (not need_lower_bound) {
            results[lane] = {distance, 0.0F, centered_ips[lane]};
            continue;
        }
        const float error = 2.0F * metadata.norm * query_norm * K_ERROR_RATE *
                            metadata.lower_bound_error / metadata.filter_error;
        const float estimate = distance - error;
        results[lane] = {
            distance, estimate - 1e-5F * std::max(1.0F, std::fabs(estimate)), centered_ips[lane]};
    }
    return results;
}

inline float
full_distance(const std::vector<float>& query,
              float query_norm,
              const EncodedView& code,
              float centered_filter_ip,
              float query_sum) {
    const float filter_ip = centered_filter_ip + 3.5F * query_sum;
    const float code_ip = filter_ip * static_cast<float>(1U << K_SUPPLEMENT_BITS) +
                          supplement_ip(query, code.supplement);
    const float base_error = std::fabs(code.metadata.error) < 1e-5F ? 1.0F : code.metadata.error;
    const float normalized_ip =
        (code_ip - 127.5F * query_sum) / code.metadata.code_norm / base_error;
    return l2_distance(code.metadata.norm, query_norm, normalized_ip);
}

// Keep the standalone distance interface for callers without a query context.
inline float
full_distance(const std::vector<float>& query,
              float query_norm,
              const EncodedView& code,
              float centered_filter_ip) {
    return full_distance(query,
                         query_norm,
                         code,
                         centered_filter_ip,
                         std::accumulate(query.begin(), query.end(), 0.0F));
}

struct Candidate {
    uint64_t id;
    float distance;
    float centered_ip{};
};

inline bool
better(const Candidate& left, const Candidate& right) {
    return left.distance < right.distance or
           (left.distance == right.distance and left.id < right.id);
}

inline bool
farther(const Candidate& left, const Candidate& right) {
    return left.distance > right.distance or
           (left.distance == right.distance and left.id > right.id);
}
struct Better {
    bool
    operator()(const Candidate& left, const Candidate& right) const {
        return better(left, right);
    }
};

struct Farther {
    bool
    operator()(const Candidate& left, const Candidate& right) const {
        return farther(left, right);
    }
};

template <typename Distance>
inline std::vector<Candidate>
top_k(uint64_t count, uint64_t k, Distance distance) {
    std::priority_queue<Candidate, std::vector<Candidate>, Better> heap;
    for (uint64_t id = 0; id < count; ++id) {
        const Candidate next{id, distance(id)};
        if (heap.size() < k) {
            heap.push(next);
        } else if (better(next, heap.top())) {
            heap.pop();
            heap.push(next);
        }
    }
    std::vector<Candidate> result(heap.size());
    for (uint64_t i = result.size(); i > 0; --i) {
        result[i - 1] = heap.top();
        heap.pop();
    }
    return result;
}

struct SearchResult {
    std::vector<Candidate> neighbors;
    uint64_t reordered{};
};

template <bool batch4 = true>
inline void
score_mutation_candidates(const std::vector<float>& query,
                          float query_norm,
                          float query_sum,
                          const EncodedRecords& records,
                          std::vector<Candidate>& ranked) {
    uint64_t offset = 0;
    if constexpr (batch4) {
        for (; offset + 4 <= ranked.size(); offset += 4) {
            std::array<EncodedView, 4> codes;
            for (uint64_t lane = 0; lane < 4; ++lane) {
                codes[lane] = records.At(ranked[offset + lane].id);
            }
            float centered_ips[4];
            filter_centered_ip_batch4(query, codes, centered_ips);
            for (uint64_t lane = 0; lane < 4; ++lane) {
                ranked[offset + lane].distance =
                    full_distance(query, query_norm, codes[lane], centered_ips[lane], query_sum);
            }
        }
    }
    for (; offset < ranked.size(); ++offset) {
        const auto code = records.At(ranked[offset].id);
        const auto coarse = filter_estimate(query, query_norm, code);
        ranked[offset].distance =
            full_distance(query, query_norm, code, coarse.centered_ip, query_sum);
    }
}

inline SearchResult
filtered_search(const std::vector<float>& query,
                float query_norm,
                const EncodedRecords& codes,
                uint64_t k) {
    const float query_sum = std::accumulate(query.begin(), query.end(), 0.0F);
    std::priority_queue<Candidate, std::vector<Candidate>, Better> heap;
    uint64_t reordered = 0;
    for (uint64_t id = 0; id < codes.Size(); ++id) {
        const auto code = codes.At(id);
        const auto coarse = filter_estimate(query, query_norm, code);
        if (heap.size() == k and coarse.lower_bound >= heap.top().distance) {
            continue;
        }
        ++reordered;
        const Candidate next{id,
                             full_distance(query, query_norm, code, coarse.centered_ip, query_sum)};
        if (heap.size() < k) {
            heap.push(next);
        } else if (better(next, heap.top())) {
            heap.pop();
            heap.push(next);
        }
    }
    std::vector<Candidate> result(heap.size());
    for (uint64_t i = result.size(); i > 0; --i) {
        result[i - 1] = heap.top();
        heap.pop();
    }
    return {std::move(result), reordered};
}

struct GraphTopology {
    [[nodiscard]] uint64_t
    Size() const {
        return offsets.empty() ? 0 : offsets.size() - 1;
    }

    std::vector<uint64_t> offsets;
    std::vector<uint64_t> neighbors;
};

struct GraphSearchResult {
    std::vector<Candidate> neighbors;
    uint64_t visited{};
    uint64_t reordered{};
};

template <typename NeighborRange,
          bool prune_rejected = true,
          bool prefetch_codes = true,
          bool select_without_pop = true,
          bool packed_visited = false,
          bool evaluate_filter = true>
inline GraphSearchResult
graph_search_impl(const std::vector<float>& query,
                  float query_norm,
                  const EncodedRecords& codes,
                  NeighborRange neighbors,
                  uint64_t k,
                  uint64_t ef_search,
                  const std::vector<int64_t>* external_ids = nullptr,
                  const std::function<bool(int64_t)>& filter = {}) {
    codec_require(external_ids == nullptr or external_ids->size() == codes.Size(),
                  "search ID count mismatch");
    codec_require(not filter or external_ids != nullptr, "filtered search needs external IDs");
    k = std::min(k, codes.Size());
    if (k == 0) {
        return {};
    }
    const float query_sum = std::accumulate(query.begin(), query.end(), 0.0F);
    const uint64_t ef = std::min(codes.Size(), std::max(k, ef_search));
    std::conditional_t<select_without_pop,
                       QueryHeap<Candidate, Better>,
                       std::priority_queue<Candidate, std::vector<Candidate>, Better>>
        best;
    std::priority_queue<Candidate, std::vector<Candidate>, Farther> candidates;
    QueryVisited<packed_visited> visited(codes.Size());
    uint64_t visited_count = 0;
    uint64_t filtered_reorders = 0;
    const auto rank = [external_ids](const Candidate& left, const Candidate& right) {
        if (left.distance != right.distance) {
            return left.distance < right.distance;
        }
        return external_ids == nullptr ? left.id < right.id
                                       : (*external_ids)[left.id] < (*external_ids)[right.id];
    };
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(rank)> accepted(rank);

    auto process_visited = [&](uint64_t slot, const EncodedView& code, FilterEstimate estimate) {
        // Rejected nodes remain traversable. Rank every visited allowed ID using
        // the complete 8-bit estimate; filtering must not sever routing paths.
        if (evaluate_filter and filter and filter((*external_ids)[slot])) {
            ++filtered_reorders;
            const Candidate value{
                slot, full_distance(query, query_norm, code, estimate.centered_ip, query_sum)};
            if (accepted.size() < k) {
                accepted.push(value);
            } else if (rank(value, accepted.top())) {
                accepted.pop();
                accepted.push(value);
            }
        }
        const Candidate next{slot, estimate.distance, estimate.centered_ip};
        if constexpr (prune_rejected) {
            // As in native BasicSearcher, bound heap admission before insertion.
            // The full best-heap boundary only improves: a rejected candidate can
            // never be expanded later. Preserve the existing distance/slot tie.
            if (best.size() == ef and not better(next, best.top())) {
                return;
            }
        }
        candidates.push(next);
        if constexpr (prune_rejected and select_without_pop and not packed_visited) {
            if (best.size() == ef) {
                best.ReplaceTop(next);
            } else {
                best.push(next);
            }
        } else {
            best.push(next);
            if (best.size() > ef) {
                best.pop();
            }
        }
    };
    auto visit = [&](uint64_t slot) {
        if (visited.TestAndSet(slot)) {
            return;
        }
        ++visited_count;
        const auto code = codes.At(slot);
        process_visited(slot, code, filter_estimate<false>(query, query_norm, code));
    };
    auto visit_range = [&](auto first, auto last) {
        std::array<uint64_t, 4> slots{};
        uint32_t count = 0;
        const auto flush = [&]() {
            std::array<EncodedView, 4> batch_codes{};
            for (uint32_t lane = 0; lane < count; ++lane) {
                batch_codes[lane] = codes.At(slots[lane]);
            }
            if (count == 4) {
                const auto estimates =
                    filter_estimate_batch4<false>(query, query_norm, batch_codes);
                for (uint32_t lane = 0; lane < 4; ++lane) {
                    process_visited(slots[lane], batch_codes[lane], estimates[lane]);
                }
            } else {
                for (uint32_t lane = 0; lane < count; ++lane) {
                    process_visited(slots[lane],
                                    batch_codes[lane],
                                    filter_estimate<false>(query, query_norm, batch_codes[lane]));
                }
            }
            count = 0;
        };
        for (auto edge = first; edge != last; ++edge) {
            const uint64_t slot = *edge;
            if (visited.TestAndSet(slot)) {
                continue;
            }
            ++visited_count;
            if constexpr (prefetch_codes) {
                // Match native HGraph RaBitQ: bring filter records in before batch scoring.
                const auto* data = codes.filters.data() + slot * codes.FilterBytes();
                for (uint64_t offset = 0; offset < codes.FilterBytes(); offset += 64) {
                    __builtin_prefetch(data + offset, 0, 3);
                }
                __builtin_prefetch(codes.metadata.data() + slot, 0, 3);
            }
            slots[count++] = slot;
            if (count == 4) {
                flush();
            }
        }
        if (count != 0) {
            flush();
        }
    };

    visit(0);
    const uint64_t last = codes.Size() - 1;
    if (last != 0) {
        visit(last);
    }
    constexpr uint64_t k_extra_entry_points = 6;
    for (uint64_t i = 1; i <= k_extra_entry_points; ++i) {
        const uint64_t entry = i * last / (k_extra_entry_points + 1);
        if (entry != 0 and entry != last) {
            visit(entry);
        }
    }

    while (not candidates.empty()) {
        const Candidate current = candidates.top();
        candidates.pop();
        if (best.size() == ef and better(best.top(), current)) {
            break;
        }
        if (codes.Size() > 1) {
            visit((current.id + codes.Size() - 1) % codes.Size());
            visit((current.id + 1) % codes.Size());
        }
        const auto range = neighbors(current.id);
        visit_range(range.first, range.second);
    }

    if (evaluate_filter and filter) {
        std::vector<Candidate> result(accepted.size());
        for (uint64_t i = result.size(); i > 0; --i) {
            result[i - 1] = accepted.top();
            accepted.pop();
        }
        return {std::move(result), visited_count, filtered_reorders};
    }
    const uint64_t reorder_count = best.size();
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(rank)> reordered(rank);
    const auto reorder = [&](const Candidate& coarse) {
        const uint64_t slot = coarse.id;
        const auto code = codes.At(slot);
        const Candidate next{slot,
                             full_distance(query, query_norm, code, coarse.centered_ip, query_sum)};
        if (reordered.size() < k) {
            reordered.push(next);
        } else if (rank(next, reordered.top())) {
            reordered.pop();
            reordered.push(next);
        }
    };
    if constexpr (select_without_pop) {
        for (const auto& coarse : best.TakeData()) {
            reorder(coarse);
        }
    } else {
        while (not best.empty()) {
            const Candidate coarse = best.top();
            best.pop();
            reorder(coarse);
        }
    }
    std::vector<Candidate> result(reordered.size());
    for (uint64_t i = result.size(); i > 0; --i) {
        result[i - 1] = reordered.top();
        reordered.pop();
    }
    return {std::move(result), visited_count, reorder_count};
}

inline GraphSearchResult
graph_search(const std::vector<float>& query,
             float query_norm,
             const EncodedRecords& codes,
             const GraphTopology& graph,
             uint64_t k,
             uint64_t ef_search) {
    codec_require(graph.Size() == codes.Size(), "graph and encoded records disagree");
    const auto range = [&graph](uint64_t slot) {
        return std::make_pair(
            graph.neighbors.begin() + static_cast<int64_t>(graph.offsets[slot]),
            graph.neighbors.begin() + static_cast<int64_t>(graph.offsets[slot + 1]));
    };
    return graph_search_impl(query, query_norm, codes, range, k, ef_search);
}

inline GraphSearchResult
graph_search_adjacency(const std::vector<float>& query,
                       float query_norm,
                       const EncodedRecords& codes,
                       const std::vector<std::vector<uint64_t>>& adjacency,
                       uint64_t k,
                       uint64_t ef_search,
                       const std::vector<int64_t>* external_ids = nullptr,
                       const std::function<bool(int64_t)>& filter = {}) {
    codec_require(adjacency.size() == codes.Size(), "graph and encoded records disagree");
    const auto range = [&adjacency](uint64_t slot) {
        return std::make_pair(adjacency[slot].begin(), adjacency[slot].end());
    };
    // Select once per query: small graphs retain byte checks without per-slot policy branches.
    if (codes.Size() >= 65536) {
        if (not filter) {
            return graph_search_impl<decltype(range), true, true, true, true, false>(
                query, query_norm, codes, range, k, ef_search, external_ids, filter);
        }
        return graph_search_impl<decltype(range), true, true, true, true>(
            query, query_norm, codes, range, k, ef_search, external_ids, filter);
    }
    return graph_search_impl(query, query_norm, codes, range, k, ef_search, external_ids, filter);
}

inline std::vector<std::vector<uint64_t>>
expand_graph(const GraphTopology& graph) {
    // Validate before iterator arithmetic, including empty graph payloads.
    codec_require(not graph.offsets.empty() and graph.offsets.front() == 0 and
                      graph.offsets.back() == graph.neighbors.size(),
                  "invalid graph offset boundaries");
    for (uint64_t i = 1; i < graph.offsets.size(); ++i) {
        codec_require(
            graph.offsets[i - 1] <= graph.offsets[i] and graph.offsets[i] <= graph.neighbors.size(),
            "invalid graph offset range");
    }
    std::vector<std::vector<uint64_t>> result(graph.Size());
    for (uint64_t slot = 0; slot < graph.Size(); ++slot) {
        result[slot].assign(
            graph.neighbors.begin() + static_cast<int64_t>(graph.offsets[slot]),
            graph.neighbors.begin() + static_cast<int64_t>(graph.offsets[slot + 1]));
    }
    return result;
}

inline GraphTopology
compact_graph(const std::vector<std::vector<uint64_t>>& adjacency) {
    GraphTopology result;
    result.offsets.reserve(adjacency.size() + 1);
    result.offsets.push_back(0);
    for (const auto& neighbors : adjacency) {
        result.neighbors.insert(result.neighbors.end(), neighbors.begin(), neighbors.end());
        result.offsets.push_back(result.neighbors.size());
    }
    return result;
}

struct MutationScanTiming {
    double update_wall_us{};
    double update_cpu_us{};
    double remove_wall_us{};
    double remove_cpu_us{};
};

struct IncomingMemoryUsage {
    uint64_t edges{};
    uint64_t logical_bytes{};
    uint64_t capacity_bytes{};
};

struct IncomingDistribution {
    uint64_t nodes{};
    uint64_t zero_degree_nodes{};
    uint64_t degree_le_4{};
    uint64_t degree_le_8{};
    uint64_t degree_le_16{};
    uint64_t degree_le_32{};
    uint64_t degree_le_64{};
    uint64_t degree_gt_64{};
    uint64_t degree_p50{};
    uint64_t degree_p90{};
    uint64_t degree_p95{};
    uint64_t degree_p99{};
    uint64_t degree_max{};
    uint64_t capacity_p50{};
    uint64_t capacity_p90{};
    uint64_t capacity_p95{};
    uint64_t capacity_p99{};
    uint64_t capacity_max{};
    uint64_t slack_nodes{};
    uint64_t slack_entries{};
    uint64_t outer_vector_bytes{};
    uint64_t edge_logical_bytes{};
    uint64_t edge_capacity_bytes{};
};

struct MutableMemoryUsage {
    uint64_t model_logical_bytes{};
    uint64_t model_capacity_bytes{};
    uint64_t codes_logical_bytes{};
    uint64_t codes_capacity_bytes{};
    uint64_t ids_logical_bytes{};
    uint64_t ids_capacity_bytes{};
    uint64_t adjacency_edges{};
    uint64_t adjacency_logical_bytes{};
    uint64_t adjacency_capacity_bytes{};
    uint64_t incoming_adjacency_logical_bytes{};
    uint64_t incoming_adjacency_capacity_bytes{};
    uint64_t incoming_count_logical_bytes{};
    uint64_t incoming_count_capacity_bytes{};
    uint64_t slot_entries{};
    uint64_t slot_buckets{};

    [[nodiscard]] uint64_t
    KnownLogicalBytes() const {
        return model_logical_bytes + codes_logical_bytes + ids_logical_bytes +
               adjacency_logical_bytes + incoming_adjacency_logical_bytes +
               incoming_count_logical_bytes;
    }

    [[nodiscard]] uint64_t
    KnownCapacityBytes() const {
        return model_capacity_bytes + codes_capacity_bytes + ids_capacity_bytes +
               adjacency_capacity_bytes + incoming_adjacency_capacity_bytes +
               incoming_count_capacity_bytes;
    }
};

enum class IncomingProtection { NONE, RECOUNT, CACHED };

// Rank is distance/slot ordered. Counts describe the graph before inserting the new edge.
// Match floating GraphBackend::link: avoid dropping a target's last incoming edge when possible.
inline uint64_t
safe_incoming_drop(const std::vector<Candidate>& ranked,
                   const std::vector<uint64_t>& incoming_counts,
                   uint64_t target) {
    codec_require(not ranked.empty(), "empty incoming prune candidates");
    for (uint64_t i = ranked.size(); i > 0; --i) {
        const uint64_t candidate = ranked[i - 1].id;
        codec_require(candidate < incoming_counts.size(), "invalid incoming prune slot");
        const bool safe =
            candidate == target ? incoming_counts[candidate] > 0 : incoming_counts[candidate] > 1;
        if (safe) {
            return i - 1;
        }
    }
    return ranked.size() - 1;
}

// Native VSAG uses uint32_t InnerIdType (src/basic_types.h).
// Only the internal reverse index narrows; outgoing slots and persisted IDs stay unchanged.
using IncomingSlot = uint32_t;

inline IncomingSlot
checked_incoming_slot(uint64_t slot) {
    codec_require(slot <= std::numeric_limits<IncomingSlot>::max(), "incoming slot overflow");
    return static_cast<IncomingSlot>(slot);
}

// Build already owns row adjacency; avoid compacting and expanding it again.
struct OwnedGraphAdjacency {
    std::vector<std::vector<uint64_t>> rows;
};

// Compare reverse edges without rebuilding a second row-allocated graph.
// Outgoing rows are traversed in source order, so each target's expected
// incoming sequence is already sorted. Scratch is local to this validation.
inline void
validate_incoming_adjacency(const std::vector<std::vector<uint64_t>>& adjacency,
                            const std::vector<std::vector<IncomingSlot>>& incoming) {
    const uint64_t count = adjacency.size();
    codec_require(incoming.size() == count, "mutable incoming size mismatch");
    std::vector<uint64_t> offsets(count + 1);
    for (uint64_t target = 0; target < count; ++target) {
        codec_require(
            incoming[target].size() <= std::numeric_limits<uint64_t>::max() - offsets[target],
            "incoming validation edge count overflow");
        offsets[target + 1] = offsets[target] + incoming[target].size();
    }
    std::vector<IncomingSlot> sorted(offsets.back());
    for (uint64_t target = 0; target < count; ++target) {
        auto begin = sorted.begin() + static_cast<int64_t>(offsets[target]);
        auto end = sorted.begin() + static_cast<int64_t>(offsets[target + 1]);
        std::copy(incoming[target].begin(), incoming[target].end(), begin);
        std::sort(begin, end);
    }
    auto cursors = offsets;
    for (uint64_t source = 0; source < count; ++source) {
        for (const uint64_t target : adjacency[source]) {
            codec_require(target < count, "invalid incoming graph neighbor");
            codec_require(
                cursors[target] < offsets[target + 1] and sorted[cursors[target]] == source,
                "mutable incoming adjacency mismatch");
            ++cursors[target];
        }
    }
    for (uint64_t target = 0; target < count; ++target) {
        codec_require(cursors[target] == offsets[target + 1],
                      "mutable incoming adjacency mismatch");
    }
}

class MutableGraphState {
public:
    MutableGraphState(Model input_model,
                      EncodedRecords input_codes,
                      const GraphTopology& input_graph,
                      std::vector<int64_t> input_ids,
                      uint64_t input_max_degree,
                      uint64_t input_ef_search,
                      bool measure_mutation_scans = false,
                      bool use_incoming_adjacency = false,
                      double (*cpu_clock)() = nullptr)
        : MutableGraphState(OwnedGraphAdjacency{expand_graph(input_graph)},
                            std::move(input_model),
                            std::move(input_codes),
                            std::move(input_ids),
                            input_max_degree,
                            input_ef_search,
                            measure_mutation_scans,
                            use_incoming_adjacency,
                            cpu_clock) {
    }

    MutableGraphState(OwnedGraphAdjacency input_graph,
                      Model input_model,
                      EncodedRecords input_codes,
                      std::vector<int64_t> input_ids,
                      uint64_t input_max_degree,
                      uint64_t input_ef_search,
                      bool measure_mutation_scans = false,
                      bool use_incoming_adjacency = false,
                      double (*cpu_clock)() = nullptr)
        : model_(std::move(input_model)),
          codes_(std::move(input_codes)),
          ids_(std::move(input_ids)),
          adjacency_(std::move(input_graph.rows)),
          max_degree_(input_max_degree),
          ef_search_(input_ef_search),
          measure_mutation_scans_(measure_mutation_scans),
          use_incoming_adjacency_(use_incoming_adjacency),
          cpu_clock_(cpu_clock) {
        codec_require(max_degree_ >= 2 and max_degree_ <= 64 and ef_search_ >= max_degree_,
                      "invalid mutable graph options");
        codec_require(ids_.size() == codes_.Size() and adjacency_.size() == codes_.Size(),
                      "mutable graph storage disagrees");
        for (uint64_t slot = 0; slot < ids_.size(); ++slot) {
            codec_require(slots_.emplace(ids_[slot], slot).second, "duplicate mutable graph ID");
        }
        if (use_incoming_adjacency_) {
            if (Size() > 0) {
                (void)checked_incoming_slot(Size() - 1);
            }
            incoming_ = BuildIncomingAdjacency();
        }
        Validate();
    }

    [[nodiscard]] uint64_t
    Size() const {
        return codes_.Size();
    }

    [[nodiscard]] bool
    Contains(int64_t id) const {
        return slots_.find(id) != slots_.end();
    }

    [[nodiscard]] bool
    SameEncoding(int64_t id, const Encoded& code) const {
        const auto found = slots_.find(id);
        if (found == slots_.end() or code.filter.size() != codes_.FilterBytes() or
            code.supplement.size() != codes_.SupplementBytes()) {
            return false;
        }
        const auto stored = codes_.At(found->second);
        const EncodedMetadata metadata{code.norm,
                                       code.code_norm,
                                       code.error,
                                       code.filter_norm,
                                       code.filter_error,
                                       code.lower_bound_error};
        static_assert(sizeof(EncodedMetadata) == 6 * sizeof(uint32_t));
        std::array<uint32_t, 6> metadata_bits{};
        std::array<uint32_t, 6> stored_bits{};
        std::memcpy(metadata_bits.data(), &metadata, sizeof(metadata));
        std::memcpy(stored_bits.data(), &stored.metadata, sizeof(stored.metadata));
        return std::equal(code.filter.begin(), code.filter.end(), stored.filter) and
               std::equal(code.supplement.begin(), code.supplement.end(), stored.supplement) and
               metadata_bits == stored_bits;
    }

    // Internal policy only; not persisted or exposed by the public Lite API.
    void
    ConfigureIncomingProtection(IncomingProtection mode,
                                bool protect_add = false,
                                bool protect_remove = false,
                                bool protect_add_displaced = false,
                                bool protect_link = false) {
        codec_require(active_journal_ == nullptr and (mode == IncomingProtection::NONE or
                                                      mode == IncomingProtection::RECOUNT or
                                                      mode == IncomingProtection::CACHED),
                      "invalid incoming protection mode");
        codec_require(not protect_link or mode == IncomingProtection::CACHED,
                      "safe link pruning requires cached incoming counts");
        incoming_protection_ = mode;
        protect_added_incoming_ = protect_add;
        protect_removed_incoming_ = protect_remove;
        protect_added_displaced_ = protect_add_displaced;
        protect_link_incoming_ = protect_link;
        incoming_counts_valid_ = false;
    }

    [[nodiscard]] uint64_t
    IncomingCountRebuilds() const {
        return incoming_count_rebuilds_;
    }

    [[nodiscard]] uint64_t
    IncomingCountBytes() const {
        return incoming_counts_.capacity() * sizeof(uint64_t);
    }

    [[nodiscard]] bool
    Add(int64_t id, const float* vector) {
        codec_require(vector != nullptr, "null mutable graph vector");
        if (slots_.count(id) != 0) {
            return false;
        }
        try {
            float query_norm = 0.0F;
            const auto query = normalize(model_, vector, query_norm);
            const auto neighbors =
                nearest(query, query_norm, std::numeric_limits<uint64_t>::max(), max_degree_);
            const uint64_t slot = Size();
            if (use_incoming_adjacency_) {
                (void)checked_incoming_slot(slot);
            }
            if (incoming_counts_valid_) {
                incoming_counts_.push_back(0);
            }
            codes_.Append(encode(model_, vector));
            ids_.push_back(id);
            adjacency_.emplace_back();
            if (use_incoming_adjacency_) {
                incoming_.emplace_back();
            }
            replace_neighbors(slot, neighbors);
            slots_.emplace(id, slot);
            std::vector<uint64_t> targets;
            const bool protect_add =
                incoming_protection_ != IncomingProtection::NONE and protect_added_incoming_;
            if (protect_add) {
                targets.reserve(1 + neighbors.size());
                targets.push_back(slot);
            }
            for (uint64_t neighbor : neighbors) {
                link(neighbor, slot, protect_add and protect_added_displaced_ ? &targets : nullptr);
            }
            if (protect_add) {
                protect_incoming(targets);
            }
            return true;
        } catch (...) {
            incoming_counts_valid_ = false;
            throw;
        }
    }

    [[nodiscard]] bool
    AddTransactional(int64_t id, const float* vector) {
        codec_require(vector != nullptr, "null mutable graph vector");
        if (Contains(id)) {
            return false;
        }
        codec_require(active_journal_ == nullptr, "nested graph add transaction");
        if (use_incoming_adjacency_ and incoming_protection_ == IncomingProtection::CACHED and
            not incoming_counts_valid_) {
            auto next = *this;
            const bool added = next.Add(id, vector);
            *this = std::move(next);
            return added;
        }
        const uint64_t count = Size();
        if (use_incoming_adjacency_) {
            (void)checked_incoming_slot(count);
        }
        const auto old_fallbacks = mutation_fallbacks_;
        const bool old_counts_valid = incoming_counts_valid_;
        UpdateJournal journal;
        journal.row_limit = count;
        bool inserted = false;
        try {
            auto prepared = prepare_encoding(model_, vector);
            const auto neighbors = nearest(prepared.query,
                                           prepared.code.norm,
                                           std::numeric_limits<uint64_t>::max(),
                                           max_degree_);
            codec_require(count < std::numeric_limits<uint64_t>::max() and
                              count + 1 <= codes_.filters.max_size() / codes_.FilterBytes() and
                              count + 1 <= codes_.supplements.max_size() / codes_.SupplementBytes(),
                          "encoded add capacity exceeded");
            grow(codes_.metadata, count + 1);
            grow(codes_.filters, (count + 1) * codes_.FilterBytes());
            grow(codes_.supplements, (count + 1) * codes_.SupplementBytes());
            grow(ids_, count + 1);
            grow(adjacency_, count + 1);
            if (use_incoming_adjacency_) {
                grow(incoming_, count + 1);
            }
            if (incoming_counts_valid_) {
                grow(incoming_counts_, count + 1);
            }
            inserted = slots_.emplace(id, count).second;
            codec_require(inserted, "duplicate add transaction ID");
            active_journal_ = &journal;
            if (incoming_counts_valid_) {
                incoming_counts_.push_back(0);
            }
            codes_.Append(std::move(prepared.code));
            ids_.push_back(id);
            adjacency_.emplace_back();
            if (use_incoming_adjacency_) {
                incoming_.emplace_back();
            }
            replace_neighbors(count, neighbors);
            std::vector<uint64_t> targets;
            const bool protect_add =
                incoming_protection_ != IncomingProtection::NONE and protect_added_incoming_;
            if (protect_add) {
                targets.reserve(1 + neighbors.size());
                targets.push_back(count);
            }
            for (uint64_t neighbor : neighbors) {
                link(
                    neighbor, count, protect_add and protect_added_displaced_ ? &targets : nullptr);
            }
            if (protect_add) {
                protect_incoming(targets);
            }
            active_journal_ = nullptr;
            return true;
        } catch (...) {
            incoming_counts_valid_ = use_incoming_adjacency_ and old_counts_valid;
            active_journal_ = nullptr;
            for (auto& row : journal.rows) {
                adjacency_[row.first].swap(row.second);
            }
            for (auto& row : journal.incoming_rows) {
                incoming_[row.first].swap(row.second);
                if (incoming_counts_valid_) {
                    incoming_counts_[row.first] = incoming_[row.first].size();
                }
            }
            if (use_incoming_adjacency_) {
                incoming_.resize(count);
                if (incoming_counts_valid_) {
                    incoming_counts_.resize(count);
                }
            }
            adjacency_.resize(count);
            ids_.resize(count);
            codes_.metadata.resize(count);
            codes_.filters.resize(count * codes_.FilterBytes());
            codes_.supplements.resize(count * codes_.SupplementBytes());
            if (inserted) {
                slots_.erase(id);
            }
            mutation_fallbacks_ = old_fallbacks;
            throw;
        }
    }

    [[nodiscard]] bool
    Update(int64_t id, const float* vector) {
        codec_require(vector != nullptr, "null mutable graph vector");
        if (not Contains(id)) {
            return false;
        }
        try {
            return UpdatePrepared(id, prepare_encoding(model_, vector));
        } catch (...) {
            incoming_counts_valid_ = false;
            throw;
        }
    }

    // Update changes no model, IDs, slot mapping or container sizes. Journal
    // only the replaced code and adjacency rows, then restore without allocation.
    [[nodiscard]] bool
    UpdateTransactional(int64_t id, PreparedEncoding prepared) {
        if (not Contains(id)) {
            return false;
        }
        codec_require(active_journal_ == nullptr, "nested graph update transaction");
        // A cold cached protection rebuild touches the whole count array.
        // Retain copy isolation for that first update only.
        if (use_incoming_adjacency_ and incoming_protection_ == IncomingProtection::CACHED and
            not incoming_counts_valid_) {
            auto next = *this;
            const bool updated = next.UpdatePrepared(id, std::move(prepared));
            *this = std::move(next);
            return updated;
        }
        const uint64_t slot = slots_.at(id);
        const auto code = codes_.At(slot);
        UpdateJournal journal;
        journal.metadata = code.metadata;
        journal.filter.assign(code.filter, code.filter + codes_.FilterBytes());
        journal.supplement.assign(code.supplement, code.supplement + codes_.SupplementBytes());
        const auto old_fallbacks = mutation_fallbacks_;
        const auto old_timing = mutation_scan_timing_;
        const bool old_counts_valid = incoming_counts_valid_;
        active_journal_ = &journal;
        try {
            const bool updated = UpdatePrepared(id, std::move(prepared));
            active_journal_ = nullptr;
            return updated;
        } catch (...) {
            incoming_counts_valid_ = use_incoming_adjacency_ and old_counts_valid;
            active_journal_ = nullptr;
            codes_.metadata[slot] = journal.metadata;
            std::copy(journal.filter.begin(),
                      journal.filter.end(),
                      codes_.filters.data() + slot * codes_.FilterBytes());
            std::copy(journal.supplement.begin(),
                      journal.supplement.end(),
                      codes_.supplements.data() + slot * codes_.SupplementBytes());
            for (auto& row : journal.rows) {
                adjacency_[row.first].swap(row.second);
            }
            for (auto& row : journal.incoming_rows) {
                incoming_[row.first].swap(row.second);
                if (incoming_counts_valid_) {
                    incoming_counts_[row.first] = incoming_[row.first].size();
                }
            }
            mutation_fallbacks_ = old_fallbacks;
            mutation_scan_timing_ = old_timing;
            throw;
        }
    }

    // Trusted internal preparation must use this state's unchanged fixed model.
    // The public adapter uses UpdateTransactional for failure atomicity.
    [[nodiscard]] bool
    UpdatePrepared(int64_t id, PreparedEncoding prepared) {
        const auto found = slots_.find(id);
        if (found == slots_.end()) {
            return false;
        }
        codec_require(prepared.query.size() == model_.dim and
                          prepared.code.filter.size() == codes_.FilterBytes() and
                          prepared.code.supplement.size() == codes_.SupplementBytes() and
                          std::isfinite(prepared.code.norm) and prepared.code.norm > 0.0F,
                      "invalid prepared update shape or norm");
        codec_require(
            std::isfinite(prepared.code.code_norm) and prepared.code.code_norm > 0.0F and
                std::isfinite(prepared.code.error) and std::isfinite(prepared.code.filter_norm) and
                prepared.code.filter_norm > 0.0F and std::isfinite(prepared.code.filter_error) and
                prepared.code.filter_error >= 1e-5F and prepared.code.filter_error <= 1.0F and
                std::isfinite(prepared.code.lower_bound_error) and
                prepared.code.lower_bound_error >= 0.0F,
            "invalid prepared encoding metadata");
        for (float value : prepared.query) {
            codec_require(std::isfinite(value), "non-finite prepared query");
        }
        const uint64_t slot = found->second;
        const float query_norm = prepared.code.norm;
        const auto& query = prepared.query;
        const auto neighbors = nearest(query, query_norm, slot, max_degree_);
        const auto old_neighbors = adjacency_[slot];
        if (incoming_protection_ == IncomingProtection::CACHED) {
            prepare_incoming_counts();
        }
        std::vector<uint64_t> affected_nodes;
        affected_nodes.reserve(old_neighbors.size());
        const auto scan_start = measure_mutation_scans_ ? Clock::now() : Clock::time_point{};
        const double scan_cpu_start_us =
            measure_mutation_scans_ ? (cpu_clock_ == nullptr ? 0.0 : cpu_clock_()) : 0.0;
        if (use_incoming_adjacency_) {
            affected_nodes.assign(incoming_[slot].begin(), incoming_[slot].end());
            remember_incoming_row(slot);
            for (const uint64_t source : affected_nodes) {
                remember_row(source);
                erase_value(adjacency_[source], slot);
                if (incoming_counts_valid_) {
                    --incoming_counts_[slot];
                }
            }
            incoming_[slot].clear();
        } else {
            for (uint64_t source = 0; source < adjacency_.size(); ++source) {
                if (source == slot) {
                    continue;
                }
                auto& reverse = adjacency_[source];
                if (std::find(reverse.begin(), reverse.end(), slot) != reverse.end()) {
                    remember_row(source);
                    reverse.erase(std::remove(reverse.begin(), reverse.end(), slot), reverse.end());
                    if (incoming_counts_valid_) {
                        --incoming_counts_[slot];
                    }
                    affected_nodes.push_back(source);
                }
            }
        }
        if (measure_mutation_scans_) {
            mutation_scan_timing_.update_wall_us = microseconds(scan_start, Clock::now());
            mutation_scan_timing_.update_cpu_us =
                (cpu_clock_ == nullptr ? 0.0 : cpu_clock_()) - scan_cpu_start_us;
        }
        codes_.Replace(slot, std::move(prepared.code));
        replace_neighbors(slot, neighbors);
        for (uint64_t neighbor : neighbors) {
            link(neighbor, slot);
        }
        repair(std::move(affected_nodes), old_neighbors);
        if (incoming_protection_ != IncomingProtection::NONE) {
            auto targets = old_neighbors;
            targets.push_back(slot);
            protect_incoming(targets);
        }
        return true;
    }

    [[nodiscard]] bool
    RemoveTransactional(int64_t id) {
        const auto found = slots_.find(id);
        if (found == slots_.end()) {
            return false;
        }
        codec_require(active_journal_ == nullptr, "nested graph remove transaction");
        if (use_incoming_adjacency_ and incoming_protection_ == IncomingProtection::CACHED and
            not incoming_counts_valid_) {
            auto next = *this;
            const bool removed = next.Remove(id);
            *this = std::move(next);
            return removed;
        }
        const uint64_t count = Size();
        const uint64_t slot = found->second;
        const uint64_t last = count - 1;
        const int64_t last_id = ids_[last];
        auto saved_slot = backup_code(slot);
        auto saved_last = backup_code(last);
        UpdateJournal journal;
        journal.rows.emplace(slot, adjacency_[slot]);
        if (slot != last) {
            journal.rows.emplace(last, adjacency_[last]);
        }
        const bool old_counts_valid = incoming_counts_valid_;
        const auto old_fallbacks = mutation_fallbacks_;
        const auto old_timing = mutation_scan_timing_;
        active_journal_ = &journal;
        try {
            const bool removed = Remove(id);
            active_journal_ = nullptr;
            return removed;
        } catch (...) {
            incoming_counts_valid_ = use_incoming_adjacency_ and old_counts_valid;
            active_journal_ = nullptr;
            // Remove only shrinks these containers; original capacity remains.
            codes_.Resize(count);
            codes_.Replace(slot, std::move(saved_slot));
            if (slot != last) {
                codes_.Replace(last, std::move(saved_last));
            }
            ids_.resize(count);
            ids_[slot] = id;
            ids_[last] = last_id;
            adjacency_.resize(count);
            for (auto& row : journal.rows) {
                adjacency_[row.first].swap(row.second);
            }
            if (use_incoming_adjacency_) {
                incoming_.resize(count);
                if (old_counts_valid) {
                    incoming_counts_.resize(count);
                }
                for (auto& row : journal.incoming_rows) {
                    incoming_[row.first].swap(row.second);
                    if (old_counts_valid) {
                        incoming_counts_[row.first] = incoming_[row.first].size();
                    }
                }
            }
            if (slot != last) {
                slots_.at(last_id) = last;
            }
            if (not journal.removed_node.empty()) {
                // Same allocator; restored size cannot exceed the old load.
                // No other map insertion/rehash occurs during Remove.
                slots_.insert(std::move(journal.removed_node));
            }
            mutation_fallbacks_ = old_fallbacks;
            mutation_scan_timing_ = old_timing;
            throw;
        }
    }

    template <bool skip_unchanged = true>
    [[nodiscard]] bool
    Remove(int64_t id) {
        const auto found = slots_.find(id);
        if (found == slots_.end()) {
            return false;
        }
        try {
            const uint64_t slot = found->second;
            const uint64_t last = Size() - 1;
            const auto removed_neighbors = adjacency_[slot];
            std::vector<uint64_t> affected_nodes = removed_neighbors;
            // Snapshot reverse rows before any count delta can throw later.
            if (use_incoming_adjacency_) {
                remember_incoming_row(slot);
                remember_incoming_row(last);
                for (const uint64_t target : removed_neighbors) {
                    remember_incoming_row(target);
                }
            }
            if (incoming_counts_valid_) {
                for (const uint64_t target : removed_neighbors) {
                    --incoming_counts_[target];
                }
                incoming_counts_[slot] = 0;
            }
            const auto scan_start = measure_mutation_scans_ ? Clock::now() : Clock::time_point{};
            const double scan_cpu_start_us =
                measure_mutation_scans_ ? (cpu_clock_ == nullptr ? 0.0 : cpu_clock_()) : 0.0;
            if (use_incoming_adjacency_) {
                const auto removed_incoming = incoming_[slot];
                affected_nodes.insert(
                    affected_nodes.end(), removed_incoming.begin(), removed_incoming.end());
                for (const uint64_t target : removed_neighbors) {
                    erase_value(incoming_[target], slot);
                }
                for (const uint64_t source : removed_incoming) {
                    remember_row(source);
                    erase_value(adjacency_[source], slot);
                }
                incoming_[slot].clear();
                if (slot != last) {
                    for (const uint64_t target : adjacency_[last]) {
                        remember_incoming_row(target);
                        replace_value(incoming_[target], last, slot);
                    }
                    for (const uint64_t source : incoming_[last]) {
                        remember_row(source);
                        replace_value(adjacency_[source], last, slot);
                    }
                }
            } else {
                for (uint64_t source = 0; source < adjacency_.size(); ++source) {
                    auto& neighbors = adjacency_[source];
                    if constexpr (skip_unchanged) {
                        const auto changed = std::find_if(
                            neighbors.begin(), neighbors.end(), [slot, last](uint64_t neighbor) {
                                return neighbor == slot or neighbor == last;
                            });
                        if (changed == neighbors.end()) {
                            continue;
                        }
                        remember_row(source);
                    } else {
                        if (std::find(neighbors.begin(), neighbors.end(), slot) !=
                                neighbors.end() or
                            std::find(neighbors.begin(), neighbors.end(), last) !=
                                neighbors.end()) {
                            remember_row(source);
                        }
                    }
                    const auto old_size = neighbors.size();
                    neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), slot),
                                    neighbors.end());
                    if (neighbors.size() != old_size) {
                        affected_nodes.push_back(source);
                    }
                    for (uint64_t& neighbor : neighbors) {
                        if (neighbor == last) {
                            neighbor = slot;
                        }
                    }
                }
            }
            if (measure_mutation_scans_) {
                mutation_scan_timing_.remove_wall_us = microseconds(scan_start, Clock::now());
                mutation_scan_timing_.remove_cpu_us =
                    (cpu_clock_ == nullptr ? 0.0 : cpu_clock_()) - scan_cpu_start_us;
            }
            if (slot != last) {
                ids_[slot] = ids_[last];
                adjacency_[slot] = std::move(adjacency_[last]);
                if (use_incoming_adjacency_) {
                    incoming_[slot] = std::move(incoming_[last]);
                }
                adjacency_[slot].erase(
                    std::remove(adjacency_[slot].begin(), adjacency_[slot].end(), slot),
                    adjacency_[slot].end());
                slots_.at(ids_[slot]) = slot;
            }
            if (active_journal_ == nullptr) {
                slots_.erase(found);
            } else {
                active_journal_->removed_node = slots_.extract(found);
            }
            ids_.pop_back();
            adjacency_.pop_back();
            if (use_incoming_adjacency_) {
                incoming_.pop_back();
            }
            codes_.RemoveSwap(slot);
            if (incoming_counts_valid_) {
                if (slot != last) {
                    incoming_counts_[slot] = incoming_counts_[last];
                }
                incoming_counts_.pop_back();
            }
            auto remap = [slot, last](uint64_t node) { return node == last ? slot : node; };
            for (auto& node : affected_nodes) {
                node = remap(node);
            }
            std::vector<uint64_t> repair_candidates;
            repair_candidates.reserve(removed_neighbors.size());
            for (const uint64_t old_candidate : removed_neighbors) {
                repair_candidates.push_back(remap(old_candidate));
            }
            if (incoming_protection_ != IncomingProtection::NONE and protect_removed_incoming_) {
                std::sort(affected_nodes.begin(), affected_nodes.end());
                affected_nodes.erase(std::unique(affected_nodes.begin(), affected_nodes.end()),
                                     affected_nodes.end());
                const auto targets = affected_nodes;
                repair(std::move(affected_nodes), repair_candidates);
                protect_incoming(targets);
            } else {
                repair(std::move(affected_nodes), repair_candidates);
            }
            return true;
        } catch (...) {
            incoming_counts_valid_ = false;
            throw;
        }
    }

    [[nodiscard]] GraphSearchResult
    Search(const float* query, uint64_t k) const {
        codec_require(query != nullptr, "null mutable graph query");
        float query_norm = 0.0F;
        const auto normalized = normalize(model_, query, query_norm);
        return graph_search_adjacency(normalized, query_norm, codes_, adjacency_, k, ef_search_);
    }

    // Per-call budget and external-ID filtering for the future Backend adapter.
    // Does not change construction, mutation or persisted defaults.
    [[nodiscard]] GraphSearchResult
    SearchWithOptions(const float* query,
                      uint64_t k,
                      uint64_t budget,
                      const std::function<bool(int64_t)>& filter = {}) const {
        codec_require(query != nullptr, "null mutable graph query");
        float query_norm = 0.0F;
        const auto normalized = normalize(model_, query, query_norm);
        const uint64_t ef = budget == 0 ? ef_search_ : budget;
        return graph_search_adjacency(
            normalized, query_norm, codes_, adjacency_, k, ef, &ids_, filter);
    }

    void
    Validate() const {
        if (incoming_counts_valid_) {
            std::vector<uint64_t> actual;
            count_incoming(actual);
            codec_require(incoming_counts_ == actual, "cached incoming degree mismatch");
        }
        codec_require(
            ids_.size() == Size() and adjacency_.size() == Size() and slots_.size() == Size(),
            "mutable graph size mismatch");
        if (use_incoming_adjacency_) {
            validate_incoming_adjacency(adjacency_, incoming_);
        }
        for (uint64_t slot = 0; slot < Size(); ++slot) {
            const auto found = slots_.find(ids_[slot]);
            codec_require(found != slots_.end() and found->second == slot,
                          "mutable graph ID map mismatch");
            codec_require(adjacency_[slot].size() <= max_degree_,
                          "mutable graph degree exceeds limit");
            for (uint64_t i = 0; i < adjacency_[slot].size(); ++i) {
                const uint64_t neighbor = adjacency_[slot][i];
                codec_require(neighbor < Size() and neighbor != slot,
                              "invalid mutable graph neighbor");
                codec_require(
                    std::find(adjacency_[slot].begin(),
                              adjacency_[slot].begin() + static_cast<int64_t>(i),
                              neighbor) == adjacency_[slot].begin() + static_cast<int64_t>(i),
                    "duplicate mutable graph neighbor");
            }
        }
    }

    [[nodiscard]] const Model&
    GetModel() const {
        return model_;
    }

    [[nodiscard]] const EncodedRecords&
    GetCodes() const {
        return codes_;
    }

    [[nodiscard]] GraphTopology
    GetGraph() const {
        return compact_graph(adjacency_);
    }

    [[nodiscard]] const std::vector<std::vector<uint64_t>>&
    GetAdjacency() const {
        return adjacency_;
    }

    [[nodiscard]] const std::vector<int64_t>&
    GetIds() const {
        return ids_;
    }

    [[nodiscard]] std::vector<std::vector<IncomingSlot>>
    BuildIncomingAdjacency() const {
        std::vector<uint64_t> counts(Size());
        for (const auto& neighbors : adjacency_) {
            for (const uint64_t target : neighbors) {
                codec_require(target < Size(), "invalid incoming graph neighbor");
                ++counts[target];
            }
        }
        if (Size() > 0) {
            (void)checked_incoming_slot(Size() - 1);
        }
        std::vector<std::vector<IncomingSlot>> incoming(Size());
        for (uint64_t target = 0; target < Size(); ++target) {
            incoming[target].reserve(counts[target]);
        }
        for (uint64_t source = 0; source < Size(); ++source) {
            for (const uint64_t target : adjacency_[source]) {
                incoming[target].push_back(checked_incoming_slot(source));
            }
        }
        return incoming;
    }

    [[nodiscard]] IncomingMemoryUsage
    GetIncomingMemoryUsage() const {
        IncomingMemoryUsage result;
        result.logical_bytes = incoming_.size() * sizeof(std::vector<IncomingSlot>);
        result.capacity_bytes = incoming_.capacity() * sizeof(std::vector<IncomingSlot>);
        for (const auto& sources : incoming_) {
            result.edges += sources.size();
            result.logical_bytes += sources.size() * sizeof(IncomingSlot);
            result.capacity_bytes += sources.capacity() * sizeof(IncomingSlot);
        }
        return result;
    }

    void
    CompactIncoming() {
        codec_require(use_incoming_adjacency_, "incoming adjacency is not enabled");
        for (auto& sources : incoming_) {
            sources.shrink_to_fit();
        }
        incoming_.shrink_to_fit();
    }

    [[nodiscard]] IncomingDistribution
    GetIncomingDistribution() const {
        codec_require(use_incoming_adjacency_, "incoming adjacency is not enabled");
        IncomingDistribution result;
        result.nodes = incoming_.size();
        result.outer_vector_bytes = incoming_.capacity() * sizeof(std::vector<IncomingSlot>);
        std::vector<uint64_t> degrees;
        std::vector<uint64_t> capacities;
        degrees.reserve(incoming_.size());
        capacities.reserve(incoming_.size());
        for (const auto& sources : incoming_) {
            const uint64_t degree = sources.size();
            const uint64_t capacity = sources.capacity();
            degrees.push_back(degree);
            capacities.push_back(capacity);
            result.zero_degree_nodes += degree == 0 ? 1 : 0;
            result.degree_le_4 += degree <= 4 ? 1 : 0;
            result.degree_le_8 += degree <= 8 ? 1 : 0;
            result.degree_le_16 += degree <= 16 ? 1 : 0;
            result.degree_le_32 += degree <= 32 ? 1 : 0;
            result.degree_le_64 += degree <= 64 ? 1 : 0;
            result.degree_gt_64 += degree > 64 ? 1 : 0;
            result.slack_nodes += capacity > degree ? 1 : 0;
            result.slack_entries += capacity - degree;
            result.edge_logical_bytes += degree * sizeof(IncomingSlot);
            result.edge_capacity_bytes += capacity * sizeof(IncomingSlot);
        }
        if (degrees.empty()) {
            return result;
        }
        std::sort(degrees.begin(), degrees.end());
        std::sort(capacities.begin(), capacities.end());
        auto quantile =
            [](const std::vector<uint64_t>& values, uint64_t numerator, uint64_t denominator) {
                return values[(values.size() - 1) * numerator / denominator];
            };
        result.degree_p50 = quantile(degrees, 50, 100);
        result.degree_p90 = quantile(degrees, 90, 100);
        result.degree_p95 = quantile(degrees, 95, 100);
        result.degree_p99 = quantile(degrees, 99, 100);
        result.degree_max = degrees.back();
        result.capacity_p50 = quantile(capacities, 50, 100);
        result.capacity_p90 = quantile(capacities, 90, 100);
        result.capacity_p95 = quantile(capacities, 95, 100);
        result.capacity_p99 = quantile(capacities, 99, 100);
        result.capacity_max = capacities.back();
        return result;
    }

    [[nodiscard]] uint64_t
    LinkCountAt(uint64_t slot) const {
        return adjacency_.at(slot).size();
    }

    [[nodiscard]] uint64_t
    LinkAt(uint64_t slot, uint64_t edge) const {
        return adjacency_.at(slot).at(edge);
    }

    [[nodiscard]] int64_t
    IdAt(uint64_t slot) const {
        codec_require(slot < ids_.size(), "mutable graph ID outside storage");
        return ids_[slot];
    }

    [[nodiscard]] uint64_t
    GetMaxDegree() const {
        return max_degree_;
    }

    [[nodiscard]] uint64_t
    GetEfSearch() const {
        return ef_search_;
    }

    [[nodiscard]] uint64_t
    GetMutationFallbacks() const {
        return mutation_fallbacks_;
    }

    [[nodiscard]] const MutationScanTiming&
    GetMutationScanTiming() const {
        return mutation_scan_timing_;
    }

    [[nodiscard]] MutableMemoryUsage
    GetMemoryUsage() const {
        MutableMemoryUsage result;
        result.model_logical_bytes =
            model_.centroid.size() * sizeof(float) + model_.flips.size() * sizeof(uint8_t);
        result.model_capacity_bytes =
            model_.centroid.capacity() * sizeof(float) + model_.flips.capacity() * sizeof(uint8_t);
        result.codes_logical_bytes = codes_.metadata.size() * sizeof(EncodedMetadata) +
                                     codes_.filters.size() * sizeof(uint8_t) +
                                     codes_.supplements.size() * sizeof(uint8_t);
        result.codes_capacity_bytes = codes_.metadata.capacity() * sizeof(EncodedMetadata) +
                                      codes_.filters.capacity() * sizeof(uint8_t) +
                                      codes_.supplements.capacity() * sizeof(uint8_t);
        result.ids_logical_bytes = ids_.size() * sizeof(int64_t);
        result.ids_capacity_bytes = ids_.capacity() * sizeof(int64_t);
        result.adjacency_logical_bytes = adjacency_.size() * sizeof(std::vector<uint64_t>);
        result.adjacency_capacity_bytes = adjacency_.capacity() * sizeof(std::vector<uint64_t>);
        for (const auto& neighbors : adjacency_) {
            result.adjacency_edges += neighbors.size();
            result.adjacency_logical_bytes += neighbors.size() * sizeof(uint64_t);
            result.adjacency_capacity_bytes += neighbors.capacity() * sizeof(uint64_t);
        }
        const auto incoming_memory = GetIncomingMemoryUsage();
        result.incoming_adjacency_logical_bytes = incoming_memory.logical_bytes;
        result.incoming_adjacency_capacity_bytes = incoming_memory.capacity_bytes;
        result.incoming_count_logical_bytes = incoming_counts_.size() * sizeof(uint64_t);
        result.incoming_count_capacity_bytes = IncomingCountBytes();
        result.slot_entries = slots_.size();
        result.slot_buckets = slots_.bucket_count();
        return result;
    }

private:
    struct UpdateJournal {
        EncodedMetadata metadata;
        std::vector<uint8_t> filter;
        std::vector<uint8_t> supplement;
        std::unordered_map<uint64_t, std::vector<uint64_t>> rows;
        std::unordered_map<uint64_t, std::vector<IncomingSlot>> incoming_rows;
        uint64_t row_limit = std::numeric_limits<uint64_t>::max();
        std::unordered_map<int64_t, uint64_t>::node_type removed_node;
    };

    void
    remember_row(uint64_t source) {
        if (active_journal_ != nullptr and source < active_journal_->row_limit and
            active_journal_->rows.count(source) == 0) {
            active_journal_->rows.emplace(source, adjacency_[source]);
        }
    }

    void
    remember_incoming_row(uint64_t target) {
        if (active_journal_ != nullptr and use_incoming_adjacency_ and
            target < active_journal_->row_limit and
            active_journal_->incoming_rows.count(target) == 0) {
            active_journal_->incoming_rows.emplace(target, incoming_[target]);
        }
    }

    [[nodiscard]] Encoded
    backup_code(uint64_t slot) const {
        const auto view = codes_.At(slot);
        Encoded code;
        code.norm = view.metadata.norm;
        code.code_norm = view.metadata.code_norm;
        code.error = view.metadata.error;
        code.filter_norm = view.metadata.filter_norm;
        code.filter_error = view.metadata.filter_error;
        code.lower_bound_error = view.metadata.lower_bound_error;
        code.filter.assign(view.filter, view.filter + codes_.FilterBytes());
        code.supplement.assign(view.supplement, view.supplement + codes_.SupplementBytes());
        return code;
    }

    template <typename T>
    static void
    grow(std::vector<T>& values, uint64_t required) {
        if (required > values.capacity()) {
            const uint64_t maximum = values.max_size();
            const uint64_t capacity = values.capacity();
            const uint64_t doubled =
                capacity == 0 ? 1 : (capacity > maximum / 2 ? maximum : capacity * 2);
            values.reserve(std::max(required, doubled));
        }
    }

    template <typename T>
    static void
    erase_value(std::vector<T>& values, uint64_t value) {
        values.erase(std::remove(values.begin(), values.end(), value), values.end());
    }

    template <typename T>
    static void
    replace_value(std::vector<T>& values, uint64_t old_value, uint64_t new_value) {
        codec_require(new_value <= std::numeric_limits<T>::max(), "neighbor slot overflow");
        for (T& value : values) {
            if (value == old_value) {
                value = static_cast<T>(new_value);
            }
        }
    }

    void
    sync_incoming(uint64_t source, const std::vector<uint64_t>& old_neighbors) {
        if (use_incoming_adjacency_ and active_journal_ != nullptr) {
            const auto& neighbors = adjacency_[source];
            for (const uint64_t target : old_neighbors) {
                if (std::find(neighbors.begin(), neighbors.end(), target) == neighbors.end()) {
                    remember_incoming_row(target);
                }
            }
            for (const uint64_t target : neighbors) {
                if (std::find(old_neighbors.begin(), old_neighbors.end(), target) ==
                    old_neighbors.end()) {
                    remember_incoming_row(target);
                }
            }
        }
        if (incoming_counts_valid_) {
            const auto& neighbors = adjacency_[source];
            for (const uint64_t target : old_neighbors) {
                if (std::find(neighbors.begin(), neighbors.end(), target) == neighbors.end()) {
                    --incoming_counts_[target];
                }
            }
            for (const uint64_t target : neighbors) {
                if (std::find(old_neighbors.begin(), old_neighbors.end(), target) ==
                    old_neighbors.end()) {
                    ++incoming_counts_[target];
                }
            }
        }
        if (not use_incoming_adjacency_) {
            return;
        }
        const auto& neighbors = adjacency_[source];
        for (const uint64_t target : old_neighbors) {
            if (std::find(neighbors.begin(), neighbors.end(), target) == neighbors.end()) {
                erase_value(incoming_[target], source);
            }
        }
        for (const uint64_t target : neighbors) {
            if (std::find(old_neighbors.begin(), old_neighbors.end(), target) ==
                old_neighbors.end()) {
                incoming_[target].push_back(checked_incoming_slot(source));
            }
        }
    }

    void
    replace_neighbors(uint64_t source, const std::vector<uint64_t>& neighbors) {
        remember_row(source);
        const auto old_neighbors = adjacency_[source];
        adjacency_[source] = neighbors;
        sync_incoming(source, old_neighbors);
    }

    [[nodiscard]] std::vector<float>
    decode_query(uint64_t slot) const {
        const auto code = codes_.At(slot);
        const uint64_t plane_bytes = (model_.dim + 7) / 8;
        std::vector<float> result(model_.dim);
        for (uint64_t d = 0; d < model_.dim; ++d) {
            const uint32_t high = read_plane_code(code.filter, plane_bytes, d, K_FILTER_BITS, true);
            const uint32_t low =
                read_plane_code(code.supplement, plane_bytes, d, K_SUPPLEMENT_BITS, false);
            result[d] = (static_cast<float>((high << K_SUPPLEMENT_BITS) + low) - 127.5F) /
                        code.metadata.code_norm;
        }
        return result;
    }

    [[nodiscard]] std::vector<uint64_t>
    nearest_exhaustive(const std::vector<float>& query,
                       float query_norm,
                       uint64_t excluded,
                       uint64_t count) const {
        const float query_sum = std::accumulate(query.begin(), query.end(), 0.0F);
        std::priority_queue<Candidate, std::vector<Candidate>, Better> heap;
        for (uint64_t slot = 0; slot < Size(); ++slot) {
            if (slot == excluded) {
                continue;
            }
            const auto code = codes_.At(slot);
            const auto coarse = filter_estimate(query, query_norm, code);
            const Candidate candidate{
                slot, full_distance(query, query_norm, code, coarse.centered_ip, query_sum)};
            if (heap.size() < count) {
                heap.push(candidate);
            } else if (better(candidate, heap.top())) {
                heap.pop();
                heap.push(candidate);
            }
        }
        std::vector<uint64_t> result(heap.size());
        for (uint64_t i = result.size(); i > 0; --i) {
            result[i - 1] = heap.top().id;
            heap.pop();
        }
        return result;
    }

    [[nodiscard]] std::vector<uint64_t>
    nearest(const std::vector<float>& query, float query_norm, uint64_t excluded, uint64_t count) {
        const uint64_t available = Size() - static_cast<uint64_t>(excluded < Size());
        const uint64_t desired = std::min(count, available);
        if (desired == 0) {
            return {};
        }
        const uint64_t requested =
            std::min(Size(), desired + static_cast<uint64_t>(excluded < Size()));
        const auto found = graph_search_adjacency(
            query, query_norm, codes_, adjacency_, requested, std::max(ef_search_, count * 4));
        std::vector<uint64_t> result;
        result.reserve(desired);
        for (const auto& candidate : found.neighbors) {
            if (candidate.id != excluded) {
                result.push_back(candidate.id);
                if (result.size() == desired) {
                    return result;
                }
            }
        }
        ++mutation_fallbacks_;
        return nearest_exhaustive(query, query_norm, excluded, count);
    }

    void
    count_incoming(std::vector<uint64_t>& counts) const {
        counts.assign(Size(), 0);
        for (const auto& row : adjacency_) {
            for (const uint64_t target : row) {
                ++counts[target];
            }
        }
    }

    void
    prepare_incoming_counts() {
        if (incoming_counts_valid_) {
            return;
        }
        count_incoming(incoming_counts_);
        incoming_counts_valid_ = true;
        ++incoming_count_rebuilds_;
    }

    // Experimental adaptation of floating GraphBackend::ensure_incoming.
    // Recount reference or cached degrees; neither adds a reverse-edge index.
    void
    protect_incoming(const std::vector<uint64_t>& targets) {
        if (Size() <= 1) {
            return;
        }
        std::vector<uint64_t> recounted;
        if (incoming_protection_ == IncomingProtection::CACHED) {
            prepare_incoming_counts();
        } else {
            count_incoming(recounted);
        }
        auto& indegrees =
            incoming_protection_ == IncomingProtection::CACHED ? incoming_counts_ : recounted;
        for (const uint64_t target : targets) {
            if (target >= Size() or indegrees[target] != 0) {
                continue;
            }
            uint64_t selected_source = Size();
            uint64_t selected_edge = 0;
            float farthest_distance = -1.0F;
            for (const uint64_t source : adjacency_[target]) {
                const auto query = decode_query(source);
                const float norm = codes_.At(source).metadata.norm;
                const float sum = std::accumulate(query.begin(), query.end(), 0.0F);
                for (uint64_t edge = 0; edge < adjacency_[source].size(); ++edge) {
                    const uint64_t displaced = adjacency_[source][edge];
                    if (indegrees[displaced] <= 1) {
                        continue;
                    }
                    const auto code = codes_.At(displaced);
                    const auto coarse = filter_estimate(query, norm, code);
                    const float distance =
                        full_distance(query, norm, code, coarse.centered_ip, sum);
                    if (selected_source == Size() or distance > farthest_distance) {
                        selected_source = source;
                        selected_edge = edge;
                        farthest_distance = distance;
                    }
                }
            }
            if (selected_source == Size()) {
                continue;
            }
            remember_row(selected_source);
            const auto old_neighbors = adjacency_[selected_source];
            const uint64_t displaced = adjacency_[selected_source][selected_edge];
            adjacency_[selected_source][selected_edge] = target;
            if (incoming_protection_ != IncomingProtection::CACHED) {
                --indegrees[displaced];
                ++indegrees[target];
            }
            sync_incoming(selected_source, old_neighbors);
        }
    }

    void
    repair(std::vector<uint64_t> affected_nodes,
           const std::vector<uint64_t>& additional_candidates) {
        std::sort(affected_nodes.begin(), affected_nodes.end());
        affected_nodes.erase(std::unique(affected_nodes.begin(), affected_nodes.end()),
                             affected_nodes.end());
        for (const uint64_t source : affected_nodes) {
            if (source >= Size()) {
                continue;
            }
            // Preserve valid links to limit topology drift; rank only candidates that can
            // refill the degree lost by the mutation.
            auto& repaired = adjacency_[source];
            if (repaired.size() >= max_degree_) {
                continue;
            }
            remember_row(source);
            const auto old_neighbors = repaired;
            const auto query = decode_query(source);
            const float query_norm = codes_.At(source).metadata.norm;
            const float query_sum = std::accumulate(query.begin(), query.end(), 0.0F);
            std::vector<Candidate> ranked;
            ranked.reserve(additional_candidates.size());
            for (const uint64_t candidate : additional_candidates) {
                if (candidate < Size() and candidate != source and
                    std::find(repaired.begin(), repaired.end(), candidate) == repaired.end()) {
                    ranked.push_back({candidate, 0.0F});
                }
            }
            score_mutation_candidates(query, query_norm, query_sum, codes_, ranked);
            std::sort(ranked.begin(), ranked.end(), better);
            const uint64_t needed = max_degree_ - repaired.size();
            const uint64_t retained = std::min(needed, ranked.size());
            repaired.reserve(repaired.size() + retained);
            if (ranked.size() > retained and retained > 0) {
                uint64_t chosen = 0;
                for (const auto& candidate : ranked) {
                    if (chosen == retained) {
                        break;
                    }
                    const auto code = codes_.At(candidate.id);
                    bool diverse = true;
                    for (const uint64_t selected : repaired) {
                        if (one_bit_code_distance(codes_.At(selected), code, model_.dim) <
                            candidate.distance) {
                            diverse = false;
                            break;
                        }
                    }
                    if (diverse) {
                        repaired.push_back(candidate.id);
                        ++chosen;
                    }
                }
                // Preserve the degree-fill contract if every remaining candidate is occluded.
                for (const auto& candidate : ranked) {
                    if (chosen == retained) {
                        break;
                    }
                    if (std::find(repaired.begin(), repaired.end(), candidate.id) ==
                        repaired.end()) {
                        repaired.push_back(candidate.id);
                        ++chosen;
                    }
                }
            } else {
                for (uint64_t i = 0; i < retained; ++i) {
                    repaired.push_back(ranked[i].id);
                }
            }
            sync_incoming(source, old_neighbors);
        }
    }

    void
    link(uint64_t source, uint64_t target, std::vector<uint64_t>* displaced_targets = nullptr) {
        auto& neighbors = adjacency_[source];
        if (source == target or
            std::find(neighbors.begin(), neighbors.end(), target) != neighbors.end()) {
            return;
        }
        remember_row(source);
        const auto old_neighbors = neighbors;
        if (protect_link_incoming_) {
            prepare_incoming_counts();
        }
        neighbors.push_back(target);
        if (neighbors.size() > max_degree_) {
            const auto query = decode_query(source);
            const float query_norm = codes_.At(source).metadata.norm;
            const float query_sum = std::accumulate(query.begin(), query.end(), 0.0F);
            std::vector<Candidate> ranked;
            ranked.reserve(neighbors.size());
            for (uint64_t neighbor : neighbors) {
                ranked.push_back({neighbor, 0.0F});
            }
            score_mutation_candidates(query, query_norm, query_sum, codes_, ranked);
            std::sort(ranked.begin(), ranked.end(), better);
            const uint64_t dropped = protect_link_incoming_
                                         ? safe_incoming_drop(ranked, incoming_counts_, target)
                                         : max_degree_;
            for (uint64_t i = 0, destination = 0; i < ranked.size(); ++i) {
                if (i != dropped) {
                    neighbors[destination++] = ranked[i].id;
                }
            }
            neighbors.resize(max_degree_);
        }
        sync_incoming(source, old_neighbors);
        if (displaced_targets != nullptr) {
            for (const uint64_t old_target : old_neighbors) {
                if (std::find(neighbors.begin(), neighbors.end(), old_target) == neighbors.end()) {
                    displaced_targets->push_back(old_target);
                }
            }
        }
    }

    IncomingProtection incoming_protection_{IncomingProtection::NONE};
    bool protect_added_incoming_{};
    bool protect_removed_incoming_{};
    bool protect_added_displaced_{};
    bool protect_link_incoming_{};
    std::vector<uint64_t> incoming_counts_;
    bool incoming_counts_valid_{};
    uint64_t incoming_count_rebuilds_{};
    UpdateJournal* active_journal_{};
    Model model_;
    EncodedRecords codes_;
    std::vector<int64_t> ids_;
    std::unordered_map<int64_t, uint64_t> slots_;
    std::vector<std::vector<uint64_t>> adjacency_;
    std::vector<std::vector<IncomingSlot>> incoming_;
    uint64_t max_degree_;
    uint64_t ef_search_;
    uint64_t mutation_fallbacks_{};
    bool measure_mutation_scans_{};
    bool use_incoming_adjacency_{};
    MutationScanTiming mutation_scan_timing_{};
    double (*cpu_clock_)(){};
};

}  // namespace vsag::lite::detail::rabitq
