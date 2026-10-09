// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <chrono>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>

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

inline float
supplement_ip(const std::vector<float>& query, const uint8_t* supplement) {
    const uint64_t plane_bytes = (query.size() + 7) / 8;
    float result = 0.0F;
    for (uint64_t d = 0; d < query.size(); ++d) {
        const auto code = read_plane_code(supplement, plane_bytes, d, K_SUPPLEMENT_BITS, false);
        result += query[d] * static_cast<float>(code);
    }
    return result;
}

inline float
l2_distance(float base_norm, float query_norm, float normalized_ip) {
    return base_norm * base_norm + query_norm * query_norm -
           2.0F * base_norm * query_norm * normalized_ip;
}

struct FilterEstimate {
    float distance;
    float lower_bound;
    float centered_ip;
};

inline FilterEstimate
filter_estimate(const std::vector<float>& query, float query_norm, const EncodedView& code) {
    const float centered_ip = filter_centered_ip(query, code.filter);
    const float normalized_ip =
        centered_ip / code.metadata.filter_norm / code.metadata.filter_error;
    const float distance = l2_distance(code.metadata.norm, query_norm, normalized_ip);
    const float error = 2.0F * code.metadata.norm * query_norm * K_ERROR_RATE *
                        code.metadata.lower_bound_error / code.metadata.filter_error;
    const float estimate = distance - error;
    const float lower_bound = estimate - 1e-5F * std::max(1.0F, std::fabs(estimate));
    return {distance, lower_bound, centered_ip};
}

inline float
full_distance(const std::vector<float>& query,
              float query_norm,
              const EncodedView& code,
              float centered_filter_ip) {
    const float query_sum = std::accumulate(query.begin(), query.end(), 0.0F);
    const float filter_ip = centered_filter_ip + 3.5F * query_sum;
    const float code_ip = filter_ip * static_cast<float>(1U << K_SUPPLEMENT_BITS) +
                          supplement_ip(query, code.supplement);
    const float base_error = std::fabs(code.metadata.error) < 1e-5F ? 1.0F : code.metadata.error;
    const float normalized_ip =
        (code_ip - 127.5F * query_sum) / code.metadata.code_norm / base_error;
    return l2_distance(code.metadata.norm, query_norm, normalized_ip);
}

struct Candidate {
    uint64_t id;
    float distance;
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

template <typename Distance>
inline std::vector<Candidate>
top_k(uint64_t count, uint64_t k, Distance distance) {
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(&better)> heap(&better);
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

inline SearchResult
filtered_search(const std::vector<float>& query,
                float query_norm,
                const EncodedRecords& codes,
                uint64_t k) {
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(&better)> heap(&better);
    uint64_t reordered = 0;
    for (uint64_t id = 0; id < codes.Size(); ++id) {
        const auto code = codes.At(id);
        const auto coarse = filter_estimate(query, query_norm, code);
        if (heap.size() == k and coarse.lower_bound >= heap.top().distance) {
            continue;
        }
        ++reordered;
        const Candidate next{id, full_distance(query, query_norm, code, coarse.centered_ip)};
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

template <typename NeighborRange>
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
    const uint64_t ef = std::min(codes.Size(), std::max(k, ef_search));
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(&better)> best(&better);
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(&farther)> candidates(&farther);
    std::vector<uint8_t> visited(codes.Size(), 0);
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

    auto visit = [&](uint64_t slot) {
        if (visited[slot] != 0) {
            return;
        }
        visited[slot] = 1;
        ++visited_count;
        const auto estimate = filter_estimate(query, query_norm, codes.At(slot));
        // Rejected nodes remain traversable. Rank every visited allowed ID using
        // the complete 8-bit estimate; filtering must not sever routing paths.
        if (filter and filter((*external_ids)[slot])) {
            ++filtered_reorders;
            const Candidate value{
                slot, full_distance(query, query_norm, codes.At(slot), estimate.centered_ip)};
            if (accepted.size() < k) {
                accepted.push(value);
            } else if (rank(value, accepted.top())) {
                accepted.pop();
                accepted.push(value);
            }
        }
        const Candidate next{slot, estimate.distance};
        candidates.push(next);
        best.push(next);
        if (best.size() > ef) {
            best.pop();
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
        for (auto edge = range.first; edge != range.second; ++edge) {
            visit(*edge);
        }
    }

    if (filter) {
        std::vector<Candidate> result(accepted.size());
        for (uint64_t i = result.size(); i > 0; --i) {
            result[i - 1] = accepted.top();
            accepted.pop();
        }
        return {std::move(result), visited_count, filtered_reorders};
    }
    const uint64_t reorder_count = best.size();
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(rank)> reordered(rank);
    while (not best.empty()) {
        const uint64_t slot = best.top().id;
        best.pop();
        const auto code = codes.At(slot);
        const auto estimate = filter_estimate(query, query_norm, code);
        const Candidate next{slot, full_distance(query, query_norm, code, estimate.centered_ip)};
        if (reordered.size() < k) {
            reordered.push(next);
        } else if (rank(next, reordered.top())) {
            reordered.pop();
            reordered.push(next);
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
    uint64_t slot_entries{};
    uint64_t slot_buckets{};

    [[nodiscard]] uint64_t
    KnownLogicalBytes() const {
        return model_logical_bytes + codes_logical_bytes + ids_logical_bytes +
               adjacency_logical_bytes;
    }

    [[nodiscard]] uint64_t
    KnownCapacityBytes() const {
        return model_capacity_bytes + codes_capacity_bytes + ids_capacity_bytes +
               adjacency_capacity_bytes;
    }
};

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
        : model_(std::move(input_model)),
          codes_(std::move(input_codes)),
          ids_(std::move(input_ids)),
          adjacency_(expand_graph(input_graph)),
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
        static_assert(sizeof(EncodedMetadata) == 6 * sizeof(float));
        return std::equal(code.filter.begin(), code.filter.end(), stored.filter) and
               std::equal(code.supplement.begin(), code.supplement.end(), stored.supplement) and
               std::memcmp(&metadata, &stored.metadata, sizeof(metadata)) == 0;
    }

    [[nodiscard]] bool
    Add(int64_t id, const float* vector) {
        codec_require(vector != nullptr, "null mutable graph vector");
        if (slots_.count(id) != 0) {
            return false;
        }
        float query_norm = 0.0F;
        const auto query = normalize(model_, vector, query_norm);
        const auto neighbors =
            nearest(query, query_norm, std::numeric_limits<uint64_t>::max(), max_degree_);
        const uint64_t slot = Size();
        codes_.Append(encode(model_, vector));
        ids_.push_back(id);
        adjacency_.emplace_back();
        if (use_incoming_adjacency_) {
            incoming_.emplace_back();
        }
        replace_neighbors(slot, neighbors);
        slots_.emplace(id, slot);
        for (uint64_t neighbor : neighbors) {
            link(neighbor, slot);
        }
        return true;
    }

    [[nodiscard]] bool
    AddTransactional(int64_t id, const float* vector) {
        codec_require(vector != nullptr, "null mutable graph vector");
        if (Contains(id)) {
            return false;
        }
        codec_require(active_journal_ == nullptr, "nested graph add transaction");
        if (use_incoming_adjacency_) {
            auto next = *this;
            const bool added = next.Add(id, vector);
            *this = std::move(next);
            return added;
        }
        const uint64_t count = Size();
        const auto old_fallbacks = mutation_fallbacks_;
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
            inserted = slots_.emplace(id, count).second;
            codec_require(inserted, "duplicate add transaction ID");
            active_journal_ = &journal;
            codes_.Append(std::move(prepared.code));
            ids_.push_back(id);
            adjacency_.emplace_back();
            replace_neighbors(count, neighbors);
            for (uint64_t neighbor : neighbors) {
                link(neighbor, count);
            }
            active_journal_ = nullptr;
            return true;
        } catch (...) {
            active_journal_ = nullptr;
            for (auto& row : journal.rows) {
                adjacency_[row.first].swap(row.second);
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
        return UpdatePrepared(id, prepare_encoding(model_, vector));
    }

    // Update changes no model, IDs, slot mapping or container sizes. Journal
    // only the replaced code and adjacency rows, then restore without allocation.
    [[nodiscard]] bool
    UpdateTransactional(int64_t id, PreparedEncoding prepared) {
        if (not Contains(id)) {
            return false;
        }
        if (use_incoming_adjacency_) {
            auto next = *this;
            const bool updated = next.UpdatePrepared(id, std::move(prepared));
            *this = std::move(next);
            return updated;
        }
        codec_require(active_journal_ == nullptr, "nested graph update transaction");
        const uint64_t slot = slots_.at(id);
        const auto code = codes_.At(slot);
        UpdateJournal journal;
        journal.metadata = code.metadata;
        journal.filter.assign(code.filter, code.filter + codes_.FilterBytes());
        journal.supplement.assign(code.supplement, code.supplement + codes_.SupplementBytes());
        const auto old_fallbacks = mutation_fallbacks_;
        const auto old_timing = mutation_scan_timing_;
        active_journal_ = &journal;
        try {
            const bool updated = UpdatePrepared(id, std::move(prepared));
            active_journal_ = nullptr;
            return updated;
        } catch (...) {
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
        std::vector<uint64_t> affected_nodes;
        affected_nodes.reserve(old_neighbors.size());
        const auto scan_start = measure_mutation_scans_ ? Clock::now() : Clock::time_point{};
        const double scan_cpu_start_us =
            measure_mutation_scans_ ? (cpu_clock_ == nullptr ? 0.0 : cpu_clock_()) : 0.0;
        if (use_incoming_adjacency_) {
            affected_nodes = incoming_[slot];
            for (const uint64_t source : affected_nodes) {
                erase_value(adjacency_[source], slot);
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
        return true;
    }

    [[nodiscard]] bool
    RemoveTransactional(int64_t id) {
        const auto found = slots_.find(id);
        if (found == slots_.end()) {
            return false;
        }
        codec_require(active_journal_ == nullptr, "nested graph remove transaction");
        if (use_incoming_adjacency_) {
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
        const auto old_fallbacks = mutation_fallbacks_;
        const auto old_timing = mutation_scan_timing_;
        active_journal_ = &journal;
        try {
            const bool removed = Remove(id);
            active_journal_ = nullptr;
            return removed;
        } catch (...) {
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

    [[nodiscard]] bool
    Remove(int64_t id) {
        const auto found = slots_.find(id);
        if (found == slots_.end()) {
            return false;
        }
        const uint64_t slot = found->second;
        const uint64_t last = Size() - 1;
        const auto removed_neighbors = adjacency_[slot];
        std::vector<uint64_t> affected_nodes = removed_neighbors;
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
                erase_value(adjacency_[source], slot);
            }
            incoming_[slot].clear();
            if (slot != last) {
                for (const uint64_t target : adjacency_[last]) {
                    replace_value(incoming_[target], last, slot);
                }
                for (const uint64_t source : incoming_[last]) {
                    replace_value(adjacency_[source], last, slot);
                }
            }
        } else {
            for (uint64_t source = 0; source < adjacency_.size(); ++source) {
                auto& neighbors = adjacency_[source];
                if (std::find(neighbors.begin(), neighbors.end(), slot) != neighbors.end() or
                    std::find(neighbors.begin(), neighbors.end(), last) != neighbors.end()) {
                    remember_row(source);
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
        auto remap = [slot, last](uint64_t node) { return node == last ? slot : node; };
        for (auto& node : affected_nodes) {
            node = remap(node);
        }
        std::vector<uint64_t> repair_candidates;
        repair_candidates.reserve(removed_neighbors.size());
        for (const uint64_t old_candidate : removed_neighbors) {
            repair_candidates.push_back(remap(old_candidate));
        }
        repair(std::move(affected_nodes), repair_candidates);
        return true;
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
        codec_require(
            ids_.size() == Size() and adjacency_.size() == Size() and slots_.size() == Size(),
            "mutable graph size mismatch");
        if (use_incoming_adjacency_) {
            codec_require(incoming_.size() == Size(), "mutable incoming size mismatch");
            auto expected_incoming = BuildIncomingAdjacency();
            for (uint64_t target = 0; target < Size(); ++target) {
                auto actual = incoming_[target];
                std::sort(actual.begin(), actual.end());
                std::sort(expected_incoming[target].begin(), expected_incoming[target].end());
                codec_require(actual == expected_incoming[target],
                              "mutable incoming adjacency mismatch");
            }
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

    [[nodiscard]] const std::vector<int64_t>&
    GetIds() const {
        return ids_;
    }

    [[nodiscard]] std::vector<std::vector<uint64_t>>
    BuildIncomingAdjacency() const {
        std::vector<uint64_t> counts(Size());
        for (const auto& neighbors : adjacency_) {
            for (const uint64_t target : neighbors) {
                ++counts[target];
            }
        }
        std::vector<std::vector<uint64_t>> incoming(Size());
        for (uint64_t target = 0; target < Size(); ++target) {
            incoming[target].reserve(counts[target]);
        }
        for (uint64_t source = 0; source < Size(); ++source) {
            for (const uint64_t target : adjacency_[source]) {
                incoming[target].push_back(source);
            }
        }
        return incoming;
    }

    [[nodiscard]] IncomingMemoryUsage
    GetIncomingMemoryUsage() const {
        IncomingMemoryUsage result;
        result.logical_bytes = incoming_.size() * sizeof(std::vector<uint64_t>);
        result.capacity_bytes = incoming_.capacity() * sizeof(std::vector<uint64_t>);
        for (const auto& sources : incoming_) {
            result.edges += sources.size();
            result.logical_bytes += sources.size() * sizeof(uint64_t);
            result.capacity_bytes += sources.capacity() * sizeof(uint64_t);
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
        result.outer_vector_bytes = incoming_.capacity() * sizeof(std::vector<uint64_t>);
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
            result.edge_logical_bytes += degree * sizeof(uint64_t);
            result.edge_capacity_bytes += capacity * sizeof(uint64_t);
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

    static void
    erase_value(std::vector<uint64_t>& values, uint64_t value) {
        values.erase(std::remove(values.begin(), values.end(), value), values.end());
    }

    static void
    replace_value(std::vector<uint64_t>& values, uint64_t old_value, uint64_t new_value) {
        for (uint64_t& value : values) {
            if (value == old_value) {
                value = new_value;
            }
        }
    }

    void
    sync_incoming(uint64_t source, const std::vector<uint64_t>& old_neighbors) {
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
                incoming_[target].push_back(source);
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
        std::priority_queue<Candidate, std::vector<Candidate>, decltype(&better)> heap(&better);
        for (uint64_t slot = 0; slot < Size(); ++slot) {
            if (slot == excluded) {
                continue;
            }
            const auto code = codes_.At(slot);
            const auto coarse = filter_estimate(query, query_norm, code);
            const Candidate candidate{slot,
                                      full_distance(query, query_norm, code, coarse.centered_ip)};
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
            std::vector<Candidate> ranked;
            ranked.reserve(additional_candidates.size());
            for (const uint64_t candidate : additional_candidates) {
                if (candidate < Size() and candidate != source and
                    std::find(repaired.begin(), repaired.end(), candidate) == repaired.end()) {
                    const auto code = codes_.At(candidate);
                    const auto coarse = filter_estimate(query, query_norm, code);
                    ranked.push_back(
                        {candidate, full_distance(query, query_norm, code, coarse.centered_ip)});
                }
            }
            std::sort(ranked.begin(), ranked.end(), better);
            const uint64_t needed = max_degree_ - repaired.size();
            const uint64_t retained = std::min(needed, ranked.size());
            repaired.reserve(repaired.size() + retained);
            for (uint64_t i = 0; i < retained; ++i) {
                repaired.push_back(ranked[i].id);
            }
            sync_incoming(source, old_neighbors);
        }
    }

    void
    link(uint64_t source, uint64_t target) {
        auto& neighbors = adjacency_[source];
        if (source == target or
            std::find(neighbors.begin(), neighbors.end(), target) != neighbors.end()) {
            return;
        }
        remember_row(source);
        const auto old_neighbors = neighbors;
        neighbors.push_back(target);
        if (neighbors.size() > max_degree_) {
            const auto query = decode_query(source);
            const float query_norm = codes_.At(source).metadata.norm;
            std::vector<Candidate> ranked;
            ranked.reserve(neighbors.size());
            for (uint64_t neighbor : neighbors) {
                const auto code = codes_.At(neighbor);
                const auto coarse = filter_estimate(query, query_norm, code);
                ranked.push_back(
                    {neighbor, full_distance(query, query_norm, code, coarse.centered_ip)});
            }
            std::sort(ranked.begin(), ranked.end(), better);
            neighbors.resize(max_degree_);
            for (uint64_t i = 0; i < max_degree_; ++i) {
                neighbors[i] = ranked[i].id;
            }
        }
        sync_incoming(source, old_neighbors);
    }

    UpdateJournal* active_journal_{};
    Model model_;
    EncodedRecords codes_;
    std::vector<int64_t> ids_;
    std::unordered_map<int64_t, uint64_t> slots_;
    std::vector<std::vector<uint64_t>> adjacency_;
    std::vector<std::vector<uint64_t>> incoming_;
    uint64_t max_degree_;
    uint64_t ef_search_;
    uint64_t mutation_fallbacks_{};
    bool measure_mutation_scans_{};
    bool use_incoming_adjacency_{};
    MutationScanTiming mutation_scan_timing_{};
    double (*cpu_clock_)(){};
};

}  // namespace vsag::lite::detail::rabitq
