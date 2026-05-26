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

#include "hgraph.h"

#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>

#include "impl/heap/standard_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"

namespace vsag {

namespace {

constexpr uint64_t kBuildCacheMagic = 0x5653414743484500ULL;
constexpr uint32_t kBuildCacheVersion = 1;
constexpr uint32_t kBuildCacheFeatureModeVariableText = 1;
constexpr uint32_t kInvalidOldId = std::numeric_limits<uint32_t>::max();
constexpr int64_t kFeatureIdsFormatVersion = 1;
constexpr std::string_view kFeatureIdsFormatVersionKey = "feature_ids_format_version";

struct BuildCacheHeader {
    uint64_t magic = kBuildCacheMagic;
    uint32_t version = kBuildCacheVersion;
    uint32_t feature_id_mode = kBuildCacheFeatureModeVariableText;
    uint64_t node_count = 0;
    uint32_t max_degree = 0;
    uint32_t reserved0 = 0;
    uint64_t feature_id_count = 0;
    uint64_t feature_id_bytes = 0;
    uint64_t build_param_hash = 0;
    uint64_t create_time = 0;
    uint8_t reserved[64] = {0};
};

static_assert(sizeof(BuildCacheHeader) == 128, "BuildCacheHeader must be 128 bytes");

struct FeatureIdIndexEntry {
    uint64_t offset = 0;
    uint32_t length = 0;
    uint32_t reserved = 0;
};

static_assert(sizeof(FeatureIdIndexEntry) == 16,
              "FeatureIdIndexEntry must be 16 bytes");

struct NeighborRecordHeader {
    uint16_t degree = 0;
};

uint64_t
hash_combine_u64(uint64_t seed, uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    return seed;
}

uint64_t
now_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

uint64_t
now_unix_seconds() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count());
}

void
log_build_cache_detail(const std::string& message) {
    std::cerr << "[hgraph_build_cache] " << message << std::endl;
}

void
log_build_cache_detail_elapsed(const std::string& stage_name, uint64_t duration_us) {
    std::cerr << "[hgraph_build_cache] " << stage_name << " finished in "
              << fmt::format("{:.3f}", static_cast<double>(duration_us) / 1000000.0) << "s"
              << std::endl;
}

}  // namespace

bool
HGraph::SupportsBuildCache() const {
    return true;
}

void
HGraph::validate_feature_ids_dataset(const DatasetPtr& data) const {
    CHECK_ARGUMENT(data != nullptr, "dataset is nullptr");
    CHECK_ARGUMENT(data->GetFeatureIds() != nullptr, "base.feature_ids is nullptr");
    CHECK_ARGUMENT(data->GetIds() != nullptr, "base.ids is nullptr");
    CHECK_ARGUMENT(get_data(data) != nullptr, "base.vector is nullptr");

    const auto total = data->GetNumElements();
    const auto* feature_ids = data->GetFeatureIds();
    std::unordered_set<std::string> unique_feature_ids;
    unique_feature_ids.reserve(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
        CHECK_ARGUMENT(not feature_ids[i].empty(), "base.feature_ids contains empty value");
        auto inserted = unique_feature_ids.emplace(feature_ids[i]);
        CHECK_ARGUMENT(inserted.second,
                       fmt::format("duplicate feature_id found: {}", feature_ids[i]));
    }
}

void
HGraph::PrepareFeatureIdsForBuildCache(const DatasetPtr& data) {
    CHECK_ARGUMENT(data != nullptr, "dataset is nullptr");
    CHECK_ARGUMENT(data->GetFeatureIds() != nullptr, "base.feature_ids is nullptr");
    CHECK_ARGUMENT(data->GetIds() != nullptr, "base.ids is nullptr");
    CHECK_ARGUMENT(data->GetNumElements() > 0, "dataset is empty");
    CHECK_ARGUMENT(this->total_count_.load() > 0, "index is empty");

    const auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* feature_ids = data->GetFeatureIds();

    std::unordered_set<std::string> unique_feature_ids;
    unique_feature_ids.reserve(static_cast<size_t>(total));
    this->feature_ids_.clear();
    this->feature_ids_.resize(this->total_count_.load());

    std::unordered_map<LabelType, InnerIdType> label_to_inner_id;
    label_to_inner_id.reserve(static_cast<size_t>(this->total_count_.load()));
    for (uint64_t inner_id = 0; inner_id < static_cast<uint64_t>(this->total_count_.load()); ++inner_id) {
        auto label = this->label_table_->GetLabelById(static_cast<InnerIdType>(inner_id));
        auto [iter, inserted] = label_to_inner_id.emplace(label, static_cast<InnerIdType>(inner_id));
        CHECK_ARGUMENT(inserted || this->support_duplicate_,
                       fmt::format("duplicate label {} found in index when preparing feature ids",
                                   label));
        if (!inserted && this->support_duplicate_) {
            iter->second = static_cast<InnerIdType>(inner_id);
        }
    }

    uint64_t mapped_count = 0;
    uint64_t found_by_label_count = 0;
    bool row_order_consistent = true;
    for (int64_t i = 0; i < total; ++i) {
        CHECK_ARGUMENT(not feature_ids[i].empty(), "base.feature_ids contains empty value");
        auto inserted = unique_feature_ids.emplace(feature_ids[i]);
        CHECK_ARGUMENT(inserted.second,
                       fmt::format("duplicate feature_id found: {}", feature_ids[i]));

        auto iter = label_to_inner_id.find(labels[i]);
        if (iter == label_to_inner_id.end()) {
            continue;
        }
        auto inner_id = iter->second;
        ++found_by_label_count;
        CHECK_ARGUMENT(inner_id < this->feature_ids_.size(),
                       fmt::format("inner_id {} out of range when preparing feature ids",
                                   inner_id));
        row_order_consistent = row_order_consistent &&
                               (static_cast<uint64_t>(i) == static_cast<uint64_t>(inner_id));
        CHECK_ARGUMENT(this->feature_ids_[inner_id].empty() || this->feature_ids_[inner_id] == feature_ids[i],
                       fmt::format("label {} maps to conflicting feature ids", labels[i]));
        if (this->feature_ids_[inner_id].empty()) {
            this->feature_ids_[inner_id] = feature_ids[i];
            ++mapped_count;
        }
    }

    const auto expected_count = static_cast<uint64_t>(this->total_count_.load());
    const bool can_use_row_order_fallback = this->delete_count_.load() == 0 &&
                                            static_cast<uint64_t>(total) == expected_count &&
                                            row_order_consistent;
    if (mapped_count != expected_count && can_use_row_order_fallback) {
        for (uint64_t i = 0; i < expected_count; ++i) {
            if (this->feature_ids_[i].empty()) {
                this->feature_ids_[i] = feature_ids[i];
                ++mapped_count;
            } else {
                CHECK_ARGUMENT(this->feature_ids_[i] == feature_ids[i],
                               fmt::format("row-order fallback found conflicting feature id at row {}",
                                           i));
            }
        }
    }

    CHECK_ARGUMENT(mapped_count == expected_count,
                   fmt::format("feature_ids metadata is incomplete after preparation: mapped={}, expected={}, found_by_label={}, row_order_consistent={}",
                               mapped_count,
                               expected_count,
                               found_by_label_count,
                               row_order_consistent));
}

uint64_t
HGraph::calculate_build_cache_param_hash() const {
    uint64_t seed = 0;
    seed = hash_combine_u64(seed, static_cast<uint64_t>(this->dim_));
    seed = hash_combine_u64(seed, static_cast<uint64_t>(this->metric_));
    seed = hash_combine_u64(seed, static_cast<uint64_t>(this->data_type_));
    seed = hash_combine_u64(seed, static_cast<uint64_t>(this->bottom_graph_->MaximumDegree()));
    seed = hash_combine_u64(seed, static_cast<uint64_t>(this->ef_construct_));
    uint32_t alpha_bits = 0;
    static_assert(sizeof(alpha_bits) == sizeof(this->alpha_));
    std::memcpy(&alpha_bits, &this->alpha_, sizeof(alpha_bits));
    seed = hash_combine_u64(seed, alpha_bits);
    return seed;
}

DistHeapPtr
HGraph::collect_refine_candidates(const DatasetPtr& data,
                                  InnerIdType inner_id,
                                  uint32_t input_idx,
                                  const FlattenInterfacePtr& flatten_codes,
                                  uint32_t refine_ef,
                                  bool use_self_as_entry) const {
    const uint32_t effective_refine_ef = refine_ef == 0 ? this->ef_construct_ : refine_ef;
    CHECK_ARGUMENT(effective_refine_ef > 0, "refine ef must be greater than 0");

    auto candidates = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
    std::unordered_set<InnerIdType> seen;

    Vector<InnerIdType> current_neighbors(allocator_);
    this->bottom_graph_->GetNeighbors(inner_id, current_neighbors);
    seen.reserve(current_neighbors.size() + effective_refine_ef);
    for (const auto neighbor : current_neighbors) {
        if (neighbor == inner_id || !seen.emplace(neighbor).second) {
            continue;
        }
        candidates->Push(flatten_codes->ComputePairVectors(neighbor, inner_id), neighbor);
    }

    if (this->entry_point_id_ != INVALID_ENTRY_POINT && this->bottom_graph_->TotalCount() > 0) {
        // Honour use_self_as_entry:
        //   - true  (cache-hit nodes): start the search from inner_id itself so
        //     the warm-started stale neighbours act as the initial frontier and
        //     refine exploits the local neighbourhood efficiently.
        //   - false (cache-missed / cold-start nodes): use the global entry
        //     point for broad exploration.
        // When the node has no cached neighbours, self-entry is meaningless,
        // so we also fall back to the global entry point.
        const auto search_entry_point =
            (use_self_as_entry && !current_neighbors.empty()) ? inner_id : this->entry_point_id_;
        InnerSearchParam param;
        param.topk = static_cast<int64_t>(effective_refine_ef);
        param.ef = effective_refine_ef;
        param.ep = search_entry_point;
        param.is_inner_id_allowed = nullptr;
        auto result = this->search_one_graph(get_data(data, input_idx),
                                             this->bottom_graph_,
                                             flatten_codes,
                                             param,
                                             (VisitedListPtr) nullptr,
                                             nullptr);
        while (not result->Empty()) {
            auto candidate = result->Top().second;
            result->Pop();
            if (candidate == inner_id || !seen.emplace(candidate).second) {
                continue;
            }
            candidates->Push(flatten_codes->ComputePairVectors(candidate, inner_id), candidate);
        }
    }

    return candidates;
}

void
HGraph::refine_single_node(const DatasetPtr& data,
                           InnerIdType inner_id,
                           uint32_t input_idx,
                           const FlattenInterfacePtr& flatten_codes,
                           uint32_t refine_ef,
                           bool use_self_as_entry) {
    auto candidates =
        this->collect_refine_candidates(
            data, inner_id, input_idx, flatten_codes, refine_ef, use_self_as_entry);

    LockGuard cur_lock(neighbors_mutex_, inner_id);
    if (candidates->Empty()) {
        Vector<InnerIdType> empty_neighbors(allocator_);
        this->bottom_graph_->InsertNeighborsById(inner_id, empty_neighbors);
        return;
    }

    mutually_connect_new_element(inner_id,
                                 candidates,
                                 this->bottom_graph_,
                                 flatten_codes,
                                 neighbors_mutex_,
                                 allocator_,
                                 alpha_);
}

HGraph::RefineExecutionStats
HGraph::refine_nodes_for_build_cache(
    const DatasetPtr& data,
    const std::vector<InnerIdType>& ids_to_refine,
    std::string_view phase_name,
    uint32_t rounds,
    uint32_t refine_ef,
    bool use_self_as_entry,
    bool enable_parallel_refine,
    uint32_t requested_parallelism,
    const FlattenInterfacePtr& flatten_codes,
    const std::unordered_map<InnerIdType, uint32_t>& inner_id_to_input_idx) {
    RefineExecutionStats stats;
    if (ids_to_refine.empty() || rounds == 0) {
        return stats;
    }

    // Force sequential mode to avoid deadlock in mutually_connect
    // (multiple threads trying to lock each other's nodes)
    uint32_t effective_parallelism = 1;
    stats.effective_parallelism = effective_parallelism;

    const uint32_t effective_refine_ef = refine_ef == 0 ? this->ef_construct_ : refine_ef;
    CHECK_ARGUMENT(effective_refine_ef > 0, "refine ef must be greater than 0");

    log_build_cache_detail(fmt::format("starting {} nodes={} rounds={} parallelism={}",
                                       phase_name,
                                       ids_to_refine.size(),
                                       rounds,
                                       effective_parallelism));

    const auto begin = now_us();
    stats.round_stats.reserve(rounds);
    for (uint32_t round = 0; round < rounds; ++round) {
        const auto round_begin = now_us();
        if (effective_parallelism <= 1) {
            for (const auto inner_id : ids_to_refine) {
                auto data_iter = inner_id_to_input_idx.find(inner_id);
                CHECK_ARGUMENT(data_iter != inner_id_to_input_idx.end(),
                               fmt::format("missing input row for inner_id {}", inner_id));
                this->refine_single_node(data,
                                         inner_id,
                                         data_iter->second,
                                         flatten_codes,
                                         effective_refine_ef,
                                         use_self_as_entry);
            }
        } else {
            const size_t chunk_count =
                std::min<size_t>(effective_parallelism, ids_to_refine.size());
            const size_t chunk_size = (ids_to_refine.size() + chunk_count - 1) / chunk_count;
            std::vector<std::future<void>> futures;
            futures.reserve(chunk_count);

            for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
                const size_t begin_idx = chunk * chunk_size;
                const size_t end_idx = std::min(ids_to_refine.size(), begin_idx + chunk_size);
                if (begin_idx >= end_idx) {
                    break;
                }

                futures.emplace_back(this->thread_pool_->GeneralEnqueue(
                    [this,
                     &data,
                     &ids_to_refine,
                     &inner_id_to_input_idx,
                     &flatten_codes,
                     effective_refine_ef,
                     use_self_as_entry,
                     begin_idx,
                     end_idx]() {
                        for (size_t index = begin_idx; index < end_idx; ++index) {
                            const auto inner_id = ids_to_refine[index];
                            auto data_iter = inner_id_to_input_idx.find(inner_id);
                            CHECK_ARGUMENT(data_iter != inner_id_to_input_idx.end(),
                                           fmt::format("missing input row for inner_id {}",
                                                       inner_id));
                            this->refine_single_node(data,
                                                     inner_id,
                                                     data_iter->second,
                                                     flatten_codes,
                                                     effective_refine_ef,
                                                     use_self_as_entry);
                        }
                    }));
            }

            for (auto& future : futures) {
                future.get();
            }
        }
        const auto round_elapsed = now_us() - round_begin;
        stats.round_stats.emplace_back(
            RefineRoundStats{.elapsed_us = round_elapsed,
                             .processed_nodes = static_cast<uint64_t>(ids_to_refine.size())});
        log_build_cache_detail(fmt::format("{} round {}/{} finished in {:.3f}s processed_nodes={}",
                                           phase_name,
                                           round + 1,
                                           rounds,
                                           static_cast<double>(round_elapsed) / 1000000.0,
                                           ids_to_refine.size()));
    }
    stats.elapsed_us = now_us() - begin;
    stats.executed_rounds = rounds;
    log_build_cache_detail_elapsed(std::string(phase_name), stats.elapsed_us);
    return stats;
}

Vector<InnerIdType>
HGraph::select_refine_neighbors_for_node(const DatasetPtr& data,
                                         InnerIdType inner_id,
                                         uint32_t input_idx,
                                         const FlattenInterfacePtr& flatten_codes,
                                         uint32_t refine_ef,
                                         bool use_self_as_entry) const {
    Vector<InnerIdType> result(allocator_);
    Vector<float> distances(allocator_);
    this->select_refine_neighbors_with_distances(
        data, inner_id, input_idx, flatten_codes, refine_ef, use_self_as_entry, result, distances);
    return result;
}

void
HGraph::select_refine_neighbors_with_distances(const DatasetPtr& data,
                                               InnerIdType inner_id,
                                               uint32_t input_idx,
                                               const FlattenInterfacePtr& flatten_codes,
                                               uint32_t refine_ef,
                                               bool use_self_as_entry,
                                               Vector<InnerIdType>& out_neighbors,
                                               Vector<float>& out_distances) const {
    out_neighbors.clear();
    out_distances.clear();

    auto candidates =
        this->collect_refine_candidates(
            data, inner_id, input_idx, flatten_codes, refine_ef, use_self_as_entry);

    if (candidates->Empty()) {
        return;
    }

    const uint64_t max_size = this->bottom_graph_->MaximumDegree();

    select_edges_by_heuristic(candidates, max_size, flatten_codes, allocator_, this->alpha_);

    out_neighbors.reserve(candidates->Size());
    out_distances.reserve(candidates->Size());
    while (not candidates->Empty()) {
        out_neighbors.emplace_back(candidates->Top().second);
        out_distances.emplace_back(candidates->Top().first);
        candidates->Pop();
    }
}

HGraph::RefineExecutionStats
HGraph::refine_nodes_two_phase(const DatasetPtr& data,
                               const std::vector<InnerIdType>& ids_to_refine,
                               std::string_view phase_name,
                               uint32_t rounds,
                               uint32_t refine_ef,
                               bool use_self_as_entry,
                               bool enable_parallel_refine,
                               uint32_t requested_parallelism,
                               const FlattenInterfacePtr& flatten_codes,
                               const std::unordered_map<InnerIdType, uint32_t>& inner_id_to_input_idx) {
    RefineExecutionStats stats;
    if (ids_to_refine.empty() || rounds == 0) {
        return stats;
    }

    uint32_t effective_parallelism = 1;
    if (enable_parallel_refine && this->thread_pool_ != nullptr &&
        this->build_thread_count_ > 1 && ids_to_refine.size() > 1) {
        effective_parallelism = requested_parallelism > 0
            ? std::min<uint32_t>(requested_parallelism, this->build_thread_count_)
            : static_cast<uint32_t>(this->build_thread_count_);
        effective_parallelism =
            std::min<uint32_t>(effective_parallelism,
                               static_cast<uint32_t>(ids_to_refine.size()));
    }
    stats.effective_parallelism = effective_parallelism;

    const uint32_t effective_refine_ef = refine_ef == 0 ? this->ef_construct_ : refine_ef;
    CHECK_ARGUMENT(effective_refine_ef > 0, "refine ef must be greater than 0");

    log_build_cache_detail(fmt::format("starting {} nodes={} rounds={} parallelism={} (two-phase)",
                                       phase_name,
                                       ids_to_refine.size(),
                                       rounds,
                                       effective_parallelism));

    const auto begin = now_us();
    stats.round_stats.reserve(rounds);

    constexpr int64_t kBlockSize = 128;

    for (uint32_t round = 0; round < rounds; ++round) {
        const auto round_begin = now_us();

        // ===== Phase 1: parallel search & local select (collect distances too) =====
        Vector<Vector<InnerIdType>> selected_neighbors(
            ids_to_refine.size(), Vector<InnerIdType>(allocator_), allocator_);
        Vector<Vector<float>> selected_distances(
            ids_to_refine.size(), Vector<float>(allocator_), allocator_);

        if (effective_parallelism <= 1) {
            for (size_t i = 0; i < ids_to_refine.size(); ++i) {
                const auto inner_id = ids_to_refine[i];
                auto data_iter = inner_id_to_input_idx.find(inner_id);
                CHECK_ARGUMENT(data_iter != inner_id_to_input_idx.end(),
                               fmt::format("missing input row for inner_id {}", inner_id));
                this->select_refine_neighbors_with_distances(
                    data, inner_id, data_iter->second, flatten_codes,
                    effective_refine_ef, use_self_as_entry,
                    selected_neighbors[i], selected_distances[i]);
            }
        } else {
            std::vector<std::future<void>> futures;
            futures.reserve((ids_to_refine.size() + kBlockSize - 1) / kBlockSize);

            for (int64_t i = 0; i < static_cast<int64_t>(ids_to_refine.size()); i += kBlockSize) {
                auto end = std::min(i + kBlockSize, static_cast<int64_t>(ids_to_refine.size()));

                futures.emplace_back(this->thread_pool_->GeneralEnqueue(
                    [this, &data, &ids_to_refine, &inner_id_to_input_idx, &flatten_codes,
                     effective_refine_ef, use_self_as_entry,
                     &selected_neighbors, &selected_distances, i, end]() {
                        for (int64_t idx = i; idx < end; ++idx) {
                            const auto inner_id = ids_to_refine[idx];
                            auto data_iter = inner_id_to_input_idx.find(inner_id);
                            CHECK_ARGUMENT(data_iter != inner_id_to_input_idx.end(),
                                           fmt::format("missing input row for inner_id {}", inner_id));
                            this->select_refine_neighbors_with_distances(
                                data, inner_id, data_iter->second, flatten_codes,
                                effective_refine_ef, use_self_as_entry,
                                selected_neighbors[idx], selected_distances[idx]);
                        }
                    }));
            }

            for (auto& future : futures) {
                future.get();
            }
        }

        const auto search_elapsed = now_us() - round_begin;
        log_build_cache_detail(
            fmt::format("{} round {}/{} search phase finished in {:.3f}s",
                        phase_name,
                        round + 1,
                        rounds,
                        static_cast<double>(search_elapsed) / 1000000.0));

        // ===== Phase 2: serial writeback of selected neighbours =====
        const auto writeback_begin = now_us();

        for (size_t i = 0; i < ids_to_refine.size(); ++i) {
            LockGuard lock(neighbors_mutex_, ids_to_refine[i]);
            this->bottom_graph_->InsertNeighborsById(ids_to_refine[i], selected_neighbors[i]);
        }

        const auto writeback_elapsed = now_us() - writeback_begin;

        // ===== Phase 3: reverse edge install =====
        // Optimisations vs. the previous version:
        //   (1) Carry the (src -> neighbour) distance computed during the
        //       select phase, so the heuristic prune below does not need to
        //       recompute these distances.
        //   (2) Shard the reverse map by `target_id % N_shard` and process
        //       shards in parallel. Each target id lives in exactly one shard,
        //       so the in-shard serial processing preserves the original
        //       single-writer-per-target invariant -- no extra mutex needed
        //       beyond the existing `neighbors_mutex_` per-id lock.
        //   (3) Replace the O(M^2) `std::find` dedup with an O(M) set lookup
        //       on small fixed-size scratch buffers.
        //   (4) Build the per-shard (target -> [(src,dist), ...]) map in two
        //       fully parallel stages: every worker first scatters its slice
        //       of `ids_to_refine` into per-(worker, shard) local buffers
        //       (lock-free); afterwards each shard is materialised into its
        //       hash map by a separate worker (shard-parallel) by simply
        //       concatenating the relevant slices.
        const auto reverse_begin = now_us();

        const uint32_t reverse_shard_count =
            (effective_parallelism > 1)
                ? std::max<uint32_t>(effective_parallelism,
                                     static_cast<uint32_t>(this->build_thread_count_))
                : 1U;

        struct ReverseEdgeEntry {
            InnerIdType src_id;
            float dist;
        };

        // Scatter target_id together with src/dist so the materialise stage
        // can directly bucket-by-target. Using a tagged struct (target,src,dist)
        // is the simplest layout; tail-packing avoids padding cost vs. a
        // separate vectors-of-arrays representation.
        struct ScatterRecord {
            InnerIdType target_id;
            InnerIdType src_id;
            float dist;
        };

        struct ReverseShard {
            // target_id -> list of (src_id, dist(src->target)) to add
            std::unordered_map<InnerIdType, std::vector<ReverseEdgeEntry>> pending;
        };
        std::vector<ReverseShard> shards(reverse_shard_count);

        // ----- (3a) group reverse edges into shards (parallel) -----
        const auto reverse_group_begin = now_us();

        // Pick a worker count for the scatter stage. Each worker owns one
        // private slice of the input array, so contention is zero.
        const uint32_t scatter_worker_count =
            (effective_parallelism > 1)
                ? std::min<uint32_t>(effective_parallelism,
                                     static_cast<uint32_t>(ids_to_refine.size()))
                : 1U;

        // worker_buckets[w][s] holds the (target,src,dist) records that
        // worker `w` produced for shard `s`. After the scatter stage we
        // simply concatenate worker_buckets[*][s] into shard `s`.
        std::vector<std::vector<std::vector<ScatterRecord>>> worker_buckets(
            scatter_worker_count,
            std::vector<std::vector<ScatterRecord>>(reverse_shard_count));

        // Heuristic per-bucket reserve so push_back stays in the fast path.
        const uint64_t max_degree_for_reserve = this->bottom_graph_->MaximumDegree();
        const size_t avg_per_bucket =
            (ids_to_refine.size() * static_cast<size_t>(max_degree_for_reserve)
             + static_cast<size_t>(scatter_worker_count) * reverse_shard_count - 1) /
            (static_cast<size_t>(scatter_worker_count) * reverse_shard_count);
        for (auto& wb : worker_buckets) {
            for (auto& shard_buf : wb) {
                shard_buf.reserve(avg_per_bucket + 16);
            }
        }

        auto scatter_slice = [&](uint32_t worker_idx, size_t lo, size_t hi) {
            auto& my_buckets = worker_buckets[worker_idx];
            for (size_t i = lo; i < hi; ++i) {
                const auto src_id = ids_to_refine[i];
                const auto& neighbors = selected_neighbors[i];
                const auto& dists = selected_distances[i];
                const size_t k = neighbors.size();
                CHECK_ARGUMENT(dists.size() == k,
                               "selected_neighbors and selected_distances size mismatch");
                for (size_t j = 0; j < k; ++j) {
                    const auto neighbor_id = neighbors[j];
                    const uint32_t shard_idx =
                        static_cast<uint32_t>(neighbor_id) % reverse_shard_count;
                    my_buckets[shard_idx].push_back(
                        {neighbor_id, src_id, dists[j]});
                }
            }
        };

        if (scatter_worker_count <= 1) {
            scatter_slice(0, 0, ids_to_refine.size());
        } else {
            const size_t per_worker =
                (ids_to_refine.size() + scatter_worker_count - 1) / scatter_worker_count;
            std::vector<std::future<void>> scatter_futures;
            scatter_futures.reserve(scatter_worker_count);
            for (uint32_t w = 0; w < scatter_worker_count; ++w) {
                const size_t lo = w * per_worker;
                if (lo >= ids_to_refine.size()) {
                    break;
                }
                const size_t hi = std::min(lo + per_worker, ids_to_refine.size());
                scatter_futures.emplace_back(this->thread_pool_->GeneralEnqueue(
                    [&scatter_slice, w, lo, hi]() { scatter_slice(w, lo, hi); }));
            }
            for (auto& f : scatter_futures) {
                f.get();
            }
        }

        // Materialise each shard's per-(target) map in parallel across shards.
        auto materialise_shard = [&](uint32_t shard_idx) {
            // Pre-sum incoming entries for a sensible reserve.
            size_t total = 0;
            for (uint32_t w = 0; w < scatter_worker_count; ++w) {
                total += worker_buckets[w][shard_idx].size();
            }
            auto& pending = shards[shard_idx].pending;
            // Heuristic: average ~max_degree entries per target. Reserve at
            // total / max_degree buckets to avoid most rehashes.
            const size_t hint =
                total / std::max<size_t>(max_degree_for_reserve / 2, 1) + 16;
            pending.reserve(hint);
            for (uint32_t w = 0; w < scatter_worker_count; ++w) {
                auto& buf = worker_buckets[w][shard_idx];
                for (const auto& rec : buf) {
                    pending[rec.target_id].push_back({rec.src_id, rec.dist});
                }
                // Free worker buffer ASAP to cap peak memory.
                std::vector<ScatterRecord>().swap(buf);
            }
        };

        if (effective_parallelism <= 1 || reverse_shard_count <= 1) {
            for (uint32_t s = 0; s < reverse_shard_count; ++s) {
                materialise_shard(s);
            }
        } else {
            std::vector<std::future<void>> mat_futures;
            mat_futures.reserve(reverse_shard_count);
            for (uint32_t s = 0; s < reverse_shard_count; ++s) {
                mat_futures.emplace_back(this->thread_pool_->GeneralEnqueue(
                    [&materialise_shard, s]() { materialise_shard(s); }));
            }
            for (auto& f : mat_futures) {
                f.get();
            }
        }

        // Release the now-empty worker_buckets vector skeleton.
        std::vector<std::vector<std::vector<ScatterRecord>>>().swap(worker_buckets);

        const auto reverse_group_elapsed = now_us() - reverse_group_begin;

        // ----- (3b) per-shard merge + heuristic prune (parallel across shards) -----
        const uint64_t max_degree = this->bottom_graph_->MaximumDegree();

        auto process_shard = [this, &flatten_codes, max_degree](ReverseShard& shard) {
            Vector<InnerIdType> current_neighbors(allocator_);
            current_neighbors.reserve(max_degree + 16);
            // Reusable lookup set to dedup against existing neighbours.
            std::unordered_set<InnerIdType> existing_set;
            existing_set.reserve(max_degree * 2 + 16);

            for (auto& entry : shard.pending) {
                const auto target_id = entry.first;
                auto& reverse_adds = entry.second;

                // Hold the per-target lock for the whole merge+prune so that
                // concurrent reverse-edge processing for *other* targets does
                // not race with this target. The shard partitioning already
                // ensures no two threads touch the same target_id, but the
                // graph data cell itself may be concurrently *read* by other
                // shards while we mutate this target; the per-id LockGuard
                // matches the original code's contract.
                LockGuard lock(neighbors_mutex_, target_id);

                current_neighbors.clear();
                this->bottom_graph_->GetNeighbors(target_id, current_neighbors);

                existing_set.clear();
                for (auto nid : current_neighbors) {
                    existing_set.insert(nid);
                }

                bool changed = false;
                // Append distinct, non-self reverse adds.
                // We do not yet know their distance to target; they will only
                // be needed if a prune triggers below.
                for (const auto& add : reverse_adds) {
                    if (add.src_id == target_id) {
                        continue;
                    }
                    if (existing_set.insert(add.src_id).second) {
                        current_neighbors.push_back(add.src_id);
                        changed = true;
                    }
                }

                if (!changed) {
                    continue;
                }

                if (current_neighbors.size() > max_degree) {
                    // Prune via heuristic. Build a (distance, neighbour) heap.
                    // For neighbours that were just inserted as reverse edges
                    // we already know dist(src -> target) thanks to the
                    // distances captured during the select phase, so we can
                    // skip recomputation for those.
                    std::unordered_map<InnerIdType, float> reuse_dist;
                    reuse_dist.reserve(reverse_adds.size());
                    for (const auto& add : reverse_adds) {
                        // If duplicates exist for the same src_id, keep the
                        // smaller distance (tighter upper bound is fine for
                        // heuristic select).
                        auto [it, inserted] = reuse_dist.emplace(add.src_id, add.dist);
                        if (!inserted && add.dist < it->second) {
                            it->second = add.dist;
                        }
                    }

                    auto edges =
                        std::make_shared<StandardHeap<true, false>>(allocator_, -1);
                    for (auto neighbor_id : current_neighbors) {
                        auto rit = reuse_dist.find(neighbor_id);
                        const float d = (rit != reuse_dist.end())
                                            ? rit->second
                                            : flatten_codes->ComputePairVectors(neighbor_id,
                                                                                target_id);
                        edges->Push(d, neighbor_id);
                    }
                    select_edges_by_heuristic(
                        edges, max_degree, flatten_codes, allocator_, this->alpha_);
                    current_neighbors.clear();
                    while (not edges->Empty()) {
                        current_neighbors.emplace_back(edges->Top().second);
                        edges->Pop();
                    }
                }

                this->bottom_graph_->InsertNeighborsById(target_id, current_neighbors);
            }
        };

        const auto reverse_merge_begin = now_us();
        if (effective_parallelism <= 1 || reverse_shard_count <= 1) {
            for (auto& shard : shards) {
                process_shard(shard);
            }
        } else {
            std::vector<std::future<void>> shard_futures;
            shard_futures.reserve(shards.size());
            for (auto& shard : shards) {
                shard_futures.emplace_back(this->thread_pool_->GeneralEnqueue(
                    [&process_shard, &shard]() { process_shard(shard); }));
            }
            for (auto& f : shard_futures) {
                f.get();
            }
        }
        const auto reverse_merge_elapsed = now_us() - reverse_merge_begin;

        const auto reverse_elapsed = now_us() - reverse_begin;
        const auto round_elapsed = now_us() - round_begin;
        stats.round_stats.emplace_back(
            RefineRoundStats{.elapsed_us = round_elapsed,
                             .processed_nodes = static_cast<uint64_t>(ids_to_refine.size())});
        log_build_cache_detail(
            fmt::format("{} round {}/{} finished in {:.3f}s (search={:.3f}s writeback={:.3f}s reverse={:.3f}s [group={:.3f}s merge_prune={:.3f}s shards={}]) processed_nodes={}",
                        phase_name,
                        round + 1,
                        rounds,
                        static_cast<double>(round_elapsed) / 1000000.0,
                        static_cast<double>(search_elapsed) / 1000000.0,
                        static_cast<double>(writeback_elapsed) / 1000000.0,
                        static_cast<double>(reverse_elapsed) / 1000000.0,
                        static_cast<double>(reverse_group_elapsed) / 1000000.0,
                        static_cast<double>(reverse_merge_elapsed) / 1000000.0,
                        reverse_shard_count,
                        ids_to_refine.size()));
    }

    stats.elapsed_us = now_us() - begin;
    stats.executed_rounds = rounds;
    log_build_cache_detail_elapsed(std::string(phase_name), stats.elapsed_us);
    return stats;
}

void
HGraph::ExportBuildCache(std::ostream& out_stream) const {
    CHECK_ARGUMENT(this->total_count_.load() > 0, "index is empty");
    CHECK_ARGUMENT(this->delete_count_.load() == 0,
                   "ExportBuildCache requires index without removed nodes");
    CHECK_ARGUMENT(this->feature_ids_.size() >= this->total_count_.load(),
                   "feature_ids metadata is incomplete");

    IOStreamWriter writer(out_stream);
    BuildCacheHeader header;
    header.node_count = this->total_count_.load();
    header.max_degree = this->bottom_graph_->MaximumDegree();
    header.feature_id_count = header.node_count;
    header.build_param_hash = this->calculate_build_cache_param_hash();
    header.create_time = now_unix_seconds();

    std::vector<FeatureIdIndexEntry> index_entries(header.node_count);
    uint64_t feature_blob_bytes = 0;
    for (uint64_t i = 0; i < header.node_count; ++i) {
        const auto& feature_id = this->feature_ids_[i];
        CHECK_ARGUMENT(not feature_id.empty(),
                       fmt::format("feature_id missing for inner_id {}", i));
        index_entries[i].offset = feature_blob_bytes;
        index_entries[i].length = static_cast<uint32_t>(feature_id.size());
        feature_blob_bytes += feature_id.size();
    }
    header.feature_id_bytes = feature_blob_bytes;

    StreamWriter::WriteObj(writer, header);
    for (const auto& entry : index_entries) {
        StreamWriter::WriteObj(writer, entry);
    }
    for (uint64_t i = 0; i < header.node_count; ++i) {
        const auto& feature_id = this->feature_ids_[i];
        if (not feature_id.empty()) {
            writer.Write(feature_id.data(), static_cast<uint64_t>(feature_id.size()));
        }
    }

    for (uint64_t i = 0; i < header.node_count; ++i) {
        Vector<InnerIdType> neighbors(allocator_);
        this->bottom_graph_->GetNeighbors(static_cast<InnerIdType>(i), neighbors);
        NeighborRecordHeader record_header{.degree = static_cast<uint16_t>(neighbors.size())};
        StreamWriter::WriteObj(writer, record_header);
        for (uint32_t j = 0; j < header.max_degree; ++j) {
            uint32_t neighbor = UINT32_MAX;
            if (j < neighbors.size()) {
                neighbor = neighbors[j];
            }
            StreamWriter::WriteObj(writer, neighbor);
        }
    }
}

std::vector<int64_t>
HGraph::BuildWithCache(const DatasetPtr& data,
                       std::istream& in_stream,
                       const BuildCacheOptions& options) {
    CHECK_ARGUMENT(GetNumElements() == 0, "index is not empty");
    if (not options.enable_warm_start) {
        build_cache_stats_ = BuildCacheStats{};
        build_cache_stats_.total_nodes = data->GetNumElements();
        return this->Build(data);
    }

    this->validate_feature_ids_dataset(data);
    this->Train(data);
    this->build_cache_stats_ = BuildCacheStats{};
    this->build_cache_stats_.total_nodes = static_cast<uint64_t>(data->GetNumElements());

    IOStreamReader reader(in_stream);
    const auto cache_load_begin = now_us();

    BuildCacheHeader header;
    StreamReader::ReadObj(reader, header);
    CHECK_ARGUMENT(header.magic == kBuildCacheMagic,
                   fmt::format("Invalid cache file: magic mismatch, expected {}", kBuildCacheMagic));
    CHECK_ARGUMENT(header.version == kBuildCacheVersion,
                   fmt::format("Cache version incompatible: expected v{}, got v{}",
                               kBuildCacheVersion,
                               header.version));
    CHECK_ARGUMENT(header.feature_id_mode == kBuildCacheFeatureModeVariableText,
                   "FeatureID metadata mismatch: unsupported feature_id_mode");
    CHECK_ARGUMENT(header.feature_id_count == header.node_count,
                   "FeatureID metadata mismatch: feature_id_count != node_count");
    CHECK_ARGUMENT(header.build_param_hash == this->calculate_build_cache_param_hash(),
                   fmt::format("Build params changed: old_hash={}, new_hash={}",
                               header.build_param_hash,
                               this->calculate_build_cache_param_hash()));

    std::vector<FeatureIdIndexEntry> old_feature_index(header.node_count);
    for (uint64_t i = 0; i < header.node_count; ++i) {
        StreamReader::ReadObj(reader, old_feature_index[i]);
    }
    std::string feature_blob(header.feature_id_bytes, '\0');
    if (header.feature_id_bytes > 0) {
        reader.Read(feature_blob.data(), header.feature_id_bytes);
    }

    std::vector<std::string> old_feature_ids(header.node_count);
    for (uint64_t i = 0; i < header.node_count; ++i) {
        const auto& entry = old_feature_index[i];
        CHECK_ARGUMENT(entry.offset + entry.length <= feature_blob.size(),
                       fmt::format("Cache stream read error at feature entry {}", i));
        old_feature_ids[i] = feature_blob.substr(entry.offset, entry.length);
    }

    std::vector<std::vector<uint32_t>> old_neighbors(header.node_count);
    for (uint64_t i = 0; i < header.node_count; ++i) {
        NeighborRecordHeader record_header;
        StreamReader::ReadObj(reader, record_header);
        CHECK_ARGUMENT(record_header.degree <= header.max_degree,
                       fmt::format("Cache stream read error at neighbor record {}", i));
        old_neighbors[i].reserve(record_header.degree);
        for (uint32_t j = 0; j < header.max_degree; ++j) {
            uint32_t neighbor = UINT32_MAX;
            StreamReader::ReadObj(reader, neighbor);
            if (j < record_header.degree) {
                old_neighbors[i].push_back(neighbor);
            }
        }
    }
    build_cache_stats_.cache_load_us = now_us() - cache_load_begin;
    build_cache_stats_.cached_nodes = header.node_count;

    const auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* feature_ids = data->GetFeatureIds();
    const auto* extra_infos = data->GetExtraInfos();
    const auto* attr_sets = data->GetAttributeSets();

    auto inner_ids = this->get_unique_inner_ids(total);
    Vector<Vector<InnerIdType>> route_graph_ids(allocator_);
    this->feature_ids_.clear();
    this->feature_ids_.resize(this->total_count_.load());

    std::unordered_map<std::string, uint32_t> fid_to_new_id;
    fid_to_new_id.reserve(static_cast<size_t>(total));
    std::vector<InnerIdType> inserted_inner_ids;
    inserted_inner_ids.reserve(static_cast<size_t>(total));
    std::unordered_map<InnerIdType, uint32_t> inner_id_to_input_idx;
    inner_id_to_input_idx.reserve(static_cast<size_t>(total));
    std::vector<int64_t> failed_ids;
    for (int64_t i = 0; i < total; ++i) {
        auto [it, inserted] = fid_to_new_id.emplace(feature_ids[i], inner_ids[i]);
        CHECK_ARGUMENT(inserted, fmt::format("duplicate feature_id found: {}", feature_ids[i]));
        auto label = labels[i];
        if (this->label_table_->CheckLabel(label)) {
            failed_ids.emplace_back(label);
            continue;
        }

        InnerIdType inner_id = inner_ids[i];
        this->label_table_->Insert(inner_id, label);
        this->basic_flatten_codes_->InsertVector(get_data(data, static_cast<uint32_t>(i)), inner_id);
        if (use_reorder_) {
            this->high_precise_codes_->InsertVector(get_data(data, static_cast<uint32_t>(i)), inner_id);
        }
        if (create_new_raw_vector_) {
            this->raw_vector_->InsertVector(get_data(data, static_cast<uint32_t>(i)), inner_id);
        }
        if (this->extra_infos_ != nullptr && extra_infos != nullptr) {
            this->extra_infos_->InsertExtraInfo(extra_infos + i * extra_info_size_, inner_id);
        }
        if (attr_sets != nullptr && this->use_attribute_filter_) {
            this->attr_filter_index_->Insert(attr_sets[i], inner_id);
        }

        if (inner_id >= this->feature_ids_.size()) {
            this->feature_ids_.resize(inner_id + 1);
        }
        this->feature_ids_[inner_id] = feature_ids[i];
        inserted_inner_ids.push_back(inner_id);
        inner_id_to_input_idx.emplace(inner_id, static_cast<uint32_t>(i));

        auto level = this->get_random_level() - 1;
        if (level >= 0) {
            if (level >= static_cast<int>(route_graph_ids.size()) || route_graph_ids.empty()) {
                for (auto k = static_cast<int>(route_graph_ids.size()); k <= level; ++k) {
                    route_graph_ids.emplace_back(allocator_);
                }
                entry_point_id_ = inner_id;
            }
            for (int j = 0; j <= level; ++j) {
                route_graph_ids[j].emplace_back(inner_id);
            }
        }
    }
    this->resize(total_count_);
    if (entry_point_id_ == INVALID_ENTRY_POINT && not inserted_inner_ids.empty()) {
        entry_point_id_ = inserted_inner_ids.front();
    }

    std::vector<uint32_t> old_to_new(header.node_count, UINT32_MAX);
    std::unordered_map<std::string, uint32_t> old_fid_to_old_id;
    old_fid_to_old_id.reserve(static_cast<size_t>(header.node_count));
    for (uint32_t old_id = 0; old_id < header.node_count; ++old_id) {
        old_fid_to_old_id.emplace(old_feature_ids[old_id], old_id);
        auto iter = fid_to_new_id.find(old_feature_ids[old_id]);
        if (iter != fid_to_new_id.end()) {
            old_to_new[old_id] = iter->second;
        }
    }

    std::vector<InnerIdType> hit_ids;
    std::vector<InnerIdType> missed_ids;
    hit_ids.reserve(inserted_inner_ids.size());
    missed_ids.reserve(inserted_inner_ids.size());

    struct WarmStartChunkResult {
        std::vector<InnerIdType> hit_ids;
        std::vector<InnerIdType> missed_ids;
        uint64_t hit_seed_neighbor_total{0};
        uint64_t missed_seed_neighbor_total{0};
        uint64_t hit_empty_seed_nodes{0};
        uint64_t missed_empty_seed_nodes{0};
        uint64_t invalid_neighbors{0};
        uint64_t dropped_neighbors{0};
    };

    const auto warm_start_begin = now_us();
    log_build_cache_detail(fmt::format("starting warm_start nodes={} cached_nodes={} build_threads={}",
                                       inserted_inner_ids.size(),
                                       header.node_count,
                                       this->build_thread_count_));
    const auto process_warm_start_range =
        [&](size_t begin, size_t end) -> WarmStartChunkResult {
        WarmStartChunkResult result;
        result.hit_ids.reserve(end - begin);
        result.missed_ids.reserve(end - begin);

        for (size_t i = begin; i < end; ++i) {
            const auto inner_id = inserted_inner_ids[i];
            auto old_id_iter = old_fid_to_old_id.find(this->feature_ids_[inner_id]);
            Vector<InnerIdType> mapped_neighbors(allocator_);
            std::unordered_set<InnerIdType> dedup;
            dedup.reserve(header.max_degree);

            if (old_id_iter != old_fid_to_old_id.end()) {
                const auto old_id = old_id_iter->second;
                for (auto neighbor_old_id : old_neighbors[old_id]) {
                    if (neighbor_old_id >= old_to_new.size()) {
                        ++result.invalid_neighbors;
                        continue;
                    }
                    auto new_neighbor_id = old_to_new[neighbor_old_id];
                    if (new_neighbor_id == UINT32_MAX) {
                        ++result.invalid_neighbors;
                        if (options.drop_invalid_neighbors) {
                            ++result.dropped_neighbors;
                            continue;
                        }
                        continue;
                    }
                    if (new_neighbor_id == inner_id) {
                        continue;
                    }
                    if (dedup.emplace(new_neighbor_id).second) {
                        mapped_neighbors.emplace_back(new_neighbor_id);
                    }
                }
                if (mapped_neighbors.size() > this->bottom_graph_->MaximumDegree()) {
                    mapped_neighbors.resize(this->bottom_graph_->MaximumDegree());
                }
                result.hit_seed_neighbor_total += mapped_neighbors.size();
                if (mapped_neighbors.empty()) {
                    ++result.hit_empty_seed_nodes;
                }
                this->bottom_graph_->InsertNeighborsById(inner_id, mapped_neighbors);
                result.hit_ids.push_back(inner_id);
            } else {
                result.missed_seed_neighbor_total += mapped_neighbors.size();
                if (mapped_neighbors.empty()) {
                    ++result.missed_empty_seed_nodes;
                }
                this->bottom_graph_->InsertNeighborsById(inner_id, mapped_neighbors);
                result.missed_ids.push_back(inner_id);
            }
        }
        return result;
    };

    if (this->thread_pool_ != nullptr && this->build_thread_count_ > 1 &&
        inserted_inner_ids.size() > 1) {
        const size_t chunk_count =
            std::min<size_t>(static_cast<size_t>(this->build_thread_count_), inserted_inner_ids.size());
        const size_t chunk_size = (inserted_inner_ids.size() + chunk_count - 1) / chunk_count;

        std::vector<std::future<WarmStartChunkResult>> futures;
        futures.reserve(chunk_count);
        for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
            const size_t begin = chunk * chunk_size;
            const size_t end = std::min(inserted_inner_ids.size(), begin + chunk_size);
            if (begin >= end) {
                break;
            }
            futures.emplace_back(
                this->thread_pool_->GeneralEnqueue(process_warm_start_range, begin, end));
        }

        for (auto& future : futures) {
            auto chunk_result = future.get();
            build_cache_stats_.hit_seed_neighbor_total += chunk_result.hit_seed_neighbor_total;
            build_cache_stats_.missed_seed_neighbor_total += chunk_result.missed_seed_neighbor_total;
            build_cache_stats_.hit_empty_seed_nodes += chunk_result.hit_empty_seed_nodes;
            build_cache_stats_.missed_empty_seed_nodes += chunk_result.missed_empty_seed_nodes;
            build_cache_stats_.invalid_neighbors += chunk_result.invalid_neighbors;
            build_cache_stats_.dropped_neighbors += chunk_result.dropped_neighbors;
            hit_ids.insert(hit_ids.end(), chunk_result.hit_ids.begin(), chunk_result.hit_ids.end());
            missed_ids.insert(
                missed_ids.end(), chunk_result.missed_ids.begin(), chunk_result.missed_ids.end());
        }
    } else {
        auto chunk_result = process_warm_start_range(0, inserted_inner_ids.size());
        build_cache_stats_.hit_seed_neighbor_total += chunk_result.hit_seed_neighbor_total;
        build_cache_stats_.missed_seed_neighbor_total += chunk_result.missed_seed_neighbor_total;
        build_cache_stats_.hit_empty_seed_nodes += chunk_result.hit_empty_seed_nodes;
        build_cache_stats_.missed_empty_seed_nodes += chunk_result.missed_empty_seed_nodes;
        build_cache_stats_.invalid_neighbors += chunk_result.invalid_neighbors;
        build_cache_stats_.dropped_neighbors += chunk_result.dropped_neighbors;
        hit_ids = std::move(chunk_result.hit_ids);
        missed_ids = std::move(chunk_result.missed_ids);
    }
    build_cache_stats_.hit_nodes = hit_ids.size();
    build_cache_stats_.missed_nodes = missed_ids.size();
    build_cache_stats_.cache_hit_rate = total > 0 ? static_cast<float>(hit_ids.size()) / total : 0.0F;
    build_cache_stats_.warm_start_apply_us = now_us() - warm_start_begin;
    log_build_cache_detail(fmt::format(
        "warm_start summary hit_nodes={} missed_nodes={} hit_empty_seed_nodes={} missed_empty_seed_nodes={} hit_seed_neighbor_total={} missed_seed_neighbor_total={} invalid_neighbors={} dropped_neighbors={}",
        build_cache_stats_.hit_nodes,
        build_cache_stats_.missed_nodes,
        build_cache_stats_.hit_empty_seed_nodes,
        build_cache_stats_.missed_empty_seed_nodes,
        build_cache_stats_.hit_seed_neighbor_total,
        build_cache_stats_.missed_seed_neighbor_total,
        build_cache_stats_.invalid_neighbors,
        build_cache_stats_.dropped_neighbors));
    log_build_cache_detail_elapsed("warm_start", build_cache_stats_.warm_start_apply_us);

    auto flatten_codes = this->basic_flatten_codes_;
    if (has_precise_reorder()) {
        flatten_codes = this->high_precise_codes_;
    }

    // Refine hit nodes first to restore graph connectivity, then refine missed
    // nodes on the improved graph. Hit nodes (92%+) carry stale neighbors from
    // the previous day's cache; refining them first gives missed nodes a better
    // connected graph to search from during their own refine phase.
    auto hit_refine_stats = this->refine_nodes_two_phase(data,
                                                         hit_ids,
                                                         "hit_refine",
                                                         options.hit_refine_rounds,
                                                         options.hit_refine_ef,
                                                         true,
                                                         options.enable_parallel_refine,
                                                         options.refine_parallelism,
                                                         flatten_codes,
                                                         inner_id_to_input_idx);
    build_cache_stats_.hit_refine_us = hit_refine_stats.elapsed_us;
    build_cache_stats_.hit_refine_rounds = hit_refine_stats.executed_rounds;
    build_cache_stats_.hit_refine_ef =
        options.hit_refine_ef == 0 ? this->ef_construct_ : options.hit_refine_ef;
    build_cache_stats_.hit_refine_parallelism = hit_refine_stats.effective_parallelism;

    auto missed_refine_stats = this->refine_nodes_two_phase(data,
                                                            missed_ids,
                                                            "missed_refine",
                                                            options.missed_refine_rounds,
                                                            options.missed_refine_ef,
                                                            false,
                                                            options.enable_parallel_refine,
                                                            options.refine_parallelism,
                                                            flatten_codes,
                                                            inner_id_to_input_idx);
    build_cache_stats_.missed_refine_us = missed_refine_stats.elapsed_us;
    build_cache_stats_.missed_refine_rounds = missed_refine_stats.executed_rounds;
    build_cache_stats_.missed_refine_ef =
        options.missed_refine_ef == 0 ? this->ef_construct_ : options.missed_refine_ef;
    build_cache_stats_.missed_refine_parallelism = missed_refine_stats.effective_parallelism;

    this->route_graphs_.clear();
    if (options.build_route_graph) {
        const auto route_graph_begin = now_us();
        log_build_cache_detail(
            fmt::format("starting route_graph_build levels_to_build={}", route_graph_ids.size()));
        auto build_data = (has_precise_reorder()) ? this->high_precise_codes_
                                                               : this->basic_flatten_codes_;
        for (auto& route_graph_id : route_graph_ids) {
            odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
            ODescent sparse_odescent_builder(
                odescent_param_, build_data, allocator_, this->thread_pool_.get());
            auto graph = this->generate_one_route_graph();
            sparse_odescent_builder.Build(route_graph_id);
            sparse_odescent_builder.SaveGraph(graph);
            this->route_graphs_.emplace_back(graph);
        }
        build_cache_stats_.route_graph_build_us = now_us() - route_graph_begin;
        build_cache_stats_.route_graph_levels = this->route_graphs_.size();
        log_build_cache_detail(fmt::format("route_graph_build summary levels={}",
                                           build_cache_stats_.route_graph_levels));
        log_build_cache_detail_elapsed("route_graph_build",
                                       build_cache_stats_.route_graph_build_us);
    } else {
        build_cache_stats_.route_graph_build_us = 0;
        build_cache_stats_.route_graph_levels = 0;
        log_build_cache_detail("route_graph_build skipped");
    }

    if (use_elp_optimizer_) {
        elp_optimize();
    }

    return failed_ids;
}

BuildCacheStats
HGraph::GetBuildCacheStats() const {
    return this->build_cache_stats_;
}

void
HGraph::serialize_feature_ids(StreamWriter& writer) const {
    const auto count = this->total_count_.load();
    StreamWriter::WriteObj(writer, count);
    for (uint64_t i = 0; i < count; ++i) {
        StreamWriter::WriteString(writer, this->feature_ids_[i]);
    }
}

void
HGraph::deserialize_feature_ids(StreamReader& reader, uint64_t expected_count) {
    uint64_t count = 0;
    StreamReader::ReadObj(reader, count);
    CHECK_ARGUMENT(count == expected_count,
                   fmt::format("FeatureID count mismatch: expected {}, got {}",
                               expected_count,
                               count));

    this->feature_ids_.clear();
    this->feature_ids_.resize(count);
    for (uint64_t i = 0; i < count; ++i) {
        this->feature_ids_[i] = StreamReader::ReadString(reader);
    }
}

}  // namespace vsag
