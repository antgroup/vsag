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

#include "algorithm/hgraph/hgraph_continue_session.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <vector>

#include "algorithm/hgraph/hgraph.h"
#include "impl/query_computer_pool.h"
#include "impl/searcher/searcher_utils.h"
#include "utils/search_threshold.h"
#include "vsag_exception.h"

namespace vsag {

// No graph search invocation owns this state: the session is the traversal.
// Routing scores and bottom scores share a cache, but discovery/expansion are
// separate because an upper-layer vertex need not yet be in the bottom frontier.
struct HGraphContinueSession::Traversal {
    using Record = std::pair<float, InnerIdType>;
    using Frontier = std::priority_queue<Record, std::vector<Record>, std::greater<>>;
    explicit Traversal(const void* data) : computers(data) {
    }
    QueryComputerPool computers;
    ComputerLease coarse;
    ComputerLease precise;
    FilterPtr filter;
    std::vector<float> scores;
    std::vector<uint8_t> scored;
    std::vector<uint8_t> discovered;
    std::vector<uint8_t> delivered;
    std::vector<uint8_t> unreturnable;
    std::vector<float> precise_scores;
    std::vector<uint8_t> precisely_scored;
    uint64_t undelivered{0};
    Frontier frontier;
    std::set<Record> pending;
    std::optional<float> threshold;
    uint64_t effort{0};
    uint64_t routing_runs{0};
    uint64_t score_count{0};
    uint64_t precise_count{0};
    uint64_t expansion_count{0};
    bool routed{false};
};

std::unique_ptr<SearchSession>
HGraph::OpenSearchSession(const DatasetPtr& query,
                          int64_t k_per_call,
                          const std::string& parameters,
                          const FilterPtr& filter,
                          Allocator* allocator,
                          std::shared_ptr<const InnerIndexInterface> owner) const {
    // Internal caller contract: IndexImpl supplies its owning SessionOwner alias.
    // This only checks identity and a nonempty control block, not arbitrary deleter semantics.
    const bool valid_owner = owner.get() == this and owner.use_count() > 0;
    CHECK_ARGUMENT(valid_owner, "session requires this backend and a nonempty owner control block");
    CHECK_ARGUMENT(query != nullptr, "session query must not be null");
    this->validate_knn_args(query, k_per_call);
    auto* alloc = allocator == nullptr ? allocator_ : allocator;
    auto session = std::unique_ptr<HGraphContinueSession>(
        new HGraphContinueSession(*this, owner, query->DeepCopy(alloc), parameters, filter, alloc));
    const auto validated = session->ValidateParameters(parameters);
    session->Initialize(k_per_call, validated.search.ef_search);
    return session;
}

HGraphContinueSession::ValidatedParameters
HGraphContinueSession::ValidateParameters(const std::string& parameters) const {
    const auto& hgraph = *hgraph_;
    const auto parsed = JsonType::Parse(parameters);
    auto params = HGraphSearchParameters::FromParsedJson(parsed);
    CHECK_ARGUMENT(params.ef_search >= 1, "ef_search must be positive");
    const bool ordinary_search = params.parallel_search_thread_count == 1 and
                                 not params.enable_time_record and params.topk_factor == 0 and
                                 params.brute_force_threshold == 0 and
                                 params.hops_limit == std::numeric_limits<uint32_t>::max();
    CHECK_ARGUMENT(
        ordinary_search,
        "session does not support parallel, timed, factored, brute-force or hop-limited search");
    const bool ordinary_quantization = not params.rabitq_one_bit_search and
                                       is_nan_distance(params.rabitq_error_rate) and
                                       hgraph.rabitq_fused_datacell_ == nullptr;
    CHECK_ARGUMENT(ordinary_quantization, "session does not support specialized RaBitQ search");
    const bool ordinary_storage =
        not hgraph.deduplicate_storage_ and not hgraph.support_force_remove() and
        not(hgraph.mci_parameters_.enabled and params.use_mci) and
        not(hgraph.use_conjugate_graph_ and params.use_conjugate_graph_search);
    CHECK_ARGUMENT(ordinary_storage,
                   "session does not support shared duplicate storage, force removal, MCI or "
                   "conjugate search");
    const auto options = parsed[INDEX_TYPE_HGRAPH];
    const bool ordinary_filter =
        not options.Contains("skip_ratio") and not options.Contains("skip_strategy");
    CHECK_ARGUMENT(ordinary_filter, "session explores filtered bridges without skip strategies");
    std::optional<float> threshold;
    if (parsed.Contains(SEARCH_THRESHOLD)) {
        CHECK_ARGUMENT(parsed[SEARCH_THRESHOLD].IsNumber(), "search threshold must be a number");
        threshold = parsed[SEARCH_THRESHOLD].GetFloat();
        ValidateSearchThreshold(threshold);
    }
    return {params, threshold};
}

HGraphContinueSession::HGraphContinueSession(
    const HGraph& hgraph,
    const std::shared_ptr<const InnerIndexInterface>& owner,
    DatasetPtr query,
    std::string parameters,
    FilterPtr filter,
    Allocator* allocator)
    : hgraph_(owner, &hgraph),
      query_(std::move(query)),
      parameters_(std::move(parameters)),
      filter_(std::move(filter)),
      allocator_(allocator) {
}

HGraphContinueSession::~HGraphContinueSession() {
    Close();
}

void
HGraphContinueSession::Initialize(int64_t k, int64_t ef_search) {
    const void* data = hgraph_->get_data(query_);
    traversal_ = std::make_unique<Traversal>(data);
    auto& state = *traversal_;
    default_effort_ = static_cast<uint64_t>(std::max(k, ef_search));
    // Dense logical id domain is stable while this immutable session is open.
    // Force-removal/recycling is rejected at Open.
    const auto count = hgraph_->total_count_.load();
    state.scores.resize(count);
    state.scored.resize(count);
    state.discovered.resize(count);
    state.delivered.resize(count);
    state.unreturnable.resize(count);
    state.precise_scores.resize(count);
    state.precisely_scored.resize(count);
    if (count == 0) {
        state.routed = true;
        return;
    }
}

void
HGraphContinueSession::Close() noexcept {
    closed_counters_ = GetCounters();
    closed_ = true;
    // Computers, filter wrappers and copied query can reference backend resources.
    traversal_.reset();
    query_.reset();
    filter_.reset();
    hgraph_.reset();
}

HGraphContinueSession::Counters
HGraphContinueSession::GetCounters() const noexcept {
    if (traversal_ == nullptr) {
        return closed_counters_;
    }
    const auto& state = *traversal_;
    return {state.routing_runs,
            state.computers.Size(),
            state.score_count,
            state.precise_count,
            state.expansion_count};
}

std::string
HGraphContinueSession::Statistics(const Counters& before) const {
    const auto now = GetCounters();
    uint64_t frontier = 0;
    uint64_t pending = 0;
    uint64_t payload = 0;
    if (traversal_ != nullptr) {
        const auto& state = *traversal_;
        frontier = state.frontier.size();
        pending = state.pending.size();
        // Logical payload only: excludes container capacity/node overhead and computers.
        payload = SaturatingSessionPayload(payload, state.scores.size(), sizeof(float));
        payload = SaturatingSessionPayload(payload, state.scored.size(), 1);
        payload = SaturatingSessionPayload(payload, state.discovered.size(), 1);
        payload = SaturatingSessionPayload(payload, state.delivered.size(), 1);
        payload = SaturatingSessionPayload(payload, state.precise_scores.size(), sizeof(float));
        payload = SaturatingSessionPayload(payload, state.precisely_scored.size(), 1);
        payload = SaturatingSessionPayload(payload, state.unreturnable.size(), 1);
        payload = SaturatingSessionPayload(payload, frontier, sizeof(Traversal::Record));
        payload = SaturatingSessionPayload(payload, pending, sizeof(Traversal::Record));
    }
    // Work counters are monotonic; gauges below are not used in unsigned deltas.
    JsonType stats;
    stats["session_routing_runs"].SetUint64(now.routing);
    stats["session_computers"].SetUint64(now.computers);
    stats["session_scored"].SetUint64(now.scored);
    stats["session_reordered"].SetUint64(now.reordered);
    stats["session_expanded"].SetUint64(now.expanded);
    stats["session_round_routing_runs"].SetUint64(now.routing - before.routing);
    stats["session_round_computer_creations"].SetUint64(now.computers - before.computers);
    stats["session_round_scored"].SetUint64(now.scored - before.scored);
    stats["session_round_reordered"].SetUint64(now.reordered - before.reordered);
    stats["session_round_expanded"].SetUint64(now.expanded - before.expanded);
    stats["session_frontier_nodes"].SetUint64(frontier);
    stats["session_pending_nodes"].SetUint64(pending);
    stats["session_undelivered_nodes"].SetUint64(traversal_ == nullptr ? 0
                                                                       : traversal_->undelivered);
    stats["session_state_payload_bytes"].SetUint64(payload);
    stats["session_traversal_exhausted"].SetBool(
        traversal_ == nullptr or (traversal_->routed and traversal_->frontier.empty()));
    return stats.Dump();
}

bool
HGraphContinueSession::HasMore() const noexcept {
    return not closed_ and traversal_ != nullptr and
           (not traversal_->routed or not traversal_->frontier.empty() or
            traversal_->undelivered != 0);
}

tl::expected<DatasetPtr, Error>
HGraphContinueSession::Next(uint64_t max_candidates) {
    return NextImpl({max_candidates, parameters_, filter_}, default_effort_);
}

tl::expected<DatasetPtr, Error>
HGraphContinueSession::Next(const SearchSessionNextOptions& options) {
    return NextImpl(options, 1);
}

tl::expected<DatasetPtr, Error>
HGraphContinueSession::NextImpl(const SearchSessionNextOptions& options, uint64_t minimum_effort) {
    const auto max_candidates = options.max_candidates;
    std::optional<ValidatedParameters> validated;
    // Validation has no traversal side effects and must not close a valid session.
    if (HasMore()) {
        if (max_candidates == 0 or
            max_candidates > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return tl::unexpected(
                Error(ErrorType::INVALID_ARGUMENT, "max_candidates must be a positive int64"));
        }
        try {
            validated = ValidateParameters(options.search_parameters);
        } catch (const VsagException& e) {
            return tl::unexpected(e.error_);
        } catch (const std::bad_alloc& e) {
            return tl::unexpected(Error(ErrorType::NO_ENOUGH_MEMORY, e.what()));
        } catch (const std::exception& e) {
            return tl::unexpected(Error(ErrorType::INVALID_ARGUMENT, e.what()));
        } catch (...) {
            return tl::unexpected(Error(ErrorType::INVALID_ARGUMENT, "invalid session options"));
        }
    }
    try {
        if (not HasMore()) {
            return Dataset::Make()->NumElements(1)->Dim(0)->Statistics(Statistics(GetCounters()));
        }
        auto& state = *traversal_;
        const auto before = GetCounters();
        auto validate_id = [&](InnerIdType id) { ValidateSessionId(id, state.scores.size()); };
        if (not validated.has_value()) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "session parameters were not validated");
        }
        const auto& params = validated->search;
        state.effort = std::max(minimum_effort, static_cast<uint64_t>(params.ef_search));
        state.threshold = validated->threshold;
        // Replace with a successful construction so the old wrapper lives until swap.
        auto new_filter =
            hgraph_->create_search_filter(options.filter, params.use_extra_info_filter);
        state.filter.swap(new_filter);
        state.pending.clear();
        if (not state.routed) {
            state.coarse =
                state.computers.Acquire(hgraph_->basic_flatten_codes_, hgraph_->get_data(query_));
        }
        const bool reorder = hgraph_->use_reorder_ and params.enable_reorder;
        // Retain the precise lease across toggles to reuse the prepared computer.
        if (reorder and state.precise.computer == nullptr) {
            state.precise =
                state.computers.Acquire(hgraph_->get_reorder_codes(), hgraph_->get_data(query_));
        }
        auto score = [&](InnerIdType id) {
            validate_id(id);
            if (state.scored[id] == 0) {
                state.coarse.owner->Query(&state.scores[id], state.coarse.computer, &id, 1);
                state.scored[id] = 1;
                ++state.score_count;
            }
            return state.scores[id];
        };
        auto accept = [&](InnerIdType id, float coarse) {
            validate_id(id);
            if (state.delivered[id] != 0 or state.unreturnable[id] != 0) {
                return;
            }
            // MARK_REMOVE is permanent for this immutable session, unlike user filters.
            // Keep the vertex in the frontier so removed vertices can still bridge the graph.
            if (hgraph_->label_table_->IsRemoved(id)) {
                state.unreturnable[id] = 1;
                --state.undelivered;
                return;
            }
            // NaN cannot be returned under any threshold. Retire only when no
            // alternate output cell can rescue it; it remains a traversal bridge.
            const bool alternate =
                hgraph_->use_reorder_ and
                hgraph_->get_reorder_codes().get() != hgraph_->basic_flatten_codes_.get();
            // When reorder is off but alternate codes exist, coarse NaN with no precise
            // score yet is still unrescuable: the per-call reorder flag prevents scoring.
            const bool coarse_nan = is_nan_distance(coarse);
            if ((not reorder and alternate and coarse_nan and state.precisely_scored[id] == 0) or
                (coarse_nan and (not alternate or (state.precisely_scored[id] != 0 and
                                                   is_nan_distance(state.precise_scores[id]))))) {
                state.unreturnable[id] = 1;
                --state.undelivered;
                return;
            }
            if (state.filter != nullptr and not state.filter->CheckValid(id)) {
                return;
            }
            float distance = coarse;
            if (reorder and state.precise.owner.get() != state.coarse.owner.get()) {
                if (state.precisely_scored[id] == 0) {
                    state.precise.owner->Query(
                        &state.precise_scores[id], state.precise.computer, &id, 1);
                    state.precisely_scored[id] = 1;
                    ++state.precise_count;
                }
                distance = state.precise_scores[id];
                if (is_nan_distance(coarse) and is_nan_distance(distance)) {
                    state.unreturnable[id] = 1;
                    --state.undelivered;
                    return;
                }
            }
            if (not is_nan_distance(distance) and
                (not state.threshold.has_value() or
                 (is_finite_distance(distance) and distance <= state.threshold.value()))) {
                state.pending.emplace(distance, id);
            }
        };
        // Eligibility is call-local, never cached by filter object identity.
        // This POC deliberately scans the dense O(N) domain, including earlier rejects.
        for (uint64_t id = 0; id < state.discovered.size(); ++id) {
            if (state.discovered[id] != 0 and state.delivered[id] == 0) {
                accept(static_cast<InnerIdType>(id), state.scores[id]);
            }
        }
        auto discover = [&](InnerIdType id) {
            validate_id(id);
            if (state.discovered[id] != 0) {
                return;
            }
            const float coarse = score(id);
            // Discovery persists across calls: each vertex enters the frontier at most once.
            state.discovered[id] = 1;
            ++state.undelivered;
            // Non-finite bridges use -max_float in this min-heap and are explored promptly.
            state.frontier.emplace(-traversal_priority(coarse), id);
            accept(id, coarse);
            if (hgraph_->support_duplicate_) {
                for (auto alias : hgraph_->bottom_graph_->GetDuplicateIds(id)) {
                    validate_id(alias);
                    if (state.discovered[alias] == 0) {
                        state.discovered[alias] = 1;
                        ++state.undelivered;
                        // The representative covers the neighborhood; aliases need no frontier entry.
                        // Ordinary duplicate storage has its own score and extra info.
                        accept(alias, score(alias));
                    }
                }
            }
        };
        Vector<InnerIdType> neighbors(allocator_);
        if (not state.routed) {
            ++state.routing_runs;
            auto entry = hgraph_->entry_point_id_;
            validate_id(entry);
            for (int64_t level = static_cast<int64_t>(hgraph_->route_graphs_.size()) - 1;
                 level >= 0;
                 --level) {
                entry = FindFiniteSessionRoute(entry, score, [&](InnerIdType id) {
                    validate_id(id);
                    Vector<InnerIdType> route_neighbors(allocator_);
                    hgraph_->route_graphs_[level]->GetNeighbors(id, route_neighbors);
                    return route_neighbors;
                });
                bool improved = true;
                while (improved) {
                    improved = false;
                    float best = score(entry);
                    if (not is_finite_distance(best)) {
                        best = std::numeric_limits<float>::infinity();
                    }
                    validate_id(entry);
                    hgraph_->route_graphs_[level]->GetNeighbors(entry, neighbors);
                    for (auto id : neighbors) {
                        const float distance = score(id);
                        if (is_finite_distance(distance) and distance < best) {
                            best = distance;
                            entry = id;
                            improved = true;
                        }
                    }
                }
            }
            state.routed = true;
            discover(entry);
        }
        // Every nonterminal call resumes actual expansion, even if pending results
        // remain. ef_search is an expansion tranche, not a precomputed result size.
        const uint64_t effort = std::max(state.effort, max_candidates);
        uint64_t expanded = 0;
        while (not state.frontier.empty() and
               (expanded < effort or state.pending.size() < max_candidates)) {
            const auto id = state.frontier.top().second;
            validate_id(id);
            state.frontier.pop();
            ++state.expansion_count;
            ++expanded;
            hgraph_->bottom_graph_->GetNeighbors(id, neighbors);
            for (auto neighbor : neighbors) {
                discover(neighbor);
            }
        }
        const auto count = std::min(max_candidates, static_cast<uint64_t>(state.pending.size()));
        // Results intentionally use independent new[] ownership, never the index allocator.
        auto result =
            Dataset::Make()->Owner(true)->NumElements(1)->Dim(static_cast<int64_t>(count));
        if (count > 0) {
            const auto extra_size = hgraph_->extra_info_size_;
            // Guard Dataset int64 width and product later validated by CheckedSessionBytes.

            (void)CheckedSessionBytes(count, sizeof(int64_t));
            (void)CheckedSessionBytes(count, sizeof(float));
            const auto extra_bytes = CheckedSessionBytes(count, extra_size);
            // Guard registration too: Dataset setters may allocate their metadata entries.
            auto ids_owner = std::make_unique<int64_t[]>(count);
            result->Ids(ids_owner.get());
            auto* ids = ids_owner.release();
            auto distances_owner = std::make_unique<float[]>(count);
            result->Distances(distances_owner.get());
            auto* distances = distances_owner.release();
            char* extra_infos = nullptr;
            if (extra_size > 0) {
                auto extra_owner = std::make_unique<char[]>(extra_bytes);
                result->ExtraInfos(extra_owner.get());
                extra_infos = extra_owner.release();
                result->ExtraInfoSize(static_cast<int64_t>(extra_size));
            }
            for (uint64_t i = 0; i < count; ++i) {
                const auto record = *state.pending.begin();
                validate_id(record.second);
                ids[i] = hgraph_->label_table_->GetLabelById(record.second);
                distances[i] = record.first;
                if (extra_size > 0) {
                    hgraph_->extra_infos_->GetExtraInfoById(record.second,
                                                            extra_infos + i * extra_size);
                }
                state.delivered[record.second] = 1;
                --state.undelivered;
                state.pending.erase(state.pending.begin());
            }
        }
        result->Statistics(Statistics(before));
        state.filter.reset();  // Per-call filters need not live until the next call/Close.
        return result;
    } catch (const VsagException& e) {
        Close();
        return tl::unexpected(e.error_);
    } catch (const std::bad_alloc& e) {
        Close();
        return tl::unexpected(Error(ErrorType::NO_ENOUGH_MEMORY, e.what()));
    } catch (const std::exception& e) {
        Close();
        return tl::unexpected(Error(ErrorType::INTERNAL_ERROR, e.what()));
    } catch (...) {
        Close();
        return tl::unexpected(
            Error(ErrorType::INTERNAL_ERROR, "unknown exception during session traversal"));
    }
}
}  // namespace vsag
