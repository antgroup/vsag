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

#include <fmt/format.h>

#include <algorithm>
#include <cmath>

#include "attr/argparse.h"
#include "dataset_impl.h"
#include "hgraph.h"  // IWYU pragma: keep
#include "impl/filter/iterator_filter.h"
#include "impl/heap/standard_heap.h"
#include "impl/query_computer_pool.h"
#include "impl/reasoning/reasoning_context.h"
#include "utils/search_threshold.h"
#include "utils/util_functions.h"

namespace vsag {

static DatasetPtr
make_empty_dataset_with_stats(const SearchStatistics& stats) {
    auto dataset_result = DatasetImpl::MakeEmptyDataset();
    dataset_result->Statistics(stats.Dump());
    return dataset_result;
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter) const {
    return KnnSearch(query, k, parameters, filter, nullptr);
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter,
                  Allocator* allocator) const {
    SearchRequest req;
    req.query_ = query;
    req.topk_ = k;
    req.filter_ = filter;
    req.params_str_ = parameters;
    req.threshold_ = ParseSearchThreshold(parameters);
    req.search_allocator_ = allocator;
    return this->SearchWithRequest(req);
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter,
                  Allocator* allocator,
                  IteratorContext*& iter_ctx,
                  bool is_last_filter) const {
    SearchStatistics stats;
    QueryContext ctx{.alloc = allocator_, .stats = &stats};
    if (allocator != nullptr) {
        ctx.alloc = allocator;
    }

    if (GetNumElements() == 0) {
        return make_empty_dataset_with_stats(stats);
    }
    this->validate_knn_args(query, k);

    auto params = HGraphSearchParameters::FromJson(parameters);
    const auto threshold = ParseSearchThreshold(parameters);
    ctx.rabitq_error_rate = params.rabitq_error_rate;
    CHECK_ARGUMENT(  // NOLINT
        params.ef_search >= 1,
        fmt::format("ef_search({}) must be at least 1", params.ef_search));

    std::shared_lock<std::shared_mutex> force_remove_rlock;
    std::shared_lock<std::shared_mutex> shared_lock;
    if (!this->immutable_.load(std::memory_order_acquire)) {
        if (this->support_force_remove()) {
            force_remove_rlock = std::shared_lock<std::shared_mutex>(this->force_remove_mutex_);
        }
        shared_lock = this->acquire_global_read_lock();
    }
    k = std::min(k, GetNumElements());

    FilterPtr ft = this->create_search_filter(filter, params.use_extra_info_filter);

    if (iter_ctx == nullptr) {
        auto cur_count = this->total_count_.load();

        if (cur_count == 0) {
            return make_empty_dataset_with_stats(stats);
        }
        auto* new_ctx = new IteratorFilterContext();
        if (auto ret = new_ctx->init(cur_count, params.ef_search, ctx.alloc); not ret.has_value()) {
            delete new_ctx;
            throw vsag::VsagException(ErrorType::INTERNAL_ERROR,
                                      "failed to init IteratorFilterContext");
        }
        iter_ctx = new_ctx;
    }

    auto* iter_filter_ctx = static_cast<IteratorFilterContext*>(iter_ctx);
    const auto* query_data = get_data(query);
    QueryComputerPool query_computer_pool(query_data, &stats);
    ctx.computer_pool = &query_computer_pool;
    // Note: brute_force_threshold is intentionally not applied here. The
    // iterator KnnSearch API pages results across multiple calls via
    // iter_filter_ctx; a single brute-force sweep would either need to drive
    // that pagination state itself or be wasted on subsequent calls. The
    // non-iterator KnnSearch overload (which delegates to SearchWithRequest)
    // still benefits from the brute-force fallback.
    while (true) {
        auto search_result = DistanceHeap::MakeInstanceBySize<true, false>(ctx.alloc, k);
        if (is_last_filter) {
            while (!iter_filter_ctx->Empty()) {
                uint32_t cur_inner_id = iter_filter_ctx->GetTopID();
                float cur_dist = iter_filter_ctx->GetTopDist();
                search_result->Push(cur_dist, cur_inner_id);
                iter_filter_ctx->PopDiscard();
            }
        } else {
            InnerSearchParam search_param;
            search_param.ep = this->entry_point_id_;
            search_param.topk = 1;
            search_param.ef = 1;
            search_param.is_inner_id_allowed = nullptr;
            search_param.enable_rabitq_one_bit_search = params.rabitq_one_bit_search;
            if (search_param.ep == INVALID_ENTRY_POINT) {
                return make_empty_dataset_with_stats(stats);
            }
            if (iter_filter_ctx->IsFirstUsed()) {
                ScopedDistancePhase routing_phase(ctx, DistanceEvaluationPhase::ROUTING);
                for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
                    auto result = this->search_one_graph(query_data,
                                                         this->route_graphs_[i],
                                                         this->basic_flatten_codes_,
                                                         search_param,
                                                         (VisitedListPtr) nullptr,
                                                         &ctx);
                    // An unrankable route seed can still bridge to finite bottom-layer results.
                    if (not result->Empty()) {
                        search_param.ep = result->Top().second;
                    }
                }
            }

            search_param.ef = std::max(params.ef_search, k);
            search_param.is_inner_id_allowed = ft;
            search_param.distance_threshold = threshold;
            search_param.topk = static_cast<int64_t>(search_param.ef);
            search_param.parallel_search_thread_count = params.parallel_search_thread_count;
            search_param.enable_reorder = params.enable_reorder;
            search_param.enable_rabitq_one_bit_search = params.rabitq_one_bit_search;
            search_param.skip_ratio = params.skip_ratio;
            search_param.skip_strategy_type = params.skip_strategy_type;

            DistanceRecordVector rabitq_lower_bound_candidates(ctx.alloc);
            auto* rabitq_lower_bound_candidates_ptr =
                search_param.enable_rabitq_one_bit_search and use_reorder_ and
                        search_param.enable_reorder and reorder_by_base_
                    ? &rabitq_lower_bound_candidates
                    : nullptr;

            search_result = this->search_one_graph(query_data,
                                                   this->bottom_graph_,
                                                   this->basic_flatten_codes_,
                                                   search_param,
                                                   iter_filter_ctx,
                                                   &ctx,
                                                   rabitq_lower_bound_candidates_ptr);

            if (use_reorder_ and search_param.enable_reorder) {
                this->reorder(query_data,
                              this->get_reorder_codes(),
                              search_result,
                              k,
                              iter_filter_ctx,
                              ctx,
                              rabitq_lower_bound_candidates_ptr,
                              threshold);
            } else if (search_param.enable_reorder and params.rabitq_one_bit_search) {
                this->reorder(query_data,
                              this->basic_flatten_codes_,
                              search_result,
                              k,
                              iter_filter_ctx,
                              ctx,
                              nullptr,
                              threshold);
            }
        }

        if (threshold.has_value()) {
            DistanceRecordVector valid_records(ctx.alloc);
            valid_records.reserve(search_result->Size());
            while (not search_result->Empty()) {
                const auto record = search_result->Top();
                search_result->Pop();
                if (std::isfinite(record.first) and record.first <= threshold.value()) {
                    valid_records.push_back(record);
                } else {
                    iter_filter_ctx->SetPoint(record.second);
                }
            }
            for (const auto& record : valid_records) {
                search_result->Push(record);
            }
        }
        while (search_result->Size() > k) {
            auto curr = search_result->Top();
            iter_filter_ctx->AddDiscardNode(curr.first, curr.second);
            search_result->Pop();
        }

        // An empty page is terminal to iterator callers, so consume retained traversal state
        // internally until an eligible result is found or the discard heap is exhausted.
        if (search_result->Empty()) {
            iter_filter_ctx->SetOFFFirstUsed();
            if (not iter_filter_ctx->Empty()) {
                continue;
            }
            return make_empty_dataset_with_stats(stats);
        }
        auto count = static_cast<const int64_t>(search_result->Size());
        auto [dataset_results, dists, ids] = create_fast_dataset(count, ctx.alloc);
        char* extra_infos = nullptr;
        if (extra_info_size_ > 0) {
            extra_infos =
                static_cast<char*>(ctx.alloc->Allocate(extra_info_size_ * search_result->Size()));
            dataset_results->ExtraInfos(extra_infos)
                ->ExtraInfoSize(static_cast<int64_t>(extra_info_size_));
        }
        for (int64_t j = count - 1; j >= 0; --j) {
            dists[j] = search_result->Top().first;
            ids[j] = this->label_table_->GetLabelById(search_result->Top().second);
            iter_filter_ctx->SetPoint(search_result->Top().second);
            if (extra_infos != nullptr) {
                this->extra_infos_->GetExtraInfoById(search_result->Top().second,
                                                     extra_infos + extra_info_size_ * j);
            }
            search_result->Pop();
        }
        iter_filter_ctx->SetOFFFirstUsed();

        dataset_results->Statistics(stats.Dump());
        return std::move(dataset_results);
    }
}

template <InnerSearchMode mode>
DistHeapPtr
HGraph::search_one_graph(const void* query,
                         const GraphInterfacePtr& graph,
                         const FlattenInterfacePtr& flatten,
                         InnerSearchParam& inner_search_param,
                         const VisitedListPtr& vt,
                         QueryContext* ctx,
                         DistanceRecordVector* rabitq_lower_bound_candidates) const {
    bool new_visited_list = vt == nullptr;
    VisitedListPtr visited_list;
    if (new_visited_list) {
        visited_list = this->pool_->TakeOne();
    } else {
        visited_list = vt;
        visited_list->Reset();
    }
    DistHeapPtr result = nullptr;
    const bool parallel_search_requested = inner_search_param.parallel_search_thread_count > 1;
    const bool has_parallel_search_executor =
        this->thread_pool_ != nullptr and this->parallel_searcher_ != nullptr;
    if (parallel_search_requested and has_parallel_search_executor) {
        result = this->parallel_searcher_->Search(graph,
                                                  flatten,
                                                  visited_list,
                                                  query,
                                                  inner_search_param,
                                                  this->label_table_,
                                                  ctx,
                                                  rabitq_lower_bound_candidates);
    } else {
        if (parallel_search_requested and ctx != nullptr and ctx->stats != nullptr) {
            ctx->stats->parallel_search_fallback_count.fetch_add(1, std::memory_order_relaxed);
        }
        result = this->searcher_->Search(graph,
                                         flatten,
                                         visited_list,
                                         query,
                                         inner_search_param,
                                         this->label_table_,
                                         ctx,
                                         rabitq_lower_bound_candidates);
    }
    if (new_visited_list) {
        this->pool_->ReturnOne(visited_list);
    }
    return result;
}

template <InnerSearchMode mode>
DistHeapPtr
HGraph::search_one_graph(const void* query,
                         const GraphInterfacePtr& graph,
                         const FlattenInterfacePtr& flatten,
                         InnerSearchParam& inner_search_param,
                         IteratorFilterContext* iter_ctx,
                         QueryContext* ctx,
                         DistanceRecordVector* rabitq_lower_bound_candidates) const {
    auto visited_list = this->pool_->TakeOne();
    if (inner_search_param.parallel_search_thread_count > 1 and ctx != nullptr and
        ctx->stats != nullptr) {
        ctx->stats->parallel_search_fallback_count.fetch_add(1, std::memory_order_relaxed);
    }
    auto result = this->searcher_->Search(graph,
                                          flatten,
                                          visited_list,
                                          query,
                                          inner_search_param,
                                          iter_ctx,
                                          ctx,
                                          rabitq_lower_bound_candidates);
    this->pool_->ReturnOne(visited_list);
    return result;
}

template <InnerSearchMode mode>
DistHeapPtr
HGraph::brute_force_search(const void* query,
                           const FilterPtr& filter,
                           int64_t topk,
                           float radius,
                           QueryContext* ctx,
                           const std::optional<float>& threshold) const {
    std::shared_lock codes_lock(this->persistent_codes_mutex_);
    Allocator* alloc = (ctx != nullptr && ctx->alloc != nullptr) ? ctx->alloc : this->allocator_;

    auto flatten = this->basic_flatten_codes_;
    if (this->has_precise_reorder()) {
        flatten = this->high_precise_codes_;
    }
    if (this->create_new_raw_vector_ && this->raw_vector_ != nullptr) {
        flatten = this->raw_vector_;
    }

    DistHeapPtr result;
    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        result = DistanceHeap::MakeInstanceBySize<true, false>(alloc, -1);
    } else {
        result = DistanceHeap::MakeInstanceBySize<true, true>(alloc, topk);
    }
    if (flatten == nullptr) {
        return result;
    }

    // Add reserves logical ids before their code slots are published.
    auto total =
        this->using_dedup_storage() ? this->GetCodeStorageCounts().first : flatten->TotalCount();
    total = std::min(total, static_cast<InnerIdType>(this->total_count_.load()));
    if (total == 0) {
        return result;
    }

    auto computer_lease = AcquireQueryComputer(flatten, query, ctx);
    const auto& computer = computer_lease.computer;

    constexpr InnerIdType brute_force_batch_size = 64;
    Vector<InnerIdType> batch_ids(brute_force_batch_size, alloc);
    Vector<float> batch_dists(brute_force_batch_size, alloc);

    InnerIdType cursor = 0;
    while (cursor < total) {
        InnerIdType batch_count = 0;
        while (cursor < total && batch_count < brute_force_batch_size) {
            if (filter == nullptr || filter->CheckValid(cursor)) {
                batch_ids[batch_count++] = cursor;
            }
            ++cursor;
        }
        if (batch_count == 0) {
            continue;
        }
        flatten->Query(batch_dists.data(), computer, batch_ids.data(), batch_count, ctx);
        for (InnerIdType i = 0; i < batch_count; ++i) {
            float dist = batch_dists[i];
            InnerIdType inner_id = batch_ids[i];
            if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
                if (dist <= radius) {
                    result->Push(dist, inner_id);
                }
            } else if (not threshold.has_value() or
                       (std::isfinite(dist) and dist <= threshold.value())) {
                result->Push(dist, inner_id);
            }
        }
    }
    return result;
}

DatasetPtr
HGraph::RangeSearch(const DatasetPtr& query,
                    float radius,
                    const std::string& parameters,
                    const FilterPtr& filter,
                    int64_t limited_size) const {
    SearchRequest req;
    req.mode_ = SearchMode::RANGE_SEARCH;
    req.query_ = query;
    req.radius_ = radius;
    req.limited_size_ = limited_size;
    req.params_str_ = parameters;
    if (filter != nullptr) {
        req.filter_ = filter;
    }
    return this->SearchWithRequest(req);
}

[[nodiscard]] DatasetPtr
HGraph::SearchWithRequest(const SearchRequest& request) const {
    ValidateSearchThreshold(request.threshold_);
    SearchStatistics stats;
    QueryContext ctx{.alloc = this->allocator_, .stats = &stats};
    if (request.search_allocator_ != nullptr) {
        ctx.alloc = request.search_allocator_;
    }

    const auto& query = request.query_;
    bool is_range = (request.mode_ == SearchMode::RANGE_SEARCH);
    auto k = request.topk_;
    const bool use_custom_distance = request.distance_batch_func_ != nullptr;
    const bool reasoning_requested = not request.expected_labels_.empty();

    if (use_custom_distance) {
        CHECK_ARGUMENT(request.distance_batch_size_ > 0,
                       "distance_batch_size must be greater than 0");
        CHECK_ARGUMENT(not is_range, "HGraph custom distance only supports KNN search");
    }

    if (is_range) {
        if (not use_custom_distance) {
            this->validate_range_args(query, request.radius_, request.limited_size_);
        }
    } else {
        if (not use_custom_distance) {
            this->validate_knn_args(query, k);
        } else {
            CHECK_ARGUMENT(k > 0, "topk must be greater than 0");
        }
    }

    auto params = HGraphSearchParameters::FromJson(request.params_str_);
    ctx.rabitq_error_rate = params.rabitq_error_rate;

    if (use_custom_distance) {
        CHECK_ARGUMENT(params.parallel_search_thread_count == 1,
                       "HGraph custom query distance does not support parallel search");
        CHECK_ARGUMENT(params.brute_force_threshold <= 0.0F,
                       "HGraph custom query distance does not support brute_force_threshold");
    }

    CHECK_ARGUMENT(  // NOLINT
        params.ef_search >= 1,
        fmt::format("ef_search({}) must be at least 1", params.ef_search));

    std::shared_lock<std::shared_mutex> force_remove_rlock;
    std::shared_lock<std::shared_mutex> shared_lock;
    if (!this->immutable_.load(std::memory_order_acquire)) {
        if (this->support_force_remove()) {
            force_remove_rlock = std::shared_lock<std::shared_mutex>(this->force_remove_mutex_);
        }
        shared_lock = this->acquire_global_read_lock();
    }
    const auto element_count = GetNumElements();
    if (element_count == 0) {
        return make_empty_dataset_with_stats(stats);
    }
    k = std::min(k, element_count);
    const auto* raw_query = use_custom_distance ? nullptr : get_data(query);
    QueryComputerPool query_computer_pool(raw_query, &stats);
    if (not use_custom_distance) {
        ctx.computer_pool = &query_computer_pool;
    }
    if (this->entry_point_id_ == INVALID_ENTRY_POINT) {
        return make_empty_dataset_with_stats(stats);
    }

    // Setup reasoning context (KNN only)
    std::shared_ptr<ReasoningContext> reasoning_ctx;
    if (not is_range and not request.expected_labels_.empty()) {
        reasoning_ctx = std::make_shared<ReasoningContext>(this->allocator_);
        reasoning_ctx->SetSearchParams(
            k, "HGraph", use_custom_distance ? false : use_reorder_, request.filter_ != nullptr);

        UnorderedMap<int64_t, InnerIdType> label_to_inner_id(this->allocator_);
        for (const auto& label : request.expected_labels_) {
            auto [success, inner_id] = label_table_->TryGetIdByLabel(label, true);
            if (success) {
                label_to_inner_id[label] = inner_id;
            }
        }

        Vector<int64_t> expected_labels_vec(
            request.expected_labels_.begin(), request.expected_labels_.end(), this->allocator_);
        reasoning_ctx->InitializeExpectedTargets(expected_labels_vec, label_to_inner_id);

        FlattenInterfacePtr precise_flatten = nullptr;
        ComputerLease computer_lease;
        ComputerInterfacePtr computer = nullptr;
        if (not use_custom_distance) {
            precise_flatten = this->get_precise_codes();
            computer_lease = AcquireQueryComputer(precise_flatten, raw_query, &ctx);
            computer = computer_lease.computer;
        }
        for (const auto& pair : label_to_inner_id) {
            float dist = 0.0F;
            const auto inner_id = pair.second;
            if (use_custom_distance) {
                const auto label = this->label_table_->GetLabelById(inner_id);
                request.distance_batch_func_(&label, 1, &dist);
                CHECK_ARGUMENT(std::isfinite(dist), "distance callback must return finite scores");
                stats.AddDistance(SearchStatistics::DistancePhase::APPROXIMATE,
                                  DistanceEvaluationBackend::UNKNOWN);
            } else {
                precise_flatten->Query(&dist, computer, &inner_id, 1, &ctx);
            }
            // [reasoning] SetTrueDistance
            reasoning_ctx->SetTrueDistance(inner_id, dist);
        }
        ctx.reasoning_ctx = reasoning_ctx.get();
    }

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_id_;
    search_param.topk = 1;
    search_param.ef = 1;
    search_param.is_inner_id_allowed = nullptr;
    search_param.enable_rabitq_one_bit_search =
        use_custom_distance ? false : params.rabitq_one_bit_search;
    search_param.distance_batch_func = request.distance_batch_func_;
    search_param.distance_batch_size = request.distance_batch_size_;

    struct HGraphVisitedListGuard {
        std::shared_ptr<VisitedListPool> pool;
        VisitedListPtr visited_list;

        void
        Release() {
            if (visited_list != nullptr) {
                pool->ReturnOne(visited_list);
                visited_list.reset();
            }
        }

        ~HGraphVisitedListGuard() {
            Release();
        }
    };
    HGraphVisitedListGuard vt_guard{this->pool_, this->pool_->TakeOne()};
    auto& vt = vt_guard.visited_list;

    ctx.distance_phase = DistanceEvaluationPhase::ROUTING;
    for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
        auto result = this->search_one_graph(
            raw_query, this->route_graphs_[i], this->basic_flatten_codes_, search_param, vt, &ctx);
        // An unrankable route seed can still bridge to finite bottom-layer results.
        if (not result->Empty()) {
            search_param.ep = result->Top().second;
        }
    }
    ctx.distance_phase = DistanceEvaluationPhase::APPROXIMATE;

    FilterPtr ft = this->create_search_filter(request.filter_, params.use_extra_info_filter);

    if (request.enable_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        auto& schema = this->attr_filter_index_->field_type_map_;
        auto expr = AstParse(request.attribute_filter_str_, &schema);
        auto executor = Executor::MakeInstance(this->allocator_, expr, this->attr_filter_index_);
        executor->Init();
        search_param.executors.emplace_back(executor);
    }

    if (is_range) {
        search_param.ef = std::max(params.ef_search, request.limited_size_);
        search_param.is_inner_id_allowed = ft;
        search_param.radius = request.radius_;
        search_param.search_mode = RANGE_SEARCH;
        search_param.consider_duplicate = true;
        search_param.range_search_limit_size = static_cast<int>(request.limited_size_);
        search_param.parallel_search_thread_count = params.parallel_search_thread_count;
        search_param.enable_reorder = use_custom_distance ? false : params.enable_reorder;
        search_param.enable_rabitq_one_bit_search =
            use_custom_distance ? false : params.rabitq_one_bit_search;
    } else {
        search_param.ef = std::max(params.ef_search, k);
        if (this->use_conjugate_graph_ and params.use_conjugate_graph_search) {
            search_param.ef = std::max(search_param.ef, static_cast<uint64_t>(LOOK_AT_K));
        }
        search_param.is_inner_id_allowed = ft;
        search_param.distance_threshold = request.threshold_;
        search_param.topk = static_cast<int64_t>(search_param.ef);
        if (params.topk_factor > 1.0F) {
            search_param.topk =
                std::min(search_param.topk,
                         static_cast<int64_t>(static_cast<float>(k) * params.topk_factor));
        }
        if (this->use_conjugate_graph_ and params.use_conjugate_graph_search) {
            search_param.topk = std::max(search_param.topk, LOOK_AT_K);
        }
        search_param.enable_reorder = use_custom_distance ? false : params.enable_reorder;
        search_param.consider_duplicate = true;
        search_param.enable_rabitq_one_bit_search =
            use_custom_distance ? false : params.rabitq_one_bit_search;
        if (params.enable_time_record) {
            search_param.time_cost = std::make_shared<Timer>();
            search_param.time_cost->SetThreshold(params.timeout_ms);
            stats.is_timeout.store(false, std::memory_order_relaxed);
        }
        search_param.parallel_search_thread_count = params.parallel_search_thread_count;

        if (static_cast<uint64_t>(params.hops_limit) <= static_cast<uint64_t>(params.ef_search)) {
            search_param.hops_limit = std::numeric_limits<uint32_t>::max();
            if (params.hops_limit != std::numeric_limits<uint32_t>::max()) {
                logger::warn(fmt::format(
                    "hops_limit({}) is not greater than ef_search({}), ignoring hops_limit",
                    params.hops_limit,
                    params.ef_search));
            }
        } else {
            search_param.hops_limit = params.hops_limit;
        }
    }

    search_param.skip_ratio = params.skip_ratio;
    search_param.skip_strategy_type = params.skip_strategy_type;

    DistanceRecordVector rabitq_lower_bound_candidates(ctx.alloc);
    auto* rabitq_lower_bound_candidates_ptr =
        search_param.enable_rabitq_one_bit_search and use_reorder_ and
                search_param.enable_reorder and reorder_by_base_
            ? &rabitq_lower_bound_candidates
            : nullptr;

    DistHeapPtr search_result;
    bool brute_force_used = false;
    MCIHybridSearchResult mci_result(params, ft);
    if (not use_custom_distance) {
        if (params.brute_force_threshold > 0.0F and
            mci_result.valid_ratio <= params.brute_force_threshold) {
            if (is_range) {
                search_result = this->brute_force_search<InnerSearchMode::RANGE_SEARCH>(
                    raw_query, ft, request.limited_size_, request.radius_, &ctx);
            } else {
                search_result = this->brute_force_search<InnerSearchMode::KNN_SEARCH>(
                    raw_query, ft, k, 0.0F, &ctx, request.threshold_);
            }
            brute_force_used = true;
            mci_result.route = "brute_force";
        } else {
            mci_result = this->try_mci_search(request, params, ft, raw_query, search_param, &ctx);
            if (mci_result.route == "mci") {
                search_result = std::move(mci_result.result);
            } else {
                search_result = this->search_one_graph(raw_query,
                                                       this->bottom_graph_,
                                                       this->basic_flatten_codes_,
                                                       search_param,
                                                       vt,
                                                       &ctx,
                                                       rabitq_lower_bound_candidates_ptr);
            }
        }
    } else {
        search_result = this->search_one_graph(raw_query,
                                               this->bottom_graph_,
                                               this->basic_flatten_codes_,
                                               search_param,
                                               vt,
                                               &ctx,
                                               rabitq_lower_bound_candidates_ptr);
    }
    vt_guard.Release();

    // Reorder
    if (mci_result.route != "mci" and not brute_force_used and use_reorder_ and
        search_param.enable_reorder) {
        auto limit = is_range ? request.limited_size_ : k;
        if (not is_range and this->use_conjugate_graph_ and params.use_conjugate_graph_search) {
            limit = std::max(limit, LOOK_AT_K);
        }
        auto reorder_threshold = is_range ? std::nullopt : request.threshold_;
        this->reorder(raw_query,
                      this->get_reorder_codes(),
                      search_result,
                      limit,
                      nullptr,
                      ctx,
                      rabitq_lower_bound_candidates_ptr,
                      reorder_threshold);
    } else if (mci_result.route != "mci" and not brute_force_used and
               search_param.enable_reorder and params.rabitq_one_bit_search) {
        auto limit = is_range ? request.limited_size_ : k;
        if (not is_range and this->use_conjugate_graph_ and params.use_conjugate_graph_search) {
            limit = std::max(limit, LOOK_AT_K);
        }
        auto reorder_threshold = is_range ? std::nullopt : request.threshold_;
        this->reorder(raw_query,
                      this->basic_flatten_codes_,
                      search_result,
                      limit,
                      nullptr,
                      ctx,
                      nullptr,
                      reorder_threshold);
    }

    if (not is_range and this->use_conjugate_graph_ and params.use_conjugate_graph_search and
        not search_result->Empty()) {
        std::priority_queue<std::pair<float, LabelType>> label_results;
        std::shared_lock label_lock(this->label_lookup_mutex_);
        while (not search_result->Empty()) {
            const auto record = search_result->Top();
            search_result->Pop();
            label_results.emplace(record.first, this->label_table_->GetLabelById(record.second));
        }

        const auto flatten = use_custom_distance ? nullptr : this->get_precise_codes();
        ComputerLease computer_lease;
        ComputerInterfacePtr computer = nullptr;
        if (not use_custom_distance) {
            computer_lease = AcquireQueryComputer(flatten, raw_query, &ctx);
            computer = computer_lease.computer;
        }
        Filter* attribute_filter = nullptr;
        if (not search_param.executors.empty() and search_param.executors[0] != nullptr) {
            search_param.executors[0]->Clear();
            attribute_filter = search_param.executors[0]->Run();
        }
        const auto is_allowed = [&](InnerIdType inner_id) {
            return (ft == nullptr or ft->CheckValid(inner_id)) and
                   (attribute_filter == nullptr or attribute_filter->CheckValid(inner_id));
        };
        const auto distance_of_label = [&](int64_t label) {
            const auto [found, inner_id] = this->label_table_->TryGetIdByLabel(label, true);
            if (not found or not is_allowed(inner_id)) {
                return std::numeric_limits<float>::max();
            }
            float distance = std::numeric_limits<float>::max();
            if (use_custom_distance) {
                request.distance_batch_func_(&label, 1, &distance);
            } else {
                flatten->Query(&distance, computer, &inner_id, 1, &ctx);
            }
            if (request.threshold_.has_value() and distance > request.threshold_.value()) {
                return std::numeric_limits<float>::max();
            }
            return distance;
        };
        {
            std::shared_lock graph_lock(this->conjugate_graph_mutex_);
            (void)this->conjugate_graph_->EnhanceResult(label_results, distance_of_label);
        }
        while (not label_results.empty()) {
            const auto record = label_results.top();
            label_results.pop();
            const auto [found, inner_id] = this->label_table_->TryGetIdByLabel(record.second, true);
            if (found and is_allowed(inner_id)) {
                search_result->Push(record.first, inner_id);
            }
        }
    }

    // Trim and pack results
    if (is_range) {
        while (not search_result->Empty() and
               search_result->Top().first > request.radius_ + THRESHOLD_ERROR) {
            search_result->Pop();
        }
        if (request.limited_size_ > 0) {
            while (search_result->Size() > static_cast<uint64_t>(request.limited_size_)) {
                search_result->Pop();
            }
        }
        auto result = this->pack_knn_result_with_extra_info(search_result, ctx.alloc);
        result->Statistics(mci_result.MakeStatistics(stats).Dump());
        if (reasoning_requested) {
            // [reasoning] HGraph range search does not build a reasoning context;
            // attach a minimal status report so callers can distinguish "skipped"
            // from "ok".
            result->Reasoning(ReasoningContext::MakeStatusReport(
                ReasoningReportStatus::kSkippedRangeSearch, "HGraph"));
        }
        return result;
    }

    // NaN is unordered and cannot be returned. Infinity remains a valid legacy result only when
    // threshold filtering is absent; the searcher has already kept it out of threshold heaps.
    DistanceRecordVector finite_records(ctx.alloc);
    finite_records.reserve(search_result->Size());
    while (not search_result->Empty()) {
        const auto record = search_result->Top();
        search_result->Pop();
        if (not std::isnan(record.first) and
            (not request.threshold_.has_value() or std::isfinite(record.first))) {
            finite_records.push_back(record);
        }
    }
    for (const auto& record : finite_records) {
        search_result->Push(record);
    }
    filter_search_result_by_threshold(search_result, request.threshold_, ctx.alloc);
    while (search_result->Size() > static_cast<uint64_t>(k)) {
        search_result->Pop();
    }

    // return an empty dataset directly if searcher returns nothing
    if (search_result->Empty()) {
        auto dataset_result = DatasetImpl::MakeEmptyDataset();
        dataset_result->Statistics(mci_result.MakeStatistics(stats).Dump());
        if (reasoning_ctx) {
            reasoning_ctx->DiagnoseExpectedTargets();
            dataset_result->Reasoning(reasoning_ctx->GenerateReport());
        }
        return dataset_result;
    }
    auto count = static_cast<const int64_t>(search_result->Size());

    Vector<InnerIdType> result_inner_ids(static_cast<size_t>(count), this->allocator_);

    auto [dataset_results, dists, ids] = create_fast_dataset(count, ctx.alloc);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        extra_infos =
            static_cast<char*>(ctx.alloc->Allocate(extra_info_size_ * search_result->Size()));
        dataset_results->ExtraInfos(extra_infos)
            ->ExtraInfoSize(static_cast<int64_t>(extra_info_size_));
    }
    for (int64_t j = count - 1; j >= 0; --j) {
        const auto& top = search_result->Top();
        dists[j] = top.first;
        ids[j] = this->label_table_->GetLabelById(top.second);
        result_inner_ids[j] = top.second;
        if (extra_infos != nullptr) {
            this->extra_infos_->GetExtraInfoById(top.second, extra_infos + extra_info_size_ * j);
        }
        search_result->Pop();
    }
    dataset_results->Statistics(mci_result.MakeStatistics(stats).Dump());

    // Generate reasoning report if reasoning context was created
    if (reasoning_ctx) {
        reasoning_ctx->MarkResult(result_inner_ids);
        reasoning_ctx->DiagnoseExpectedTargets();
        dataset_results->Reasoning(reasoning_ctx->GenerateReport());
    }

    return std::move(dataset_results);
}

}  // namespace vsag
