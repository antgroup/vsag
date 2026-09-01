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
#include "impl/reasoning/search_reasoning.h"
#include "utils/search_threshold.h"
#include "utils/util_functions.h"

namespace vsag {

static DatasetPtr
make_empty_dataset_with_stats() {
    SearchStatistics stats;
    auto dataset_result = DatasetImpl::MakeEmptyDataset();
    dataset_result->Statistics(stats.Dump());
    return dataset_result;
}

static DatasetPtr
make_empty_dataset_with_stats(const SearchStatistics& stats) {
    auto dataset_result = DatasetImpl::MakeEmptyDataset();
    dataset_result->Statistics(stats.Dump());
    return dataset_result;
}

static void
apply_hops_limit(InnerSearchParam& search_param, const HGraphSearchParameters& params) {
    if (static_cast<uint64_t>(params.hops_limit) <= static_cast<uint64_t>(params.ef_search)) {
        search_param.hops_limit = std::numeric_limits<uint32_t>::max();
        if (params.hops_limit != std::numeric_limits<uint32_t>::max()) {
            logger::warn(
                fmt::format("hops_limit({}) is not greater than ef_search({}), ignoring hops_limit",
                            params.hops_limit,
                            params.ef_search));
        }
        return;
    }
    search_param.hops_limit = params.hops_limit;
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
        return DatasetImpl::MakeEmptyDataset();
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

    // Iterator state is maintained per query, so this overload remains single-query only.
    CHECK_ARGUMENT(query->GetNumElements() == 1,
                   "iterator-based KnnSearch only supports single query (NumElements=1)");

    FilterPtr ft = this->create_search_filter(filter, params.use_extra_info_filter);

    if (iter_ctx == nullptr) {
        auto cur_count = this->total_count_.load();

        if (cur_count == 0) {
            return make_empty_dataset_with_stats();
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
                return make_empty_dataset_with_stats();
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
    if (inner_search_param.parallel_search_thread_count > 1) {
        result = this->parallel_searcher_->Search(graph,
                                                  flatten,
                                                  visited_list,
                                                  query,
                                                  inner_search_param,
                                                  this->label_table_,
                                                  ctx,
                                                  rabitq_lower_bound_candidates);
    } else {
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

    auto computer = flatten->FactoryComputer(query);

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

DatasetPtr
HGraph::search_range_with_request(const SearchRequest& request,
                                  const HGraphSearchParameters& params,
                                  const FilterPtr& filter,
                                  QueryContext& ctx) const {
    InnerSearchParam search_param;
    search_param.ep = this->entry_point_id_;
    search_param.topk = 1;
    search_param.ef = 1;
    search_param.is_inner_id_allowed = nullptr;
    search_param.enable_rabitq_one_bit_search = params.rabitq_one_bit_search;
    if (params.enable_time_record) {
        search_param.time_cost = std::make_shared<Timer>();
        search_param.time_cost->SetThreshold(params.timeout_ms);
        ctx.stats->is_timeout.store(false, std::memory_order_relaxed);
    }

    struct visited_list_guard {
        std::shared_ptr<VisitedListPool> pool;
        VisitedListPtr visited_list;
        ~visited_list_guard() {
            if (visited_list != nullptr) {
                pool->ReturnOne(visited_list);
            }
        }
    };
    visited_list_guard vt_guard{this->pool_, this->pool_->TakeOne()};
    auto& vt = vt_guard.visited_list;
    const auto* raw_query = get_data(request.query_);
    ctx.distance_phase = DistanceEvaluationPhase::ROUTING;
    for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
        auto result = this->search_one_graph(
            raw_query, this->route_graphs_[i], this->basic_flatten_codes_, search_param, vt, &ctx);
        if (!result->Empty()) {
            search_param.ep = result->Top().second;
        }
    }
    ctx.distance_phase = DistanceEvaluationPhase::APPROXIMATE;

    if (request.enable_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        auto& schema = this->attr_filter_index_->field_type_map_;
        auto expr = AstParse(request.attribute_filter_str_, &schema);
        auto executor = Executor::MakeInstance(this->allocator_, expr, this->attr_filter_index_);
        executor->Init();
        search_param.executors.emplace_back(executor);
    }

    search_param.ef = std::max(params.ef_search, request.limited_size_);
    search_param.is_inner_id_allowed = filter;
    search_param.radius = request.radius_;
    search_param.search_mode = RANGE_SEARCH;
    search_param.consider_duplicate = true;
    search_param.range_search_limit_size = static_cast<int>(request.limited_size_);
    search_param.parallel_search_thread_count = params.parallel_search_thread_count;
    search_param.enable_reorder = params.enable_reorder;
    search_param.enable_rabitq_one_bit_search = params.rabitq_one_bit_search;
    search_param.skip_ratio = params.skip_ratio;
    search_param.skip_strategy_type = params.skip_strategy_type;

    apply_hops_limit(search_param, params);
    DistanceRecordVector rabitq_lower_bound_candidates(ctx.alloc);
    auto* rabitq_lower_bound_candidates_ptr =
        search_param.enable_rabitq_one_bit_search and use_reorder_ and
                search_param.enable_reorder and reorder_by_base_
            ? &rabitq_lower_bound_candidates
            : nullptr;

    DistHeapPtr search_result;
    bool brute_force_used = false;
    MCIHybridSearchResult mci_result(params, filter);
    if (params.brute_force_threshold > 0.0F &&
        mci_result.valid_ratio <= params.brute_force_threshold) {
        search_result = this->brute_force_search<InnerSearchMode::RANGE_SEARCH>(
            raw_query, filter, request.limited_size_, request.radius_, &ctx);
        brute_force_used = true;
        mci_result.route = "brute_force";
    } else {
        mci_result = this->try_mci_search(request, params, filter, raw_query, search_param, &ctx);
        if (mci_result.route == "mci") {
            search_result = std::move(mci_result.result);
        }
    }
    if (search_result == nullptr) {
        search_result = this->search_one_graph(raw_query,
                                               this->bottom_graph_,
                                               this->basic_flatten_codes_,
                                               search_param,
                                               vt,
                                               &ctx,
                                               rabitq_lower_bound_candidates_ptr);
    }

    if (mci_result.route != "mci" && not brute_force_used && use_reorder_ &&
        search_param.enable_reorder) {
        this->reorder(raw_query,
                      this->get_reorder_codes(),
                      search_result,
                      request.limited_size_,
                      nullptr,
                      ctx,
                      rabitq_lower_bound_candidates_ptr);
    } else if (mci_result.route != "mci" && not brute_force_used && search_param.enable_reorder &&
               params.rabitq_one_bit_search) {
        this->reorder(raw_query,
                      this->basic_flatten_codes_,
                      search_result,
                      request.limited_size_,
                      nullptr,
                      ctx);
    }

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
    result->Statistics(mci_result.MakeStatistics(*ctx.stats).Dump());
    return result;
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

    if (use_custom_distance) {
        CHECK_ARGUMENT(request.distance_batch_size_ > 0,
                       "distance_batch_size must be greater than 0");
        CHECK_ARGUMENT(not is_range, "HGraph custom query distance only supports KNN search");
    }

    if (is_range) {
        // Range search remains single-query only (validate_range_args enforces NumElements==1).
        if (not use_custom_distance) {
            this->validate_range_args(query, request.radius_, request.limited_size_);
        } else {
            CHECK_ARGUMENT(query != nullptr, "query dataset cannot be null");
            CHECK_ARGUMENT(query->GetNumElements() == 1,
                           "HGraph range search only supports a single query");
        }
    } else {
        // KNN search supports multi-query batch: validate_knn_args enforces NumElements==1,
        // so use inline checks that allow NumElements >= 1.
        if (not use_custom_distance) {
            CHECK_ARGUMENT(query != nullptr, "query dataset cannot be null");
            if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
                CHECK_ARGUMENT(
                    query->GetDim() == dim_,
                    fmt::format(
                        "query.dim({}) must be equal to index.dim({})", query->GetDim(), dim_));
            }
            CHECK_ARGUMENT(get_data(query) != nullptr,
                           "query vector storage must match index data type");
        }
        CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
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
    if (use_custom_distance && query != nullptr) {
        CHECK_ARGUMENT(query->GetNumElements() == 1,
                       "HGraph custom query distance only supports a single query");
    }
    int64_t query_count = use_custom_distance ? 1 : query->GetNumElements();
    CHECK_ARGUMENT(query_count >= 1,
                   fmt::format("query count({}) must be at least 1", query_count));
    if (is_range) {
        CHECK_ARGUMENT(query_count == 1, "range search only supports single query (NumElements=1)");
    }
    if (query_count > 1) {
        CHECK_ARGUMENT(request.expected_labels_.empty(),
                       "reasoning (expected_labels_) is only supported for single-query search");
        CHECK_ARGUMENT(!this->label_table_->HasActivePaddingLabel(),
                       "batch KNN does not support an index containing external label -1");
    }
    if (element_count == 0) {
        if (!is_range && query_count > 1) {
            CHECK_ARGUMENT(query_count <= std::numeric_limits<int64_t>::max() / k,
                           fmt::format("query_count({}) * k({}) would overflow", query_count, k));
            const auto total_slots = query_count * k;
            auto [result, distances, ids] = create_fast_dataset(total_slots, ctx.alloc);
            std::fill_n(distances, total_slots, std::numeric_limits<float>::infinity());
            std::fill_n(ids, total_slots, -1);
            result->NumElements(query_count);
            result->Dim(k);
            result->Statistics(stats.Dump());
            return result;
        }
        return make_empty_dataset_with_stats();
    }
    k = std::min(k, element_count);

    if (!is_range && query_count > 1 && k == 0) {
        auto result = create_fast_dataset(0, ctx.alloc);
        std::get<0>(result)->NumElements(query_count);
        std::get<0>(result)->Dim(0);
        std::get<0>(result)->Statistics(stats.Dump());
        return std::get<0>(result);
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
        ComputerInterfacePtr computer = nullptr;
        if (not use_custom_distance) {
            precise_flatten = this->basic_flatten_codes_;
            if (use_reorder_) {
                precise_flatten = this->high_precise_codes_;
            }
            if (create_new_raw_vector_) {
                precise_flatten = this->raw_vector_;
            }
            computer = precise_flatten->FactoryComputer(get_data(query));
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
            reasoning_ctx->SetTrueDistance(inner_id, dist);
        }

        ctx.reasoning_ctx = reasoning_ctx.get();
    }

    if (this->entry_point_id_ == INVALID_ENTRY_POINT) {
        if (query_count > 1) {
            // Return batch-shaped empty result preserving documented layout.
            CHECK_ARGUMENT(
                query_count <= std::numeric_limits<int64_t>::max() / std::max(k, (int64_t)1),
                fmt::format("query_count({}) * k({}) would overflow", query_count, k));
            int64_t batch_count = query_count * k;
            auto [empty_ds, empty_dists, empty_ids] = create_fast_dataset(batch_count, ctx.alloc);
            std::fill_n(empty_dists, batch_count, std::numeric_limits<float>::infinity());
            std::fill_n(empty_ids, batch_count, -1);
            empty_ds->NumElements(query_count);
            empty_ds->Dim(k);
            empty_ds->Statistics(stats.Dump());
            return empty_ds;
        }
        return make_empty_dataset_with_stats();
    }

    FilterPtr ft = this->create_search_filter(request.filter_, params.use_extra_info_filter);

    if (is_range) {
        return this->search_range_with_request(request, params, ft, ctx);
    }

    // ---- KNN search: multi-query batch path (PR #1685) ----

    // Build a shared base search_param; per-query fields (ep) are set inside the loop.
    InnerSearchParam base_search_param;
    base_search_param.is_inner_id_allowed = ft;
    base_search_param.distance_threshold = request.threshold_;
    base_search_param.ef = std::max(params.ef_search, k);
    if (this->use_conjugate_graph_ and params.use_conjugate_graph_search) {
        base_search_param.ef = std::max(base_search_param.ef, static_cast<uint64_t>(LOOK_AT_K));
    }
    base_search_param.topk = static_cast<int64_t>(base_search_param.ef);
    if (params.topk_factor > 1.0F) {
        base_search_param.topk =
            std::min(base_search_param.topk,
                     static_cast<int64_t>(static_cast<float>(k) * params.topk_factor));
    }
    if (this->use_conjugate_graph_ and params.use_conjugate_graph_search) {
        base_search_param.topk = std::max(base_search_param.topk, LOOK_AT_K);
    }
    base_search_param.consider_duplicate = true;
    base_search_param.enable_reorder = use_custom_distance ? false : params.enable_reorder;
    base_search_param.enable_rabitq_one_bit_search =
        use_custom_distance ? false : params.rabitq_one_bit_search;
    base_search_param.skip_ratio = params.skip_ratio;
    base_search_param.skip_strategy_type = params.skip_strategy_type;
    base_search_param.distance_batch_func = request.distance_batch_func_;
    base_search_param.distance_batch_size = request.distance_batch_size_;
    if (params.enable_time_record) {
        base_search_param.time_cost = std::make_shared<Timer>();
        base_search_param.time_cost->SetThreshold(params.timeout_ms);
        stats.is_timeout.store(false, std::memory_order_relaxed);
    }
    base_search_param.parallel_search_thread_count = params.parallel_search_thread_count;

    // hops_limit only takes effect when it's greater than ef_search
    apply_hops_limit(base_search_param, params);

    if (request.enable_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        auto& schema = this->attr_filter_index_->field_type_map_;
        auto expr = AstParse(request.attribute_filter_str_, &schema);
        auto executor = Executor::MakeInstance(this->allocator_, expr, this->attr_filter_index_);
        executor->Init();
        base_search_param.executors.emplace_back(executor);
    }

    // Single-query preserves the original "dim = actual result count" contract; multi-query
    // uses a fixed query_count x k rectangular layout. Guard the multiplication against overflow.
    int64_t total_result_count = 0;
    if (query_count > 1) {
        CHECK_ARGUMENT(
            query_count <= std::numeric_limits<int64_t>::max() / k,
            fmt::format("query_count({}) * k({}) would overflow int64_t", query_count, k));
        total_result_count = query_count * k;
    }
    // Validate that byte-level allocations do not overflow size_t.
    if (total_result_count > 0) {
        constexpr auto k_id_size = sizeof(int64_t);
        constexpr auto k_dist_size = sizeof(float);
        CHECK_ARGUMENT(total_result_count <= std::numeric_limits<size_t>::max() / k_id_size,
                       fmt::format("total_result_count({}) * sizeof(int64_t) would overflow size_t",
                                   total_result_count));
        CHECK_ARGUMENT(total_result_count <= std::numeric_limits<size_t>::max() / k_dist_size,
                       fmt::format("total_result_count({}) * sizeof(float) would overflow size_t",
                                   total_result_count));
        if (extra_info_size_ > 0) {
            constexpr auto k_extra_size = sizeof(char);
            CHECK_ARGUMENT(
                total_result_count <= std::numeric_limits<size_t>::max() /
                                          (static_cast<int64_t>(extra_info_size_) * k_extra_size),
                fmt::format("total_result_count({}) * extra_info_size({}) would overflow size_t",
                            total_result_count,
                            extra_info_size_));
        }
    }
    auto [dataset_results, dists, ids] = create_fast_dataset(total_result_count, ctx.alloc);
    char* extra_infos = nullptr;
    if (query_count > 1 && extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        extra_infos = static_cast<char*>(ctx.alloc->Allocate(
            static_cast<size_t>(extra_info_size_) * static_cast<size_t>(total_result_count)));
        std::memset(extra_infos,
                    0,
                    static_cast<size_t>(static_cast<size_t>(extra_info_size_) *
                                        static_cast<size_t>(total_result_count)));
        dataset_results->ExtraInfos(extra_infos);
        dataset_results->ExtraInfoSize(static_cast<int64_t>(extra_info_size_));
    }

    // Pre-fill sentinels: ids = -1 (authoritative signal for "no result") and
    // dists = +infinity (unambiguous for inner-product / cosine metrics that may produce
    // negative distances). Callers MUST detect padding via ids[i] == -1 rather than by
    // distance comparison.
    std::fill_n(dists, total_result_count, std::numeric_limits<float>::infinity());
    std::fill_n(ids, total_result_count, -1);

    Vector<InnerIdType> reasoning_result_inner_ids(this->allocator_);

    struct visited_list_guard {
        std::shared_ptr<VisitedListPool> pool;
        VisitedListPtr visited_list;
        ~visited_list_guard() {
            if (visited_list != nullptr) {
                pool->ReturnOne(visited_list);
            }
        }
    };
    visited_list_guard vt_guard{this->pool_, this->pool_->TakeOne()};
    auto& vt = vt_guard.visited_list;

    // Hoist per-query search_param and rabitq candidate buffer out of the loop:
    // the searcher only mutates `duplicate_id` (declared `mutable` on the const
    // InnerSearchParam&) and callers only tweak `ep` per query, so a single instance
    // reused across queries avoids copying the base_search_param (including its
    // `executors` vector) on every iteration.
    InnerSearchParam search_param = base_search_param;
    DistanceRecordVector rabitq_lower_bound_candidates(ctx.alloc);
    auto* rabitq_lower_bound_candidates_ptr =
        search_param.enable_rabitq_one_bit_search and use_reorder_ and
                search_param.enable_reorder and reorder_by_base_
            ? &rabitq_lower_bound_candidates
            : nullptr;

    InnerSearchParam ep_search_param;
    ep_search_param.ep = this->entry_point_id_;
    ep_search_param.topk = 1;
    ep_search_param.ef = 1;
    ep_search_param.is_inner_id_allowed = nullptr;
    ep_search_param.enable_rabitq_one_bit_search =
        use_custom_distance ? false : params.rabitq_one_bit_search;
    ep_search_param.distance_batch_func = request.distance_batch_func_;
    ep_search_param.distance_batch_size = request.distance_batch_size_;

    for (int64_t q_idx = 0; q_idx < query_count; ++q_idx) {
        const auto* raw_query = use_custom_distance ? nullptr : get_data(query, q_idx);

        // Reset per-query mutable state before each query.
        search_param.duplicate_id = -1;
        // Per-query entry point search through hierarchical graphs.
        ctx.distance_phase = DistanceEvaluationPhase::ROUTING;
        ep_search_param.ep = this->entry_point_id_;
        rabitq_lower_bound_candidates.clear();

        for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
            auto result = this->search_one_graph(raw_query,
                                                 this->route_graphs_[i],
                                                 this->basic_flatten_codes_,
                                                 ep_search_param,
                                                 vt,
                                                 &ctx);
            if (not result->Empty()) {
                ep_search_param.ep = result->Top().second;
            }
        }
        search_param.ep = ep_search_param.ep;
        ctx.distance_phase = DistanceEvaluationPhase::APPROXIMATE;
        if (search_param.time_cost != nullptr) {
            search_param.time_cost->Reset();
        }

        DistHeapPtr search_result;
        bool brute_force_used = false;
        MCIHybridSearchResult mci_result(params, ft);
        if (not use_custom_distance) {
            if (params.brute_force_threshold > 0.0F &&
                mci_result.valid_ratio <= params.brute_force_threshold) {
                search_result = this->brute_force_search<InnerSearchMode::KNN_SEARCH>(
                    raw_query, ft, k, 0.0F, &ctx, request.threshold_);
                brute_force_used = true;
                mci_result.route = "brute_force";
            } else {
                mci_result =
                    this->try_mci_search(request, params, ft, raw_query, search_param, &ctx);
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

        if (mci_result.route != "mci" && !brute_force_used && use_reorder_ &&
            search_param.enable_reorder) {
            auto reorder_limit = k;
            if (this->use_conjugate_graph_ and params.use_conjugate_graph_search) {
                reorder_limit = std::max(reorder_limit, LOOK_AT_K);
            }
            this->reorder(raw_query,
                          this->get_reorder_codes(),
                          search_result,
                          reorder_limit,
                          nullptr,
                          ctx,
                          rabitq_lower_bound_candidates_ptr,
                          request.threshold_);
        } else if (mci_result.route != "mci" && !brute_force_used && search_param.enable_reorder &&
                   params.rabitq_one_bit_search) {
            auto reorder_limit = k;
            if (this->use_conjugate_graph_ and params.use_conjugate_graph_search) {
                reorder_limit = std::max(reorder_limit, LOOK_AT_K);
            }
            this->reorder(raw_query,
                          this->basic_flatten_codes_,
                          search_result,
                          reorder_limit,
                          nullptr,
                          ctx,
                          nullptr,
                          request.threshold_);
        }

        if (this->use_conjugate_graph_ and params.use_conjugate_graph_search and
            not search_result->Empty()) {
            std::priority_queue<std::pair<float, LabelType>> label_results;
            std::shared_lock label_lock(this->label_lookup_mutex_);
            while (not search_result->Empty()) {
                const auto record = search_result->Top();
                search_result->Pop();
                label_results.emplace(record.first,
                                      this->label_table_->GetLabelById(record.second));
            }

            const auto flatten = use_custom_distance ? nullptr : this->get_precise_codes();
            const auto computer =
                use_custom_distance ? nullptr : flatten->FactoryComputer(raw_query);
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
                const auto [found, inner_id] =
                    this->label_table_->TryGetIdByLabel(record.second, true);
                if (found and is_allowed(inner_id)) {
                    search_result->Push(record.first, inner_id);
                }
            }
        }

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
        while (search_result->Size() > k) {
            search_result->Pop();
        }

        // Single-query preserves the original contract: an empty result returns an empty dataset.
        if (query_count == 1 && search_result->Empty()) {
            auto dataset_result = DatasetImpl::MakeEmptyDataset();
            dataset_result->Statistics(mci_result.MakeStatistics(stats).Dump());
            if (reasoning_ctx) {
                reasoning_ctx->DiagnoseExpectedTargets();
                dataset_result->Reasoning(reasoning_ctx->GenerateReport());
            }
            return dataset_result;
        }

        auto count = static_cast<int64_t>(search_result->Size());
        if (reasoning_ctx) {
            reasoning_result_inner_ids.resize(static_cast<size_t>(count));
        }

        if (query_count == 1) {
            // Single-query path may shrink the dataset to the actual neighbor count.
            if (dataset_results->GetDim() != count) {
                auto [single_results, single_dists, single_ids] =
                    create_fast_dataset(count, ctx.alloc);
                dataset_results = single_results;
                dists = single_dists;
                ids = single_ids;
            }
            if (extra_info_size_ > 0 && this->extra_infos_ != nullptr && count > 0) {
                extra_infos = static_cast<char*>(ctx.alloc->Allocate(
                    static_cast<size_t>(extra_info_size_) * static_cast<size_t>(count)));
                dataset_results->ExtraInfos(extra_infos);
                dataset_results->ExtraInfoSize(static_cast<int64_t>(extra_info_size_));
            }
            for (int64_t j = count - 1; j >= 0; --j) {
                const auto& top = search_result->Top();
                dists[j] = top.first;
                ids[j] = this->label_table_->GetLabelById(top.second);
                if (reasoning_ctx) {
                    reasoning_result_inner_ids[static_cast<size_t>(j)] = top.second;
                }
                if (extra_infos != nullptr) {
                    this->extra_infos_->GetExtraInfoById(top.second,
                                                         extra_infos + extra_info_size_ * j);
                }
                search_result->Pop();
            }
            dataset_results->Statistics(mci_result.MakeStatistics(stats).Dump());
        } else {
            int64_t offset = q_idx * k;
            for (int64_t j = count - 1; j >= 0; --j) {
                const auto& top = search_result->Top();
                dists[offset + j] = top.first;
                ids[offset + j] = this->label_table_->GetLabelById(top.second);
                if (reasoning_ctx) {
                    reasoning_result_inner_ids[static_cast<size_t>(j)] = top.second;
                }
                if (extra_infos != nullptr) {
                    this->extra_infos_->GetExtraInfoById(
                        top.second, extra_infos + extra_info_size_ * (offset + j));
                }
                search_result->Pop();
            }
        }
    }

    dataset_results->NumElements(query_count);
    if (query_count > 1) {
        dataset_results->Dim(k);
    }
    if (query_count > 1) {
        dataset_results->Statistics(stats.Dump());
    }

    // Generate reasoning report if reasoning context was created.
    if (reasoning_ctx) {
        reasoning_ctx->MarkResult(reasoning_result_inner_ids);
        reasoning_ctx->DiagnoseExpectedTargets();
        dataset_results->Reasoning(reasoning_ctx->GenerateReport());
    }

    return std::move(dataset_results);
}

}  // namespace vsag
