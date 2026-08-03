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
#include "impl/filter/black_list_filter.h"
#include "impl/filter/iterator_filter.h"
#include "impl/heap/standard_heap.h"
#include "impl/reasoning/search_reasoning.h"
#include "utils/search_threshold.h"
#include "utils/util_functions.h"

namespace vsag {

enum class search_plan {
    K_BOTTOM_GRAPH,
    K_BRUTE_FORCE,
    K_MCI,
};

struct search_plan_input {
    bool is_range_search{false};
    float valid_ratio{1.0F};
    float brute_force_threshold{0.0F};
    bool use_mci{false};
    bool mci_enabled{false};
    bool mci_has_clique_index{false};
    bool has_attribute_executor{false};
    bool has_valid_id_source{false};
    bool has_bitset_source{false};
    float mci_valid_ratio_threshold{0.0F};
};

static bool
is_mci_available(const search_plan_input& input) {
    return input.use_mci and input.mci_enabled and input.mci_has_clique_index and
           not input.has_attribute_executor and
           (input.has_valid_id_source or input.has_bitset_source);
}

static search_plan
select_search_plan(const search_plan_input& input, bool mci_available) {
    if (input.brute_force_threshold > 0.0F and input.valid_ratio <= input.brute_force_threshold) {
        return search_plan::K_BRUTE_FORCE;
    }
    // A NaN ratio preserves the existing MCI attempt behavior.
    if (not input.is_range_search and mci_available and
        not(input.valid_ratio >= input.mci_valid_ratio_threshold)) {
        return search_plan::K_MCI;
    }
    return search_plan::K_BOTTOM_GRAPH;
}

static bool
has_valid_id_source(const FilterPtr& filter) {
    if (filter == nullptr) {
        return false;
    }
    const int64_t* valid_ids = nullptr;
    int64_t valid_count = 0;
    filter->GetValidIds(&valid_ids, valid_count);
    return valid_ids != nullptr and valid_count > 0;
}

static bool
has_bitset_source(const FilterPtr& filter) {
    if (filter == nullptr) {
        return false;
    }
    const auto bitset_filter = std::dynamic_pointer_cast<BlackListFilter>(filter);
    return bitset_filter != nullptr and bitset_filter->IsBitsetFilter();
}

class VisitedListGuard {
public:
    VisitedListGuard(std::shared_ptr<VisitedListPool> pool, VisitedListPtr visited_list)
        : pool_(std::move(pool)), visited_list_(std::move(visited_list)) {
    }

    VisitedListGuard(const VisitedListGuard&) = delete;
    VisitedListGuard&
    operator=(const VisitedListGuard&) = delete;

    VisitedListPtr&
    Get() {
        return visited_list_;
    }

    void
    Release() {
        if (visited_list_ != nullptr) {
            pool_->ReturnOne(visited_list_);
            visited_list_.reset();
        }
    }

    ~VisitedListGuard() {
        Release();
    }

private:
    std::shared_ptr<VisitedListPool> pool_;
    VisitedListPtr visited_list_;
};

static DatasetPtr
make_empty_dataset_with_stats() {
    SearchStatistics stats;
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
                for (auto i = static_cast<int64_t>(this->route_graphs_.size()) - 1; i >= 0; --i) {
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
            return DatasetImpl::MakeEmptyDataset();
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

    auto total = static_cast<InnerIdType>(this->total_count_.load());
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

QueryContext
HGraph::create_query_context(const SearchRequest& request,
                             const HGraphSearchParameters& params,
                             int64_t k,
                             bool use_custom_distance,
                             SearchStatistics* stats,
                             std::shared_ptr<ReasoningContext>& reasoning_ctx) const {
    QueryContext ctx;
    ctx.alloc = this->allocator_;
    ctx.stats = stats;
    ctx.rabitq_error_rate = params.rabitq_error_rate;
    if (request.search_allocator_ != nullptr) {
        ctx.alloc = request.search_allocator_;
    }
    reasoning_ctx = this->initialize_reasoning_context(request, k, use_custom_distance);
    if (reasoning_ctx != nullptr) {
        ctx.reasoning_ctx = reasoning_ctx.get();
    }
    return ctx;
}

void
HGraph::search_route_graphs(const SearchRequest& request,
                            const HGraphSearchParameters& params,
                            InnerIdType entry_point,
                            bool use_custom_distance,
                            const void* query,
                            const VisitedListPtr& visited_list,
                            QueryContext* ctx,
                            InnerSearchParam& search_param) const {
    search_param.ep = entry_point;
    search_param.topk = 1;
    search_param.ef = 1;
    search_param.is_inner_id_allowed = nullptr;
    search_param.enable_rabitq_one_bit_search =
        use_custom_distance ? false : params.rabitq_one_bit_search;
    search_param.distance_batch_func = request.distance_batch_func_;
    search_param.distance_batch_size = request.distance_batch_size_;

    for (auto i = static_cast<int64_t>(this->route_graphs_.size()) - 1; i >= 0; --i) {
        auto result = this->search_one_graph(query,
                                             this->route_graphs_[i],
                                             this->basic_flatten_codes_,
                                             search_param,
                                             visited_list,
                                             ctx);
        // An unrankable route seed can still bridge to finite bottom-layer results.
        if (not result->Empty()) {
            search_param.ep = result->Top().second;
        }
    }
}

void
HGraph::configure_bottom_graph_search(const SearchRequest& request,
                                      const HGraphSearchParameters& params,
                                      bool is_range,
                                      int64_t k,
                                      bool use_custom_distance,
                                      const FilterPtr& filter,
                                      const std::optional<float>& threshold,
                                      QueryContext* ctx,
                                      InnerSearchParam& search_param) {
    search_param.is_inner_id_allowed = filter;
    search_param.enable_reorder = use_custom_distance ? false : params.enable_reorder;
    search_param.consider_duplicate = true;
    search_param.enable_rabitq_one_bit_search =
        use_custom_distance ? false : params.rabitq_one_bit_search;
    search_param.parallel_search_thread_count = params.parallel_search_thread_count;

    if (is_range) {
        search_param.ef = std::max(params.ef_search, request.limited_size_);
        search_param.radius = request.radius_;
        search_param.search_mode = RANGE_SEARCH;
        search_param.range_search_limit_size = static_cast<int>(request.limited_size_);
    } else {
        search_param.ef = std::max(params.ef_search, k);
        search_param.distance_threshold = threshold;
        search_param.topk = static_cast<int64_t>(search_param.ef);
        if (params.topk_factor > 1.0F) {
            search_param.topk =
                std::min(search_param.topk,
                         static_cast<int64_t>(static_cast<float>(k) * params.topk_factor));
        }
        if (params.enable_time_record) {
            search_param.time_cost = std::make_shared<Timer>();
            search_param.time_cost->SetThreshold(params.timeout_ms);
            ctx->stats->is_timeout.store(false, std::memory_order_relaxed);
        }
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
}

DatasetPtr
HGraph::pack_search_result(const SearchRequest& request,
                           int64_t k,
                           DistHeapPtr search_result,
                           const QueryContext& ctx,
                           const MCIHybridSearchResult& mci_result,
                           const SearchStatistics& stats,
                           const std::shared_ptr<ReasoningContext>& reasoning_ctx) const {
    if (request.mode_ == SearchMode::RANGE_SEARCH) {
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
    if (extra_info_size_ > 0 and this->extra_infos_ != nullptr) {
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

    if (reasoning_ctx) {
        reasoning_ctx->MarkResult(result_inner_ids);
        reasoning_ctx->DiagnoseExpectedTargets();
        dataset_results->Reasoning(reasoning_ctx->GenerateReport());
    }
    return std::move(dataset_results);
}

HGraphSearchParameters
HGraph::parse_and_validate_search_params(const SearchRequest& request,
                                         bool is_range,
                                         int64_t k,
                                         bool use_custom_distance) const {
    if (use_custom_distance) {
        CHECK_ARGUMENT(request.distance_batch_size_ > 0,
                       "distance_batch_size must be greater than 0");
        CHECK_ARGUMENT(not is_range, "HGraph custom distance only supports KNN search");
    }

    if (is_range) {
        if (not use_custom_distance) {
            this->validate_range_args(request.query_, request.radius_, request.limited_size_);
        }
    } else if (not use_custom_distance) {
        this->validate_knn_args(request.query_, k);
    } else {
        CHECK_ARGUMENT(k > 0, "topk must be greater than 0");
    }

    auto params = HGraphSearchParameters::FromJson(request.params_str_);
    if (use_custom_distance) {
        CHECK_ARGUMENT(params.parallel_search_thread_count == 1,
                       "HGraph custom query distance does not support parallel search");
        CHECK_ARGUMENT(params.brute_force_threshold <= 0.0F,
                       "HGraph custom query distance does not support brute_force_threshold");
    }
    CHECK_ARGUMENT(  // NOLINT
        params.ef_search >= 1,
        fmt::format("ef_search({}) must be at least 1", params.ef_search));
    return params;
}

std::shared_ptr<ReasoningContext>
HGraph::initialize_reasoning_context(const SearchRequest& request,
                                     int64_t k,
                                     bool use_custom_distance) const {
    if (request.mode_ == SearchMode::RANGE_SEARCH or request.expected_labels_.empty()) {
        return nullptr;
    }

    auto reasoning_ctx = std::make_shared<ReasoningContext>(this->allocator_);
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
        computer = precise_flatten->FactoryComputer(get_data(request.query_));
    }
    for (const auto& pair : label_to_inner_id) {
        float dist = 0.0F;
        const auto inner_id = pair.second;
        if (use_custom_distance) {
            const auto label = this->label_table_->GetLabelById(inner_id);
            request.distance_batch_func_(&label, 1, &dist);
            CHECK_ARGUMENT(std::isfinite(dist), "distance callback must return finite scores");
        } else {
            precise_flatten->Query(&dist, computer, &inner_id, 1);
        }
        reasoning_ctx->SetTrueDistance(inner_id, dist);
    }
    return reasoning_ctx;
}

[[nodiscard]] DatasetPtr
HGraph::SearchWithRequest(const SearchRequest& request) const {
    ValidateSearchThreshold(request.threshold_);
    SearchStatistics stats;

    const auto& query = request.query_;
    bool is_range = (request.mode_ == SearchMode::RANGE_SEARCH);
    auto k = request.topk_;
    const bool use_custom_distance = request.distance_batch_func_ != nullptr;

    /***** Step 1: Parse and validate request-specific search parameters. *****/
    auto params = this->parse_and_validate_search_params(request, is_range, k, use_custom_distance);

    /***** Step 2: Keep the index state stable while searching mutable indexes. *****/
    std::shared_lock<std::shared_mutex> force_remove_rlock;
    std::shared_lock<std::shared_mutex> shared_lock;
    if (!this->immutable_.load(std::memory_order_acquire)) {
        if (this->support_force_remove()) {
            force_remove_rlock = std::shared_lock<std::shared_mutex>(this->force_remove_mutex_);
        }
        shared_lock = this->acquire_global_read_lock();
    }
    const auto element_count = GetNumElements();
    const auto entry_point = this->entry_point_id_;
    if (element_count == 0 or entry_point == INVALID_ENTRY_POINT) {
        return make_empty_dataset_with_stats();
    }
    k = std::min(k, element_count);

    /***** Step 3: Set up query-local allocation, statistics, and optional reasoning. *****/
    std::shared_ptr<ReasoningContext> reasoning_ctx;
    auto ctx =
        this->create_query_context(request, params, k, use_custom_distance, &stats, reasoning_ctx);

    /***** Step 4: Navigate upper route graphs to obtain the bottom-graph entry point. *****/
    VisitedListGuard vt_guard{this->pool_, this->pool_->TakeOne()};
    auto& vt = vt_guard.Get();
    const auto* raw_query = use_custom_distance ? nullptr : get_data(query);
    InnerSearchParam search_param;
    this->search_route_graphs(
        request, params, entry_point, use_custom_distance, raw_query, vt, &ctx, search_param);

    /***** Step 5: Build filters and configure the bottom-graph search parameters. *****/
    FilterPtr ft = this->create_search_filter(request.filter_, params.use_extra_info_filter);
    if (request.enable_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        auto& schema = this->attr_filter_index_->field_type_map_;
        auto expr = AstParse(request.attribute_filter_str_, &schema);
        auto executor = Executor::MakeInstance(this->allocator_, expr, this->attr_filter_index_);
        executor->Init();
        search_param.executors.emplace_back(executor);
    }

    this->configure_bottom_graph_search(request,
                                        params,
                                        is_range,
                                        k,
                                        use_custom_distance,
                                        ft,
                                        request.threshold_,
                                        &ctx,
                                        search_param);

    /***** Step 6: Select and execute brute-force, MCI, or bottom-graph search. *****/
    DistanceRecordVector rabitq_lower_bound_candidates(ctx.alloc);
    auto* rabitq_lower_bound_candidates_ptr =
        search_param.enable_rabitq_one_bit_search and use_reorder_ and
                search_param.enable_reorder and reorder_by_base_
            ? &rabitq_lower_bound_candidates
            : nullptr;

    bool brute_force_used = false;
    MCIHybridSearchResult mci_result(params, ft);
    DistHeapPtr search_result;
    if (not use_custom_distance) {
        search_plan_input plan_input;
        plan_input.is_range_search = is_range;
        plan_input.valid_ratio = mci_result.valid_ratio;
        plan_input.brute_force_threshold = params.brute_force_threshold;
        plan_input.use_mci = params.use_mci;
        plan_input.mci_enabled = this->mci_parameters_.enabled;
        plan_input.has_attribute_executor = not search_param.executors.empty();
        plan_input.mci_valid_ratio_threshold = params.mci_hgraph_valid_ratio_threshold;
        const bool brute_force_route =
            select_search_plan(plan_input, false) == search_plan::K_BRUTE_FORCE;
        bool mci_available = false;
        if (not brute_force_route) {
            // MCI seeds use the original external-label filter; ft wraps it for inner-ID search.
            const auto bitset_seed_source = has_bitset_source(request.filter_);
            bool valid_id_seed_source = false;
            if (params.use_mci and this->mci_parameters_.enabled and
                search_param.executors.empty()) {
                valid_id_seed_source = has_valid_id_source(request.filter_);
                if (valid_id_seed_source or bitset_seed_source) {
                    plan_input.mci_has_clique_index =
                        this->mci_cliques_ != nullptr and
                        this->mci_cliques_->HasCliqueIndex(this->total_count_.load());
                }
            }
            plan_input.has_valid_id_source = valid_id_seed_source;
            plan_input.has_bitset_source = bitset_seed_source;
            mci_available = is_mci_available(plan_input);
        }

        switch (select_search_plan(plan_input, mci_available)) {
            case search_plan::K_BRUTE_FORCE:
                if (is_range) {
                    search_result = this->brute_force_search<InnerSearchMode::RANGE_SEARCH>(
                        raw_query, ft, request.limited_size_, request.radius_, &ctx);
                } else {
                    search_result = this->brute_force_search<InnerSearchMode::KNN_SEARCH>(
                        raw_query, ft, k, 0.0F, &ctx, request.threshold_);
                }
                brute_force_used = true;
                mci_result.route = "brute_force";
                break;
            case search_plan::K_MCI:
                mci_result =
                    this->try_mci_search(request, params, ft, raw_query, search_param, &ctx);
                if (mci_result.route == "mci") {
                    search_result = std::move(mci_result.result);
                    break;
                }
                [[fallthrough]];
            case search_plan::K_BOTTOM_GRAPH:
                if (mci_available) {
                    mci_result.route = "hgraph";
                }
                search_result = this->search_one_graph(raw_query,
                                                       this->bottom_graph_,
                                                       this->basic_flatten_codes_,
                                                       search_param,
                                                       vt,
                                                       &ctx,
                                                       rabitq_lower_bound_candidates_ptr);
                break;
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
    /***** Step 7: Return the pooled visited list before post-processing results. *****/
    vt_guard.Release();

    /***** Step 8: Reorder candidates when the selected route permits it. *****/
    if (mci_result.route != "mci" and not brute_force_used and use_reorder_ and
        search_param.enable_reorder) {
        auto limit = is_range ? request.limited_size_ : k;
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

    /***** Step 9: Trim, pack, and annotate the final dataset. *****/
    return this->pack_search_result(
        request, k, search_result, ctx, mci_result, stats, reasoning_ctx);
}

}  // namespace vsag
