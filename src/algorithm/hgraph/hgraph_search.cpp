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

#include "attr/argparse.h"
#include "dataset_impl.h"
#include "hgraph.h"  // IWYU pragma: keep
#include "impl/filter/filter_headers.h"
#include "impl/heap/standard_heap.h"
#include "impl/reasoning/search_reasoning.h"
#include "index/iterator_filter.h"
#include "utils/util_functions.h"

namespace vsag {

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
    int64_t query_dim = query->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }

    auto params = HGraphSearchParameters::FromJson(parameters);
    auto ef_search_threshold = std::max<int64_t>(AMPLIFICATION_FACTOR * k, 1000);
    CHECK_ARGUMENT(  // NOLINT
        (1 <= params.ef_search) and (params.ef_search <= ef_search_threshold),
        fmt::format("ef_search({}) must in range[1, {}]", params.ef_search, ef_search_threshold));

    std::shared_lock force_remove_rlock(this->force_remove_mutex_);
    std::shared_lock shared_lock(this->global_mutex_);
    // check k
    CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
    k = std::min(k, GetNumElements());

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    auto combined_filter = std::make_shared<CombinedFilter>();
    combined_filter->AppendFilter(this->label_table_->GetDeletedIdsFilter());
    if (filter != nullptr) {
        if (params.use_extra_info_filter) {
            combined_filter->AppendFilter(
                std::make_shared<ExtraInfoWrapperFilter>(filter, this->extra_infos_));
        } else {
            combined_filter->AppendFilter(
                std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_));
        }
    }
    FilterPtr ft = nullptr;
    if (not combined_filter->IsEmpty()) {
        ft = combined_filter;
    }

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
    auto search_result = DistanceHeap::MakeInstanceBySize<true, false>(ctx.alloc, k);
    const auto* query_data = get_data(query);
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
        if (search_param.ep == INVALID_ENTRY_POINT) {
            return make_empty_dataset_with_stats();
        }
        if (iter_filter_ctx->IsFirstUsed()) {
            for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
                auto result = this->search_one_graph(query_data,
                                                     this->route_graphs_[i],
                                                     this->basic_flatten_codes_,
                                                     search_param,
                                                     (VisitedListPtr) nullptr,
                                                     &ctx);
                search_param.ep = result->Top().second;
            }
        }

        search_param.ef = std::max(params.ef_search, k);
        search_param.is_inner_id_allowed = ft;
        search_param.topk = static_cast<int64_t>(search_param.ef);
        search_param.parallel_search_thread_count = params.parallel_search_thread_count;
        search_param.enable_rabitq_one_bit_search = params.rabitq_one_bit_search;

        DistanceRecordVector rabitq_lower_bound_candidates(ctx.alloc);
        auto* rabitq_lower_bound_candidates_ptr =
            search_param.enable_rabitq_one_bit_search and use_reorder_ and reorder_by_base_
                ? &rabitq_lower_bound_candidates
                : nullptr;

        search_result = this->search_one_graph(query_data,
                                               this->bottom_graph_,
                                               this->basic_flatten_codes_,
                                               search_param,
                                               iter_filter_ctx,
                                               &ctx,
                                               rabitq_lower_bound_candidates_ptr);

        if (use_reorder_) {
            this->reorder(query_data,
                          this->get_reorder_codes(),
                          search_result,
                          k,
                          iter_filter_ctx,
                          ctx,
                          rabitq_lower_bound_candidates_ptr);
        } else if (params.rabitq_one_bit_search) {
            this->reorder(
                query_data, this->basic_flatten_codes_, search_result, k, iter_filter_ctx, ctx);
        }
    }

    while (search_result->Size() > k) {
        auto curr = search_result->Top();
        iter_filter_ctx->AddDiscardNode(curr.first, curr.second);
        search_result->Pop();
    }

    // return an empty dataset directly if searcher returns nothing
    if (search_result->Empty()) {
        return DatasetImpl::MakeEmptyDataset();
    }
    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, ids] = create_fast_dataset(count, ctx.alloc);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0) {
        extra_infos =
            static_cast<char*>(ctx.alloc->Allocate(extra_info_size_ * search_result->Size()));
        dataset_results->ExtraInfos(extra_infos);
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

DatasetPtr
HGraph::RangeSearch(const DatasetPtr& query,
                    float radius,
                    const std::string& parameters,
                    const FilterPtr& filter,
                    int64_t limited_size) const {
    SearchStatistics stats;
    QueryContext ctx{.stats = &stats};

    auto combined_filter = std::make_shared<CombinedFilter>();
    combined_filter->AppendFilter(this->label_table_->GetDeletedIdsFilter());
    if (filter != nullptr) {
        combined_filter->AppendFilter(
            std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_));
    }
    FilterPtr ft = nullptr;
    if (not combined_filter->IsEmpty()) {
        ft = combined_filter;
    }

    int64_t query_dim = query->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }
    // check radius
    CHECK_ARGUMENT(radius >= 0, fmt::format("radius({}) must be greater equal than 0", radius))

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    // check limited_size
    CHECK_ARGUMENT(limited_size != 0,
                   fmt::format("limited_size({}) must not be equal to 0", limited_size));

    std::shared_lock force_remove_rlock(this->force_remove_mutex_);
    std::shared_lock shared_lock(this->global_mutex_);

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_id_;
    search_param.topk = 1;
    search_param.ef = 1;

    if (search_param.ep == INVALID_ENTRY_POINT) {
        return make_empty_dataset_with_stats();
    }

    const auto* raw_query = get_data(query);
    for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
        auto result = this->search_one_graph(raw_query,
                                             this->route_graphs_[i],
                                             this->basic_flatten_codes_,
                                             search_param,
                                             (VisitedListPtr) nullptr,
                                             &ctx);
        search_param.ep = result->Top().second;
    }

    auto params = HGraphSearchParameters::FromJson(parameters);

    CHECK_ARGUMENT((1 <= params.ef_search) and (params.ef_search <= 1000),  // NOLINT
                   fmt::format("ef_search({}) must in range[1, 1000]", params.ef_search));
    search_param.ef = std::max(params.ef_search, limited_size);
    search_param.is_inner_id_allowed = ft;
    search_param.radius = radius;
    search_param.search_mode = RANGE_SEARCH;
    search_param.consider_duplicate = true;
    search_param.range_search_limit_size = static_cast<int>(limited_size);
    search_param.parallel_search_thread_count = params.parallel_search_thread_count;
    search_param.enable_rabitq_one_bit_search = params.rabitq_one_bit_search;

    auto search_result = this->search_one_graph(raw_query,
                                                this->bottom_graph_,
                                                this->basic_flatten_codes_,
                                                search_param,
                                                (VisitedListPtr) nullptr,
                                                &ctx);

    if (use_reorder_) {
        this->reorder(
            raw_query, this->get_reorder_codes(), search_result, limited_size, nullptr, ctx);
    } else if (params.rabitq_one_bit_search) {
        this->reorder(
            raw_query, this->basic_flatten_codes_, search_result, limited_size, nullptr, ctx);
    }

    if (limited_size > 0) {
        while (search_result->Size() > limited_size) {
            search_result->Pop();
        }
    }

    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, ids] = create_fast_dataset(count, allocator_);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0) {
        extra_infos =
            static_cast<char*>(allocator_->Allocate(extra_info_size_ * search_result->Size()));
        dataset_results->ExtraInfos(extra_infos);
    }
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        ids[j] = this->label_table_->GetLabelById(search_result->Top().second);
        if (extra_infos != nullptr) {
            this->extra_infos_->GetExtraInfoById(search_result->Top().second,
                                                 extra_infos + extra_info_size_ * j);
        }
        search_result->Pop();
    }

    dataset_results->Statistics(stats.Dump());
    return std::move(dataset_results);
}

[[nodiscard]] DatasetPtr
HGraph::SearchWithRequest(const SearchRequest& request) const {
    SearchStatistics stats;
    QueryContext ctx{.alloc = this->allocator_, .stats = &stats};
    if (request.search_allocator_ != nullptr) {
        ctx.alloc = request.search_allocator_;
    }

    const auto& query = request.query_;
    int64_t query_dim = query->GetDim();
    auto k = request.topk_;
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }

    auto params = HGraphSearchParameters::FromJson(request.params_str_);

    auto ef_search_threshold = std::max<int64_t>(AMPLIFICATION_FACTOR * k, 1000);
    CHECK_ARGUMENT(  // NOLINT
        (1 <= params.ef_search) and (params.ef_search <= ef_search_threshold),
        fmt::format("ef_search({}) must in range[1, {}]", params.ef_search, ef_search_threshold));

    std::shared_lock force_remove_rlock(this->force_remove_mutex_);
    std::shared_lock shared_lock(this->global_mutex_);

    // check k
    CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
    k = std::min(k, GetNumElements());

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    // Setup reasoning context if expected labels are provided.
    std::shared_ptr<ReasoningContext> reasoning_ctx;
    if (not request.expected_labels_.empty()) {
        reasoning_ctx = std::make_shared<ReasoningContext>(this->allocator_);
        reasoning_ctx->SetSearchParams(k, "HGraph", use_reorder_, request.filter_ != nullptr);

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

        const auto* const query_vector = get_data(query);
        auto precise_flatten = this->basic_flatten_codes_;
        if (use_reorder_) {
            precise_flatten = this->high_precise_codes_;
        }
        if (create_new_raw_vector_) {
            precise_flatten = this->raw_vector_;
        }
        auto computer = precise_flatten->FactoryComputer(query_vector);
        for (const auto& pair : label_to_inner_id) {
            float dist = 0.0F;
            const auto inner_id = pair.second;
            precise_flatten->Query(&dist, computer, &inner_id, 1);
            reasoning_ctx->SetTrueDistance(inner_id, dist);
        }
        ctx.reasoning_ctx = reasoning_ctx.get();
    }

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_id_;
    search_param.topk = 1;
    search_param.ef = 1;
    search_param.is_inner_id_allowed = nullptr;

    if (search_param.ep == INVALID_ENTRY_POINT) {
        return make_empty_dataset_with_stats();
    }

    auto vt = this->pool_->TakeOne();

    const auto* raw_query = get_data(query);
    for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
        auto result = this->search_one_graph(
            raw_query, this->route_graphs_[i], this->basic_flatten_codes_, search_param, vt, &ctx);
        search_param.ep = result->Top().second;
    }

    auto combined_filter = std::make_shared<CombinedFilter>();
    combined_filter->AppendFilter(this->label_table_->GetDeletedIdsFilter());
    if (request.filter_ != nullptr) {
        if (params.use_extra_info_filter) {
            combined_filter->AppendFilter(
                std::make_shared<ExtraInfoWrapperFilter>(request.filter_, this->extra_infos_));
        } else {
            combined_filter->AppendFilter(
                std::make_shared<InnerIdWrapperFilter>(request.filter_, *this->label_table_));
        }
    }
    FilterPtr ft = nullptr;
    if (not combined_filter->IsEmpty()) {
        ft = combined_filter;
    }

    if (request.enable_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        auto& schema = this->attr_filter_index_->field_type_map_;
        auto expr = AstParse(request.attribute_filter_str_, &schema);
        auto executor = Executor::MakeInstance(this->allocator_, expr, this->attr_filter_index_);
        executor->Init();
        search_param.executors.emplace_back(executor);
    }

    search_param.ef = std::max(params.ef_search, k);
    search_param.is_inner_id_allowed = ft;
    search_param.topk = static_cast<int64_t>(search_param.ef);
    if (params.topk_factor > 1.0F) {
        search_param.topk = std::min(
            search_param.topk, static_cast<int64_t>(static_cast<float>(k) * params.topk_factor));
    }
    search_param.consider_duplicate = true;
    search_param.enable_rabitq_one_bit_search = params.rabitq_one_bit_search;
    if (params.enable_time_record) {
        search_param.time_cost = std::make_shared<Timer>();
        search_param.time_cost->SetThreshold(params.timeout_ms);
        stats.is_timeout.store(false, std::memory_order_relaxed);
    }
    search_param.parallel_search_thread_count = params.parallel_search_thread_count;

    // hops_limit only takes effect when it's greater than ef_search
    if (params.hops_limit <= static_cast<uint32_t>(params.ef_search)) {
        search_param.hops_limit = std::numeric_limits<uint32_t>::max();
        if (params.hops_limit != std::numeric_limits<uint32_t>::max()) {
            logger::warn(
                fmt::format("hops_limit({}) is not greater than ef_search({}), ignoring hops_limit",
                            params.hops_limit,
                            params.ef_search));
        }
    } else {
        search_param.hops_limit = params.hops_limit;
    }

    DistanceRecordVector rabitq_lower_bound_candidates(ctx.alloc);
    auto* rabitq_lower_bound_candidates_ptr =
        search_param.enable_rabitq_one_bit_search and use_reorder_ and reorder_by_base_
            ? &rabitq_lower_bound_candidates
            : nullptr;

    auto search_result = this->search_one_graph(raw_query,
                                                this->bottom_graph_,
                                                this->basic_flatten_codes_,
                                                search_param,
                                                vt,
                                                &ctx,
                                                rabitq_lower_bound_candidates_ptr);

    this->pool_->ReturnOne(vt);

    if (use_reorder_) {
        this->reorder(raw_query,
                      this->get_reorder_codes(),
                      search_result,
                      k,
                      nullptr,
                      ctx,
                      rabitq_lower_bound_candidates_ptr);
    } else if (params.rabitq_one_bit_search) {
        this->reorder(raw_query, this->basic_flatten_codes_, search_result, k, nullptr, ctx);
    }

    while (search_result->Size() > k) {
        search_result->Pop();
    }

    // return an empty dataset directly if searcher returns nothing
    if (search_result->Empty()) {
        auto dataset_result = DatasetImpl::MakeEmptyDataset();
        dataset_result->Statistics(stats.Dump());
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
        dataset_results->ExtraInfos(extra_infos);
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
    dataset_results->Statistics(stats.Dump());

    // Generate reasoning report if reasoning context was created
    if (reasoning_ctx) {
        reasoning_ctx->MarkResult(result_inner_ids);
        reasoning_ctx->DiagnoseExpectedTargets();
        dataset_results->Reasoning(reasoning_ctx->GenerateReport());
    }

    return std::move(dataset_results);
}

}  // namespace vsag
