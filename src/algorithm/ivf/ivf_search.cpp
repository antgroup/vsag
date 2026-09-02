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
#include <atomic>
#include <cmath>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "attr/argparse.h"
#include "attr/executor/executor.h"
#include "dataset_impl.h"
#include "impl/heap/standard_heap.h"
#include "impl/inner_search_param.h"
#include "impl/reasoning/search_reasoning.h"
#include "impl/reorder/bucket_reorder.h"
#include "impl/reorder/flatten_reorder.h"
#include "inner_string_params.h"
#include "ivf.h"  // IWYU pragma: keep
#include "query_context.h"
#include "simd/normalize.h"
#include "utils/search_threshold.h"
#include "utils/util_functions.h"

namespace vsag {

static constexpr BucketIdType INVALID_BUCKET_ID = static_cast<BucketIdType>(-1);

DatasetPtr
IVF::KnnSearch(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const FilterPtr& filter) const {
    SearchRequest req;
    req.mode_ = SearchMode::KNN_SEARCH;
    req.query_ = query;
    req.topk_ = k;
    req.params_str_ = parameters;
    req.threshold_ = ParseSearchThreshold(parameters);
    if (filter != nullptr) {
        req.filter_ = filter;
    }
    return this->SearchWithRequest(req);
}

DatasetPtr
IVF::RangeSearch(const DatasetPtr& query,
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

InnerSearchParam
IVF::create_search_param(const std::string& parameters, const FilterPtr& filter) const {
    InnerSearchParam param;
    param.is_inner_id_allowed = this->create_search_filter(filter);
    auto search_param = IVFSearchParameters::FromJson(parameters);
    if (search_param.disable_bucket_scan) {
        param.scan_bucket_size = static_cast<BucketIdType>(search_param.scan_buckets_count);
    } else {
        param.scan_bucket_size = std::min(
            static_cast<BucketIdType>(search_param.scan_buckets_count), bucket_->bucket_count_);
    }
    param.disable_bucket_scan = search_param.disable_bucket_scan;
    param.factor = search_param.topk_factor;
    param.enable_reorder = search_param.enable_reorder;
    param.first_order_scan_ratio = search_param.first_order_scan_ratio;
    param.parallel_search_thread_count = search_param.parallel_search_thread_count;
    param.ef = static_cast<uint64_t>(search_param.ef_search);
    if (search_param.enable_time_record) {
        param.time_cost = std::make_shared<Timer>();
        param.time_cost->SetThreshold(search_param.timeout_ms);
    }
    return param;
}

DatasetPtr
IVF::route_buckets_only(const DatasetPtr& query,
                        const InnerSearchParam& param,
                        QueryContext& ctx) const {
    const auto num_queries = query->GetNumElements();
    const auto* query_data = query->GetFloat32Vectors();
    const auto buckets_per_query = param.scan_bucket_size;
    const auto candidate_buckets =
        partition_strategy_->ClassifyDatasForSearch(query_data, num_queries, param, &ctx);

    auto result = Dataset::Make();
    if (num_queries == 0 || buckets_per_query == 0) {
        return result->NumElements(0)->Dim(0);
    }

    auto* alloc = (ctx.alloc != nullptr) ? ctx.alloc : allocator_;
    const auto total_slots = num_queries * buckets_per_query;
    auto* ids = static_cast<int64_t*>(alloc->Allocate(sizeof(int64_t) * total_slots));
    auto* distances = static_cast<float*>(alloc->Allocate(sizeof(float) * total_slots));
    const auto dim = partition_strategy_->dim_;
    const auto metric = partition_strategy_->metric_type_;

    Vector<float> centroid(dim, allocator_);
    Vector<float> norm_query(dim, allocator_);
    Vector<float> norm_centroid(dim, allocator_);
    for (int64_t q = 0; q < num_queries; ++q) {
        const auto* query_vec = query_data + q * dim;
        if (metric == MetricType::METRIC_TYPE_COSINE) {
            Normalize(query_vec, norm_query.data(), dim);
        }
        for (int64_t b = 0; b < buckets_per_query; ++b) {
            const auto idx = q * buckets_per_query + b;
            const auto bucket_id = candidate_buckets[idx];
            if (bucket_id == INVALID_BUCKET_ID) {
                ids[idx] = -1;
                distances[idx] = std::numeric_limits<float>::infinity();
                continue;
            }
            partition_strategy_->GetCentroid(bucket_id, centroid);
            float dist = 0.0F;
            if (metric == MetricType::METRIC_TYPE_L2SQR) {
                for (int64_t d = 0; d < dim; ++d) {
                    auto diff = query_vec[d] - centroid[d];
                    dist += diff * diff;
                }
            } else if (metric == MetricType::METRIC_TYPE_COSINE) {
                Normalize(centroid.data(), norm_centroid.data(), dim);
                for (int64_t d = 0; d < dim; ++d) {
                    dist += norm_query[d] * norm_centroid[d];
                }
                dist = 1.0F - dist;
            } else {
                for (int64_t d = 0; d < dim; ++d) {
                    dist += query_vec[d] * centroid[d];
                }
                dist = 1.0F - dist;
            }
            ids[idx] = static_cast<int64_t>(bucket_id);
            distances[idx] = dist;
            if (ctx.stats != nullptr) {
                ctx.stats->AddDistance(SearchStatistics::DistancePhase::ROUTING,
                                       DistanceEvaluationBackend::FP32);
            }
        }
    }

    return result->NumElements(num_queries)
        ->Dim(buckets_per_query)
        ->Ids(ids)
        ->Distances(distances)
        ->Owner(true, alloc);
}

DatasetPtr
IVF::reorder(int64_t topk,
             DistHeapPtr& input,
             const float* query,
             const InnerSearchParam& param,
             QueryContext& ctx,
             ReasoningContext* reasoning_ctx,
             const std::optional<float>& distance_threshold) const {
    auto reorder_heap =
        reorder_->Reorder(input, query, topk, ctx, nullptr, nullptr, distance_threshold);
    auto dataset_results = this->pack_knn_result(reorder_heap, ctx.alloc);

    return dataset_results;
}

template <InnerSearchMode mode>
DistHeapPtr
IVF::search(const DatasetPtr& query,
            const InnerSearchParam& param,
            QueryContext& ctx,
            ReasoningContext* reasoning_ctx) const {
    const auto* query_data = query->GetFloat32Vectors();
    Vector<BucketIdType> candidate_buckets(allocator_);
    if (not param.bucket_ids.empty()) {
        candidate_buckets.reserve(param.bucket_ids.size());
        for (auto id : param.bucket_ids) {
            candidate_buckets.push_back(static_cast<BucketIdType>(id));
        }
    } else {
        candidate_buckets = partition_strategy_->ClassifyDatasForSearch(query_data, 1, param, &ctx);
    }
    if (reasoning_ctx != nullptr) {
        reasoning_ctx->RecordBucketSelection(candidate_buckets);
    }
    auto computer = bucket_->FactoryComputer(query_data);

    int64_t topk = param.topk;
    if constexpr (mode == RANGE_SEARCH) {
        topk = param.range_search_limit_size;
        if (topk < 0) {
            topk = this->GetNumElements();
        }
    }
    // Scale topk to ensure sufficient candidates after deduplication when buckets_per_data_ > 1
    int64_t origin_topk = topk;
    if (buckets_per_data_ > 1) {
        if (topk <= std::numeric_limits<int64_t>::max() / buckets_per_data_) {
            topk *= buckets_per_data_;
        } else {
            topk = std::numeric_limits<int64_t>::max();
        }
    }

    DistHeapPtr search_result = nullptr;

    auto bucket_count = candidate_buckets.size();
    auto search_thread_count = param.parallel_search_thread_count;
    if (this->thread_pool_ == nullptr) {
        search_thread_count = 1;
    }
    std::vector<DistHeapPtr> heaps(search_thread_count);
    std::atomic<uint64_t> cur_bucket_num(0);
    auto search_func = [&](int64_t thread_id) -> void {
        heaps[thread_id] = DistanceHeap::MakeInstanceBySize<true, false>(this->allocator_, topk);
        auto& heap = heaps[thread_id];
        Vector<float> dist(allocator_);
        uint64_t i = cur_bucket_num.fetch_add(1);
        for (; i < bucket_count; i = cur_bucket_num.fetch_add(1)) {
            if (param.time_cost != nullptr and param.time_cost->CheckOvertime() and
                ctx.stats != nullptr) {
                ctx.stats->is_timeout.store(true, std::memory_order_relaxed);
                break;
            }
            auto bucket_id = candidate_buckets[i];
            if (bucket_id == INVALID_BUCKET_ID) {
                break;
            }
            bucket_searcher_->Search(bucket_id,
                                     bucket_,
                                     computer,
                                     param,
                                     thread_id,
                                     topk,
                                     buckets_per_data_,
                                     heap,
                                     dist,
                                     reasoning_ctx);
        }
    };
    std::vector<std::future<void>> futures;
    if (this->thread_pool_ != nullptr and search_thread_count > 1) {
        for (int64_t thread_id = 0; thread_id < search_thread_count; ++thread_id) {
            auto future = this->thread_pool_->GeneralEnqueue(search_func, thread_id);
            futures.emplace_back(std::move(future));
        }
    } else {
        search_func(0);
        search_result = heaps[0];
    }

    if (this->thread_pool_ != nullptr and search_thread_count > 1) {
        for (auto& future : futures) {
            future.get();
        }
        search_result = DistanceHeap::MakeInstanceBySize<true, true>(this->allocator_, topk);
        for (auto& heap : heaps) {
            auto size = heap->Size();
            const auto* data = heap->GetData();
            for (int i = 0; i < size; ++i) {
                if (reasoning_ctx != nullptr and
                    search_result->Size() >= static_cast<uint64_t>(topk) and
                    data[i].first < search_result->Top().first) {
                    reasoning_ctx->RecordEviction(search_result->Top().second / buckets_per_data_,
                                                  1);
                }
                search_result->Push(data[i]);
            }
        }
    }

    // Deduplicate ids when buckets_per_data_ > 1
    if (buckets_per_data_ > 1) {
        std::unordered_map<InnerIdType, float> id_to_min_dist;
        while (!search_result->Empty()) {
            const auto& [dist_val, id] = search_result->Top();
            auto origin_id = id / buckets_per_data_;
            // Keep the smallest distance for each id
            if (id_to_min_dist.find(origin_id) == id_to_min_dist.end() ||
                dist_val < id_to_min_dist[origin_id]) {
                id_to_min_dist[origin_id] = dist_val;
            }
            search_result->Pop();
        }

        auto cur_heap_top2 = std::numeric_limits<float>::max();
        for (const auto& [origin_id, dist_val] : id_to_min_dist) {
            if (dist_val < cur_heap_top2) {
                search_result->Push(dist_val, origin_id);
            }
            if (search_result->Size() > origin_topk) {
                search_result->Pop();
            }
            if (not search_result->Empty() and search_result->Size() == origin_topk) {
                cur_heap_top2 = search_result->Top().first;
            }
        }
    }

    return search_result;
}

DistHeapPtr
IVF::search_with_custom_distance(const DatasetPtr& query,
                                 const SearchRequest& request,
                                 const InnerSearchParam& param,
                                 QueryContext& ctx,
                                 ReasoningContext* reasoning_ctx) const {
    const auto* query_data = query->GetFloat32Vectors();
    Vector<BucketIdType> candidate_buckets(allocator_);
    if (not param.bucket_ids.empty()) {
        candidate_buckets.reserve(param.bucket_ids.size());
        for (auto id : param.bucket_ids) {
            candidate_buckets.push_back(static_cast<BucketIdType>(id));
        }
    } else {
        candidate_buckets = partition_strategy_->ClassifyDatasForSearch(query_data, 1, param, &ctx);
    }
    if (reasoning_ctx != nullptr) {
        reasoning_ctx->RecordBucketSelection(candidate_buckets);
    }

    int64_t topk = request.topk_;
    const int64_t origin_topk = topk;
    if (buckets_per_data_ > 1) {
        CHECK_ARGUMENT(topk <= std::numeric_limits<int64_t>::max() / buckets_per_data_,
                       "topk is too large for multi-bucket IVF search");
        topk *= buckets_per_data_;
    }

    auto search_result = DistanceHeap::MakeInstanceBySize<true, false>(this->allocator_, topk);
    const auto& filter = param.is_inner_id_allowed;
    Filter* attr_filter = nullptr;

    Vector<InnerIdType> candidate_ids(this->allocator_);
    Vector<int64_t> candidate_labels(this->allocator_);
    Vector<float> scores(this->allocator_);
    const uint64_t batch_capacity = std::min<uint64_t>(
        request.distance_batch_size_, std::max<uint64_t>(1, this->GetNumElements()));
    candidate_ids.reserve(batch_capacity);
    candidate_labels.reserve(batch_capacity);
    scores.resize(batch_capacity);

    auto is_timed_out = [&]() {
        if (param.time_cost == nullptr or not param.time_cost->CheckOvertime()) {
            return false;
        }
        if (ctx.stats != nullptr) {
            ctx.stats->is_timeout.store(true, std::memory_order_relaxed);
        }
        return true;
    };

    auto submit_batch = [&]() {
        if (candidate_ids.empty()) {
            return true;
        }
        if (is_timed_out()) {
            return false;
        }
        request.distance_batch_func_(
            candidate_labels.data(), candidate_labels.size(), scores.data());
        if (ctx.stats != nullptr) {
            ctx.stats->AddDistance(SearchStatistics::DistancePhase::APPROXIMATE,
                                   DistanceEvaluationBackend::UNKNOWN,
                                   candidate_ids.size());
        }
        for (uint64_t i = 0; i < candidate_ids.size(); ++i) {
            CHECK_ARGUMENT(std::isfinite(scores[i]),
                           "custom query distance callback must return finite scores");
            const auto origin_id = candidate_ids[i] / buckets_per_data_;
            if (filter != nullptr and not filter->CheckValid(origin_id)) {
                if (reasoning_ctx != nullptr) {
                    reasoning_ctx->RecordFilterReject(origin_id);
                }
                continue;
            }
            if (reasoning_ctx != nullptr) {
                reasoning_ctx->RecordVisit(origin_id, scores[i], 0);
            }
            search_result->Push(scores[i], candidate_ids[i]);
            while (search_result->Size() > static_cast<uint64_t>(topk)) {
                if (reasoning_ctx != nullptr) {
                    reasoning_ctx->RecordEviction(search_result->Top().second / buckets_per_data_,
                                                  0);
                }
                search_result->Pop();
            }
        }
        candidate_ids.clear();
        candidate_labels.clear();
        return true;
    };

    bool timed_out = false;
    for (const auto bucket_id : candidate_buckets) {
        if (is_timed_out()) {
            timed_out = true;
            break;
        }
        if (bucket_id == INVALID_BUCKET_ID) {
            continue;
        }
        if (not param.executors.empty()) {
            param.executors[0]->Clear();
            attr_filter = param.executors[0]->Run(bucket_id);
        }
        const auto bucket_size = bucket_->GetBucketSize(bucket_id);
        const auto* ids = bucket_->GetInnerIds(bucket_id);
        for (InnerIdType offset = 0; offset < bucket_size; ++offset) {
            const auto inner_id = ids[offset];
            if (inner_id == std::numeric_limits<InnerIdType>::max()) {
                continue;
            }
            const auto origin_id = inner_id / buckets_per_data_;
            if (attr_filter != nullptr and not attr_filter->CheckValid(offset)) {
                if (reasoning_ctx != nullptr) {
                    reasoning_ctx->RecordFilterReject(origin_id);
                }
                continue;
            }
            candidate_ids.push_back(inner_id);
            candidate_labels.push_back(label_table_->GetLabelById(origin_id));
            if (candidate_ids.size() == batch_capacity and not submit_batch()) {
                timed_out = true;
                break;
            }
        }
        if (timed_out) {
            break;
        }
    }
    if (not timed_out) {
        submit_batch();
    }

    if (buckets_per_data_ == 1) {
        return search_result;
    }

    std::unordered_map<InnerIdType, float> id_to_min_score;
    while (not search_result->Empty()) {
        const auto& [score, inner_id] = search_result->Top();
        const auto origin_id = inner_id / buckets_per_data_;
        auto iter = id_to_min_score.find(origin_id);
        if (iter == id_to_min_score.end() or score < iter->second) {
            id_to_min_score[origin_id] = score;
        }
        search_result->Pop();
    }

    for (const auto& [origin_id, score] : id_to_min_score) {
        search_result->Push(score, origin_id);
        if (search_result->Size() > static_cast<uint64_t>(origin_topk)) {
            search_result->Pop();
        }
    }
    return search_result;
}

DatasetPtr
IVF::SearchWithRequest(const SearchRequest& request) const {
    ValidateSearchThreshold(request.threshold_);
    SearchStatistics stats;
    QueryContext ctx{.alloc = request.search_allocator_, .stats = &stats};

    bool is_range = (request.mode_ == SearchMode::RANGE_SEARCH);

    auto param = this->create_search_param(request.params_str_, request.filter_);
    const bool use_custom_distance = request.distance_batch_func_ != nullptr;
    if (use_custom_distance) {
        CHECK_ARGUMENT(request.distance_batch_size_ > 0,
                       "custom query distance batch size must be greater than 0");
        CHECK_ARGUMENT(not is_range, "IVF custom query distance only supports KNN search");
        CHECK_ARGUMENT(not param.disable_bucket_scan,
                       "IVF custom query distance does not support disable_bucket_scan");
        CHECK_ARGUMENT(request.topk_ > 0, "topk must be greater than 0");
        CHECK_ARGUMENT(param.parallel_search_thread_count == 1,
                       "IVF custom query distance does not support parallel search");
        param.enable_reorder = false;
    }
    param.query_context = &ctx;

    if (not request.bucket_ids_.empty()) {
        auto query_check = request.query_;
        CHECK_ARGUMENT(query_check != nullptr, "query dataset cannot be null");
        CHECK_ARGUMENT(request.bucket_ids_.size() == query_check->GetNumElements(),
                       fmt::format("bucket_ids_ size must match the number of query vectors; "
                                   "got {} bucket lists for {} queries",
                                   request.bucket_ids_.size(),
                                   query_check->GetNumElements()));
        CHECK_ARGUMENT(query_check->GetFloat32Vectors() != nullptr,
                       "query float32 vectors cannot be null");
        CHECK_ARGUMENT(query_check->GetDim() == this->dim_,
                       "query dimension must match index dimension");
        CHECK_ARGUMENT(not param.disable_bucket_scan,
                       "bucket_ids_ is incompatible with disable_bucket_scan mode");
        for (uint64_t query_idx = 0; query_idx < request.bucket_ids_.size(); ++query_idx) {
            const auto& ids = request.bucket_ids_[query_idx];
            CHECK_ARGUMENT(not ids.empty(),
                           fmt::format("bucket_ids_[{}] must not be empty; "
                                       "use empty outer vector for default routing",
                                       query_idx));
            std::set<int64_t> seen_ids;
            for (auto id : ids) {
                CHECK_ARGUMENT(
                    id >= 0,
                    fmt::format(
                        "bucket_id {} out of range [0, {})", id, this->bucket_->bucket_count_));
                CHECK_ARGUMENT(
                    id < static_cast<int64_t>(this->bucket_->bucket_count_),
                    fmt::format(
                        "bucket_id {} out of range [0, {})", id, this->bucket_->bucket_count_));
                CHECK_ARGUMENT(seen_ids.insert(id).second,
                               fmt::format("duplicate bucket_id {}", id));
            }
        }
        if (query_check->GetNumElements() == 1) {
            param.bucket_ids.assign(request.bucket_ids_[0].begin(), request.bucket_ids_[0].end());
        }
    }

    auto query = request.query_;
    if (use_custom_distance) {
        CHECK_ARGUMENT(query != nullptr, "query dataset cannot be null");
        CHECK_ARGUMENT(query->GetNumElements() == 1,
                       "IVF custom search requires exactly one query");
        CHECK_ARGUMENT(query->GetFloat32Vectors() != nullptr,
                       "query float32 vectors cannot be null");
        CHECK_ARGUMENT(query->GetDim() == this->dim_, "query dimension must match index dimension");
    }
    if (param.disable_bucket_scan) {
        CHECK_ARGUMENT(query != nullptr, "query dataset cannot be null");
        CHECK_ARGUMENT(query->GetNumElements() >= 1,
                       "disable bucket scan requires at least one query");
        CHECK_ARGUMENT(query->GetFloat32Vectors() != nullptr,
                       "query float32 vectors cannot be null");
        CHECK_ARGUMENT(query->GetDim() == this->dim_, "query dimension must match index dimension");
        CHECK_ARGUMENT(not request.threshold_.has_value(),
                       "threshold filtering is not supported with disable_bucket_scan");
        auto result = this->route_buckets_only(query, param, ctx);
        result->Statistics(stats.Dump());
        return result;
    }

    if (query != nullptr && query->GetNumElements() > 1) {
        CHECK_ARGUMENT(not is_range, "IVF batch search only supports KNN search");
        CHECK_ARGUMENT(not use_custom_distance,
                       "IVF batch search does not support custom query distance");
        CHECK_ARGUMENT(request.expected_labels_.empty(),
                       "IVF batch search does not support expected labels");
        CHECK_ARGUMENT(request.topk_ > 0, "topk must be greater than 0");
        CHECK_ARGUMENT(query->GetFloat32Vectors() != nullptr,
                       "query float32 vectors cannot be null");
        CHECK_ARGUMENT(query->GetDim() == this->dim_, "query dimension must match index dimension");

        const auto num_queries = query->GetNumElements();
        const auto total_slots = num_queries * request.topk_;
        auto* alloc = select_query_allocator(ctx.alloc, this->allocator_);
        auto* ids = static_cast<int64_t*>(alloc->Allocate(sizeof(int64_t) * total_slots));
        auto* distances = static_cast<float*>(alloc->Allocate(sizeof(float) * total_slots));
        std::fill_n(ids, total_slots, -1);
        std::fill_n(distances, total_slots, std::numeric_limits<float>::infinity());

        const auto* query_data = query->GetFloat32Vectors();
        auto base_request = request;
        base_request.expected_labels_.clear();
        if (not request.bucket_ids_.empty()) {
            base_request.bucket_ids_.clear();
        }
        auto search_func = [&](int64_t query_idx) -> void {
            auto one_query = Dataset::Make();
            one_query->NumElements(1)
                ->Dim(query->GetDim())
                ->Float32Vectors(query_data + query_idx * query->GetDim())
                ->Owner(false);

            auto one_request = base_request;
            one_request.query_ = one_query;
            if (not request.bucket_ids_.empty()) {
                one_request.bucket_ids_ = {request.bucket_ids_[query_idx]};
            }
            JsonType json = JsonType::Parse(base_request.params_str_);
            if (json.Contains("ivf")) {
                json["ivf"]["parallelism"].SetInt64(1);
            }
            one_request.params_str_ = json.Dump();
            auto one_result = this->SearchWithRequest(one_request);
            const auto count = std::min(request.topk_, one_result->GetDim());
            if (count > 0) {
                std::copy_n(one_result->GetIds(), count, ids + query_idx * request.topk_);
                std::copy_n(
                    one_result->GetDistances(), count, distances + query_idx * request.topk_);
            }
        };

        if (this->thread_pool_ != nullptr and param.parallel_search_thread_count > 1) {
            std::vector<std::future<void>> futures;
            for (int64_t query_idx = 0; query_idx < num_queries; ++query_idx) {
                auto future = this->thread_pool_->GeneralEnqueue(search_func, query_idx);
                futures.emplace_back(std::move(future));
            }
            for (auto& future : futures) {
                future.get();
            }
        } else {
            for (int64_t query_idx = 0; query_idx < num_queries; ++query_idx) {
                search_func(query_idx);
            }
        }
        auto result = Dataset::Make()
                          ->NumElements(num_queries)
                          ->Dim(request.topk_)
                          ->Ids(ids)
                          ->Distances(distances)
                          ->Owner(true, alloc);
        result->Statistics(stats.Dump());
        return result;
    }

    if (request.enable_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        auto& schema = this->attr_filter_index_->field_type_map_;
        auto expr = AstParse(request.attribute_filter_str_, &schema);
        for (int64_t i = 0; i < param.parallel_search_thread_count; ++i) {
            auto executor =
                Executor::MakeInstance(this->allocator_, expr, this->attr_filter_index_);
            executor->Init();
            param.executors.emplace_back(executor);
        }
    }
    std::shared_ptr<ReasoningContext> reasoning_ctx;
    if (not request.expected_labels_.empty()) {
        reasoning_ctx = std::make_shared<ReasoningContext>(this->allocator_);
        reasoning_ctx->SetSearchParams(
            request.topk_, "IVF", use_reorder_, request.filter_ != nullptr);

        UnorderedMap<int64_t, InnerIdType> label_to_inner_id(this->allocator_);
        std::vector<std::tuple<InnerIdType, BucketIdType, InnerIdType>> locations;
        {
            std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
            locations.reserve(request.expected_labels_.size());
            for (const auto& label : request.expected_labels_) {
                auto [success, inner_id] = this->label_table_->TryGetIdByLabel(label, true);
                if (success) {
                    label_to_inner_id[label] = inner_id;
                    auto [bucket_id, offset_id] = this->get_location(inner_id);
                    locations.emplace_back(inner_id, bucket_id, offset_id);
                }
            }
        }

        Vector<int64_t> expected_labels_vec(this->allocator_);
        expected_labels_vec.reserve(request.expected_labels_.size());
        for (const auto& label : request.expected_labels_) {
            expected_labels_vec.push_back(label);
        }
        reasoning_ctx->InitializeExpectedTargets(expected_labels_vec, label_to_inner_id);

        const auto* query_data = query->GetFloat32Vectors();
        auto computer = this->bucket_->FactoryComputer(query_data);
        for (const auto& [inner_id, bucket_id, offset_id] : locations) {
            float dist = this->bucket_->QueryOneById(computer, bucket_id, offset_id);
            if (ctx.stats != nullptr) {
                ctx.stats->AddDistance(SearchStatistics::DistancePhase::APPROXIMATE,
                                       this->bucket_->backend_);
            }
            reasoning_ctx->SetTrueDistance(inner_id, dist);
        }
        ctx.reasoning_ctx = reasoning_ctx.get();
    }

    if (use_custom_distance) {
        param.search_mode = KNN_SEARCH;
        param.topk = request.topk_;
        auto search_result =
            search_with_custom_distance(query, request, param, ctx, reasoning_ctx.get());
        filter_search_result_by_threshold(
            search_result, request.threshold_, select_query_allocator(ctx.alloc, this->allocator_));
        if (search_result == nullptr || search_result->Empty()) {
            auto dataset_results = DatasetImpl::MakeEmptyDataset();
            this->AttachReasoningReport(dataset_results, reasoning_ctx.get());
            dataset_results->Statistics(stats.Dump());
            return dataset_results;
        }
        auto dataset_results = this->pack_knn_result(search_result, ctx.alloc);
        this->AttachReasoningReport(dataset_results, reasoning_ctx.get());
        dataset_results->Statistics(stats.Dump());
        return dataset_results;
    }

    if (is_range) {
        param.search_mode = RANGE_SEARCH;
        param.radius = request.radius_;
        param.range_search_limit_size = static_cast<int>(request.limited_size_);
        if (use_reorder_ and param.enable_reorder and request.limited_size_ > 0) {
            CHECK_ARGUMENT(param.factor > 0.0F,
                           fmt::format("factor must be positive when use_reorder is true, got {}",
                                       param.factor));
            param.range_search_limit_size =
                static_cast<int>(param.factor * static_cast<float>(request.limited_size_));
        }
        auto search_result = this->search<RANGE_SEARCH>(query, param, ctx, reasoning_ctx.get());
        if (use_reorder_ and param.enable_reorder) {
            int64_t k = (request.limited_size_ > 0) ? request.limited_size_
                                                    : static_cast<int64_t>(search_result->Size());
            auto result = reorder(
                k, search_result, query->GetFloat32Vectors(), param, ctx, reasoning_ctx.get());
            result->Statistics(stats.Dump());
            this->AttachReasoningReport(result, reasoning_ctx.get());
            return result;
        }
        auto dataset_results = this->pack_knn_result(search_result, ctx.alloc);
        dataset_results->Statistics(stats.Dump());
        this->AttachReasoningReport(dataset_results, reasoning_ctx.get());
        return dataset_results;
    }

    // KNN mode
    param.search_mode = KNN_SEARCH;
    param.topk = request.topk_;
    if (use_reorder_ and param.enable_reorder) {
        CHECK_ARGUMENT(
            param.factor > 0.0F,
            fmt::format("factor must be positive when use_reorder is true, got {}", param.factor));
        param.topk = static_cast<int64_t>(param.factor * static_cast<float>(request.topk_));
        if (request.threshold_.has_value()) {
            param.topk = std::max(param.topk, request.topk_);
        }
    }
    const bool reorder_enabled = use_reorder_ and param.enable_reorder;
    // Reordered searches defer the finite bound to exact distances, but bucket selection still
    // needs threshold-mode state so non-finite approximations cannot consume the rerank pool.
    param.distance_threshold = request.threshold_;
    auto search_result = this->search<KNN_SEARCH>(query, param, ctx, reasoning_ctx.get());
    if (reorder_enabled) {
        auto result = reorder(request.threshold_.has_value() ? param.topk : request.topk_,
                              search_result,
                              query->GetFloat32Vectors(),
                              param,
                              ctx,
                              reasoning_ctx.get(),
                              request.threshold_);
        result = FilterDatasetByThreshold(result, request.threshold_, ctx.alloc, request.topk_);
        AttachReasoningReport(result, reasoning_ctx.get());
        result->Statistics(stats.Dump());
        return result;
    }
    filter_search_result_by_threshold(
        search_result, request.threshold_, select_query_allocator(ctx.alloc, this->allocator_));
    if (search_result == nullptr || search_result->Empty()) {
        auto dataset_results = DatasetImpl::MakeEmptyDataset();
        this->AttachReasoningReport(dataset_results, reasoning_ctx.get());
        dataset_results->Statistics(stats.Dump());
        return dataset_results;
    }

    auto dataset_results = this->pack_knn_result(search_result, ctx.alloc);
    dataset_results->Statistics(stats.Dump());

    this->AttachReasoningReport(dataset_results, reasoning_ctx.get());

    return dataset_results;
}

void
IVF::AttachReasoningReport(const DatasetPtr& dataset_results,
                           ReasoningContext* reasoning_ctx) const {
    if (reasoning_ctx == nullptr) {
        return;
    }
    auto count = dataset_results->GetDim();
    if (count > 0 and dataset_results->GetIds() != nullptr) {
        Vector<InnerIdType> result_inner_ids(static_cast<uint64_t>(count), this->allocator_);
        {
            std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
            for (int64_t i = 0; i < count; ++i) {
                result_inner_ids[i] =
                    this->label_table_->GetIdByLabel(dataset_results->GetIds()[i]);
            }
        }
        reasoning_ctx->MarkResult(result_inner_ids);
    }
    reasoning_ctx->DiagnoseExpectedTargets();
    dataset_results->Reasoning(reasoning_ctx->GenerateReport());
}
}  // namespace vsag
