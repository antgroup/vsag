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

#include "bruteforce.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <tuple>

#include "attr/argparse.h"
#include "attr/executor/executor.h"
#include "datacell/attribute_inverted_interface.h"
#include "datacell/flatten_datacell.h"
#include "datacell/flatten_interface.h"
#include "fmt/chrono.h"
#include "impl/heap/standard_heap.h"
#include "impl/reasoning/search_reasoning.h"
#include "index_common_param.h"
#include "index_feature_list.h"
#include "inner_string_params.h"
#include "storage/serialization.h"
#include "typing.h"
#include "utils/slow_task_timer.h"
#include "utils/util_functions.h"
namespace vsag {

namespace {
constexpr const char* WARP_MODE_MARKER = "_warp_mode";
}  // namespace

BruteForce::BruteForce(const BruteForceParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param) {
    inner_codes_ = FlattenInterface::MakeInstance(param->base_codes_param, common_param);
    is_multi_vector_ = (param->base_codes_param->name == MULTI_VECTOR_DATA_CELL);
    auto code_size = this->inner_codes_->code_size_;
    auto increase_count = Options::Instance().block_size_limit() / std::max(code_size, 1U);
    this->resize_increase_count_bit_ = std::max(
        DEFAULT_RESIZE_BIT, static_cast<uint64_t>(log2(static_cast<double>(increase_count))));
    this->use_attribute_filter_ = param->use_attribute_filter;
    this->has_raw_vector_ = !is_multi_vector_;
}

uint64_t
BruteForce::EstimateMemory(uint64_t num_elements) const {
    if (is_multi_vector_) {
        uint64_t avg_vectors_per_doc = 10;
        return num_elements * (avg_vectors_per_doc * this->dim_ * sizeof(float) +
                               sizeof(LabelType) * 2 + sizeof(InnerIdType) + sizeof(uint32_t) * 2);
    }
    return num_elements *
           (this->dim_ * sizeof(float) + sizeof(LabelType) * 2 + sizeof(InnerIdType));
}

std::vector<int64_t>
BruteForce::Build(const vsag::DatasetPtr& data) {
    this->Train(data);
    return this->Add(data);
}

void
BruteForce::train_multi_vector(const DatasetPtr& data) {
    const MultiVector* multi_vectors = data->GetMultiVectors();
    CHECK_ARGUMENT(multi_vectors != nullptr, "data.multi_vectors is nullptr");
    int64_t mv_dim = data->GetMultiVectorDim();
    CHECK_ARGUMENT(
        mv_dim == dim_,
        fmt::format("data.multi_vector_dim({}) must be equal to index.dim({})", mv_dim, dim_));
    int64_t num_elements = data->GetNumElements();
    uint64_t total_vectors = 0;
    for (int64_t i = 0; i < num_elements; ++i) {
        total_vectors += multi_vectors[i].len_;
    }
    Vector<float> buffer(total_vectors * mv_dim, allocator_);
    uint64_t offset = 0;
    for (int64_t i = 0; i < num_elements; ++i) {
        uint64_t num_floats = static_cast<uint64_t>(multi_vectors[i].len_) * mv_dim;
        std::memcpy(buffer.data() + offset, multi_vectors[i].vectors_, num_floats * sizeof(float));
        offset += num_floats;
    }
    this->inner_codes_->Train(buffer.data(), total_vectors);
}

void
BruteForce::Train(const DatasetPtr& data) {
    if (is_multi_vector_) {
        this->train_multi_vector(data);
    } else {
        this->inner_codes_->Train(data->GetFloat32Vectors(), data->GetNumElements());
    }
}

std::optional<InnerIdType>
BruteForce::claim_slot(int64_t label, const AttributeSet* attr) {
    InnerIdType inner_id;
    {
        std::scoped_lock add_lock(this->label_lookup_mutex_, this->add_mutex_);
        if (this->label_table_->CheckLabel(label)) {
            return std::nullopt;
        }
        inner_id = this->total_count_.load();
        ++this->total_count_;
        this->resize(total_count_.load());
        this->label_table_->Insert(inner_id, label);
    }
    std::shared_lock global_lock(this->global_mutex_);
    if (use_attribute_filter_ && attr != nullptr) {
        this->attr_filter_index_->Insert(*attr, inner_id);
    }
    return inner_id;
}

std::vector<int64_t>
BruteForce::add_multi_vector(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;
    const MultiVector* multi_vectors = data->GetMultiVectors();
    CHECK_ARGUMENT(multi_vectors != nullptr, "data.multi_vectors is nullptr");
    int64_t mv_dim = data->GetMultiVectorDim();
    CHECK_ARGUMENT(
        mv_dim == dim_,
        fmt::format("data.multi_vector_dim({}) must be equal to index.dim({})", mv_dim, dim_));

    {
        std::lock_guard lock(this->add_mutex_);
        if (this->total_count_.load() == 0) {
            this->Train(data);
        }
    }

    const int64_t num_elements = data->GetNumElements();
    const int64_t* labels = data->GetIds();
    const AttributeSet* attrs = data->GetAttributeSets();

    auto add_func = [&](const MultiVector* mv,
                        const int64_t label,
                        const AttributeSet* attr) -> std::optional<int64_t> {
        auto slot = this->claim_slot(label, attr);
        if (not slot.has_value()) {
            return label;
        }
        this->inner_codes_->InsertVector(mv, slot.value());
        return std::nullopt;
    };

    std::vector<std::future<std::optional<int64_t>>> futures;
    for (int64_t i = 0; i < num_elements; ++i) {
        const int64_t label = labels[i];
        {
            std::lock_guard label_lock(this->label_lookup_mutex_);
            if (this->label_table_->CheckLabel(label)) {
                failed_ids.emplace_back(label);
                continue;
            }
        }
        if (this->thread_pool_ != nullptr) {
            auto future = this->thread_pool_->GeneralEnqueue(
                add_func, &multi_vectors[i], label, attrs == nullptr ? nullptr : attrs + i);
            futures.emplace_back(std::move(future));
        } else {
            if (auto add_res =
                    add_func(&multi_vectors[i], label, attrs == nullptr ? nullptr : attrs + i);
                add_res.has_value()) {
                failed_ids.emplace_back(add_res.value());
            }
        }
    }
    if (this->thread_pool_ != nullptr) {
        for (auto& future : futures) {
            if (auto reply = future.get(); reply.has_value()) {
                failed_ids.emplace_back(reply.value());
            }
        }
    }
    return failed_ids;
}

std::vector<int64_t>
BruteForce::Add(const DatasetPtr& data) {
    if (is_multi_vector_) {
        return this->add_multi_vector(data);
    }

    std::vector<int64_t> failed_ids;
    auto base_dim = data->GetDim();
    CHECK_ARGUMENT(base_dim == dim_,
                   fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));
    CHECK_ARGUMENT(data->GetFloat32Vectors() != nullptr, "base.float_vector is nullptr");

    {
        std::lock_guard lock(this->add_mutex_);
        if (this->total_count_.load() == 0) {
            this->Train(data);
        }
    }

    auto add_func = [&](const float* data,
                        const int64_t label,
                        const AttributeSet* attr,
                        const char* extra_info) -> std::optional<int64_t> {
        auto slot = this->claim_slot(label, attr);
        if (not slot.has_value()) {
            return label;
        }
        if (this->extra_infos_ != nullptr) {
            std::lock_guard lock(this->add_mutex_);
            this->extra_infos_->InsertExtraInfo(extra_info, slot.value());
        }
        this->add_one(data, slot.value());
        return std::nullopt;
    };

    std::vector<std::future<std::optional<int64_t>>> futures;
    const auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* vectors = data->GetFloat32Vectors();
    const auto* attrs = data->GetAttributeSets();
    const auto* extra_info = data->GetExtraInfos();
    const auto extra_info_size = data->GetExtraInfoSize();
    if (this->extra_infos_ != nullptr) {
        CHECK_ARGUMENT(extra_info != nullptr, "extra_infos is nullptr");
        CHECK_ARGUMENT(extra_info_size == static_cast<int64_t>(this->extra_info_size_),
                       "extra_infos size mismatch");
    }
    for (int64_t j = 0; j < total; ++j) {
        const auto label = labels[j];
        {
            std::lock_guard label_lock(this->label_lookup_mutex_);
            if (this->label_table_->CheckLabel(label)) {
                failed_ids.emplace_back(label);
                continue;
            }
        }
        const auto* ei_ptr = extra_info == nullptr ? nullptr : extra_info + j * extra_info_size;
        if (this->thread_pool_ != nullptr) {
            auto future = this->thread_pool_->GeneralEnqueue(add_func,
                                                             vectors + j * dim_,
                                                             label,
                                                             attrs == nullptr ? nullptr : attrs + j,
                                                             ei_ptr);
            futures.emplace_back(std::move(future));
        } else {
            if (auto add_res = add_func(
                    vectors + j * dim_, label, attrs == nullptr ? nullptr : attrs + j, ei_ptr);
                add_res.has_value()) {
                failed_ids.emplace_back(add_res.value());
            }
        }
    }
    if (this->thread_pool_ != nullptr) {
        for (auto& future : futures) {
            if (auto reply = future.get(); reply.has_value()) {
                failed_ids.emplace_back(reply.value());
            }
        }
    }
    return failed_ids;
}

uint32_t
BruteForce::Remove(const std::vector<int64_t>& ids, RemoveMode mode) {
    if (is_multi_vector_ && mode != RemoveMode::MARK_REMOVE) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "multi-vector mode only supports MARK_REMOVE");
    }
    if (not is_multi_vector_) {
        CHECK_ARGUMENT(not use_attribute_filter_,
                       "remove is not supported when use_attribute_filter is true");
    }

    uint32_t delete_count = 0;
    if (mode == RemoveMode::MARK_REMOVE) {
        std::scoped_lock label_lock(this->label_lookup_mutex_);
        delete_count = this->label_table_->MarkRemove(ids);
        delete_count_.fetch_add(delete_count, std::memory_order_relaxed);
        return delete_count;
    }

    std::scoped_lock add_label_lock(this->add_mutex_, this->label_lookup_mutex_);
    std::unique_lock global_lock(this->global_mutex_);
    for (auto label : ids) {
        InnerIdType inner_id;
        bool found = false;
        std::tie(found, inner_id) = this->label_table_->TryGetIdByLabel(label, true);
        auto current_total = this->total_count_.load();
        if (not found or inner_id >= current_total) {
            continue;
        }

        const auto last_inner_id = static_cast<InnerIdType>(current_total - 1);
        bool was_mark_removed = this->label_table_->IsRemoved(inner_id);
        this->label_table_->ForceRemove(label, inner_id);

        if (inner_id < last_inner_id) {
            Vector<float> data(dim_, allocator_);
            GetVectorByInnerId(last_inner_id, data.data());
            this->inner_codes_->InsertVector(data.data(), inner_id);
            this->label_table_->Move(last_inner_id, inner_id);
            if (this->extra_infos_ != nullptr) {
                this->extra_infos_->Move(last_inner_id, inner_id);
            }
        }

        if (was_mark_removed and this->delete_count_.load(std::memory_order_relaxed) > 0) {
            this->delete_count_.fetch_sub(1, std::memory_order_relaxed);
        }
        this->total_count_.fetch_sub(1);
        delete_count++;
    }
    if (delete_count != 0) {
        this->shrink_to_fit();
    }
    return delete_count;
}

DatasetPtr
BruteForce::KnnSearch(const DatasetPtr& query,
                      int64_t k,
                      const std::string& parameters,
                      const FilterPtr& filter) const {
    SearchRequest req;
    req.query_ = query;
    req.topk_ = k;
    req.params_str_ = parameters;
    if (filter != nullptr) {
        req.filter_ = filter;
    }
    return this->SearchWithRequest(req);
}

DatasetPtr
BruteForce::SearchWithRequest(const SearchRequest& request) const {
    std::shared_lock read_lock(this->global_mutex_);

    auto computer = this->make_search_computer(request.query_);

    bool is_range = (request.mode_ == SearchMode::RANGE_SEARCH);
    if (is_range) {
        if (not is_multi_vector_) {
            this->validate_range_args(request.query_, request.radius_, request.limited_size_);
        }
    } else {
        if (not is_multi_vector_) {
            this->validate_knn_args(request.query_, request.topk_);
        }
    }

    auto heap_size = is_range ? request.limited_size_ : request.topk_;
    if (heap_size < 0) {
        heap_size = static_cast<int64_t>(this->total_count_.load());
    }
    auto radius = is_range ? request.radius_ : std::numeric_limits<float>::max();

    if (total_count_.load() == 0) {
        return make_empty_result();
    }

    DistHeapPtr heap = nullptr;
    ExecutorPtr executor = nullptr;
    Filter* attr_filter = nullptr;

    auto brute_force_params = BruteForceSearchParameters::FromJson(request.params_str_);
    FilterPtr ft =
        this->create_search_filter(request.filter_, brute_force_params.use_extra_info_filter);

    if (request.enable_attribute_filter_) {
        auto& schema = this->attr_filter_index_->field_type_map_;
        auto expr = AstParse(request.attribute_filter_str_, &schema);
        executor = Executor::MakeInstance(this->allocator_, expr, this->attr_filter_index_);
        executor->Init();
        executor->Clear();
        attr_filter = executor->Run();
    }

    // Setup reasoning context if expected labels are provided.
    std::shared_ptr<ReasoningContext> reasoning_ctx;
    if (not request.expected_labels_.empty()) {
        reasoning_ctx = std::make_shared<ReasoningContext>(this->allocator_);
        reasoning_ctx->SetSearchParams(
            request.topk_, is_multi_vector_ ? "WARP" : "BruteForce", false, ft != nullptr);

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

        // Compute true distances for expected targets.
        // BruteForce uses inner_codes_ directly; for multi-vector, this is the
        // same tokenizer used during search, so the distance is already precise.
        for (const auto& pair : label_to_inner_id) {
            float dist = 0.0F;
            const auto inner_id = pair.second;
            this->inner_codes_->Query(&dist, computer, &inner_id, 1);
            reasoning_ctx->SetTrueDistance(inner_id, dist);
        }
    }

    std::atomic<uint32_t> dist_cmp{0};

    auto parallel_count = brute_force_params.parallel_search_thread_count;
    std::vector<DistHeapPtr> heaps(parallel_count);
    for (auto& cur_heap : heaps) {
        cur_heap = DistanceHeap::MakeInstanceBySize<true, true>(this->allocator_, heap_size);
    }
    auto* reasoning = reasoning_ctx ? reasoning_ctx.get() : nullptr;

    auto search_func = [&](InnerIdType start, InnerIdType end, const DistHeapPtr& cur_heap) {
        uint32_t dist_cmp_local = 0;
        for (InnerIdType i = start; i < end; ++i) {
            float dist = 0.0F;
            if (attr_filter != nullptr and not attr_filter->CheckValid(i)) {
                if (reasoning != nullptr) {
                    reasoning->RecordFilterReject(i);
                }
                continue;
            }
            if (ft == nullptr or ft->CheckValid(i)) {
                inner_codes_->Query(&dist, computer, &i, 1);
                ++dist_cmp_local;
                if (reasoning != nullptr) {
                    reasoning->RecordVisit(i, dist, 0);
                }
                if (is_range and dist > radius) {
                    continue;
                }
                cur_heap->Push(dist, i);
            } else {
                if (reasoning != nullptr) {
                    reasoning->RecordFilterReject(i);
                }
            }
        }
        dist_cmp.fetch_add(dist_cmp_local, std::memory_order_relaxed);
    };

    auto count = total_count_.load();
    // Reasoning context is not thread-safe; force single-threaded search when
    // reasoning is enabled so that RecordVisit / RecordFilterReject calls are
    // serialized.  Reasoning is a diagnostic tool and does not need maximum
    // throughput.
    if (parallel_count == 1 || this->thread_pool_ == nullptr || reasoning != nullptr) {
        search_func(0, count, heaps[0]);
        heap = heaps[0];
    } else {
        std::vector<std::future<void>> futures;
        auto chunk_size = (count + parallel_count - 1) / parallel_count;
        for (auto i = 0; i < parallel_count; ++i) {
            auto start = i * chunk_size;
            auto end = std::min(start + chunk_size, count);
            auto future = this->thread_pool_->GeneralEnqueue(search_func, start, end, heaps[i]);
            futures.emplace_back(std::move(future));
        }
        for (auto& future : futures) {
            future.get();
        }
        heap = heaps[0];
        for (auto i = 1; i < parallel_count; ++i) {
            heap->Merge(*heaps[i]);
        }
    }

    // Collect result inner IDs before pack_knn_result_with_extra_info consumes the heap,
    // so we can call MarkResult for reasoning analysis.
    Vector<InnerIdType> result_inner_ids(this->allocator_);
    if (reasoning_ctx && heap != nullptr && !heap->Empty()) {
        auto heap_size = static_cast<uint64_t>(heap->Size());
        result_inner_ids.reserve(heap_size);
        const auto* data = heap->GetData();
        for (uint64_t i = 0; i < heap_size; ++i) {
            result_inner_ids.push_back(data[i].second);
        }
    }

    auto result = this->pack_knn_result_with_extra_info(heap);

    // Generate reasoning report if reasoning context was created.
    if (reasoning_ctx) {
        if (not result_inner_ids.empty()) {
            reasoning_ctx->MarkResult(result_inner_ids);
        }
        reasoning_ctx->SetTermination(ReasoningContext::kTerminationLowerBoundReached);
        reasoning_ctx->DiagnoseExpectedTargets();
        result->Reasoning(reasoning_ctx->GenerateReport());
    }

    JsonType stats;
    stats["dist_cmp"].SetInt(dist_cmp.load(std::memory_order_relaxed));
    result->Statistics(stats.Dump());

    return result;
}

DatasetPtr
BruteForce::RangeSearch(const vsag::DatasetPtr& query,
                        float radius,
                        const std::string& parameters,
                        const vsag::FilterPtr& filter,
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

ComputerInterfacePtr
BruteForce::make_search_computer(const DatasetPtr& query) const {
    if (is_multi_vector_) {
        const MultiVector* query_multi_vectors = query->GetMultiVectors();
        CHECK_ARGUMENT(query_multi_vectors != nullptr, "query.multi_vectors is nullptr");
        return this->inner_codes_->FactoryComputer(&query_multi_vectors[0]);
    }
    return this->inner_codes_->FactoryComputer(query->GetFloat32Vectors());
}

float
BruteForce::CalcDistanceById(const float* vector,
                             int64_t id,
                             bool calculate_precise_distance) const {
    auto computer = this->inner_codes_->FactoryComputer(vector);
    float result = 0.0F;
    InnerIdType inner_id = this->label_table_->GetIdByLabel(id);
    this->inner_codes_->Query(&result, computer, &inner_id, 1);
    return result;
}

void
BruteForce::Serialize(StreamWriter& writer) const {
    if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        this->attr_filter_index_->Serialize(writer);
    }
    if (this->extra_info_size_ > 0 and this->extra_infos_ != nullptr) {
        this->extra_infos_->Serialize(writer);
    }
    this->inner_codes_->Serialize(writer);
    this->label_table_->Serialize(writer);

    // serialize footer (introduced since v0.15)
    JsonType basic_info;
    basic_info["dim"].SetInt(dim_);
    basic_info["total_count"].SetUint64(total_count_.load());
    basic_info["is_multi_vector"].SetBool(is_multi_vector_);
    basic_info["extra_info_size"].SetUint64(this->extra_info_size_);
    basic_info["has_extra_info"].SetBool(this->extra_info_size_ > 0 and
                                         this->extra_infos_ != nullptr);
    basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    write_index_footer(writer, basic_info);
}

void
BruteForce::Deserialize(StreamReader& reader) {
    JsonType basic_info;
    bool has_footer = read_index_footer(reader, basic_info);

    BufferStreamReader buffer_reader(
        &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);

    if (not has_footer) {
        logger::debug("parse with v0.13 version format");

        StreamReader::ReadObj(buffer_reader, dim_);
        uint64_t count = 0;
        StreamReader::ReadObj(buffer_reader, count);
        total_count_.store(count);
        this->inner_codes_->Deserialize(buffer_reader);
        this->label_table_->Deserialize(buffer_reader);
    } else {
        logger::debug("parse with new version format");

        if (basic_info.Contains(INDEX_PARAM)) {
            std::string index_param_string = basic_info[INDEX_PARAM].GetString();
            auto index_param = std::make_shared<BruteForceParameter>();
            index_param->FromString(index_param_string);
            if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
                auto message =
                    fmt::format("BruteForce index parameter not match, current: {}, new: {}",
                                this->create_param_ptr_->ToString(),
                                index_param->ToString());
                logger::error(message);
                throw VsagException(ErrorType::INVALID_ARGUMENT, message);
            }
        }
        dim_ = basic_info["dim"].GetInt();
        total_count_.store(basic_info["total_count"].GetUint64());
        if (basic_info.Contains("extra_info_size")) {
            CHECK_ARGUMENT(basic_info["extra_info_size"].GetUint64() == this->extra_info_size_,
                           "BruteForce index extra_info_size not match");
        }

        if (basic_info.Contains("is_multi_vector")) {
            is_multi_vector_ = basic_info["is_multi_vector"].GetBool();
            this->has_raw_vector_ = !is_multi_vector_;
        }

        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Deserialize(buffer_reader);
        }
        const bool has_extra_info =
            basic_info.Contains("has_extra_info") and basic_info["has_extra_info"].GetBool();
        if (has_extra_info) {
            CHECK_ARGUMENT(this->extra_info_size_ > 0,
                           "BruteForce serialized extra_info is not supported by current index");
            CHECK_ARGUMENT(this->extra_infos_ != nullptr,
                           "BruteForce serialized extra_info is not supported by current index");
            this->extra_infos_->Deserialize(buffer_reader);
        }

        this->inner_codes_->Deserialize(buffer_reader);
        this->label_table_->Deserialize(buffer_reader);
    }
    delete_count_.store(label_table_->GetAllDeletedIds().size(), std::memory_order_relaxed);
    this->cal_memory_usage();
}

void
BruteForce::InitFeatures() {
    auto name = this->inner_codes_->GetQuantizerName();
    if (is_multi_vector_) {
        if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16) {
            this->index_feature_list_->SetFeature(IndexFeature::NEED_TRAIN);
        } else {
            this->index_feature_list_->SetFeatures(
                {IndexFeature::SUPPORT_ADD_FROM_EMPTY,
                 IndexFeature::SUPPORT_RANGE_SEARCH,
                 IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID,
                 IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER});
        }
    } else {
        if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16 and
            name != QUANTIZATION_TYPE_VALUE_FP16) {
            this->index_feature_list_->SetFeature(IndexFeature::NEED_TRAIN);
        } else {
            this->index_feature_list_->SetFeatures(
                {IndexFeature::SUPPORT_ADD_FROM_EMPTY,
                 IndexFeature::SUPPORT_RANGE_SEARCH,
                 IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID,
                 IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER});
        }
        if (name == QUANTIZATION_TYPE_VALUE_FP32 and
            (metric_ != MetricType::METRIC_TYPE_COSINE || this->inner_codes_->HoldMolds())) {
            this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS);
        }
    }

    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
        IndexFeature::SUPPORT_DELETE_BY_ID,
    });

    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
    });

    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_SEARCH_CONCURRENT,
        IndexFeature::SUPPORT_ADD_CONCURRENT,
        IndexFeature::SUPPORT_DELETE_CONCURRENT,
    });

    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
        IndexFeature::SUPPORT_SERIALIZE_WRITE_FUNC,
    });

    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_ESTIMATE_MEMORY,
        IndexFeature::SUPPORT_GET_MEMORY_USAGE,
        IndexFeature::SUPPORT_CHECK_ID_EXIST,
        IndexFeature::SUPPORT_CLONE,
    });

    if (this->extra_infos_ != nullptr) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_EXTRA_INFO_BY_ID);
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_KNN_SEARCH_WITH_EX_FILTER);
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_UPDATE_EXTRA_INFO_CONCURRENT);
    }
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_UPDATE_VECTOR_CONCURRENT);
}

static const std::string BRUTE_FORCE_PARAMS_TEMPLATE =
    R"(
    {
        "{TYPE_KEY}": "{INDEX_BRUTE_FORCE}",
        "{USE_REORDER_KEY}": false,
        "{BASE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY}": 0.05,
                "{PCA_DIM_KEY}": 0,
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY}": 32,
                "{TQ_CHAIN_KEY}": "",
                "nbits": 8,
                "{PRODUCT_QUANTIZATION_DIM_KEY}": 1,
                "{HOLD_MOLDS}": false
            }
        },
        "{PRECISE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY}": 0.05,
                "{PCA_DIM_KEY}": 0,
                "{PRODUCT_QUANTIZATION_DIM_KEY}": 1,
                "{HOLD_MOLDS}": false
            }
        },
        "{BUILD_THREAD_COUNT_KEY}": 1,
        "{EXTRA_INFO_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            }
        },
        "{USE_ATTRIBUTE_FILTER_KEY}": false,
        "{ATTR_PARAMS_KEY}": {
            "{ATTR_HAS_BUCKETS_KEY}": true
        }
    })";

static const std::string WARP_PARAMS_TEMPLATE =
    R"(
    {
        "{TYPE_KEY}": "{INDEX_BRUTE_FORCE}",
        "{USE_REORDER_KEY}": false,
        "{BASE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "multi_vector",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}"
            }
        },
        "{BUILD_THREAD_COUNT_KEY}": 1,
        "{USE_ATTRIBUTE_FILTER_KEY}": false,
        "{ATTR_PARAMS_KEY}": {
            "{ATTR_HAS_BUCKETS_KEY}": true
        }
    })";

ParamPtr
BruteForce::CheckAndMappingExternalParam(const JsonType& external_param,
                                         const IndexCommonParam& common_param) {
    // Detect if this is a WARP (multi-vector) index request
    bool is_warp = external_param.Contains(WARP_MODE_MARKER);

    if (is_warp) {
        // Remove the marker key before mapping (it would cause "invalid config param" error)
        JsonType warp_external_param = external_param;
        warp_external_param.Erase(WARP_MODE_MARKER);

        const ConstParamMap external_mapping = {
            {
                BRUTE_FORCE_BASE_QUANTIZATION_TYPE,
                {
                    BASE_CODES_KEY,
                    QUANTIZATION_PARAMS_KEY,
                    TYPE_KEY,
                },
            },
            {
                BRUTE_FORCE_BASE_IO_TYPE,
                {
                    BASE_CODES_KEY,
                    IO_PARAMS_KEY,
                    TYPE_KEY,
                },
            },
            {
                BRUTE_FORCE_BASE_FILE_PATH,
                {
                    BASE_CODES_KEY,
                    IO_PARAMS_KEY,
                    IO_FILE_PATH_KEY,
                },
            },
        };

        if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("WARP not support {} datatype", DATATYPE_INT8));
        }

        std::string str = format_map(WARP_PARAMS_TEMPLATE, DEFAULT_MAP);
        auto inner_json = JsonType::Parse(str);
        mapping_external_param_to_inner(warp_external_param, external_mapping, inner_json);

        auto brute_force_parameter = std::make_shared<BruteForceParameter>();
        brute_force_parameter->FromJson(inner_json);
        return brute_force_parameter;
    }

    const ConstParamMap external_mapping = {
        {
            BRUTE_FORCE_BASE_QUANTIZATION_TYPE,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            BRUTE_FORCE_BASE_IO_TYPE,
            {
                BASE_CODES_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            BRUTE_FORCE_BASE_PQ_DIM,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                PRODUCT_QUANTIZATION_DIM_KEY,
            },
        },
        {
            BRUTE_FORCE_BASE_FILE_PATH,
            {
                BASE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            BRUTE_FORCE_PRECISE_QUANTIZATION_TYPE,
            {
                PRECISE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            BRUTE_FORCE_PRECISE_IO_TYPE,
            {
                PRECISE_CODES_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            BRUTE_FORCE_PRECISE_FILE_PATH,
            {
                PRECISE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            BRUTE_FORCE_THREAD_COUNT,
            {
                BUILD_THREAD_COUNT_KEY,
            },
        },
        {
            STORE_RAW_VECTOR,
            {
                QUANTIZATION_PARAMS_KEY,
                HOLD_MOLDS,
            },
        },
        {
            USE_ATTRIBUTE_FILTER,
            {
                USE_ATTRIBUTE_FILTER_KEY,
            },
        },
        {
            BRUTE_FORCE_USE_RESIDUAL,
            {
                USE_REORDER_KEY,
            },
        },
    };

    if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("BruteForce not support {} datatype", DATATYPE_INT8));
    }

    std::string str = format_map(BRUTE_FORCE_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto inner_json = JsonType::Parse(str);
    mapping_external_param_to_inner(external_param, external_mapping, inner_json);

    auto brute_force_parameter = std::make_shared<BruteForceParameter>();
    brute_force_parameter->FromJson(inner_json);

    return brute_force_parameter;
}

void
BruteForce::resize(uint64_t new_size) {
    uint64_t new_size_power_2 =
        next_multiple_of_power_of_two(new_size, this->resize_increase_count_bit_);
    auto cur_size = this->max_capacity_.load();
    if (cur_size >= new_size_power_2) {
        return;
    }
    std::lock_guard lock(this->global_mutex_);
    cur_size = this->max_capacity_.load();
    if (cur_size < new_size_power_2) {
        this->inner_codes_->Resize(new_size_power_2);
        if (this->extra_infos_ != nullptr) {
            this->extra_infos_->Resize(new_size_power_2);
        }
        this->max_capacity_.store(new_size_power_2);
        this->cal_memory_usage();
    }
}

void
BruteForce::shrink_to_fit() {
    auto total_count = this->total_count_.load();
    auto current_capacity = this->max_capacity_.load();
    if (total_count != 0 and total_count > current_capacity / 2) {
        return;
    }

    this->inner_codes_->ShrinkToFit(total_count);
    this->label_table_->ShrinkToFit(total_count);
    if (this->extra_infos_ != nullptr) {
        this->extra_infos_->ShrinkToFit(total_count);
    }
    this->max_capacity_.store(total_count);
    this->cal_memory_usage();
}

void
BruteForce::add_one(const float* data, InnerIdType inner_id) {
    this->inner_codes_->InsertVector(data, inner_id);
}

void
BruteForce::GetVectorByInnerId(InnerIdType inner_id, float* data) const {
    if (is_multi_vector_) {
        bool need_release = false;
        const uint8_t* codes = inner_codes_->GetCodesById(inner_id, need_release);
        if (codes == nullptr) {
            std::memset(data, 0, dim_ * sizeof(float));
            return;
        }
        uint32_t token_count = 0;
        std::memcpy(&token_count, codes, sizeof(uint32_t));
        if (token_count > 0) {
            std::memcpy(data, codes + sizeof(uint32_t), dim_ * sizeof(float));
        } else {
            std::memset(data, 0, dim_ * sizeof(float));
        }
        if (need_release) {
            inner_codes_->Release(codes);
        }
    } else {
        Vector<uint8_t> codes(inner_codes_->code_size_, allocator_);
        inner_codes_->GetCodesById(inner_id, codes.data());
        inner_codes_->Decode(codes.data(), data);
    }
}

bool
BruteForce::UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update) {
    (void)force_update;
    CHECK_ARGUMENT(new_base != nullptr, "new_base is nullptr");
    auto base_dim = new_base->GetDim();
    CHECK_ARGUMENT(base_dim == dim_,
                   fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));
    CHECK_ARGUMENT(new_base->GetFloat32Vectors() != nullptr, "base.float_vector is nullptr");

    std::shared_lock add_lock(this->add_mutex_, std::defer_lock);
    std::shared_lock label_lock(this->label_lookup_mutex_, std::defer_lock);
    std::lock(add_lock, label_lock);
    std::unique_lock global_lock(this->global_mutex_);
    InnerIdType inner_id = this->label_table_->GetIdByLabel(id);
    return this->inner_codes_->UpdateVector(new_base->GetFloat32Vectors(), inner_id);
}

void
BruteForce::UpdateAttribute(int64_t id, const AttributeSet& new_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, inner_id, 0);
}

void
BruteForce::UpdateAttribute(int64_t id,
                            const AttributeSet& new_attrs,
                            const AttributeSet& origin_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, inner_id, 0, origin_attrs);
}

void
BruteForce::GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const {
    this->attr_filter_index_->GetAttribute(0, inner_id, attr);
}

void
BruteForce::cal_memory_usage() {
    auto memory_usage = this->inner_codes_->GetMemoryUsage();
    memory_usage += sizeof(BruteForce);
    memory_usage += this->label_table_->GetMemoryUsage();
    if (this->extra_infos_ != nullptr) {
        memory_usage += this->extra_infos_->GetMemoryUsage();
    }
    std::unique_lock lock(this->memory_usage_mutex_);
    this->current_memory_usage_.store(memory_usage);
}

int64_t
BruteForce::GetMemoryUsage() const {
    int64_t memory = 0;
    {
        std::shared_lock lock(this->memory_usage_mutex_);
        memory = this->current_memory_usage_.load();
    }
    if (this->attr_filter_index_ != nullptr) {
        memory += this->attr_filter_index_->GetMemoryUsage();
    }
    return memory;
}

}  // namespace vsag
