
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

#include "brute_force.h"

#include <atomic>
#include <mutex>
#include <optional>

#include "attr/argparse.h"
#include "attr/executor/executor.h"
#include "datacell/attribute_inverted_interface.h"
#include "datacell/flatten_datacell.h"
#include "datacell/flatten_interface.h"
#include "fmt/chrono.h"
#include "impl/heap/standard_heap.h"
#include "index_common_param.h"
#include "index_feature_list.h"
#include "inner_string_params.h"
#include "storage/serialization.h"
#include "typing.h"
#include "utils/slow_task_timer.h"
#include "utils/util_functions.h"
namespace vsag {

BruteForce::BruteForce(const BruteForceParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param) {
    inner_codes_ = FlattenInterface::MakeInstance(param->base_codes_param, common_param);
    auto code_size = this->inner_codes_->code_size_;
    auto increase_count = Options::Instance().block_size_limit() / code_size;
    this->resize_increase_count_bit_ = std::max(
        DEFAULT_RESIZE_BIT, static_cast<uint64_t>(log2(static_cast<double>(increase_count))));
    this->use_attribute_filter_ = param->use_attribute_filter;
    this->has_raw_vector_ = true;
}

uint64_t
BruteForce::EstimateMemory(uint64_t num_elements) const {
    return num_elements *
           (this->dim_ * sizeof(float) + sizeof(LabelType) * 2 + sizeof(InnerIdType));
}

std::vector<int64_t>
BruteForce::Build(const vsag::DatasetPtr& data) {
    this->Train(data);
    return this->Add(data);
}

void
BruteForce::Train(const DatasetPtr& data) {
    this->inner_codes_->Train(data->GetFloat32Vectors(), data->GetNumElements());
}

std::vector<int64_t>
BruteForce::Add(const DatasetPtr& data, AddMode mode) {
    std::vector<int64_t> failed_ids;
    auto base_dim = data->GetDim();
    CHECK_ARGUMENT(base_dim == dim_,
                   fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));
    CHECK_ARGUMENT(data->GetFloat32Vectors() != nullptr, "base.float_vector is nullptr");

    {
        std::lock_guard lock(this->add_mutex_);
        if (this->total_count_ == 0) {
            this->Train(data);
        }
    }

    auto add_func = [&](const float* data,
                        const int64_t label,
                        const AttributeSet* attr,
                        const char* extra_info) -> std::optional<int64_t> {
        InnerIdType inner_id;
        {
            std::scoped_lock add_lock(this->label_lookup_mutex_, this->add_mutex_);
            if (this->label_table_->CheckLabel(label)) {
                return label;
            }
            inner_id = this->total_count_;
            this->total_count_++;
            this->resize(total_count_);
            this->label_table_->Insert(inner_id, label);
        }
        std::shared_lock global_lock(this->global_mutex_);
        if (use_attribute_filter_ && attr != nullptr) {
            this->attr_filter_index_->Insert(*attr, inner_id);
        }

        this->add_one(data, inner_id);
        return std::nullopt;
    };

    std::vector<std::future<std::optional<int64_t>>> futures;
    const auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* vectors = data->GetFloat32Vectors();
    const auto* attrs = data->GetAttributeSets();
    const auto* extra_info = data->GetExtraInfos();
    const auto extra_info_size = data->GetExtraInfoSize();
    for (int64_t j = 0; j < total; ++j) {
        const auto label = labels[j];
        {
            std::lock_guard label_lock(this->label_lookup_mutex_);
            if (this->label_table_->CheckLabel(label)) {
                failed_ids.emplace_back(label);
                continue;
            }
        }
        if (this->thread_pool_ != nullptr) {
            auto future = this->thread_pool_->GeneralEnqueue(add_func,
                                                             vectors + j * dim_,
                                                             label,
                                                             attrs == nullptr ? nullptr : attrs + j,
                                                             extra_info + j * extra_info_size);
            futures.emplace_back(std::move(future));
        } else {
            if (auto add_res = add_func(vectors + j * dim_,
                                        label,
                                        attrs == nullptr ? nullptr : attrs + j,
                                        extra_info + j * extra_info_size);
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
    CHECK_ARGUMENT(not use_attribute_filter_,
                   "remove is not supported when use_attribute_filter is true");

    uint32_t delete_count = 0;
    if (mode == RemoveMode::MARK_REMOVE) {
        std::scoped_lock label_lock(this->label_lookup_mutex_);
        delete_count = this->label_table_->MarkRemove(ids);
        delete_count_ += delete_count;
        return delete_count;
    }

    std::scoped_lock lock(this->add_mutex_, this->label_lookup_mutex_);
    for (auto label : ids) {
        const auto last_inner_id = static_cast<InnerIdType>(this->total_count_ - 1);
        const auto inner_id = this->label_table_->GetIdByLabel(label);

        CHECK_ARGUMENT(inner_id <= last_inner_id, "the element to be remove is invalid");

        const auto last_label = this->label_table_->GetLabelById(last_inner_id);
        this->label_table_->MarkRemove(label);
        --this->label_table_->total_count_;

        if (inner_id < last_inner_id) {
            Vector<float> data(dim_, allocator_);
            GetVectorByInnerId(last_inner_id, data.data());

            this->label_table_->MarkRemove(last_label);
            --this->label_table_->total_count_;

            this->inner_codes_->InsertVector(data.data(), inner_id);
            this->label_table_->Insert(inner_id, last_label);
        }

        this->total_count_--;
    }
    return 1;
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

    auto computer = this->inner_codes_->FactoryComputer(request.query_->GetFloat32Vectors());
    DistHeapPtr heap = nullptr;
    ExecutorPtr executor = nullptr;
    Filter* attr_filter = nullptr;

    auto combined_filter = std::make_shared<CombinedFilter>();
    combined_filter->AppendFilter(this->label_table_->GetDeletedIdsFilter());
    if (request.filter_ != nullptr) {
        combined_filter->AppendFilter(
            std::make_shared<InnerIdWrapperFilter>(request.filter_, *this->label_table_));
    }
    FilterPtr ft = nullptr;
    if (not combined_filter->IsEmpty()) {
        ft = combined_filter;
    }

    if (request.enable_attribute_filter_) {
        auto& schema = this->attr_filter_index_->field_type_map_;
        auto expr = AstParse(request.attribute_filter_str_, &schema);
        executor = Executor::MakeInstance(this->allocator_, expr, this->attr_filter_index_);
        executor->Init();
        executor->Clear();
        attr_filter = executor->Run();
    }

    std::atomic<uint32_t> dist_cmp{0};

    auto brute_force_params = BruteForceSearchParameters::FromJson(request.params_str_);
    auto parallel_count = brute_force_params.parallel_search_thread_count;
    std::vector<DistHeapPtr> heaps(parallel_count);
    for (auto& cur_heap : heaps) {
        cur_heap = DistanceHeap::MakeInstanceBySize<true, true>(this->allocator_, request.topk_);
    }
    auto search_func = [&](InnerIdType start, InnerIdType end, const DistHeapPtr& cur_heap) {
        float cur_min_dist = std::numeric_limits<float>::max();
        uint32_t dist_cmp_local = 0;
        for (InnerIdType i = start; i < end; ++i) {
            float dist = 0.0F;
            if (attr_filter != nullptr and not attr_filter->CheckValid(i)) {
                continue;
            }
            if (ft == nullptr or ft->CheckValid(i)) {
                inner_codes_->Query(&dist, computer, &i, 1);
                ++dist_cmp_local;
                cur_heap->Push(dist, i);
            }
        }

        dist_cmp.fetch_add(dist_cmp_local, std::memory_order_relaxed);
    };

    if (parallel_count == 1 || this->thread_pool_ == nullptr) {
        search_func(0, total_count_, heaps[0]);
        heap = heaps[0];
    } else {
        std::vector<std::future<void>> futures;
        auto chunk_size = (total_count_ + parallel_count - 1) / parallel_count;
        for (auto i = 0; i < parallel_count; ++i) {
            auto start = i * chunk_size;
            auto end = std::min(start + chunk_size, total_count_);
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

    auto [dataset_results, dists, ids] =
        create_fast_dataset(static_cast<int64_t>(heap->Size()), allocator_);
    for (auto j = static_cast<int64_t>(heap->Size() - 1); j >= 0; --j) {
        dists[j] = heap->Top().first;
        ids[j] = this->label_table_->GetLabelById(heap->Top().second);
        heap->Pop();
    }

    JsonType stats;
    stats["dist_cmp"].SetInt(dist_cmp.load(std::memory_order_relaxed));
    dataset_results->Statistics(stats.Dump());

    return std::move(dataset_results);
}

DatasetPtr
BruteForce::RangeSearch(const vsag::DatasetPtr& query,
                        float radius,
                        const std::string& parameters,
                        const vsag::FilterPtr& filter,
                        int64_t limited_size) const {
    std::shared_lock read_lock(this->global_mutex_);
    auto computer = this->inner_codes_->FactoryComputer(query->GetFloat32Vectors());
    if (limited_size < 0) {
        limited_size = std::numeric_limits<int64_t>::max();
    }
    auto heap = std::make_shared<StandardHeap<true, true>>(this->allocator_, limited_size);
    for (InnerIdType i = 0; i < total_count_; ++i) {
        float dist;
        if (filter == nullptr or filter->CheckValid(this->label_table_->GetLabelById(i))) {
            inner_codes_->Query(&dist, computer, &i, 1);
            if (dist > radius) {
                continue;
            }
            heap->Push(dist, i);
        }
    }

    auto [dataset_results, dists, ids] =
        create_fast_dataset(static_cast<int64_t>(heap->Size()), allocator_);
    for (auto j = static_cast<int64_t>(heap->Size() - 1); j >= 0; --j) {
        dists[j] = heap->Top().first;
        ids[j] = this->label_table_->GetLabelById(heap->Top().second);
        heap->Pop();
    }
    return std::move(dataset_results);
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
    // FIXME(wxyu): only for testing, remove before merge into the main branch
    // if (not Options::Instance().new_version()) {
    //     StreamWriter::WriteObj(writer, dim_);
    //     StreamWriter::WriteObj(writer, total_count_);
    //     this->inner_codes_->Serialize(writer);
    //     this->label_table_->Serialize(writer);
    //     return;
    // }
    if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        this->attr_filter_index_->Serialize(writer);
    }
    this->inner_codes_->Serialize(writer);
    this->label_table_->Serialize(writer);

    // serialize footer (introduced since v0.15)
    auto metadata = std::make_shared<Metadata>();
    JsonType basic_info;
    basic_info["dim"].SetInt(dim_);
    basic_info["total_count"].SetInt(total_count_);
    basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    metadata->Set("basic_info", basic_info);
    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

void
BruteForce::Deserialize(StreamReader& reader) {
    // try to deserialize footer (only in new version)
    auto footer = Footer::Parse(reader);

    BufferStreamReader buffer_reader(
        &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);

    if (footer == nullptr) {  // old format, DON'T EDIT, remove in the future
        logger::debug("parse with v0.13 version format");

        StreamReader::ReadObj(buffer_reader, dim_);
        StreamReader::ReadObj(buffer_reader, total_count_);
    } else {  // create like `else if ( ver in [v0.15, v0.17] )` here if need in the future
        logger::debug("parse with new version format");

        auto metadata = footer->GetMetadata();
        auto basic_info = metadata->Get("basic_info");
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
        total_count_ = basic_info["total_count"].GetInt();

        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Deserialize(buffer_reader);
        }

        this->inner_codes_->Deserialize(buffer_reader);
        this->label_table_->Deserialize(buffer_reader);
    }
    this->cal_memory_usage();
}

void
BruteForce::InitFeatures() {
    // About Train
    auto name = this->inner_codes_->GetQuantizerName();
    if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16) {
        this->index_feature_list_->SetFeature(IndexFeature::NEED_TRAIN);
    } else {
        this->index_feature_list_->SetFeatures({IndexFeature::SUPPORT_ADD_FROM_EMPTY,
                                                IndexFeature::SUPPORT_RANGE_SEARCH,
                                                IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID,
                                                IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER});
    }
    if (name == QUANTIZATION_TYPE_VALUE_FP32 and
        (metric_ != MetricType::METRIC_TYPE_COSINE || this->inner_codes_->HoldMolds())) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS);
    }

    // add & build & delete
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
        IndexFeature::SUPPORT_DELETE_BY_ID,
    });

    // search
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
    });

    // concurrency
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_SEARCH_CONCURRENT,
        IndexFeature::SUPPORT_ADD_CONCURRENT,
        IndexFeature::SUPPORT_DELETE_CONCURRENT,
    });

    // serialize
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
        IndexFeature::SUPPORT_SERIALIZE_WRITE_FUNC,
    });

    // others
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_ESTIMATE_MEMORY,
        IndexFeature::SUPPORT_GET_MEMORY_USAGE,
        IndexFeature::SUPPORT_CHECK_ID_EXIST,
        IndexFeature::SUPPORT_CLONE,
    });
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
        "{USE_ATTRIBUTE_FILTER_KEY}": false,
        "{ATTR_PARAMS_KEY}": {
            "{ATTR_HAS_BUCKETS_KEY}": true
        }
    })";

ParamPtr
BruteForce::CheckAndMappingExternalParam(const JsonType& external_param,
                                         const IndexCommonParam& common_param) {
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
        this->max_capacity_.store(new_size_power_2);
        this->cal_memory_usage();
    }
}

void
BruteForce::add_one(const float* data, InnerIdType inner_id) {
    this->inner_codes_->InsertVector(data, inner_id);
}

void
BruteForce::GetVectorByInnerId(InnerIdType inner_id, float* data) const {
    Vector<uint8_t> codes(inner_codes_->code_size_, allocator_);
    inner_codes_->GetCodesById(inner_id, codes.data());
    inner_codes_->Decode(codes.data(), data);
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
