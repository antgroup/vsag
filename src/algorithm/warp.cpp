
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

#include "warp.h"

#include <atomic>
#include <mutex>
#include <numeric>
#include <optional>

#include "attr/argparse.h"
#include "attr/executor/executor.h"
#include "datacell/attribute_inverted_interface.h"
#include "datacell/flatten_datacell.h"
#include "datacell/flatten_datacell_parameter.h"
#include "datacell/flatten_interface.h"
#include "fmt/chrono.h"
#include "impl/heap/distance_heap.h"
#include "impl/heap/standard_heap.h"
#include "index_common_param.h"
#include "index_feature_list.h"
#include "inner_string_params.h"
#include "storage/serialization.h"
#include "typing.h"
#include "utils/slow_task_timer.h"
#include "utils/util_functions.h"
namespace vsag {

WARP::WARP(const WarpParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param), doc_offsets_(allocator_) {
    inner_codes_ = FlattenInterface::MakeInstance(param->base_codes_param, common_param);
    auto code_size = this->inner_codes_->code_size_;
    auto increase_count = Options::Instance().block_size_limit() / code_size;
    this->resize_increase_count_bit_ = std::max(
        DEFAULT_RESIZE_BIT, static_cast<uint64_t>(log2(static_cast<double>(increase_count))));
    this->use_attribute_filter_ = param->use_attribute_filter;
    this->has_raw_vector_ = true;
}

uint64_t
WARP::EstimateMemory(uint64_t num_elements) const {
    // For WARP, we need to account for multi-vector documents
    // Assuming average of 10 vectors per document
    uint64_t avg_vectors_per_doc = 10;
    return num_elements * (avg_vectors_per_doc * this->dim_ * sizeof(float) +
                           sizeof(LabelType) * 2 + sizeof(InnerIdType) + sizeof(uint32_t) * 2);
}

std::vector<int64_t>
WARP::Build(const vsag::DatasetPtr& data) {
    this->Train(data);
    return this->Add(data);
}

void
WARP::Train(const DatasetPtr& data) {
    this->inner_codes_->Train(data->GetFloat32Vectors(), data->GetNumElements());
}

std::vector<int64_t>
WARP::Add(const DatasetPtr& data, AddMode mode) {
    std::vector<int64_t> failed_ids;
    auto base_dim = data->GetDim();
    CHECK_ARGUMENT(base_dim == dim_,
                   fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));
    CHECK_ARGUMENT(data->GetFloat32Vectors() != nullptr, "base.float_vector is nullptr");

    // For multi-vector support, vector counts per document is required
    const uint32_t* vector_counts = data->GetVectorCounts();
    CHECK_ARGUMENT(vector_counts != nullptr, "base.vector_counts is nullptr");

    int64_t num_elements = data->GetNumElements();

    {
        std::lock_guard lock(this->add_mutex_);
        if (this->total_count_ == 0) {
            this->Train(data);
        }
    }

    // Multi-vector document handling
    const auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* vectors = data->GetFloat32Vectors();
    const auto* attrs = data->GetAttributeSets();
    const auto* extra_info = data->GetExtraInfos();
    const auto extra_info_size = data->GetExtraInfoSize();

    auto add_func = [&](const float* doc_vectors,
                        uint32_t doc_vec_count,
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

        this->add_one_doc(doc_vectors, doc_vec_count, inner_id);
        return std::nullopt;
    };

    std::vector<std::future<std::optional<int64_t>>> futures;
    uint64_t vec_offset = 0;

    for (int64_t j = 0; j < total; ++j) {
        const auto label = labels[j];
        uint32_t doc_vec_count = vector_counts[j];

        {
            std::lock_guard label_lock(this->label_lookup_mutex_);
            if (this->label_table_->CheckLabel(label)) {
                failed_ids.emplace_back(label);
                vec_offset += doc_vec_count;
                continue;
            }
        }

        if (this->thread_pool_ != nullptr) {
            auto future = this->thread_pool_->GeneralEnqueue(add_func,
                                                             vectors + vec_offset * dim_,
                                                             doc_vec_count,
                                                             label,
                                                             attrs == nullptr ? nullptr : attrs + j,
                                                             extra_info + j * extra_info_size);
            futures.emplace_back(std::move(future));
        } else {
            if (auto add_res = add_func(vectors + vec_offset * dim_,
                                        doc_vec_count,
                                        label,
                                        attrs == nullptr ? nullptr : attrs + j,
                                        extra_info + j * extra_info_size);
                add_res.has_value()) {
                failed_ids.emplace_back(add_res.value());
            }
        }

        vec_offset += doc_vec_count;
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

float
WARP::compute_maxsin_similarity(const float* query_vectors,
                                uint32_t query_vec_count,
                                uint32_t doc_start_vec_idx,
                                uint32_t doc_vec_count) const {
    if (doc_vec_count == 0 || query_vec_count == 0) {
        return 0.0f;
    }

    float total_score = 0.0f;

    // For each query vector, find max similarity with any document vector
    for (uint32_t q = 0; q < query_vec_count; ++q) {
        const float* query_vec = query_vectors + q * dim_;
        auto computer = this->inner_codes_->FactoryComputer(query_vec);

        float min_sim = std::numeric_limits<float>::max();
        Vector<InnerIdType> vec_indices(doc_vec_count, allocator_);
        Vector<float> dists(doc_vec_count, allocator_);
        std::iota(vec_indices.begin(), vec_indices.end(), doc_start_vec_idx);
        this->inner_codes_->Query(dists.data(), computer, vec_indices.data(), doc_vec_count);
        for (const auto& sim : dists) {
            min_sim = std::min(min_sim, sim);
        }
        total_score += min_sim;
    }
    return total_score;
}

DatasetPtr
WARP::KnnSearch(const DatasetPtr& query,
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
WARP::SearchWithRequest(const SearchRequest& request) const {
    std::shared_lock read_lock(this->global_mutex_);

    // Get query information
    const float* query_vectors = request.query_->GetFloat32Vectors();
    const uint32_t* query_vec_counts = request.query_->GetVectorCounts();
    CHECK_ARGUMENT(query_vec_counts != nullptr, "query.vector_counts is nullptr");

    int64_t query_num = request.query_->GetNumElements();
    uint32_t query_vec_count = query_vec_counts[0];

    FilterPtr ft = nullptr;
    auto combined_filter = std::make_shared<CombinedFilter>();
    combined_filter->AppendFilter(this->label_table_->GetDeletedIdsFilter());
    if (request.filter_ != nullptr) {
        combined_filter->AppendFilter(
            std::make_shared<InnerIdWrapperFilter>(request.filter_, *this->label_table_));
    }
    if (not combined_filter->IsEmpty()) {
        ft = combined_filter;
    }

    // Parse search parameters
    auto warp_params = WarpSearchParameters::FromJson(request.params_str_);
    auto parallel_count = warp_params.parallel_search_thread_count;

    std::atomic<uint32_t> dist_cmp{0};

    // For each query, compute maxsin score for all documents
    // Use a heap to maintain top-k results
    auto search_func = [&](InnerIdType start_doc, InnerIdType end_doc) -> DistHeapPtr {
        auto heap = DistanceHeap::MakeInstanceBySize<true, true>(this->allocator_, request.topk_);

        for (InnerIdType doc_id = start_doc; doc_id < end_doc; ++doc_id) {
            if (ft != nullptr && not ft->CheckValid(doc_id)) {
                continue;
            }

            uint32_t doc_start_vec = doc_offsets_[doc_id];
            uint32_t doc_vec_count = doc_offsets_[doc_id + 1] - doc_offsets_[doc_id];

            float score = compute_maxsin_similarity(
                query_vectors, query_vec_count, doc_start_vec, doc_vec_count);

            // For L2, we use -score as distance; for IP/Cosine, score is already the similarity
            // The heap expects distance (smaller is better for L2)
            float heap_dist = score;
            heap->Push(heap_dist, doc_id);
            dist_cmp.fetch_add(query_vec_count * doc_vec_count, std::memory_order_relaxed);
        }

        return heap;
    };

    DistHeapPtr heap = nullptr;

    if (parallel_count == 1 || this->thread_pool_ == nullptr || total_count_ < 1000) {
        heap = search_func(0, total_count_);
    } else {
        std::vector<std::future<DistHeapPtr>> futures;
        auto chunk_size = (total_count_ + parallel_count - 1) / parallel_count;
        for (auto i = 0; i < static_cast<int>(parallel_count); ++i) {
            auto start = i * chunk_size;
            auto end = std::min(start + chunk_size, static_cast<uint64_t>(total_count_));
            if (start < end) {
                auto future = this->thread_pool_->GeneralEnqueue(search_func, start, end);
                futures.emplace_back(std::move(future));
            }
        }

        heap = futures[0].get();
        for (size_t i = 1; i < futures.size(); ++i) {
            auto partial_heap = futures[i].get();
            heap->Merge(*partial_heap);
        }
    }

    auto [dataset_results, dists, ids] =
        create_fast_dataset(static_cast<int64_t>(heap->Size()), allocator_);
    for (auto j = static_cast<int64_t>(heap->Size() - 1); j >= 0; --j) {
        float dist = heap->Top().first;
        dists[j] = dist;
        ids[j] = this->label_table_->GetLabelById(heap->Top().second);
        heap->Pop();
    }

    JsonType stats;
    stats["dist_cmp"].SetInt(dist_cmp.load(std::memory_order_relaxed));
    dataset_results->Statistics(stats.Dump());

    return std::move(dataset_results);
}

DatasetPtr
WARP::RangeSearch(const vsag::DatasetPtr& query,
                  float radius,
                  const std::string& parameters,
                  const vsag::FilterPtr& filter,
                  int64_t limited_size) const {
    std::shared_lock read_lock(this->global_mutex_);

    // Get query information
    const float* query_vectors = query->GetFloat32Vectors();
    const uint32_t* query_vec_counts = query->GetVectorCounts();
    CHECK_ARGUMENT(query_vec_counts != nullptr, "query.vector_counts is nullptr");

    uint32_t query_vec_count = query_vec_counts[0];

    if (limited_size < 0) {
        limited_size = std::numeric_limits<int64_t>::max();
    }

    auto heap = std::make_shared<StandardHeap<true, true>>(this->allocator_, limited_size);

    for (InnerIdType doc_id = 0; doc_id < total_count_; ++doc_id) {
        if (filter != nullptr and
            not filter->CheckValid(this->label_table_->GetLabelById(doc_id))) {
            continue;
        }

        uint32_t doc_start_vec = doc_offsets_[doc_id];
        uint32_t doc_vec_count_val = doc_offsets_[doc_id + 1] - doc_offsets_[doc_id];

        float score = compute_maxsin_similarity(
            query_vectors, query_vec_count, doc_start_vec, doc_vec_count_val);

        float heap_dist = score;
        if (heap_dist > radius) {
            continue;
        }
        heap->Push(heap_dist, doc_id);
    }

    auto [dataset_results, dists, ids] =
        create_fast_dataset(static_cast<int64_t>(heap->Size()), allocator_);
    for (auto j = static_cast<int64_t>(heap->Size() - 1); j >= 0; --j) {
        float dist = heap->Top().first;
        dists[j] = dist;
        ids[j] = this->label_table_->GetLabelById(heap->Top().second);
        heap->Pop();
    }
    return std::move(dataset_results);
}

void
WARP::Serialize(StreamWriter& writer) const {
    // Serialize document offsets (size = total_count_ + 1)
    // doc_offsets_[i] is the starting vector index of document i
    // doc_offsets_[total_count_] = total_vector_count_
    StreamWriter::WriteObj(writer, total_count_);
    StreamWriter::WriteObj(writer, total_vector_count_);

    for (uint64_t i = 0; i <= total_count_; ++i) {
        uint32_t offset = (i < doc_offsets_.size()) ? doc_offsets_[i] : 0;
        StreamWriter::WriteObj(writer, offset);
    }

    this->inner_codes_->Serialize(writer);
    this->label_table_->Serialize(writer);

    // Serialize footer
    auto metadata = std::make_shared<Metadata>();
    JsonType basic_info;
    basic_info["dim"].SetInt(dim_);
    basic_info["total_count"].SetInt(total_count_);
    basic_info["total_vector_count"].SetInt(total_vector_count_);
    basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    metadata->Set("basic_info", basic_info);
    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

void
WARP::Deserialize(StreamReader& reader) {
    // Try to deserialize footer
    auto footer = Footer::Parse(reader);

    BufferStreamReader buffer_reader(
        &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);

    auto metadata = footer->GetMetadata();
    auto basic_info = metadata->Get("basic_info");
    if (basic_info.Contains(INDEX_PARAM)) {
        std::string index_param_string = basic_info[INDEX_PARAM].GetString();
        auto index_param = std::make_shared<WarpParameter>();
        index_param->FromString(index_param_string);
        if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
            auto message = fmt::format("WARP index parameter not match, current: {}, new: {}",
                                       this->create_param_ptr_->ToString(),
                                       index_param->ToString());
            logger::error(message);
            throw VsagException(ErrorType::INVALID_ARGUMENT, message);
        }
    }
    dim_ = basic_info["dim"].GetInt();

    StreamReader::ReadObj(buffer_reader, total_count_);
    StreamReader::ReadObj(buffer_reader, total_vector_count_);

    // doc_offsets_ size is total_count_ + 1
    doc_offsets_.resize(total_count_ + 1);

    for (uint64_t i = 0; i <= total_count_; ++i) {
        StreamReader::ReadObj(buffer_reader, doc_offsets_[i]);
    }

    this->inner_codes_->Deserialize(buffer_reader);
    this->label_table_->Deserialize(buffer_reader);
    this->cal_memory_usage();
}

void
WARP::resize(uint64_t new_size) {
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
        // doc_offsets_ size is new_size_power_2 + 1 (to store total_vector_count_ at the end)
        doc_offsets_.resize(new_size_power_2 + 1);
        this->max_capacity_.store(new_size_power_2);
        this->cal_memory_usage();
    }
}

void
WARP::add_one_doc(const float* data, uint32_t vec_count, InnerIdType inner_id) {
    // Store the document's vectors
    for (uint32_t i = 0; i < vec_count; ++i) {
        this->inner_codes_->InsertVector(data + i * dim_, total_vector_count_ + i);
    }

    // Update document offsets
    // doc_offsets_[inner_id] is the starting vector index
    // doc_offsets_[inner_id + 1] will be the ending vector index (= total_vector_count_ + vec_count)
    doc_offsets_[inner_id] = total_vector_count_;
    doc_offsets_[inner_id + 1] = total_vector_count_ + vec_count;

    total_vector_count_ += vec_count;
}

void
WARP::GetVectorByInnerId(InnerIdType inner_id, float* data) const {
    // For multi-vector docs, return the first vector
    uint32_t vec_idx = doc_offsets_[inner_id];
    Vector<uint8_t> codes(inner_codes_->code_size_, allocator_);
    inner_codes_->GetCodesById(vec_idx, codes.data());
    inner_codes_->Decode(codes.data(), data);
}

void
WARP::cal_memory_usage() {
    auto memory_usage = this->inner_codes_->GetMemoryUsage();
    memory_usage += sizeof(WARP);
    memory_usage += this->label_table_->GetMemoryUsage();
    memory_usage += doc_offsets_.size() * sizeof(uint32_t);
    std::unique_lock lock(this->memory_usage_mutex_);
    this->current_memory_usage_.store(memory_usage);
}

int64_t
WARP::GetMemoryUsage() const {
    int64_t memory = 0;
    {
        std::shared_lock lock(this->memory_usage_mutex_);
        memory = this->current_memory_usage_.load();
    }
    return memory;
}

void
WARP::InitFeatures() {
    auto name = this->inner_codes_->GetQuantizerName();
    if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16) {
        this->index_feature_list_->SetFeature(IndexFeature::NEED_TRAIN);
    } else {
        this->index_feature_list_->SetFeatures({IndexFeature::SUPPORT_ADD_FROM_EMPTY,
                                                IndexFeature::SUPPORT_RANGE_SEARCH,
                                                IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID,
                                                IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER});
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

static const std::string WARP_PARAMS_TEMPLATE =
    R"(
    {
        "{TYPE_KEY}": "{INDEX_WARP}",
        "{BASE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}"
            }
        }
    })";

ParamPtr
WARP::CheckAndMappingExternalParam(const JsonType& external_param,
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
    mapping_external_param_to_inner(external_param, external_mapping, inner_json);

    auto warp_parameter = std::make_shared<WarpParameter>();
    warp_parameter->FromJson(inner_json);

    return warp_parameter;
}

}  // namespace vsag
