
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

#include "data_cell/flatten_datacell.h"
#include "impl/heap/standard_heap.h"
#include "inner_string_params.h"
#include "storage/serialization.h"
#include "utils/slow_task_timer.h"
#include "utils/util_functions.h"

namespace vsag {

BruteForce::BruteForce(const BruteForceParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param) {
    inner_codes_ = FlattenInterface::MakeInstance(param->flatten_param, common_param);
    auto code_size = this->inner_codes_->code_size_;
    auto increase_count = Options::Instance().block_size_limit() / code_size;
    this->resize_increase_count_bit_ = std::max(
        DEFAULT_RESIZE_BIT, static_cast<uint64_t>(log2(static_cast<double>(increase_count))));
    this->build_pool_ = common_param.thread_pool_;
    if (this->build_pool_ == nullptr) {
        this->build_pool_ = SafeThreadPool::FactoryDefaultThreadPool();
    }
}

int64_t
BruteForce::GetMemoryUsage() const {
    return static_cast<int64_t>(this->CalSerializeSize());
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
BruteForce::Add(const DatasetPtr& data) {
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

    auto add_func = [&](const float* data, InnerIdType inner_id) -> void {
        this->add_one(data, inner_id);
    };

    std::vector<std::future<void>> futures;
    auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* vectors = data->GetFloat32Vectors();
    Vector<std::pair<InnerIdType, LabelType>> inner_ids(allocator_);
    for (int64_t j = 0; j < total; ++j) {
        auto label = labels[j];
        InnerIdType inner_id;
        {
            std::lock_guard label_lock(this->label_lookup_mutex_);
            if (this->label_table_->CheckLabel(label)) {
                failed_ids.emplace_back(label);
                continue;
            }
            {
                std::lock_guard lock(this->add_mutex_);
                inner_id = this->total_count_;
                total_count_++;
                this->resize(total_count_);
            }
            this->label_table_->Insert(inner_id, label);
            inner_ids.emplace_back(inner_id, j);
        }
    }
    for (auto& [inner_id, local_idx] : inner_ids) {
        if (this->build_pool_ != nullptr) {
            auto future =
                this->build_pool_->GeneralEnqueue(add_func, vectors + local_idx * dim_, inner_id);
            futures.emplace_back(std::move(future));
        } else {
            add_func(vectors + local_idx * dim_, inner_id);
        }
    }
    if (this->build_pool_ != nullptr) {
        for (auto& future : futures) {
            future.get();
        }
    }
    return failed_ids;
}

DatasetPtr
BruteForce::KnnSearch(const DatasetPtr& query,
                      int64_t k,
                      const std::string& parameters,
                      const FilterPtr& filter) const {
    std::shared_lock read_lock(this->global_mutex_);
    auto computer = this->inner_codes_->FactoryComputer(query->GetFloat32Vectors());
    auto heap = std::make_shared<StandardHeap<true, true>>(this->allocator_, k);
    for (InnerIdType i = 0; i < total_count_; ++i) {
        float dist;
        if (filter == nullptr or filter->CheckValid(this->label_table_->GetLabelById(i))) {
            inner_codes_->Query(&dist, computer, &i, 1);
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
BruteForce::CalcDistanceById(const float* vector, int64_t id) const {
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

    this->inner_codes_->Serialize(writer);
    this->label_table_->Serialize(writer);

    // serialize footer (introduced since v0.15)
    auto metadata = std::make_shared<Metadata>();
    JsonType basic_info;
    basic_info["dim"] = dim_;
    basic_info["total_count"] = total_count_;
    metadata->Set("basic_info", basic_info);
    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

void
BruteForce::Deserialize(StreamReader& reader) {
    // try to deserialize footer (only in new version)
    auto footer = Footer::Parse(reader);

    if (footer == nullptr) {  // old format, DON'T EDIT, remove in the future
        logger::debug("parse with v0.13 version format");

        StreamReader::ReadObj(reader, dim_);
        StreamReader::ReadObj(reader, total_count_);
    } else {  // create like `else if ( ver in [v0.15, v0.17] )` here if need in the future
        logger::debug("parse with new version format");

        auto metadata = footer->GetMetadata();
        auto basic_info = metadata->Get("basic_info");
        dim_ = basic_info["dim"];
        total_count_ = basic_info["total_count"];

        this->inner_codes_->Deserialize(reader);
        this->label_table_->Deserialize(reader);
    }

    // post serialize procedure
}

void
BruteForce::InitFeatures() {
    // About Train
    auto name = this->inner_codes_->GetQuantizerName();
    if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16) {
        this->index_feature_list_->SetFeature(IndexFeature::NEED_TRAIN);
    } else {
        this->index_feature_list_->SetFeatures({
            IndexFeature::SUPPORT_ADD_FROM_EMPTY,
            IndexFeature::SUPPORT_RANGE_SEARCH,
            IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID,
            IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
        });
    }

    // add & build
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
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
    });

    // serialize
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
    });

    // others
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_ESTIMATE_MEMORY,
        IndexFeature::SUPPORT_CHECK_ID_EXIST,
        IndexFeature::SUPPORT_CLONE,
    });
}

static const std::string BRUTE_FORCE_PARAMS_TEMPLATE =
    R"(
    {
        "type": "{INDEX_BRUTE_FORCE}",
        "{IO_PARAMS_KEY}": {
            "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}"
        },
        "{QUANTIZATION_PARAMS_KEY}": {
            "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
            "subspace": 64,
            "nbits": 8
        }
    })";

ParamPtr
BruteForce::CheckAndMappingExternalParam(const JsonType& external_param,
                                         const IndexCommonParam& common_param) {
    const std::unordered_map<std::string, std::vector<std::string>> external_mapping = {
        {
            BRUTE_FORCE_QUANTIZATION_TYPE,
            {
                QUANTIZATION_PARAMS_KEY,
                QUANTIZATION_TYPE_KEY,
            },
        },
        {
            BRUTE_FORCE_IO_TYPE,
            {
                IO_PARAMS_KEY,
                IO_TYPE_KEY,
            },
        }};

    if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("BruteForce not support {} datatype", DATATYPE_INT8));
    }

    std::string str = format_map(BRUTE_FORCE_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto inner_json = JsonType::parse(str);
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
    }
}

void
BruteForce::add_one(const float* data, InnerIdType inner_id) {
    this->inner_codes_->InsertVector(data, inner_id);
}

}  // namespace vsag
