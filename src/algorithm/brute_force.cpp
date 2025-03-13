
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
#include "inner_string_params.h"
#include "utils/slow_task_timer.h"
#include "utils/standard_heap.h"
#include "utils/util_functions.h"

namespace vsag {

BruteForce::BruteForce(const BruteForceParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param),
      dim_(common_param.dim_),
      allocator_(common_param.allocator_.get()) {
    label_table_ = std::make_shared<LabelTable>(common_param.allocator_.get());
    inner_codes_ = FlattenInterface::MakeInstance(param->flatten_param, common_param);
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
    return this->Add(data);
}

std::vector<int64_t>
BruteForce::Add(const DatasetPtr& data) {
    this->inner_codes_->Train(data->GetFloat32Vectors(), data->GetNumElements());
    std::vector<int64_t> failed_ids;

    const auto& datasets = this->split_dataset_by_duplicate_label(data, failed_ids);
    for (const auto& per_dataset : datasets) {
        auto start_id = this->GetNumElements();
        this->inner_codes_->BatchInsertVector(per_dataset->GetFloat32Vectors(),
                                              per_dataset->GetNumElements());
        for (uint64_t i = 0; i < per_dataset->GetNumElements(); ++i) {
            const auto& label = per_dataset->GetIds()[i];
            this->label_table_->Insert(start_id + i, label);
        }
        this->total_count_ += per_dataset->GetNumElements();
    }

    return failed_ids;
}

DatasetPtr
BruteForce::KnnSearch(const DatasetPtr& query,
                      int64_t k,
                      const std::string& parameters,
                      const FilterPtr& filter) const {
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
        CreateFastDataset(static_cast<int64_t>(heap->Size()), allocator_);
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
        CreateFastDataset(static_cast<int64_t>(heap->Size()), allocator_);
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
    StreamWriter::WriteObj(writer, dim_);
    StreamWriter::WriteObj(writer, total_count_);

    this->inner_codes_->Serialize(writer);
    this->label_table_->Serialize(writer);
}

void
BruteForce::Deserialize(StreamReader& reader) {
    StreamReader::ReadObj(reader, dim_);
    StreamReader::ReadObj(reader, total_count_);
    this->inner_codes_->Deserialize(reader);
    this->label_table_->Deserialize(reader);
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
    // Add & Build
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
    });
    // Search
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
    });
    // concurrency
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_SEARCH_CONCURRENT,
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
    });
}

Vector<DatasetPtr>
BruteForce::split_dataset_by_duplicate_label(const DatasetPtr& dataset,
                                             std::vector<LabelType>& failed_ids) const {
    Vector<DatasetPtr> return_datasets(0, this->allocator_);
    auto count = dataset->GetNumElements();
    auto dim = dataset->GetDim();
    const auto* labels = dataset->GetIds();
    const auto* vec = dataset->GetFloat32Vectors();
    UnorderedSet<LabelType> temp_labels(allocator_);

    for (uint64_t i = 0; i < count; ++i) {
        if (label_table_->CheckLabel(labels[i]) or
            temp_labels.find(labels[i]) != temp_labels.end()) {
            failed_ids.emplace_back(i);
            continue;
        }
        temp_labels.emplace(labels[i]);
    }
    failed_ids.emplace_back(count);

    if (failed_ids.size() == 1) {
        return_datasets.emplace_back(dataset);
        return return_datasets;
    }
    int64_t start = -1;
    for (auto end : failed_ids) {
        if (end - start == 1) {
            start = end;
            continue;
        }
        auto new_dataset = Dataset::Make();
        new_dataset->NumElements(end - start - 1)
            ->Dim(dim)
            ->Ids(labels + start + 1)
            ->Float32Vectors(vec + dim * (start + 1))
            ->Owner(false);
        return_datasets.emplace_back(new_dataset);
        start = end;
    }
    failed_ids.pop_back();
    for (auto& failed_id : failed_ids) {
        failed_id = labels[failed_id];
    }
    return return_datasets;
}

static const std::unordered_map<std::string, std::vector<std::string>> EXTERNAL_MAPPING = {
    {BRUTE_FORCE_QUANTIZATION_TYPE, {QUANTIZATION_PARAMS_KEY, QUANTIZATION_TYPE_KEY}},
    {BRUTE_FORCE_IO_TYPE, {IO_PARAMS_KEY, IO_TYPE_KEY}}};

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
    if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw std::invalid_argument(
            fmt::format("BruteForce not support {} datatype", DATATYPE_INT8));
    }

    std::string str = format_map(BRUTE_FORCE_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto inner_json = JsonType::parse(str);
    mapping_external_param_to_inner(external_param, EXTERNAL_MAPPING, inner_json);

    auto brute_force_parameter = std::make_shared<BruteForceParameter>();
    brute_force_parameter->FromJson(inner_json);

    return brute_force_parameter;
}

}  // namespace vsag
