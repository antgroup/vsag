
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
#include "utils/util_functions.h"

namespace vsag {

BruteForce::BruteForce(const BruteForceParameter& param, const IndexCommonParam& common_param)
    : Index(), dim_(common_param.dim_), allocator_(common_param.allocator_) {
    label_table_ = std::make_shared<LabelTable>(common_param.allocator_.get());
    inner_codes_ = FlattenInterface::MakeInstance(param.flatten_param_, common_param);
    this->init_feature_list();
}

BruteForce::~BruteForce() {
    label_table_.reset();
    inner_codes_.reset();
}

int64_t
BruteForce::GetMemoryUsage() const {
    return static_cast<int64_t>(this->cal_serialize_size());
}

uint64_t
BruteForce::EstimateMemory(uint64_t num_elements) const {
    return num_elements *
           (this->dim_ * sizeof(float) + sizeof(LabelType) * 2 + sizeof(InnerIdType));
}

bool
BruteForce::CheckFeature(IndexFeature feature) const {
    return feature_list_.CheckFeature(feature);
}

std::vector<int64_t>
BruteForce::build(const DatasetPtr& data) {
    return this->add(data);
}

std::vector<int64_t>
BruteForce::add(const DatasetPtr& data) {
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
BruteForce::knn_search(const DatasetPtr& query,
                       int64_t k,
                       const std::string& parameters,
                       const std::function<bool(int64_t)>& filter) const {
    auto computer = this->inner_codes_->FactoryComputer(query->GetFloat32Vectors());
    MaxHeap heap(this->allocator_.get());
    auto cur_heap_top = std::numeric_limits<float>::max();
    for (InnerIdType i = 0; i < total_count_; ++i) {
        float dist;
        if (filter == nullptr or not filter(this->label_table_->GetLabelById(i))) {
            inner_codes_->Query(&dist, computer, &i, 1);
            if (heap.size() < k or dist < cur_heap_top) {
                heap.emplace(dist, i);
            }
            if (heap.size() > k) {
                heap.pop();
            }
            cur_heap_top = heap.top().first;
        }
    }
    auto [dataset_results, dists, ids] =
        CreateFastDataset(static_cast<int64_t>(heap.size()), allocator_.get());
    for (auto j = static_cast<int64_t>(heap.size() - 1); j >= 0; --j) {
        dists[j] = heap.top().first;
        ids[j] = this->label_table_->GetLabelById(heap.top().second);
        heap.pop();
    }
    return std::move(dataset_results);
}

DatasetPtr
BruteForce::range_search(const DatasetPtr& query,
                         float radius,
                         const std::string& parameters,
                         BaseFilterFunctor* filter_ptr,
                         int64_t limited_size) const {
    auto computer = this->inner_codes_->FactoryComputer(query->GetFloat32Vectors());
    MaxHeap heap(this->allocator_.get());
    auto cur_heap_top = radius;
    if (limited_size < 0) {
        limited_size = std::numeric_limits<int64_t>::max();
    }
    for (InnerIdType i = 0; i < total_count_; ++i) {
        float dist;
        if (filter_ptr == nullptr or (*filter_ptr)(this->label_table_->GetLabelById(i))) {
            inner_codes_->Query(&dist, computer, &i, 1);
            if (dist > radius) {
                continue;
            }
            heap.emplace(dist, i);
            if (heap.size() > limited_size) {
                heap.pop();
            }
            cur_heap_top = heap.top().first;
        }
    }
    auto [dataset_results, dists, ids] =
        CreateFastDataset(static_cast<int64_t>(heap.size()), allocator_.get());
    for (auto j = static_cast<int64_t>(heap.size() - 1); j >= 0; --j) {
        dists[j] = heap.top().first;
        ids[j] = this->label_table_->GetLabelById(heap.top().second);
        heap.pop();
    }
    return std::move(dataset_results);
}

float
BruteForce::calculate_distance_by_id(const float* vector, int64_t id) const {
    auto computer = this->inner_codes_->FactoryComputer(vector);
    float result = 0.0F;
    InnerIdType inner_id = this->label_table_->GetIdByLabel(id);
    this->inner_codes_->Query(&result, computer, &inner_id, 1);
    return result;
}

BinarySet
BruteForce::serialize() const {
    SlowTaskTimer t("brute force Serialize");
    size_t num_bytes = this->cal_serialize_size();
    std::shared_ptr<int8_t[]> bin(new int8_t[num_bytes]);
    auto* buffer = reinterpret_cast<char*>(const_cast<int8_t*>(bin.get()));
    BufferStreamWriter writer(buffer);
    this->serialize(writer);
    Binary b{
        .data = bin,
        .size = num_bytes,
    };
    BinarySet bs;
    bs.Set(INDEX_BRUTE_FORCE, b);

    return bs;
}

void
BruteForce::serialize(std::ostream& out_stream) const {
    SlowTaskTimer t("brute force Serialize");
    IOStreamWriter writer(out_stream);
    this->serialize(writer);
}

void
BruteForce::serialize(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, dim_);
    StreamWriter::WriteObj(writer, total_count_);

    this->inner_codes_->Serialize(writer);
    this->label_table_->Serialize(writer);
}

void
BruteForce::deserialize(std::istream& in_stream) {
    SlowTaskTimer t("brute force Deserialize");
    IOStreamReader reader(in_stream);
    this->deserialize(reader);
}

void
BruteForce::deserialize(const BinarySet& binary_set) {
    SlowTaskTimer t("brute force Deserialize");
    Binary b = binary_set.Get(INDEX_BRUTE_FORCE);
    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        std::memcpy(dest, b.data.get() + offset, len);
    };
    uint64_t cursor = 0;
    auto reader = ReadFuncStreamReader(func, cursor);
    this->deserialize(reader);
}

void
BruteForce::deserialize(const ReaderSet& reader_set) {
    SlowTaskTimer t("brute force Deserialize");
    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        reader_set.Get(INDEX_BRUTE_FORCE)->Read(offset, len, dest);
    };
    uint64_t cursor = 0;
    auto reader = ReadFuncStreamReader(func, cursor);
    this->deserialize(reader);
}

void
BruteForce::deserialize(StreamReader& reader) {
    StreamReader::ReadObj(reader, dim_);
    StreamReader::ReadObj(reader, total_count_);
    this->inner_codes_->Deserialize(reader);
    this->label_table_->Deserialize(reader);
}

uint64_t
BruteForce::cal_serialize_size() const {
    auto cal_size_func = [](uint64_t cursor, uint64_t size, void* buf) { return; };
    WriteFuncStreamWriter writer(cal_size_func, 0);
    this->serialize(writer);
    return writer.cursor_;
}

Vector<DatasetPtr>
BruteForce::split_dataset_by_duplicate_label(const DatasetPtr& dataset,
                                             std::vector<LabelType>& failed_ids) const {
    Vector<DatasetPtr> return_datasets(0, this->allocator_.get());
    auto count = dataset->GetNumElements();
    auto dim = dataset->GetDim();
    const auto* labels = dataset->GetIds();
    const auto* vec = dataset->GetFloat32Vectors();
    UnorderedSet<LabelType> temp_labels(allocator_.get());

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

void
BruteForce::init_feature_list() {
    // About Train
    auto name = this->inner_codes_->GetQuantizerName();
    if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16) {
        feature_list_.SetFeature(IndexFeature::NEED_TRAIN);
    } else {
        feature_list_.SetFeatures({
            IndexFeature::SUPPORT_ADD_FROM_EMPTY,
            IndexFeature::SUPPORT_RANGE_SEARCH,
            IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID,
            IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
        });
    }
    // Add & Build
    feature_list_.SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
    });
    // Search
    feature_list_.SetFeatures({
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
    });
    // concurrency
    feature_list_.SetFeatures({
        IndexFeature::SUPPORT_SEARCH_CONCURRENT,
    });

    // serialize
    feature_list_.SetFeatures({
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
    });
    // others
    feature_list_.SetFeatures({
        IndexFeature::SUPPORT_ESTIMATE_MEMORY,
        IndexFeature::SUPPORT_CHECK_ID_EXIST,
    });
}
bool
BruteForce::CheckIdExist(int64_t id) const {
    return this->label_table_->CheckLabel(id);
}

}  // namespace vsag
