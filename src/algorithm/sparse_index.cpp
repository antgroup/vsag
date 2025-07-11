
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

#include "sparse_index.h"

#include "impl/heap/standard_heap.h"
#include "utils/util_functions.h"

namespace vsag {

float
get_distance(uint32_t len1,
             const uint32_t* ids1,
             const float* vals1,
             uint32_t len2,
             const uint32_t* ids2,
             const float* vals2) {
    float sum = 0.0F;
    uint32_t i = 0;
    uint32_t j = 0;

    while (i < len1 && j < len2) {
        if (ids1[i] < ids2[j]) {
            i++;
        } else if (ids1[i] > ids2[j]) {
            j++;
        } else {
            sum += vals1[i] * vals2[j];
            i++;
            j++;
        }
    }

    return 1 - sum;
}

ParamPtr
SparseIndex::CheckAndMappingExternalParam(const JsonType& external_param,
                                          const IndexCommonParam& common_param) {
    auto ptr = std::make_shared<SparseIndexParameters>();
    ptr->FromJson(external_param);
    return ptr;
}

std::tuple<Vector<uint32_t>, Vector<float>>
SparseIndex::sort_sparse_vector(const SparseVector& vector) const {
    Vector<uint32_t> indices(vector.len_, allocator_);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](uint32_t a, uint32_t b) {
        return vector.ids_[a] < vector.ids_[b];
    });
    Vector<uint32_t> sorted_ids(vector.len_, allocator_);
    Vector<float> sorted_vals(vector.len_, allocator_);
    for (size_t j = 0; j < vector.len_; ++j) {
        sorted_ids[j] = vector.ids_[indices[j]];
        sorted_vals[j] = vector.vals_[indices[j]];
    }
    return std::make_tuple(sorted_ids, sorted_vals);
}

std::vector<int64_t>
SparseIndex::Add(const DatasetPtr& base) {
    const auto* sparse_vectors = base->GetSparseVectors();
    auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(data_num > 0, "data_num is zero when add vectors");
    const auto* ids = base->GetIds();
    if (max_capacity_ == 0) {
        auto new_capacity = std::max(INIT_CAPACITY, data_num);
        resize(new_capacity);
    }

    if (max_capacity_ < data_num + cur_element_count_) {
        auto extend_size = std::min(MAX_CAPACITY_EXTEND, max_capacity_);
        auto new_capacity =
            std::max(data_num + cur_element_count_ - max_capacity_, extend_size) + max_capacity_;
        resize(new_capacity);
    }

    for (int64_t i = 0; i < data_num; ++i) {
        const auto& vector = sparse_vectors[i];
        auto size = (vector.len_ + 1) * sizeof(uint32_t);  // vector index + array size
        size += (vector.len_) * sizeof(float);             // vector value
        datas_[i + cur_element_count_] = (uint32_t*)allocator_->Allocate(size);
        datas_[i + cur_element_count_][0] = vector.len_;
        auto* data = datas_[i + cur_element_count_] + 1;
        label_table_->Insert(i + cur_element_count_, ids[i]);
        if (need_sort_) {
            auto [sorted_ids, sorted_vals] = sort_sparse_vector(vector);
            std::memcpy(data, sorted_ids.data(), vector.len_ * sizeof(uint32_t));
            std::memcpy(data + vector.len_, sorted_vals.data(), vector.len_ * sizeof(float));
        } else {
            std::memcpy(data, vector.ids_, vector.len_ * sizeof(uint32_t));
            std::memcpy(data + vector.len_, vector.vals_, vector.len_ * sizeof(float));
        }
    }
    cur_element_count_ += data_num;
    return {};
}

DatasetPtr
SparseIndex::KnnSearch(const DatasetPtr& query,
                       int64_t k,
                       const std::string& parameters,
                       const FilterPtr& filter) const {
    const auto* sparse_vectors = query->GetSparseVectors();
    CHECK_ARGUMENT(query->GetNumElements() == 1, "num of query should be 1");
    auto results = std::make_shared<StandardHeap<true, false>>(allocator_, -1);

    auto [sorted_ids, sorted_vals] = sort_sparse_vector(sparse_vectors[0]);
    for (int j = 0; j < cur_element_count_; ++j) {
        auto distance = get_distance(sorted_ids.size(),
                                     sorted_ids.data(),
                                     sorted_vals.data(),
                                     datas_[j][0],
                                     datas_[j] + 1,
                                     (float*)(datas_[j] + 1 + datas_[j][0]));
        auto label = label_table_->GetLabelById(j);
        if (not filter || filter->CheckValid(label)) {
            results->Push(distance, label);
            if (results->Size() > k) {
                results->Pop();
            }
        }
    }
    // return result
    return collect_results(results);
}

DatasetPtr
SparseIndex::RangeSearch(const DatasetPtr& query,
                         float radius,
                         const std::string& parameters,
                         const FilterPtr& filter,
                         int64_t limited_size) const {
    const auto* sparse_vectors = query->GetSparseVectors();
    CHECK_ARGUMENT(query->GetNumElements() == 1, "num of query should be 1");
    auto results = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
    auto [sorted_ids, sorted_vals] = sort_sparse_vector(sparse_vectors[0]);
    for (int j = 0; j < cur_element_count_; ++j) {
        auto distance = get_distance(sorted_ids.size(),
                                     sorted_ids.data(),
                                     sorted_vals.data(),
                                     datas_[j][0],
                                     datas_[j] + 1,
                                     (float*)(datas_[j] + 1 + datas_[j][0]));
        auto label = label_table_->GetLabelById(j);
        if ((not filter || filter->CheckValid(label)) && distance <= radius + 2e-6) {
            results->Push(distance, label);
        }
    }

    while (results->Size() > limited_size) {
        results->Pop();
    }

    // return result
    return collect_results(results);
}

DatasetPtr
SparseIndex::collect_results(const DistHeapPtr& results) const {
    auto [result, dists, ids] =
        create_fast_dataset(static_cast<int64_t>(results->Size()), allocator_);
    if (results->Empty()) {
        result->Dim(0)->NumElements(1);
        return result;
    }

    for (auto j = static_cast<int64_t>(results->Size() - 1); j >= 0; --j) {
        dists[j] = results->Top().first;
        ids[j] = results->Top().second;
        results->Pop();
    }
    return result;
}

}  // namespace vsag
