
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

#include "utils/util_functions.h"

namespace vsag {

float
get_distance(uint32_t len1,
             const uint32_t* ids1,
             const float* vals1,
             uint32_t len2,
             const uint32_t* ids2,
             const float* vals2) {
    float sum = 0.0f;
    uint32_t i = 0, j = 0;

    while (i < len1 && j < len2) {
        if (ids1[i] == ids2[j]) {
            sum += vals1[i] * vals2[j];
            i++;
            j++;
        } else if (ids1[i] < ids2[j]) {
            i++;
        } else {
            j++;
        }
    }

    return 1 - sum;
}

ParamPtr
SparseIndex::CheckAndMappingExternalParam(const JsonType& external_param,
                                          const IndexCommonParam& common_param) {
    return std::make_shared<SparseIndexParameters>();
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
    auto sparse_vectors = base->GetSparseVectors();
    auto data_num = base->GetNumElements();
    auto ids = base->GetIds();
    auto cur_size = datas_.size();
    datas_.resize(cur_size + data_num);

    for (int64_t i = 0; i < data_num; ++i) {
        const auto& vector = sparse_vectors[i];
        auto [sorted_ids, sorted_vals] = sort_sparse_vector(vector);
        datas_[i + cur_size] =
            (uint32_t*)allocator_->Allocate((2 * vector.len_ + 1) * sizeof(uint32_t));
        datas_[i + cur_size][0] = vector.len_;
        auto* data = datas_[i + cur_size] + 1;
        label_table_->Insert(i + cur_size, ids[i]);
        std::memcpy(data, sorted_ids.data(), vector.len_ * sizeof(uint32_t));
        std::memcpy(data + vector.len_, sorted_vals.data(), vector.len_ * sizeof(float));
    }
    return {};
}

DatasetPtr
SparseIndex::KnnSearch(const DatasetPtr& query,
                       int64_t k,
                       const std::string& parameters,
                       const FilterPtr& filter) const {
    auto sparse_vectors = query->GetSparseVectors();
    MaxHeap results(allocator_);
    auto [sorted_ids, sorted_vals] = sort_sparse_vector(sparse_vectors[0]);
    for (int j = 0; j < datas_.size(); ++j) {
        auto distance = get_distance(sorted_ids.size(),
                                     sorted_ids.data(),
                                     sorted_vals.data(),
                                     datas_[j][0],
                                     datas_[j] + 1,
                                     (float*)datas_[j] + 1 + datas_[j][0]);
        auto id = label_table_->GetLabelById(j);
        if (not filter || filter->CheckValid(id)) {
            results.emplace(distance, id);
            if (results.size() > k) {
                results.pop();
            }
        }
    }

    while (results.size() > k) {
        results.pop();
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
    auto sparse_vectors = query->GetSparseVectors();
    MaxHeap results(allocator_);
    auto [sorted_ids, sorted_vals] = sort_sparse_vector(sparse_vectors[0]);
    for (int j = 0; j < datas_.size(); ++j) {
        auto distance = get_distance(sorted_ids.size(),
                                     sorted_ids.data(),
                                     sorted_vals.data(),
                                     datas_[j][0],
                                     datas_[j] + 1,
                                     (float*)datas_[j] + 1 + datas_[j][0]);
        auto id = label_table_->GetLabelById(j);
        if ((not filter || filter->CheckValid(id)) && distance <= radius + 2e-6) {
            results.emplace(distance, id);
        }
    }

    while (results.size() > limited_size) {
        results.pop();
    }

    // return result
    return collect_results(results);
}

DatasetPtr
SparseIndex::collect_results(MaxHeap& results) const {
    auto [result, ids, dists] = CreateFastDataset(results.size(), allocator_);
    if (results.empty()) {
        result->Dim(0)->NumElements(1);
        return result;
    }

    for (auto j = static_cast<int64_t>(results.size() - 1); j >= 0; --j) {
        dists[j] = results.top().first;
        ids[j] = results.top().second;
        results.pop();
    }
    return result;
    return vsag::DatasetPtr();
}

}  // namespace vsag