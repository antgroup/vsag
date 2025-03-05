
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

#pragma once

#include "index/sparse_index_parameters.h"
#include "inner_index_interface.h"

namespace vsag {

float
get_distance(
    uint32_t len1, uint32_t* ids1, float* vals1, uint32_t len2, uint32_t* ids2, float* vals2) {
    std::unordered_map<uint32_t, float> hashmap;
    for (uint32_t i = 0; i < len1; ++i) {
        hashmap[ids1[i]] = vals1[i];
    }

    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < len2; ++i) {
        uint32_t id = ids2[i];
        float val2 = vals2[i];
        auto it = hashmap.find(id);
        if (it != hashmap.end()) {
            float val1 = it->second;
            float diff = val1 - val2;
            sum_sq += diff * diff;
            hashmap.erase(it);
        } else {
            sum_sq += val2 * val2;
        }
    }
    for (const auto& entry : hashmap) {
        float val1 = entry.second;
        sum_sq += val1 * val1;
    }

    return sum_sq;
}

class SparseIndex : public InnerIndexInterface {
public:
    explicit SparseIndex(const SparseIndexParameterPtr& param, const IndexCommonParam& common_param)
        : InnerIndexInterface(param, common_param), datas_(allocator_), label_table_(allocator_) {
    }

    virtual ~SparseIndex() {
        for (int i = 0; i < datas_.size(); ++i) {
            allocator_->Deallocate(datas_[i]);
        }
    }

    [[nodiscard]] std::string
    GetName() const override {
        return "SparseIndex";
    }

    std::vector<int64_t>
    Add(const DatasetPtr& base) override {
        auto sparse_vectors = base->GetSparseVectors();
        auto data_num = base->GetNumElements();
        auto ids = base->GetIds();
        auto cur_size = datas_.size();
        datas_.resize(cur_size + data_num);
        for (int64_t i = 0; i < data_num; ++i) {
            const auto& vector = sparse_vectors[i];
            datas_[i + cur_size] = (uint32_t*)allocator_->Allocate(2 * vector.len_ + 1);
            datas_[i + cur_size][0] = vector.len_;
            auto* data = datas_[i + cur_size] + 1;
            label_table_.Insert(i + cur_size, ids[i]);
            std::memcpy(data, vector.ids_, vector.len_ * sizeof(uint32_t));
            std::memcpy(data + vector.len_, vector.vals_, vector.len_ * sizeof(float));
        }
        return {};
    }

    DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override {
        auto sparse_vectors = query->GetSparseVectors();
        MaxHeap results(allocator_);
        for (int j = 0; j < datas_.size(); ++j) {
            auto distance = get_distance(sparse_vectors[0].len_,
                                         sparse_vectors[0].ids_,
                                         sparse_vectors[0].vals_,
                                         datas_[j][0],
                                         datas_[j] + 1,
                                         (float*)datas_[j] + 1 + datas_[j][0]);
            auto id = label_table_.GetIdByLabel(results.top().second);
            if (filter->CheckValid(id)) {
                results.push({distance, id});
                if (results.size() > k) {
                    results.pop();
                }
            }
        }
        // return result
        auto result = Dataset::Make();

        while (results.size() > k) {
            results.pop();
        }

        if (results.empty()) {
            result->Dim(0)->NumElements(1);
            return result;
        }

        result->Dim(static_cast<int64_t>(results.size()))->NumElements(1)->Owner(true, allocator_);

        auto* ids = (int64_t*)allocator_->Allocate(sizeof(int64_t) * results.size());
        result->Ids(ids);
        auto* dists = (float*)allocator_->Allocate(sizeof(float) * results.size());
        result->Distances(dists);

        for (auto j = static_cast<int64_t>(results.size() - 1); j >= 0; --j) {
            dists[j] = results.top().first;
            ids[j] = label_table_.GetLabelById(results.top().second);
            results.pop();
        }
        return result;
    }

    DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override {
        auto sparse_vectors = query->GetSparseVectors();
        MaxHeap results(allocator_);
        for (int j = 0; j < datas_.size(); ++j) {
            auto distance = get_distance(sparse_vectors[0].len_,
                                         sparse_vectors[0].ids_,
                                         sparse_vectors[0].vals_,
                                         datas_[j][0],
                                         datas_[j] + 1,
                                         (float*)datas_[j] + 1 + datas_[j][0]);
            auto id = label_table_.GetIdByLabel(results.top().second);
            if (filter->CheckValid(id) && distance < radius) {
                results.push({distance, id});
            }
        }

        // return result
        auto result = Dataset::Make();
        if (results.empty()) {
            result->Dim(0)->NumElements(1);
            return result;
        }

        result->Dim(static_cast<int64_t>(results.size()))->NumElements(1)->Owner(true, allocator_);

        auto* ids = (int64_t*)allocator_->Allocate(sizeof(int64_t) * results.size());
        result->Ids(ids);
        auto* dists = (float*)allocator_->Allocate(sizeof(float) * results.size());
        result->Distances(dists);

        for (auto j = static_cast<int64_t>(results.size() - 1); j >= 0; --j) {
            dists[j] = results.top().first;
            ids[j] = results.top().second;
            results.pop();
        }
        return result;
    }

    void
    Serialize(StreamWriter& writer) const override {
        size_t data_num = datas_.size();
        StreamWriter::WriteObj(writer, data_num);
        for (int i = 0; i < data_num; ++i) {
            uint32_t len = datas_[i][0];
            writer.Write((char*)datas_[i], (2 * len + 1) * sizeof(uint32_t));
        }
        label_table_.Serialize(writer);
    }

    void
    Deserialize(StreamReader& reader) override {
        size_t data_num;
        StreamReader::ReadObj(reader, data_num);
        datas_.resize(data_num);
        for (int i = 0; i < data_num; ++i) {
            uint32_t len;
            StreamReader::ReadObj(reader, len);
            datas_[i] = (uint32_t*)allocator_->Allocate((2 * len + 1) * sizeof(uint32_t));
            datas_[i][0] = len;
            reader.Read((char*)(datas_[i] + 1), 2 * len * sizeof(uint32_t));
        }
        label_table_.Deserialize(reader);
    }

    int64_t
    GetNumElements() const override {
        return 0;
    }

    DatasetPtr
    CalDistanceById(const float* query, const int64_t* ids, int64_t count) const override {
        return 0;
    }

private:
    Vector<uint32_t*> datas_;
    LabelTable label_table_;
};

}  // namespace vsag
