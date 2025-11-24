
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

#include "impl/heap/distance_heap.h"
#include "inner_index_interface.h"
#include "sparse_index_parameters.h"

namespace vsag {

class SparseIndex : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    explicit SparseIndex(const SparseIndexParameterPtr& param,
                         const IndexCommonParam& common_param);

    SparseIndex(const ParamPtr& param, const IndexCommonParam& common_param);

    ~SparseIndex() override;

    std::vector<int64_t>
    Add(const DatasetPtr& base) override;

    DatasetPtr
    CalDistanceById(const DatasetPtr& query, const int64_t* ids, int64_t count) const override;

    float
    CalcDistanceById(const DatasetPtr& vector, int64_t id) const override;

    void
    Deserialize(StreamReader& reader) override;

    InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return std::make_shared<SparseIndex>(this->create_param_ptr_, param);
    }

    void
    GetSparseVectorByInnerId(InnerIdType inner_id, SparseVector* data) const override;

    IndexType
    GetIndexType() const override {
        return IndexType::SPARSE;
    }

    std::string
    GetName() const override {
        return INDEX_SPARSE;
    }

    int64_t
    GetNumElements() const override {
        return cur_element_count_;
    }

    DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    InitFeatures() override;

    float
    CalDistanceByIdUnsafe(Vector<uint32_t>& sorted_ids,
                          Vector<float>& sorted_vals,
                          uint32_t inner_id) const;

    DatasetPtr
    collect_results(const DistHeapPtr& results) const;

    std::tuple<Vector<uint32_t>, Vector<float>>
    sort_sparse_vector(const SparseVector& vector) const;

private:
    void
    resize(int64_t new_capacity) {
        if (new_capacity <= max_capacity_) {
            return;
        }
        datas_.resize(new_capacity);
        max_capacity_ = new_capacity;
    }

private:
    Vector<uint32_t*> datas_;
    bool need_sort_;
    int64_t cur_element_count_{0};
    int64_t max_capacity_{0};
};

}  // namespace vsag
