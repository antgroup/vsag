
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

#include "algorithm/inner_index_interface.h"
#include "impl/heap/distance_heap.h"
#include "sparse_index_parameters.h"

namespace vsag {

/**
 * @brief SparseIndex: a flat brute-force index for sparse vectors.
 *
 * Stores raw sparse vectors (dimension ids as uint32_t, values as float
 * bit-cast into uint32_t) and performs exhaustive linear scan for KNN / range queries.
 * Suitable as a reranking back-end for SINDI or for small sparse datasets.
 *
 * Distance metric: inner product (IP).
 */
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
    CalDistanceById(const DatasetPtr& query,
                    const int64_t* ids,
                    int64_t count,
                    bool calculate_precise_distance = true) const override;

    float
    CalcDistanceById(const DatasetPtr& vector,
                     int64_t id,
                     bool calculate_precise_distance = true) const override;

    void
    Deserialize(StreamReader& reader) override;

    InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return std::make_shared<SparseIndex>(this->create_param_ptr_, param);
    }

    void
    GetSparseVectorByInnerId(InnerIdType inner_id,
                             SparseVector* data,
                             Allocator* specified_allocator) const override;

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

    /**
     * @brief Compute the IP distance between a pre-sorted query and a stored
     *        vector without acquiring any lock.
     *
     * The caller must guarantee that the index is not mutated concurrently
     * and that inner_id < cur_element_count_.
     *
     * @param sorted_ids  sorted dimension ids of the query (uint32).
     * @param sorted_vals corresponding values, parallel to sorted_ids.
     * @param inner_id    internal id of the target vector.
     * @return inner-product distance.
     */
    float
    CalDistanceByIdUnsafe(Vector<uint32_t>& sorted_ids,
                          Vector<float>& sorted_vals,
                          uint32_t inner_id) const;

    int64_t
    GetMemoryUsage() const override;

    /**
     * @brief Pack a distance heap into a Dataset with (id, distance) pairs.
     */
    DatasetPtr
    collect_results(const DistHeapPtr& results) const;

    /**
     * @brief Return a sorted copy of the given sparse vector by dimension id.
     */
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
    Vector<uint32_t*> datas_;       // raw sparse vectors; each entry encodes
                                    // [dim_count, id0, val0_as_uint32, id1, val1_as_uint32, ...]
    bool need_sort_;                // true if stored vectors may be unsorted by dim id
    int64_t cur_element_count_{0};  // number of valid entries in datas_
    int64_t max_capacity_{0};       // allocated capacity of datas_
};

}  // namespace vsag
