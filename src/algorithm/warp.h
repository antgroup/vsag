
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
#include "impl/label_table.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "vsag/filter.h"
#include "warp_parameter.h"

namespace vsag {

class SafeThreadPool;

DEFINE_POINTER2(AttrInvertedInterface, AttributeInvertedInterface);
DEFINE_POINTER(FlattenInterface);

// WARP index for multi-vector documents with maxsin similarity (ColBERT-style retrieval)
class WARP : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    explicit WARP(const WarpParameterPtr& param, const IndexCommonParam& common_param);

    explicit WARP(const ParamPtr& param, const IndexCommonParam& common_param)
        : WARP(std::dynamic_pointer_cast<WarpParameter>(param), common_param){};

    ~WARP() override = default;

    std::vector<int64_t>
    Add(const DatasetPtr& data, AddMode mode = AddMode::DEFAULT) override;

    std::vector<int64_t>
    Build(const DatasetPtr& data) override;

    void
    Deserialize(StreamReader& reader) override;

    uint64_t
    EstimateMemory(uint64_t num_elements) const override;

    [[nodiscard]] InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return std::make_shared<WARP>(this->create_param_ptr_, param);
    }

    [[nodiscard]] IndexType
    GetIndexType() const override {
        return IndexType::WARP;
    }

    std::string
    GetName() const override {
        return INDEX_WARP;
    }

    [[nodiscard]] int64_t
    GetNumElements() const override {
        return this->total_count_ - this->delete_count_;
    }

    [[nodiscard]] int64_t
    GetNumberRemoved() const override {
        return this->delete_count_;
    }

    void
    GetVectorByInnerId(InnerIdType inner_id, float* data) const override;

    void
    InitFeatures() override;

    [[nodiscard]] DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    [[nodiscard]] DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    [[nodiscard]] DatasetPtr
    SearchWithRequest(const SearchRequest& request) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Train(const DatasetPtr& data) override;

    int64_t
    GetMemoryUsage() const override;

private:
    void
    resize(uint64_t new_size);

    void
    add_one_doc(const float* data, uint32_t vec_count, InnerIdType inner_id);

    void
    cal_memory_usage();

    // Compute maxsin similarity between query vectors and a document's vectors
    float
    compute_maxsin_similarity(const float* query_vectors,
                              uint32_t query_vec_count,
                              uint32_t doc_start_vec_idx,
                              uint32_t doc_vec_count) const;

private:
    FlattenInterfacePtr inner_codes_{nullptr};

    uint64_t total_count_{0};  // Number of documents (not vectors)

    uint64_t delete_count_{0};

    uint64_t total_vector_count_{0};  // Total number of vectors across all docs

    uint64_t resize_increase_count_bit_{DEFAULT_RESIZE_BIT};

    mutable std::shared_mutex global_mutex_;
    mutable std::shared_mutex add_mutex_;

    std::atomic<InnerIdType> max_capacity_{0};

    // Document offsets: size is total_count_ + 1
    // doc_offsets_[i] is the starting vector index of document i
    // doc_offsets_[total_count_] = total_vector_count_
    // Vector count for document i: doc_offsets_[i+1] - doc_offsets_[i]
    Vector<uint32_t> doc_offsets_;

    static constexpr uint64_t DEFAULT_RESIZE_BIT = 10;
};

}  // namespace vsag
