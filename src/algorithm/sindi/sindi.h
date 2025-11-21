
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
#include "algorithm/sparse_index.h"
#include "datacell/sparse_term_datacell.h"

namespace vsag {

class SINDI : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

    explicit SINDI(const SINDIParameterPtr& param, const IndexCommonParam& common_param);

    SINDI(const ParamPtr& param, const IndexCommonParam& common_param)
        : SINDI(std::dynamic_pointer_cast<SINDIParameter>(param), common_param){};

    ~SINDI() = default;

    std::string
    GetName() const override {
        return "sindi";
    }

    void
    InitFeatures() override;

    std::string
    GetMemoryUsageDetail() const override {
        return "";
    }

    std::vector<int64_t>
    Add(const DatasetPtr& base) override;

    std::vector<int64_t>
    Build(const DatasetPtr& base) override;

    DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    DatasetPtr
    KnnSearch(const vsag::DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const vsag::FilterPtr& filter,
              vsag::Allocator* allocator) const override;

    DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return nullptr;
    };

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    void
    GetSparseVectorByInnerId(InnerIdType inner_id, SparseVector* data) const override;

    IndexType
    GetIndexType() const override {
        return IndexType::SINDI;
    }

    int64_t
    GetNumElements() const override {
        return cur_element_count_;
    }

    [[nodiscard]] uint64_t
    EstimateMemory(uint64_t num_elements) const override;

    float
    CalcDistanceById(const DatasetPtr& vector, int64_t id) const override;

    DatasetPtr
    CalDistanceById(const DatasetPtr& query, const int64_t* ids, int64_t count) const override;

    bool
    UpdateId(int64_t old_id, int64_t new_id) override;

    std::pair<int64_t, int64_t>
    GetMinAndMaxId() const override;

    void
    SetImmutable() override;

private:
    template <InnerSearchMode mode>
    DatasetPtr
    search_impl(const SparseTermComputerPtr& computer,
                const InnerSearchParam& inner_param,
                Allocator* allocator) const;

private:
    mutable std::shared_mutex global_mutex_;

    uint32_t term_id_limit_{0};

    uint32_t window_size_{0};

    Vector<SparseTermDataCellPtr> window_term_list_;

    int64_t cur_element_count_{0};

    bool use_reorder_{false};

    float doc_retain_ratio_{0};

    std::shared_ptr<SparseIndex> rerank_flat_index_{nullptr};
    bool deserialize_without_footer_{false};
};

}  // namespace vsag
