
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
#include "data_cell/sparse_term_datacell.h"

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

    IndexType
    GetIndexType() override {
        return IndexType::SINDI;
    }

    int64_t
    GetNumElements() const override {
        return cur_element_count_;
    }

private:
    template <InnerSearchMode mode>
    DatasetPtr
    search_impl(const SparseTermComputerPtr& computer, const InnerSearchParam& inner_param) const;

private:
    mutable std::shared_mutex global_mutex_;

    uint32_t window_size_{0};

    Vector<SparseTermDataCellPtr> window_term_list_;

    int64_t cur_element_count_{0};

    bool use_reorder_{false};

    float doc_retain_ratio_{0};

    std::shared_ptr<SparseIndex> rerank_flat_index_{nullptr};
    bool deserialize_without_footer_{false};
};

}  // namespace vsag
