
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
#include "utils/distance_heap.h"
#include "utils/standard_heap.h"

namespace vsag {

class SparseTermIndex : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

    explicit SparseTermIndex(const SparseTermIndexParameterPtr& param,
                             const IndexCommonParam& common_param);

    SparseTermIndex(const ParamPtr& param, const IndexCommonParam& common_param)
        : SparseTermIndex(std::dynamic_pointer_cast<SparseTermIndexParameters>(param),
                          common_param){};

    ~SparseTermIndex() {
    }

    std::string
    GetName() const override {
        return "sparse_term_index";
    }

    void
    InitFeatures() override {
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
                int64_t limited_size = -1) const override {
        return nullptr;
    }

    InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return nullptr;
    };

    void
    Serialize(StreamWriter& writer) const override {
        StreamWriter::WriteObj(writer, cur_element_count_);

        uint32_t window_term_list_size = window_term_list_.size();
        StreamWriter::WriteObj(writer, window_term_list_size);
        for (auto i = 0; i < window_term_list_.size(); i++) {
            window_term_list_[i]->Serialize(writer);
        }

        label_table_->Serialize(writer);

        if (use_reorder_) {
            rerank_flat_index_->Serialize(writer);
        }
    }

    void
    Deserialize(StreamReader& reader) override {
        StreamReader::ReadObj(reader, cur_element_count_);

        uint32_t window_term_list_size = 0;
        StreamReader::ReadObj(reader, window_term_list_size);
        window_term_list_.resize(window_term_list_size);
        for (auto i = 0; i < window_term_list_.size(); i++) {
            if (i != 0) {
                window_term_list_[i] = this->window_term_list_[0]->Clone();
            }
            window_term_list_[i]->Deserialize(reader);
        }

        label_table_->Deserialize(reader);

        if (use_reorder_) {
            rerank_flat_index_->Deserialize(reader);
        }
    }

    int64_t
    GetNumElements() const override {
        return cur_element_count_;
    }

private:
    uint32_t window_size_{0};

    Vector<SparseTermDataCellPtr> window_term_list_;

    int64_t cur_element_count_{0};

    bool use_reorder_{false};

    std::shared_ptr<SparseIndex> rerank_flat_index_{nullptr};
};

}  // namespace vsag
