
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
#include "brute_force_parameter.h"
#include "faiss/faiss/Index.h"
#include "faiss/faiss/impl/io.h"
#include "faiss_parameter.h"
#include "impl/label_table.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "vsag/filter.h"

namespace vsag {

class SafeThreadPool;

DEFINE_POINTER(FlattenInterface);
// Faiss index was introduced since v0.13
class Faiss : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    explicit Faiss(const FaissParameterPtr& param, const IndexCommonParam& common_param);

    explicit Faiss(const ParamPtr& param, const IndexCommonParam& common_param)
        : Faiss(std::dynamic_pointer_cast<FaissParameter>(param), common_param){};

    ~Faiss() override = default;

    std::vector<int64_t>
    Add(const DatasetPtr& data) override;

    std::vector<int64_t>
    Build(const DatasetPtr& data) override;

    void
    Deserialize(StreamReader& reader) override;

    [[nodiscard]] IndexType
    GetIndexType() override {
        return IndexType::FAISS;
    }

    [[nodiscard]] std::string
    GetName() const override {
        return INDEX_FAISS;
    }

    [[nodiscard]] int64_t
    GetNumElements() const override {
        return this->index_->ntotal;
    }

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

    void
    Serialize(StreamWriter& writer) const override;

    void
    Serialize(std::ostream& out_stream) const override;

    void
    Deserialize(std::istream& in_stream) override;

    void
    Train(const DatasetPtr& data) override;

private:
    uint64_t total_count_{0};

    std::string faiss_string_{};

    faiss::Index* index_{nullptr};

    mutable std::shared_mutex global_mutex_;
    mutable std::shared_mutex add_mutex_;

    std::atomic<InnerIdType> max_capacity_{0};

    std::string index_path_{};
};
}  // namespace vsag
