
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

#include "inner_index_interface.h"
#include "sparse_index_parameters.h"

namespace vsag {

class SparseIndex : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    explicit SparseIndex(const SparseIndexParameterPtr& param, const IndexCommonParam& common_param)
        : InnerIndexInterface(param, common_param), datas_(common_param.allocator_.get()) {
    }

    SparseIndex(const ParamPtr& param, const IndexCommonParam& common_param)
        : SparseIndex(std::dynamic_pointer_cast<SparseIndexParameters>(param), common_param){};

    ~SparseIndex() override {
        for (auto& data : datas_) {
            allocator_->Deallocate(data);
        }
    }

    [[nodiscard]] std::string
    GetName() const override {
        return INDEX_SPARSE;
    }

    std::vector<int64_t>
    Add(const DatasetPtr& base) override;

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
    Serialize(StreamWriter& writer) const override {
        size_t data_num = datas_.size();
        StreamWriter::WriteObj(writer, data_num);
        for (int i = 0; i < data_num; ++i) {
            uint32_t len = datas_[i][0];
            writer.Write((char*)datas_[i], (2 * len + 1) * sizeof(uint32_t));
        }
        label_table_->Serialize(writer);
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
        label_table_->Deserialize(reader);
    }

    int64_t
    GetNumElements() const override {
        return datas_.size();
    }

    DatasetPtr
    CalDistanceById(const float* query, const int64_t* ids, int64_t count) const override {
        throw VsagException(vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "no support CalDistanceById in " + GetName());
    }

    void
    InitFeatures() override {
    }

private:
    DatasetPtr
    collect_results(MaxHeap& results) const;

    std::tuple<Vector<uint32_t>, Vector<float>>
    sort_sparse_vector(const SparseVector& vector) const;

private:
    Vector<uint32_t*> datas_;
};

}  // namespace vsag
