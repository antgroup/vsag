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

#include "flatten_interface.h"
#include "index_common_param.h"
#include "quantization/sparse_quantization/sparse_dmq_quantizer.h"

namespace vsag {

class SparseDmqDataCell : public FlattenInterface {
public:
    SparseDmqDataCell(uint32_t term_id_limit, const IndexCommonParam& common_param);

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) override;

    ComputerInterfacePtr
    FactoryComputer(const void* query) override;

    void
    Train(const void* data, uint64_t count) override;

    void
    InsertVector(const void* vector,
                 InnerIdType idx = std::numeric_limits<InnerIdType>::max()) override;

    void
    BatchInsertVector(const void* vectors,
                      InnerIdType count,
                      InnerIdType* idx_vec = nullptr) override;

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override;

    void
    GetSparseVectorByInnerId(InnerIdType inner_id,
                             SparseVector* data,
                             Allocator* specified_allocator) const override;

    void
    Prefetch(InnerIdType id) override;

    [[nodiscard]] std::string
    GetQuantizerName() override;

    [[nodiscard]] MetricType
    GetMetricType() override;

    void
    Resize(InnerIdType capacity) override;

    void
    ExportModel(const FlattenInterfacePtr& other) const override;

    bool
    Decode(const uint8_t* codes, float* vector) override;

    bool
    Encode(const float* vector, uint8_t* codes) override;

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override;

    void
    Release(const uint8_t* data) const override;

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

    [[nodiscard]] uint64_t
    GetMemoryUsage() const override;

private:
    [[nodiscard]] const uint8_t*
    GetCode(InnerIdType id) const;

private:
    Allocator* allocator_{nullptr};
    std::shared_ptr<SparseDmqQuantizer> quantizer_;
    Vector<uint64_t> offsets_;
    Vector<uint8_t> codes_;
};

}  // namespace vsag
