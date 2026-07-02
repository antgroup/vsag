
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
#include "io/common/basic_io.h"
#include "io/memory_block_io/memory_block_io.h"
#include "quantization/multi_vector_computer.h"
#include "vsag/dataset.h"

namespace vsag {

template <typename QuantTmpl, typename IOTmpl>
class MultiVectorDataCell : public FlattenInterface {
public:
    MultiVectorDataCell() = default;

    MultiVectorDataCell(const QuantizerParamPtr& quantization_param,
                        const IOParamPtr& io_param,
                        const IndexCommonParam& common_param);

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) override;

    ComputerInterfacePtr
    FactoryComputer(const void* query) override;

    bool
    Decode(const uint8_t* codes, float* vector) override {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Decode is not supported for MultiVectorDataCell");
    }

    bool
    Encode(const float* vector, uint8_t* codes) override {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Encode is not supported for MultiVectorDataCell");
    }

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "ComputePairVectors is not supported for MultiVectorDataCell");
    }

    void
    Train(const void* data, uint64_t count) override;

    void
    InsertVector(const void* vector, InnerIdType idx) override;

    void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec) override;

    void
    Resize(InnerIdType new_capacity) override;

    void
    Prefetch(InnerIdType id) override {
    }

    void
    ExportModel(const FlattenInterfacePtr& other) const override {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "ExportModel is not supported for MultiVectorDataCell");
    }

    [[nodiscard]] std::string
    GetQuantizerName() override;

    [[nodiscard]] MetricType
    GetMetricType() override;

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override;

    void
    Release(const uint8_t* data) const override;

    [[nodiscard]] bool
    InMemory() const override;

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "fixed-length GetCodesById is not supported for MultiVectorDataCell");
    }

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

    int64_t
    GetMemoryUsage() const override;

private:
    std::shared_ptr<Quantizer<QuantTmpl>> quantizer_{nullptr};
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    Allocator* const allocator_{nullptr};
    std::shared_ptr<MemoryBlockIO> offset_io_{nullptr};
    uint64_t current_offset_{0};
    std::mutex current_offset_mutex_;

    uint32_t multi_vector_dim_{0};
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
};

}  // namespace vsag

#include "multi_vector_datacell.inl"
