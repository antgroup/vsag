
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
#include "io/basic_io.h"
#include "io/memory_block_io.h"
#include "vsag/dataset.h"

namespace vsag {

template <typename QuantTmpl, typename IOTmpl>
class SparseVectorDataCell : public FlattenInterface {
public:
    SparseVectorDataCell() = default;

    SparseVectorDataCell(const QuantizerParamPtr& quantization_param,
                         const IOParamPtr& io_param,
                         const IndexCommonParam& common_param);

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          Allocator* allocator = nullptr) override {
        auto comp = std::static_pointer_cast<Computer<QuantTmpl>>(computer);
        this->query(result_dists, comp, idx, id_count);
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        return this->factory_computer((const float*)query);
    }

    bool
    Decode(const uint8_t* codes, DataType* vector) override {
        // TODO(inabao): Implement the decode function
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Decode function is not implemented for SparseVectorDataCell");
    }

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override;

    void
    Train(const void* data, uint64_t count) override;

    void
    InsertVector(const void* vector, InnerIdType idx) override;

    void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec) override;

    void
    Resize(InnerIdType new_capacity) override {
        if (new_capacity <= this->max_capacity_) {
            return;
        }
        size_t io_size = (new_capacity - total_count_) * max_code_size_ + current_offset_;
        this->max_capacity_ = new_capacity;
        uint8_t end_flag =
            127;  // the value is meaingless, only to occupy the position for io allocate
        this->io_->Write(&end_flag, 1, io_size);
        this->offset_io_->Write(&end_flag, 1, new_capacity * sizeof(uint32_t));
    }

    void
    Prefetch(InnerIdType id) override{};

    void
    ExportModel(const FlattenInterfacePtr& other) const override {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        this->quantizer_->Serialize(writer);
        ss.seekg(0, std::ios::beg);
        IOStreamReader reader(ss);
        auto ptr = std::dynamic_pointer_cast<FlattenDataCell<QuantTmpl, IOTmpl>>(other);
        if (ptr == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Export model's sparse flatten datacell failed");
        }
        ptr->quantizer_->Deserialize(reader);
    }

    [[nodiscard]] std::string
    GetQuantizerName() override;

    [[nodiscard]] MetricType
    GetMetricType() override;

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override;

    [[nodiscard]] bool
    InMemory() const override;

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

    inline void
    SetQuantizer(std::shared_ptr<Quantizer<QuantTmpl>> quantizer) {
        this->quantizer_ = quantizer;
    }

    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>> io) {
        this->io_ = io;
    }

private:
    inline void
    query(float* result_dists,
          const std::shared_ptr<Computer<QuantTmpl>>& computer,
          const InnerIdType* idx,
          InnerIdType id_count);

    ComputerInterfacePtr
    factory_computer(const float* query) {
        auto computer = this->quantizer_->FactoryComputer();
        computer->SetQuery(query);
        return computer;
    }

    std::shared_ptr<Quantizer<QuantTmpl>> quantizer_{nullptr};
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    Allocator* const allocator_{nullptr};
    std::shared_ptr<MemoryBlockIO> offset_io_{nullptr};
    uint32_t current_offset_{0};
    uint64_t max_code_size_{0};
    std::mutex current_offset_mutex_;
};

}  // namespace vsag

#include "sparse_vector_datacell.inl"
