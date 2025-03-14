
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
          InnerIdType id_count) override {
        auto comp = std::static_pointer_cast<Computer<QuantTmpl>>(computer);
        this->query(result_dists, comp, idx, id_count);
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        return this->factory_computer((const float*)query);
    }

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override;

    void
    Train(const void* data, uint64_t count) override;

    void
    InsertVector(const void* vector, InnerIdType idx) override;

    void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx) override;

    void
    SetMaxCapacity(InnerIdType capacity) override {
        this->max_capacity_ = std::max(capacity, this->total_count_);
    }

    void
    Resize(InnerIdType new_capacity) override {
        if (new_capacity <= this->max_capacity_) {
            return;
        }
        this->max_capacity_ = new_capacity;
        uint64_t io_size = new_capacity * code_size_;
        uint8_t end_flag =
            127;  // the value is meaningless, only to occupy the position for io allocate
        this->io_->Write(&end_flag, 1, io_size);
    }

    void
    Prefetch(InnerIdType id) override{};

    [[nodiscard]] std::string
    GetQuantizerName() override;

    [[nodiscard]] MetricType
    GetMetricType() override;

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override;

    [[nodiscard]] bool
    InMemory() const override;

    void
    EnableForceInMemory() override {
    }

    void
    DisableForceInMemory() override {
    }

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(StreamReader& reader) override;

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
};

template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::query(float* result_dists,
                                               const std::shared_ptr<Computer<QuantTmpl>>& computer,
                                               const InnerIdType* idx,
                                               InnerIdType id_count) {
    for (int i = 0; i < id_count; ++i) {
        bool need_release;
        auto codes = this->GetCodesById(idx[i], need_release);
        result_dists[i] = this->quantizer_->Compute(computer->buf_, codes);
        allocator_->Deallocate((void*)codes);
    }
}
template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::Deserialize(StreamReader& reader) {
    FlattenInterface::Deserialize(reader);
    StreamReader::ReadObj(reader, current_offset_);
    this->io_->Deserialize(reader);
    this->offset_io_->Deserialize(reader);
    this->quantizer_->Deserialize(reader);
}

template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::Serialize(StreamWriter& writer) {
    FlattenInterface::Serialize(writer);
    StreamWriter::WriteObj(writer, current_offset_);
    this->io_->Serialize(writer);
    this->offset_io_->Serialize(writer);
    this->quantizer_->Serialize(writer);
}

template <typename QuantTmpl, typename IOTmpl>
bool
SparseVectorDataCell<QuantTmpl, IOTmpl>::GetCodesById(InnerIdType id, uint8_t* codes) const {
    throw VsagException(
        ErrorType::INTERNAL_ERROR,
        "no implement in SparseVectorDataCell for GetCodesById without need_release");
}

template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::BatchInsertVector(const void* vectors,
                                                           InnerIdType count,
                                                           InnerIdType* idx) {
    const auto* sparse_array = reinterpret_cast<const SparseVector*>(vectors);
    Vector<InnerIdType> idx_ptr(count, allocator_);
    if (idx == nullptr) {
        idx = idx_ptr.data();
        for (InnerIdType i = 0; i < count; ++i) {
            idx[i] = total_count_ + i;
        }
    }
    for (InnerIdType i = 0; i < count; ++i) {
        this->InsertVector(sparse_array + i, idx[i]);
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::InsertVector(const void* vector, InnerIdType idx) {
    if (idx == std::numeric_limits<InnerIdType>::max()) {
        idx = total_count_;
        ++total_count_;
    } else {
        total_count_ = std::max(total_count_, idx + 1);
    }
    auto sparse_vector = (const SparseVector*)vector;
    size_t code_size = (sparse_vector->len_ * 2 + 1) * sizeof(uint32_t);
    auto* codes = reinterpret_cast<uint8_t*>(allocator_->Allocate(code_size));
    quantizer_->EncodeOne((const float*)vector, codes);
    io_->Write(codes, code_size, current_offset_);
    offset_io_->Write(
        (uint8_t*)&current_offset_, sizeof(current_offset_), idx * sizeof(current_offset_));
    current_offset_ += code_size;
    allocator_->Deallocate(codes);
}

template <typename QuantTmpl, typename IOTmpl>
bool
SparseVectorDataCell<QuantTmpl, IOTmpl>::InMemory() const {
    return FlattenInterface::InMemory();
}

template <typename QuantTmpl, typename IOTmpl>
const uint8_t*
SparseVectorDataCell<QuantTmpl, IOTmpl>::GetCodesById(InnerIdType id, bool& need_release) const {
    uint32_t offset;
    offset_io_->Read(sizeof(offset), id * sizeof(offset), (uint8_t*)&offset);
    uint32_t length;
    io_->Read(sizeof(length), offset, (uint8_t*)&length);
    need_release = true;
    size_t read_size = sizeof(uint32_t) * (2 * length + 1);
    auto* codes = (uint8_t*)allocator_->Allocate(read_size);
    io_->Read(read_size, offset, codes);
    return codes;
}

template <typename QuantTmpl, typename IOTmpl>
MetricType
SparseVectorDataCell<QuantTmpl, IOTmpl>::GetMetricType() {
    return this->quantizer_->Metric();
}

template <typename QuantTmpl, typename IOTmpl>
std::string
SparseVectorDataCell<QuantTmpl, IOTmpl>::GetQuantizerName() {
    return this->quantizer_->Name();
}

template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::Train(const void* data, uint64_t count) {
    this->quantizer_->Train((const float*)data, count);
}

template <typename QuantTmpl, typename IOTmpl>
float
SparseVectorDataCell<QuantTmpl, IOTmpl>::ComputePairVectors(InnerIdType id1, InnerIdType id2) {
    bool release1, release2;
    auto* codes1 = this->GetCodesById(id1, release1);
    auto* codes2 = this->GetCodesById(id2, release2);
    auto result = this->quantizer_->Compute(codes1, codes2);
    allocator_->Deallocate((void*)codes1);
    allocator_->Deallocate((void*)codes2);
    return result;
}

template <typename QuantTmpl, typename IOTmpl>
SparseVectorDataCell<QuantTmpl, IOTmpl>::SparseVectorDataCell(
    const QuantizerParamPtr& quantization_param,
    const IOParamPtr& io_param,
    const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()) {
    this->quantizer_ = std::make_shared<QuantTmpl>(quantization_param, common_param);
    this->io_ = std::make_shared<IOTmpl>(io_param, common_param);
    this->offset_io_ =
        std::make_shared<MemoryBlockIO>(allocator_, Options::Instance().block_size_limit());
}

}  // namespace vsag
