
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

#include "sparse_vector_datacell.h"

namespace vsag {
template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::query(float* result_dists,
                                               const std::shared_ptr<Computer<QuantTmpl>>& computer,
                                               const InnerIdType* idx,
                                               InnerIdType id_count) {
    for (int i = 0; i < id_count; ++i) {
        bool need_release{true};
        auto codes = this->GetCodesById(idx[i], need_release);
        result_dists[i] = this->quantizer_->ComputeDist(computer, codes);
        if (need_release) {
            allocator_->Deallocate((void*)codes);
        }
    }
}
template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
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
                                                           InnerIdType* idx_vec) {
    const auto* sparse_array = reinterpret_cast<const SparseVector*>(vectors);
    Vector<InnerIdType> idx_ptr(count, allocator_);
    if (idx_vec == nullptr) {
        idx_vec = idx_ptr.data();
        for (InnerIdType i = 0; i < count; ++i) {
            idx_vec[i] = total_count_ + i;
        }
    }
    for (InnerIdType i = 0; i < count; ++i) {
        this->InsertVector(sparse_array + i, idx_vec[i]);
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::InsertVector(const void* vector, InnerIdType idx) {
    {
        std::lock_guard lock(mutex_);
        total_count_ = std::max(total_count_, idx + 1);
    }
    auto sparse_vector = (const SparseVector*)vector;
    size_t code_size = (sparse_vector->len_ * 2 + 1) * sizeof(uint32_t);
    if (code_size > max_code_size_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, fmt::format("code size ({}) of sparse vector more than max code size ({})", code_size, max_code_size_));
    }
    auto* codes = reinterpret_cast<uint8_t*>(allocator_->Allocate(code_size));
    quantizer_->EncodeOne((const float*)vector, codes);
    uint32_t old_offset = 0;
    {
        std::lock_guard lock(current_offset_mutex_);
        old_offset = current_offset_;
        current_offset_ += code_size;
    }
    offset_io_->Write(
        (uint8_t*)&old_offset, sizeof(current_offset_), idx * sizeof(current_offset_));
    io_->Write(codes, code_size, old_offset);
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
    this->max_code_size_ = (this->quantizer_->GetDim() * 2 + 1) * sizeof(uint32_t);
    this->max_capacity_ = 0;
    this->code_size_ = this->quantizer_->GetCodeSize();
}

}
