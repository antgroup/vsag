
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

#include <algorithm>
#include <cstring>

#include "common.h"
#include "multi_vector_datacell.h"
#include "utils/byte_buffer.h"
#include "vsag/options.h"

namespace vsag {

template <typename QuantTmpl, typename IOTmpl>
MultiVectorDataCell<QuantTmpl, IOTmpl>::MultiVectorDataCell(
    const QuantizerParamPtr& quantization_param,
    const IOParamPtr& io_param,
    const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()),
      multi_vector_dim_(static_cast<uint32_t>(common_param.dim_)),
      metric_(common_param.metric_) {
    this->quantizer_ = std::make_shared<QuantTmpl>(quantization_param, common_param);
    this->io_ = std::make_shared<IOTmpl>(io_param, common_param);
    this->offset_io_ =
        std::make_shared<MemoryBlockIO>(Options::Instance().block_size_limit(), allocator_);
    this->max_capacity_ = 0;
    this->code_size_ = 0;
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::Train(const void* data, uint64_t count) {
    this->quantizer_->Train(static_cast<const float*>(data), count);
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::InsertVector(const void* vector, InnerIdType idx) {
    CHECK_ARGUMENT(vector != nullptr, "multi-vector data is nullptr");
    const MultiVector* multi_vector = static_cast<const MultiVector*>(vector);
    CHECK_ARGUMENT(multi_vector->len_ > 0, "multi-vector token count must be greater than 0");
    CHECK_ARGUMENT(multi_vector->vectors_ != nullptr, "multi-vector tokens are nullptr");
    CHECK_ARGUMENT(multi_vector_dim_ > 0, "multi-vector dim must be greater than 0");

    {
        std::lock_guard lock(mutex_);
        if (idx == std::numeric_limits<InnerIdType>::max()) {
            idx = total_count_;
            ++total_count_;
        } else {
            total_count_ = std::max(total_count_, idx + 1);
        }
    }

    const uint64_t vector_bytes = static_cast<uint64_t>(multi_vector->len_) *
                                  static_cast<uint64_t>(multi_vector_dim_) * sizeof(float);
    const uint64_t code_size = sizeof(uint32_t) + vector_bytes;
    ByteBuffer codes(code_size, allocator_);
    std::memcpy(codes.data, &multi_vector->len_, sizeof(uint32_t));
    std::memcpy(codes.data + sizeof(uint32_t), multi_vector->vectors_, vector_bytes);

    uint64_t old_offset = 0;
    {
        std::lock_guard lock(current_offset_mutex_);
        old_offset = current_offset_;
        current_offset_ += code_size;
    }
    offset_io_->Write(reinterpret_cast<const uint8_t*>(&old_offset),
                      sizeof(old_offset),
                      static_cast<uint64_t>(idx) * sizeof(old_offset));
    io_->Write(codes.data, code_size, old_offset);
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::BatchInsertVector(const void* vectors,
                                                          InnerIdType count,
                                                          InnerIdType* idx_vec) {
    CHECK_ARGUMENT(vectors != nullptr, "multi-vector array is nullptr");
    const MultiVector* multi_vectors = static_cast<const MultiVector*>(vectors);
    Vector<InnerIdType> reserved_idx(count, allocator_);
    if (idx_vec == nullptr) {
        idx_vec = reserved_idx.data();
        {
            std::lock_guard lock(mutex_);
            for (InnerIdType i = 0; i < count; ++i) {
                idx_vec[i] = total_count_ + i;
            }
            total_count_ += count;
        }
    }
    for (InnerIdType i = 0; i < count; ++i) {
        this->InsertVector(multi_vectors + i, idx_vec[i]);
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::Resize(InnerIdType new_capacity) {
    if (new_capacity <= this->max_capacity_) {
        return;
    }
    this->offset_io_->Resize(static_cast<uint64_t>(new_capacity) * sizeof(uint64_t));
    this->max_capacity_ = new_capacity;
}

template <typename QuantTmpl, typename IOTmpl>
std::string
MultiVectorDataCell<QuantTmpl, IOTmpl>::GetQuantizerName() {
    return this->quantizer_->Name();
}

template <typename QuantTmpl, typename IOTmpl>
MetricType
MultiVectorDataCell<QuantTmpl, IOTmpl>::GetMetricType() {
    return this->metric_;
}

template <typename QuantTmpl, typename IOTmpl>
const uint8_t*
MultiVectorDataCell<QuantTmpl, IOTmpl>::GetCodesById(InnerIdType id, bool& need_release) const {
    uint64_t offset = 0;
    offset_io_->Read(sizeof(offset), static_cast<uint64_t>(id) * sizeof(offset), (uint8_t*)&offset);
    uint32_t len = 0;
    io_->Read(sizeof(len), offset, (uint8_t*)&len);
    uint64_t read_size =
        sizeof(uint32_t) + static_cast<uint64_t>(len) * multi_vector_dim_ * sizeof(float);
    auto* codes = static_cast<uint8_t*>(allocator_->Allocate(read_size));
    io_->Read(read_size, offset, codes);
    need_release = true;
    return codes;
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::Release(const uint8_t* data) const {
    allocator_->Deallocate(const_cast<uint8_t*>(data));
}

template <typename QuantTmpl, typename IOTmpl>
bool
MultiVectorDataCell<QuantTmpl, IOTmpl>::InMemory() const {
    return FlattenInterface::InMemory();
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::Serialize(StreamWriter& writer) {
    FlattenInterface::Serialize(writer);
    StreamWriter::WriteObj(writer, multi_vector_dim_);
    StreamWriter::WriteObj(writer, current_offset_);
    this->offset_io_->Serialize(writer);
    this->io_->Serialize(writer);
    this->quantizer_->Serialize(writer);
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    FlattenInterface::Deserialize(reader);
    StreamReader::ReadObj(reader, multi_vector_dim_);
    StreamReader::ReadObj(reader, current_offset_);
    this->offset_io_->Deserialize(reader);
    this->io_->Deserialize(reader);
    this->quantizer_->Deserialize(reader);
}

template <typename QuantTmpl, typename IOTmpl>
ComputerInterfacePtr
MultiVectorDataCell<QuantTmpl, IOTmpl>::FactoryComputer(const void* query) {
    CHECK_ARGUMENT(query != nullptr, "query is nullptr");
    const MultiVector* multi_vector = static_cast<const MultiVector*>(query);
    CHECK_ARGUMENT(multi_vector->len_ > 0, "query token count must be greater than 0");
    CHECK_ARGUMENT(multi_vector->vectors_ != nullptr, "query vectors are nullptr");

    auto computer = std::make_shared<MultiVectorComputer>(multi_vector_dim_, metric_, allocator_);
    computer->SetQuery(multi_vector->vectors_, multi_vector->len_);
    return computer;
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::Query(float* result_dists,
                                              const ComputerInterfacePtr& computer,
                                              const InnerIdType* idx,
                                              InnerIdType id_count,
                                              QueryContext* ctx) {
    auto* mv_computer = dynamic_cast<MultiVectorComputer*>(computer.get());
    CHECK_ARGUMENT(mv_computer != nullptr, "computer is not a MultiVectorComputer");

    if (id_count == 0) {
        return;
    }

    // Step 1: Read all offsets (offset_io_ is MemoryBlockIO, in-memory, fast)
    std::vector<uint64_t> offsets(id_count);
    for (InnerIdType i = 0; i < id_count; ++i) {
        bool ok = offset_io_->Read(sizeof(uint64_t),
                                   static_cast<uint64_t>(idx[i]) * sizeof(uint64_t),
                                   reinterpret_cast<uint8_t*>(&offsets[i]));
        CHECK_ARGUMENT(ok, "MultiVectorDataCell: failed to read offset");
    }

    // Step 2: Batch read all token counts via MultiRead (async IO)
    std::vector<uint32_t> lens(id_count);
    std::vector<uint64_t> len_sizes(id_count, sizeof(uint32_t));
    if (!this->io_->MultiRead(reinterpret_cast<uint8_t*>(lens.data()),
                              len_sizes.data(),
                              offsets.data(),
                              static_cast<uint64_t>(id_count))) {
        throw VsagException(ErrorType::READ_ERROR,
                            "MultiVectorDataCell: failed to read token counts");
    }

    // Step 3: Batch read all data via MultiRead (async IO)
    std::vector<uint64_t> data_sizes(id_count);
    uint64_t total_size = 0;
    for (InnerIdType i = 0; i < id_count; ++i) {
        data_sizes[i] = sizeof(uint32_t) +
                        static_cast<uint64_t>(lens[i]) * multi_vector_dim_ * sizeof(float);
        total_size += data_sizes[i];
    }
    ByteBuffer all_codes(total_size, this->allocator_);
    if (!this->io_->MultiRead(all_codes.data,
                              data_sizes.data(),
                              offsets.data(),
                              static_cast<uint64_t>(id_count))) {
        throw VsagException(ErrorType::READ_ERROR,
                            "MultiVectorDataCell: failed to read token data");
    }

    // Step 4: Compute MaxSim distances
    uint64_t cursor = 0;
    for (InnerIdType i = 0; i < id_count; ++i) {
        uint32_t token_count = lens[i];
        mv_computer->ComputeDist(
            all_codes.data + cursor + sizeof(uint32_t), token_count, result_dists + i);
        cursor += data_sizes[i];
    }
}

template <typename QuantTmpl, typename IOTmpl>
uint64_t
MultiVectorDataCell<QuantTmpl, IOTmpl>::GetMemoryUsage() const {
    uint64_t memory = sizeof(MultiVectorDataCell<QuantTmpl, IOTmpl>);
    memory += this->offset_io_->size_;
    if (IOTmpl::InMemory) {
        memory += this->io_->GetMemoryUsage();
    }
    memory += sizeof(QuantTmpl);
    return memory;
}

}  // namespace vsag
