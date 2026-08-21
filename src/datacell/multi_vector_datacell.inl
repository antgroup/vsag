
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
#include <limits>

#include "common.h"
#include "multi_vector_datacell.h"
#include "utils/byte_buffer.h"
#include "vsag/options.h"

namespace vsag {

inline uint64_t
GetMultiVectorCodeSize(uint32_t token_count, uint32_t dimension) {
    constexpr uint64_t header_size = sizeof(uint32_t);
    constexpr uint64_t value_size = sizeof(float);
    const uint64_t values_per_token = static_cast<uint64_t>(dimension) * value_size;
    if (values_per_token != 0 &&
        token_count > (std::numeric_limits<uint64_t>::max() - header_size) / values_per_token) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "MultiVectorDataCell: record size overflow");
    }
    return header_size + static_cast<uint64_t>(token_count) * values_per_token;
}

template <typename QuantTmpl, typename IOTmpl>
MultiVectorDataCell<QuantTmpl, IOTmpl>::MultiVectorDataCell(
    const QuantizerParamPtr& quantization_param,
    const IOParamPtr& io_param,
    const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()),
      multi_vector_dim_(static_cast<uint32_t>(common_param.dim_)),
      metric_(common_param.metric_) {
    this->quantizer_ = std::make_shared<QuantTmpl>(quantization_param, common_param);
    this->backend_ =
        QuantizerDistanceBackend<QuantTmpl>::Get(static_cast<const QuantTmpl&>(*this->quantizer_));
    auto io = std::make_shared<IOTmpl>(io_param, common_param);
    auto offset_io =
        std::make_shared<MemoryBlockIO>(Options::Instance().block_size_limit(), allocator_);
    layout_.SetIO(std::move(offset_io), std::move(io));
    layout_.SetLocationPolicy(
        HeaderLengthLocationPolicy{static_cast<uint64_t>(multi_vector_dim_) * sizeof(float)});
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

    const uint64_t code_size = GetMultiVectorCodeSize(multi_vector->len_, multi_vector_dim_);
    const uint64_t vector_bytes = code_size - sizeof(uint32_t);
    ByteBuffer codes(code_size, allocator_);
    std::memcpy(codes.data, &multi_vector->len_, sizeof(uint32_t));
    std::memcpy(codes.data + sizeof(uint32_t), multi_vector->vectors_, vector_bytes);

    {
        std::lock_guard lock(mutex_);
        layout_.Write(idx, codes.data, code_size);
    }
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
    layout_.ResizeLocations(new_capacity);
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
    const uint64_t offset = layout_.ReadLocation(id);
    uint32_t len = 0;
    if (not layout_.Payload().Read(offset, sizeof(len), reinterpret_cast<uint8_t*>(&len))) {
        throw VsagException(ErrorType::READ_ERROR,
                            "MultiVectorDataCell: failed to read token count");
    }
    const uint64_t read_size = GetMultiVectorCodeSize(len, multi_vector_dim_);
    const uint64_t payload_size = layout_.Payload().GetByteSize();
    if (offset > payload_size || read_size > payload_size - offset) {
        throw VsagException(ErrorType::READ_ERROR,
                            "MultiVectorDataCell: token data range exceeds payload");
    }
    auto* codes = static_cast<uint8_t*>(allocator_->Allocate(read_size));
    if (not layout_.Payload().Read(offset, read_size, codes)) {
        allocator_->Deallocate(codes);
        throw VsagException(ErrorType::READ_ERROR,
                            "MultiVectorDataCell: failed to read token data");
    }
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
    StreamWriter::WriteObj(writer, layout_.GetNextOffset());
    layout_.Locations().Serialize(writer);
    layout_.Payload().Serialize(writer);
    this->quantizer_->Serialize(writer);
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    FlattenInterface::Deserialize(reader);
    StreamReader::ReadObj(reader, multi_vector_dim_);
    uint64_t current_offset = 0;
    StreamReader::ReadObj(reader, current_offset);
    layout_.SetNextOffset(current_offset);
    layout_.Locations().Deserialize(reader);
    layout_.Payload().Deserialize(reader);
    this->quantizer_->Deserialize(reader);
    this->backend_ =
        QuantizerDistanceBackend<QuantTmpl>::Get(static_cast<const QuantTmpl&>(*this->quantizer_));
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
        offsets[i] = layout_.ReadLocation(idx[i]);
    }

    // Step 2: Batch read all token counts via MultiRead (async IO)
    std::vector<uint32_t> lens(id_count);
    std::vector<uint64_t> len_sizes(id_count, sizeof(uint32_t));
    if (!layout_.Payload().MultiRead(offsets.data(),
                                     len_sizes.data(),
                                     static_cast<uint64_t>(id_count),
                                     reinterpret_cast<uint8_t*>(lens.data()))) {
        throw VsagException(ErrorType::READ_ERROR,
                            "MultiVectorDataCell: failed to read token counts");
    }

    // Step 3: Batch read all data via MultiRead (async IO)
    std::vector<uint64_t> data_sizes(id_count);
    uint64_t total_size = 0;
    for (InnerIdType i = 0; i < id_count; ++i) {
        data_sizes[i] = GetMultiVectorCodeSize(lens[i], multi_vector_dim_);
        const uint64_t payload_size = layout_.Payload().GetByteSize();
        if (offsets[i] > payload_size || data_sizes[i] > payload_size - offsets[i]) {
            throw VsagException(ErrorType::READ_ERROR,
                                "MultiVectorDataCell: token data range exceeds payload");
        }
        if (data_sizes[i] > std::numeric_limits<uint64_t>::max() - total_size) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "MultiVectorDataCell: batch record size overflow");
        }
        total_size += data_sizes[i];
    }
    ByteBuffer all_codes(total_size, this->allocator_);
    if (!layout_.Payload().MultiRead(
            offsets.data(), data_sizes.data(), static_cast<uint64_t>(id_count), all_codes.data)) {
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
    if (ctx != nullptr and ctx->stats != nullptr and ctx->track_distance_evaluations)
        ctx->stats->AddDistance(ctx->distance_phase, backend_, id_count);
}

template <typename QuantTmpl, typename IOTmpl>
uint64_t
MultiVectorDataCell<QuantTmpl, IOTmpl>::GetMemoryUsage() const {
    uint64_t memory = sizeof(MultiVectorDataCell<QuantTmpl, IOTmpl>);
    memory += layout_.GetMemoryUsage();
    memory += sizeof(QuantTmpl);
    return memory;
}

}  // namespace vsag
