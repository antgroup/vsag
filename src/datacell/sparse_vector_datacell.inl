
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

#include <fmt/format.h>

#include "sparse_vector_datacell.h"

namespace vsag {
template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::query(float* result_dists,
                                               const std::shared_ptr<Computer<QuantTmpl>>& computer,
                                               const InnerIdType* idx,
                                               InnerIdType id_count) {
    std::shared_lock lock(mutex_);
    for (int i = 0; i < id_count; ++i) {
        bool need_release{true};
        auto codes = this->GetCodesById(idx[i], need_release);
        try {
            computer->ComputeDist(codes, result_dists + i);
        } catch (...) {
            if (need_release) {
                this->Release(codes);
            }
            throw;
        }
        if (need_release) {
            this->Release(codes);
        }
    }
}
template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    FlattenInterface::Deserialize(reader);

    uint32_t maybe_sentinel = 0;
    StreamReader::ReadObj(reader, maybe_sentinel);

    if (maybe_sentinel == SERIALIZE_FORMAT_SENTINEL) {
        // New 64-bit format. Layout written by Serialize().
        uint32_t version = 0;
        StreamReader::ReadObj(reader, version);
        if (version != SERIALIZE_FORMAT_VERSION_V2) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                fmt::format("unsupported SparseVectorDataCell serialization version: {}", version));
        }
        StreamReader::ReadObj(reader, current_offset_);
        this->io_->Deserialize(reader);
        this->offset_io_->Deserialize(reader);
    } else {
        // Legacy 32-bit format. The uint32 we just read is the old current_offset_.
        current_offset_ = static_cast<uint64_t>(maybe_sentinel);
        this->io_->Deserialize(reader);
        // Legacy offset_io_ holds an array of 8-byte LegacyDocLocation records. We
        // load them and expand each entry to the new 12-byte DocLocation in memory
        // so the rest of the code can use a single internal representation.
        uint64_t legacy_offset_io_size = 0;
        StreamReader::ReadObj(reader, legacy_offset_io_size);
        const uint64_t legacy_entry_size = sizeof(LegacyDocLocation);
        const uint64_t doc_count =
            legacy_entry_size == 0 ? 0 : legacy_offset_io_size / legacy_entry_size;
        this->offset_io_->Resize(doc_count * sizeof(DocLocation));
        if (doc_count > 0) {
            constexpr uint64_t BATCH = 4096;
            Vector<LegacyDocLocation> legacy_batch(allocator_);
            Vector<DocLocation> new_batch(allocator_);
            legacy_batch.reserve(BATCH);
            new_batch.reserve(BATCH);
            uint64_t remaining = doc_count;
            uint64_t cursor = 0;
            while (remaining > 0) {
                const uint64_t batch = std::min<uint64_t>(BATCH, remaining);
                legacy_batch.resize(batch);
                new_batch.resize(batch);
                reader->Read(reinterpret_cast<char*>(legacy_batch.data()),
                             batch * sizeof(LegacyDocLocation));
                for (uint64_t i = 0; i < batch; ++i) {
                    new_batch[i].offset = static_cast<uint64_t>(legacy_batch[i].offset);
                    new_batch[i].size = legacy_batch[i].size;
                }
                this->offset_io_->Write(reinterpret_cast<uint8_t*>(new_batch.data()),
                                        batch * sizeof(DocLocation),
                                        cursor * sizeof(DocLocation));
                cursor += batch;
                remaining -= batch;
            }
        }
    }
    this->quantizer_->Deserialize(reader);
}

template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::Serialize(StreamWriter& writer) {
    FlattenInterface::Serialize(writer);
    const uint32_t sentinel = SERIALIZE_FORMAT_SENTINEL;
    const uint32_t version = SERIALIZE_FORMAT_VERSION_V2;
    StreamWriter::WriteObj(writer, sentinel);
    StreamWriter::WriteObj(writer, version);
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
    auto sparse_vector = (const SparseVector*)vector;
    uint64_t code_size = (sparse_vector->len_ * 2 + 1) * sizeof(uint32_t);
    auto* codes = reinterpret_cast<uint8_t*>(allocator_->Allocate(code_size));
    quantizer_->EncodeOne((const float*)vector, codes);
    DocLocation location;
    {
        std::scoped_lock lock(mutex_, current_offset_mutex_);
        total_count_ = std::max(total_count_, idx + 1);
        max_code_size_ = std::max(max_code_size_, code_size);
        const auto required_size = current_offset_ + code_size;
        if (required_size > this->io_->size_) {
            this->io_->Resize(required_size);
        }
        location.offset = current_offset_;
        location.size = static_cast<uint32_t>(code_size);
        current_offset_ += code_size;
        offset_io_->Write(reinterpret_cast<uint8_t*>(&location),
                          sizeof(location),
                          static_cast<uint64_t>(idx) * sizeof(location));
        io_->Write(codes, code_size, location.offset);
    }
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
    DocLocation location;
    offset_io_->Read(sizeof(location),
                     static_cast<uint64_t>(id) * sizeof(location),
                     reinterpret_cast<uint8_t*>(&location));
    return io_->Read(location.size, location.offset, need_release);
}

template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::GetSparseVectorByInnerId(
    InnerIdType inner_id, SparseVector* data, Allocator* specified_allocator) const {
    Allocator* allocator = specified_allocator != nullptr ? specified_allocator : allocator_;

    std::shared_lock lock(mutex_);

    bool need_release{false};
    const auto* codes = this->GetCodesById(inner_id, need_release);
    data->len_ = *reinterpret_cast<const uint32_t*>(codes);
    const auto* entries = reinterpret_cast<const BufferEntry*>(codes + sizeof(uint32_t));
    data->ids_ = static_cast<uint32_t*>(allocator->Allocate(sizeof(uint32_t) * data->len_));
    try {
        data->vals_ = static_cast<float*>(allocator->Allocate(sizeof(float) * data->len_));
    } catch (...) {
        allocator->Deallocate(data->ids_);
        data->ids_ = nullptr;
        if (need_release) {
            this->Release(codes);
        }
        throw;
    }
    for (uint32_t i = 0; i < data->len_; ++i) {
        data->ids_[i] = entries[i].id;
        data->vals_[i] = entries[i].val;
    }
    if (need_release) {
        this->Release(codes);
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::Release(const uint8_t* data) const {
    io_->Release(data);
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
    std::shared_lock lock(mutex_);
    bool release1 = false, release2 = false;
    const uint8_t* codes1 = nullptr;
    const uint8_t* codes2 = nullptr;
    try {
        codes1 = this->GetCodesById(id1, release1);
        codes2 = this->GetCodesById(id2, release2);
        auto result = this->quantizer_->Compute(codes1, codes2);
        if (release1) {
            this->Release(codes1);
        }
        if (release2) {
            this->Release(codes2);
        }
        return result;
    } catch (...) {
        if (codes1 && release1) {
            this->Release(codes1);
        }
        if (codes2 && release2) {
            this->Release(codes2);
        }
        throw;
    }
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
        std::make_shared<MemoryBlockIO>(Options::Instance().block_size_limit(), allocator_);
    this->max_code_size_ = sizeof(uint32_t);
    this->max_capacity_ = 0;
    this->code_size_ = this->quantizer_->GetCodeSize();
}

template <typename QuantTmpl, typename IOTmpl>
uint64_t
SparseVectorDataCell<QuantTmpl, IOTmpl>::GetMemoryUsage() const {
    uint64_t memory = sizeof(SparseVectorDataCell<QuantTmpl, IOTmpl>);
    memory += this->offset_io_->size_;
    if (IOTmpl::InMemory) {
        memory += this->io_->GetMemoryUsage();
    }
    memory += sizeof(QuantTmpl);
    return memory;
}
}  // namespace vsag
