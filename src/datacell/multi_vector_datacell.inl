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
#include <chrono>
#include <cstring>
#include <limits>
#include <numeric>

#include "common.h"
#include "multi_vector_datacell.h"
#include "utils/byte_buffer.h"
#include "vsag/options.h"

namespace vsag {
namespace {

uint64_t
GetMultiVectorRecordSize(uint32_t token_count, uint64_t code_size_per_token) {
    constexpr uint64_t header_size = sizeof(uint32_t);
    if (code_size_per_token != 0 and
        token_count > (std::numeric_limits<uint64_t>::max() - header_size) / code_size_per_token) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "MultiVectorDataCell: record size overflow");
    }
    return header_size + static_cast<uint64_t>(token_count) * code_size_per_token;
}

}  // namespace

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
    layout_.SetLocationPolicy(HeaderLengthLocationPolicy{this->quantizer_->GetCodeSize()});
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

    const uint64_t code_size_per_token = this->quantizer_->GetCodeSize();
    const uint64_t code_size = GetMultiVectorRecordSize(multi_vector->len_, code_size_per_token);
    ByteBuffer codes(code_size, allocator_);
    std::memcpy(codes.data, &multi_vector->len_, sizeof(uint32_t));
    for (uint32_t token = 0; token < multi_vector->len_; ++token) {
        const float* token_vector =
            multi_vector->vectors_ + static_cast<uint64_t>(token) * multi_vector_dim_;
        this->quantizer_->EncodeOne(
            token_vector,
            codes.data + sizeof(uint32_t) + static_cast<uint64_t>(token) * code_size_per_token);
    }

    {
        std::lock_guard lock(mutex_);
        layout_.Write(idx, codes.data, code_size);
        if (static_cast<uint64_t>(idx) >= token_counts_.size()) {
            token_counts_.resize(static_cast<uint64_t>(idx) + 1, 0);
        }
        token_counts_[idx] = multi_vector->len_;
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
    std::lock_guard lock(mutex_);
    const InnerIdType effective_capacity = std::max(new_capacity, total_count_);
    if (effective_capacity <= this->max_capacity_) {
        return;
    }
    layout_.ResizeLocations(effective_capacity);
    if (static_cast<uint64_t>(effective_capacity) > token_counts_.size()) {
        token_counts_.resize(static_cast<uint64_t>(effective_capacity), 0);
    }
    this->max_capacity_ = effective_capacity;
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
    std::shared_lock lock(mutex_);
    CHECK_ARGUMENT(id < total_count_, "MultiVectorDataCell id is out of range");
    const uint64_t offset = layout_.ReadLocation(id);
    uint32_t length = 0;
    if (not layout_.Payload().Read(offset, sizeof(length), reinterpret_cast<uint8_t*>(&length))) {
        throw VsagException(ErrorType::READ_ERROR,
                            "MultiVectorDataCell: failed to read token count");
    }
    const uint64_t read_size = GetMultiVectorRecordSize(length, this->quantizer_->GetCodeSize());
    const uint64_t payload_size = layout_.Payload().GetByteSize();
    if (offset > payload_size or read_size > payload_size - offset) {
        throw VsagException(ErrorType::READ_ERROR,
                            "MultiVectorDataCell: token data range exceeds payload");
    }
    auto* codes = static_cast<uint8_t*>(allocator_->Allocate(read_size));
    if (codes == nullptr) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                            "MultiVectorDataCell: failed to allocate buffer for GetCodesById");
    }
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
MultiVectorDataCell<QuantTmpl, IOTmpl>::Deserialize(LvalueOrRvalue<StreamReader> reader) {
    FlattenInterface::Deserialize(reader);
    StreamReader::ReadObj(reader, multi_vector_dim_);
    uint64_t current_offset = 0;
    StreamReader::ReadObj(reader, current_offset);
    layout_.SetNextOffset(current_offset);
    layout_.Locations().Deserialize(reader);
    layout_.Payload().Deserialize(reader);
    this->quantizer_->Deserialize(reader);
    layout_.SetLocationPolicy(HeaderLengthLocationPolicy{this->quantizer_->GetCodeSize()});
    this->backend_ =
        QuantizerDistanceBackend<QuantTmpl>::Get(static_cast<const QuantTmpl&>(*this->quantizer_));

    if (this->total_count_ > 0) {
        std::vector<uint64_t> offsets(static_cast<uint64_t>(this->total_count_));
        std::vector<InnerIdType> ids(static_cast<uint64_t>(this->total_count_));
        for (InnerIdType i = 0; i < this->total_count_; ++i) {
            ids[i] = i;
        }
        if (not layout_.Locations().MultiRead(ids.data(),
                                              static_cast<uint64_t>(this->total_count_),
                                              reinterpret_cast<uint8_t*>(offsets.data()),
                                              allocator_)) {
            throw VsagException(ErrorType::READ_ERROR,
                                "MultiVectorDataCell: failed to read offsets in Deserialize");
        }

        token_counts_.resize(static_cast<uint64_t>(this->total_count_));
        std::vector<uint64_t> sizes(static_cast<uint64_t>(this->total_count_), sizeof(uint32_t));
        if (not layout_.Payload().MultiRead(offsets.data(),
                                            sizes.data(),
                                            static_cast<uint64_t>(this->total_count_),
                                            reinterpret_cast<uint8_t*>(token_counts_.data()))) {
            throw VsagException(ErrorType::READ_ERROR,
                                "MultiVectorDataCell: failed to read token counts in Deserialize");
        }
    }
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
    auto* computer_impl = dynamic_cast<MultiVectorComputer*>(computer.get());
    CHECK_ARGUMENT(computer_impl != nullptr, "computer is not a MultiVectorComputer");
    if (id_count == 0) {
        return;
    }
    CHECK_ARGUMENT(idx != nullptr, "MultiVectorDataCell query ids are null");
    std::shared_lock lock(mutex_);
    for (InnerIdType i = 0; i < id_count; ++i) {
        CHECK_ARGUMENT(idx[i] < total_count_, "MultiVectorDataCell query id is out of range");
    }

    SearchStatistics* stats = ctx != nullptr ? ctx->stats : nullptr;
    const auto io_start = std::chrono::steady_clock::now();
    std::vector<uint64_t> offsets(id_count);
    if (not layout_.Locations().MultiRead(idx,
                                          static_cast<uint64_t>(id_count),
                                          reinterpret_cast<uint8_t*>(offsets.data()),
                                          allocator_)) {
        throw VsagException(ErrorType::READ_ERROR,
                            "MultiVectorDataCell: failed to read offsets in Query");
    }

    const uint64_t code_size_per_token = this->quantizer_->GetCodeSize();
    std::vector<uint64_t> data_sizes(id_count);
    uint64_t total_size = 0;
    const uint64_t payload_size = layout_.Payload().GetByteSize();
    for (InnerIdType i = 0; i < id_count; ++i) {
        if (static_cast<uint64_t>(idx[i]) >= token_counts_.size()) {
            throw VsagException(ErrorType::READ_ERROR,
                                "MultiVectorDataCell: token count is unavailable");
        }
        data_sizes[i] = GetMultiVectorRecordSize(token_counts_[idx[i]], code_size_per_token);
        if (offsets[i] > payload_size or data_sizes[i] > payload_size - offsets[i]) {
            throw VsagException(ErrorType::READ_ERROR,
                                "MultiVectorDataCell: token data range exceeds payload");
        }
        if (data_sizes[i] > std::numeric_limits<uint64_t>::max() - total_size) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "MultiVectorDataCell: batch record size overflow");
        }
        total_size += data_sizes[i];
    }

    std::vector<uint32_t> permutation(static_cast<uint64_t>(id_count));
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(), [&offsets](uint32_t lhs, uint32_t rhs) {
        return offsets[lhs] < offsets[rhs];
    });
    std::vector<uint64_t> sorted_offsets(static_cast<uint64_t>(id_count));
    std::vector<uint64_t> sorted_sizes(static_cast<uint64_t>(id_count));
    std::vector<InnerIdType> sorted_ids(static_cast<uint64_t>(id_count));
    for (InnerIdType i = 0; i < id_count; ++i) {
        sorted_offsets[i] = offsets[permutation[i]];
        sorted_sizes[i] = data_sizes[permutation[i]];
        sorted_ids[i] = idx[permutation[i]];
    }

    ByteBuffer all_codes_buffer(total_size, allocator_);
    auto* all_codes = all_codes_buffer.data;
    if (all_codes == nullptr) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                            "MultiVectorDataCell: failed to allocate buffer for Query");
    }
    if (not layout_.Payload().MultiRead(
            sorted_offsets.data(), sorted_sizes.data(), id_count, all_codes)) {
        throw VsagException(ErrorType::READ_ERROR,
                            "MultiVectorDataCell: failed to read data in Query");
    }
    const double io_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - io_start)
            .count();

    const auto compute_start = std::chrono::steady_clock::now();
    std::vector<float> sorted_distances(static_cast<uint64_t>(id_count));
    std::vector<float> decoded_tokens;
    uint64_t cursor = 0;
    for (InnerIdType i = 0; i < id_count; ++i) {
        const uint32_t token_count = token_counts_[sorted_ids[i]];
        const uint8_t* encoded = all_codes + cursor + sizeof(uint32_t);
        decoded_tokens.resize(static_cast<uint64_t>(token_count) * multi_vector_dim_);
        for (uint32_t token = 0; token < token_count; ++token) {
            this->quantizer_->DecodeOne(
                encoded + static_cast<uint64_t>(token) * code_size_per_token,
                decoded_tokens.data() + static_cast<uint64_t>(token) * multi_vector_dim_);
        }
        computer_impl->ComputeDist(reinterpret_cast<const uint8_t*>(decoded_tokens.data()),
                                   token_count,
                                   &sorted_distances[i]);
        cursor += sorted_sizes[i];
    }
    for (InnerIdType i = 0; i < id_count; ++i) {
        result_dists[permutation[i]] = sorted_distances[i];
    }
    const double compute_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - compute_start)
            .count();

    if (stats != nullptr) {
        stats->mv_io_time_ms.fetch_add(static_cast<uint32_t>(io_ms + 0.5),
                                       std::memory_order_relaxed);
        stats->mv_compute_time_ms.fetch_add(static_cast<uint32_t>(compute_ms + 0.5),
                                            std::memory_order_relaxed);
        stats->mv_candidate_count.fetch_add(static_cast<uint32_t>(id_count),
                                            std::memory_order_relaxed);
        stats->mv_io_bytes.fetch_add(total_size, std::memory_order_relaxed);
    }
    if (ctx != nullptr and ctx->stats != nullptr) {
        ctx->stats->AddDistance(DistanceEvaluationPhase::RERANK,
                                this->quantizer_->Name(),
                                static_cast<uint64_t>(id_count));
    }
}

template <typename QuantTmpl, typename IOTmpl>
uint64_t
MultiVectorDataCell<QuantTmpl, IOTmpl>::GetMemoryUsage() const {
    uint64_t memory = sizeof(MultiVectorDataCell<QuantTmpl, IOTmpl>);
    memory += layout_.GetMemoryUsage();
    memory += sizeof(QuantTmpl);
    memory += token_counts_.capacity() * sizeof(uint32_t);
    return memory;
}

}  // namespace vsag
