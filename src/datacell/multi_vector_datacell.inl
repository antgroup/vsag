
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
#include <numeric>

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
    this->backend_ =
        QuantizerDistanceBackend<QuantTmpl>::Get(static_cast<const QuantTmpl&>(*this->quantizer_));
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

    const uint64_t code_size_per_token = this->quantizer_->GetCodeSize();
    const uint64_t payload_bytes = static_cast<uint64_t>(multi_vector->len_) * code_size_per_token;
    const uint64_t code_size = sizeof(uint32_t) + payload_bytes;
    ByteBuffer codes(code_size, allocator_);
    std::memcpy(codes.data, &multi_vector->len_, sizeof(uint32_t));

    // Encode each token through the quantizer (FP32Quantizer is a no-op memcpy)
    for (uint32_t t = 0; t < multi_vector->len_; ++t) {
        const float* token_vec =
            multi_vector->vectors_ +
            static_cast<uint64_t>(t) * static_cast<uint64_t>(multi_vector_dim_);
        this->quantizer_->EncodeOne(
            token_vec,
            codes.data + sizeof(uint32_t) + static_cast<uint64_t>(t) * code_size_per_token);
    }

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

    // Cache the token count in memory so Query can skip the disk read
    if (static_cast<uint64_t>(idx) >= token_counts_.size()) {
        token_counts_.resize(static_cast<uint64_t>(idx) + 1, 0);
    }
    token_counts_[idx] = multi_vector->len_;
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
    if (static_cast<uint64_t>(new_capacity) > token_counts_.size()) {
        token_counts_.resize(static_cast<uint64_t>(new_capacity), 0);
    }
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
    const uint64_t code_size_per_token = this->quantizer_->GetCodeSize();
    uint64_t read_size = sizeof(uint32_t) + static_cast<uint64_t>(len) * code_size_per_token;
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
    this->backend_ =
        QuantizerDistanceBackend<QuantTmpl>::Get(static_cast<const QuantTmpl&>(*this->quantizer_));

    // Rebuild token_counts_ cache using batched MultiRead so Query does not need
    // a separate io_submit to fetch token counts from disk.
    if (this->total_count_ > 0) {
        std::vector<uint64_t> offsets(static_cast<uint64_t>(this->total_count_));
        std::vector<uint64_t> off_sizes(static_cast<uint64_t>(this->total_count_),
                                        sizeof(uint64_t));
        std::vector<uint64_t> off_offs(static_cast<uint64_t>(this->total_count_));
        for (InnerIdType i = 0; i < this->total_count_; ++i) {
            off_offs[i] = static_cast<uint64_t>(i) * sizeof(uint64_t);
        }
        if (not offset_io_->MultiRead(reinterpret_cast<uint8_t*>(offsets.data()),
                                      off_sizes.data(),
                                      off_offs.data(),
                                      static_cast<uint64_t>(this->total_count_))) {
            throw VsagException(ErrorType::READ_ERROR,
                                "MultiVectorDataCell: failed to read offsets in Deserialize");
        }

        token_counts_.resize(static_cast<uint64_t>(this->total_count_));
        std::vector<uint64_t> tc_sizes(static_cast<uint64_t>(this->total_count_), sizeof(uint32_t));
        if (not this->io_->MultiRead(reinterpret_cast<uint8_t*>(token_counts_.data()),
                                     tc_sizes.data(),
                                     offsets.data(),
                                     static_cast<uint64_t>(this->total_count_))) {
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
    auto* mv_computer = dynamic_cast<MultiVectorComputer*>(computer.get());
    CHECK_ARGUMENT(mv_computer != nullptr, "computer is not a MultiVectorComputer");

    if (id_count == 0) {
        return;
    }

    SearchStatistics* stats = (ctx != nullptr) ? ctx->stats : nullptr;

    // Step 1: Batch read all offsets via MultiRead (offset_io_ is MemoryBlockIO, in-memory)
    auto t_io_start = std::chrono::steady_clock::now();
    std::vector<uint64_t> offsets(id_count);
    std::vector<uint64_t> offset_sizes(id_count, sizeof(uint64_t));
    std::vector<uint64_t> offset_offsets(id_count);
    for (InnerIdType i = 0; i < id_count; ++i) {
        offset_offsets[i] = static_cast<uint64_t>(idx[i]) * sizeof(uint64_t);
    }
    if (not offset_io_->MultiRead(reinterpret_cast<uint8_t*>(offsets.data()),
                                  offset_sizes.data(),
                                  offset_offsets.data(),
                                  static_cast<uint64_t>(id_count))) {
        throw VsagException(ErrorType::READ_ERROR,
                            "MultiVectorDataCell: failed to read offsets in Query");
    }

    // Step 2: Look up token counts from in-memory cache (no disk IO)
    //         Populated by InsertVector (Build) or rebuilt in Deserialize.
    const uint64_t code_size_per_token = this->quantizer_->GetCodeSize();
    std::vector<uint64_t> data_sizes(id_count);
    uint64_t total_size = 0;
    for (InnerIdType i = 0; i < id_count; ++i) {
        if (static_cast<uint64_t>(idx[i]) >= token_counts_.size()) {
            throw VsagException(ErrorType::READ_ERROR,
                                "MultiVectorDataCell: token_counts_ not populated for doc ID " +
                                    std::to_string(idx[i]));
        }
        const uint32_t token_count = token_counts_[idx[i]];
        data_sizes[i] = sizeof(uint32_t) + static_cast<uint64_t>(token_count) * code_size_per_token;
        total_size += data_sizes[i];
    }

    // Step 2.5: Sort requests by disk offset for sequential IO.
    //           SSD schedulers handle sorted requests more efficiently, and
    //           AsyncIO::MultiReadImpl can merge/schedule them better.
    //           We build a permutation, reorder offsets/data_sizes accordingly,
    //           then unsort the computed distances back to the caller's order.
    std::vector<uint32_t> perm(static_cast<uint64_t>(id_count));
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&offsets](uint32_t a, uint32_t b) {
        return offsets[a] < offsets[b];
    });

    std::vector<uint64_t> sorted_offsets(static_cast<uint64_t>(id_count));
    std::vector<uint64_t> sorted_data_sizes(static_cast<uint64_t>(id_count));
    std::vector<InnerIdType> sorted_idx(static_cast<uint64_t>(id_count));
    for (InnerIdType i = 0; i < id_count; ++i) {
        sorted_offsets[i] = offsets[perm[i]];
        sorted_data_sizes[i] = data_sizes[perm[i]];
        sorted_idx[i] = idx[perm[i]];
    }

    // Step 3: Batch read all data via MultiRead (async IO, now in offset-sorted order)
    auto* all_codes = static_cast<uint8_t*>(this->allocator_->Allocate(total_size));
    if (all_codes == nullptr) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                            "MultiVectorDataCell: failed to allocate buffer for Query");
    }
    if (not this->io_->MultiRead(all_codes,
                                 sorted_data_sizes.data(),
                                 sorted_offsets.data(),
                                 static_cast<uint64_t>(id_count))) {
        this->allocator_->Deallocate(all_codes);
        throw VsagException(ErrorType::READ_ERROR,
                            "MultiVectorDataCell: failed to read data in Query");
    }
    double io_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_io_start)
            .count();

    // Step 4: Compute MaxSim distances in sorted order → temp_dists
    auto t_compute_start = std::chrono::steady_clock::now();
    std::vector<float> temp_dists(static_cast<uint64_t>(id_count));
    // Decode buffer: one doc at a time (reused across iterations)
    std::vector<float> decoded_tokens;
    uint64_t cursor = 0;
    for (InnerIdType i = 0; i < id_count; ++i) {
        const uint32_t token_count = token_counts_[sorted_idx[i]];
        const uint8_t* encoded = all_codes + cursor + sizeof(uint32_t);

        // Decode quantized tokens back to float32 for ComputeDist.
        // For FP32Quantizer this is a no-op memcpy; for SQ8/FP16 it performs
        // the actual dequantization.
        decoded_tokens.resize(static_cast<uint64_t>(token_count) *
                              static_cast<uint64_t>(multi_vector_dim_));
        for (uint32_t t = 0; t < token_count; ++t) {
            this->quantizer_->DecodeOne(
                encoded + static_cast<uint64_t>(t) * code_size_per_token,
                decoded_tokens.data() +
                    static_cast<uint64_t>(t) * static_cast<uint64_t>(multi_vector_dim_));
        }
        mv_computer->ComputeDist(
            reinterpret_cast<const uint8_t*>(decoded_tokens.data()), token_count, &temp_dists[i]);
        cursor += sorted_data_sizes[i];
    }

    // Step 4.5: Unsort temp_dists back to the caller's original order.
    for (InnerIdType i = 0; i < id_count; ++i) {
        result_dists[perm[i]] = temp_dists[i];
    }
    double compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                  t_compute_start)
                            .count();

    this->allocator_->Deallocate(all_codes);

    // Populate SearchStatistics with fine-grained breakdown (rounded to ms)
    if (stats != nullptr) {
        stats->mv_io_time_ms.fetch_add(static_cast<uint32_t>(io_ms + 0.5),
                                       std::memory_order_relaxed);
        stats->mv_compute_time_ms.fetch_add(static_cast<uint32_t>(compute_ms + 0.5),
                                            std::memory_order_relaxed);
        stats->mv_candidate_count.fetch_add(static_cast<uint32_t>(id_count),
                                            std::memory_order_relaxed);
        stats->mv_io_bytes.fetch_add(total_size, std::memory_order_relaxed);
    }

    // Record distance evaluations for telemetry (compatible with upstream PR #2545)
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
    memory += this->offset_io_->size_;
    if (IOTmpl::InMemory) {
        memory += this->io_->GetMemoryUsage();
    }
    memory += sizeof(QuantTmpl);
    memory += token_counts_.capacity() * sizeof(uint32_t);
    return memory;
}

}  // namespace vsag
