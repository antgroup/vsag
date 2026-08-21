
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

#include <algorithm>
#include <limits>

#include "container_types.h"
#include "sparse_vector_datacell.h"
#include "vsag/options.h"

namespace vsag {
template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::query(float* result_dists,
                                               const std::shared_ptr<Computer<QuantTmpl>>& computer,
                                               const InnerIdType* idx,
                                               InnerIdType id_count,
                                               QueryContext* ctx) {
    if (id_count == 0) {
        return;
    }
    CHECK_ARGUMENT(idx != nullptr, "SparseVectorDataCell query ids are null");

    const auto load_location = [this](InnerIdType id) {
        const auto location = layout_.ReadLocation(id);
        if (not layout_.IsValidLocation(location)) {
            throw VsagException(ErrorType::READ_ERROR,
                                "SparseVectorDataCell read an invalid document location");
        }
        return location;
    };

    std::shared_lock lock(mutex_);

    const auto compute_direct = [&](const DocLocation& location, InnerIdType result_index) {
        bool need_release = false;
        const auto* codes = layout_.Read(location, need_release);
        if (codes == nullptr) {
            throw VsagException(ErrorType::READ_ERROR,
                                "SparseVectorDataCell failed to read vector codes");
        }
        try {
            computer->ComputeDist(codes, result_dists + result_index);
        } catch (...) {
            if (need_release) {
                this->Release(codes);
            }
            throw;
        }
        if (need_release) {
            this->Release(codes);
        }
    };

    if (query_io_strategy_ == QueryIOStrategy::DIRECT_READ) {
        for (InnerIdType i = 0; i < id_count; ++i) {
            const auto location = load_location(idx[i]);
            compute_direct(location, i);
        }
        return;
    }

    struct QueryLocation {
        DocLocation location;
        InnerIdType result_index{0};
    };

    Allocator* query_allocator = select_query_allocator(ctx, allocator_);
    Vector<QueryLocation> locations(id_count, query_allocator);
    for (InnerIdType i = 0; i < id_count; ++i) {
        locations[i] = {load_location(idx[i]), i};
    }
    std::sort(locations.begin(), locations.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.location.offset != rhs.location.offset) {
            return lhs.location.offset < rhs.location.offset;
        }
        return lhs.result_index < rhs.result_index;
    });

    if (query_io_strategy_ == QueryIOStrategy::SORTED_DIRECT_READ) {
        for (const auto& item : locations) {
            compute_direct(item.location, item.result_index);
        }
        return;
    }

    const uint64_t merge_gap_limit = 1ULL << Options::Instance().direct_IO_object_align_bit();
    constexpr uint64_t max_merged_io_len = 1ULL << 20;

    struct MergedRange {
        uint64_t offset{0};
        uint64_t size{0};
        uint64_t first_location{0};
        uint64_t last_location{0};
    };

    Vector<MergedRange> ranges(query_allocator);
    ranges.reserve(id_count);
    for (uint64_t i = 0; i < static_cast<uint64_t>(id_count); ++i) {
        const auto& location = locations[i].location;
        const uint64_t location_end = location.offset + location.length;
        if (not ranges.empty()) {
            auto& range = ranges.back();
            const uint64_t range_end = range.offset + range.size;
            const uint64_t gap = location.offset > range_end ? location.offset - range_end : 0;
            const uint64_t merged_end = std::max(range_end, location_end);
            const uint64_t merged_size = merged_end - range.offset;
            if (gap <= merge_gap_limit && merged_size <= max_merged_io_len) {
                range.size = merged_size;
                range.last_location = i;
                continue;
            }
        }
        ranges.push_back({location.offset, location.length, i, i});
    }

    const uint64_t range_count = ranges.size();
    Vector<uint64_t> read_sizes(range_count, query_allocator);
    Vector<uint64_t> read_offsets(range_count, query_allocator);
    Vector<uint64_t> scratch_offsets(range_count, query_allocator);
    uint64_t scratch_size = 0;
    for (uint64_t i = 0; i < range_count; ++i) {
        read_sizes[i] = ranges[i].size;
        read_offsets[i] = ranges[i].offset;
        scratch_offsets[i] = scratch_size;
        scratch_size += ranges[i].size;
    }

    Vector<uint8_t> scratch(scratch_size, query_allocator);
    if (not layout_.Payload().MultiRead(
            read_offsets.data(), read_sizes.data(), range_count, scratch.data())) {
        throw VsagException(ErrorType::READ_ERROR,
                            "SparseVectorDataCell failed to read vector-code batch");
    }

    for (uint64_t range_index = 0; range_index < range_count; ++range_index) {
        const auto& range = ranges[range_index];
        const auto* range_data = scratch.data() + scratch_offsets[range_index];
        for (uint64_t location_index = range.first_location; location_index <= range.last_location;
             ++location_index) {
            const auto& item = locations[location_index];
            const auto* codes = range_data + item.location.offset - range.offset;
            computer->ComputeDist(codes, result_dists + item.result_index);
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
        uint64_t current_offset = 0;
        StreamReader::ReadObj(reader, current_offset);
        layout_.SetNextOffset(current_offset);
        layout_.Payload().Deserialize(reader);
        layout_.Locations().Deserialize(reader);
    } else {
        // Legacy 32-bit format. The uint32 we just read is the old current_offset_.
        layout_.SetNextOffset(static_cast<uint64_t>(maybe_sentinel));
        layout_.Payload().Deserialize(reader);
        // Legacy offset_io_ holds an array of 8-byte LegacyDocLocation records. We
        // load them and expand each entry to the new 12-byte DocLocation in memory
        // so the rest of the code can use a single internal representation.
        uint64_t legacy_offset_io_size = 0;
        StreamReader::ReadObj(reader, legacy_offset_io_size);
        const uint64_t legacy_entry_size = sizeof(LegacyDocLocation);
        if (legacy_offset_io_size % legacy_entry_size != 0) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("invalid legacy SparseVectorDataCell offset size: {}",
                                            legacy_offset_io_size));
        }
        const uint64_t doc_count = legacy_offset_io_size / legacy_entry_size;
        if (doc_count > std::numeric_limits<InnerIdType>::max() || doc_count < total_count_) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                fmt::format("invalid legacy SparseVectorDataCell document count: {}", doc_count));
        }
        layout_.ResizeLocations(doc_count);
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
                    new_batch[i].length = legacy_batch[i].size;
                }
                layout_.Locations().WriteRange(
                    cursor, reinterpret_cast<uint8_t*>(new_batch.data()), batch);
                cursor += batch;
                remaining -= batch;
            }
        }
    }
    this->quantizer_->Deserialize(reader);
    backend_ =
        QuantizerDistanceBackend<QuantTmpl>::Get(static_cast<const QuantTmpl&>(*this->quantizer_));
}

template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::Serialize(StreamWriter& writer) {
    FlattenInterface::Serialize(writer);
    const uint32_t sentinel = SERIALIZE_FORMAT_SENTINEL;
    const uint32_t version = SERIALIZE_FORMAT_VERSION_V2;
    StreamWriter::WriteObj(writer, sentinel);
    StreamWriter::WriteObj(writer, version);
    StreamWriter::WriteObj(writer, layout_.GetNextOffset());
    layout_.Payload().Serialize(writer);
    layout_.Locations().Serialize(writer);
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
    uint64_t code_size = (static_cast<uint64_t>(sparse_vector->len_) * 2 + 1) * sizeof(uint32_t);
    if (code_size > std::numeric_limits<uint32_t>::max()) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("sparse vector code size {} exceeds uint32_t limit", code_size));
    }
    Vector<uint8_t> codes(code_size, allocator_);
    quantizer_->EncodeOne((const float*)vector, codes.data());
    {
        std::lock_guard lock(mutex_);
        total_count_ = std::max(total_count_, idx + 1);
        max_code_size_ = std::max(max_code_size_, code_size);
        layout_.Write(idx, codes.data(), code_size);
    }
}

template <typename QuantTmpl, typename IOTmpl>
bool
SparseVectorDataCell<QuantTmpl, IOTmpl>::InMemory() const {
    return FlattenInterface::InMemory();
}

template <typename QuantTmpl, typename IOTmpl>
const uint8_t*
SparseVectorDataCell<QuantTmpl, IOTmpl>::GetCodesById(InnerIdType id, bool& need_release) const {
    std::shared_lock lock(mutex_);
    return this->get_codes_by_id_no_lock(id, need_release);
}

template <typename QuantTmpl, typename IOTmpl>
const uint8_t*
SparseVectorDataCell<QuantTmpl, IOTmpl>::get_codes_by_id_no_lock(InnerIdType id,
                                                                 bool& need_release) const {
    const auto* codes = layout_.Read(id, need_release);
    if (codes == nullptr) {
        throw VsagException(ErrorType::READ_ERROR,
                            "SparseVectorDataCell failed to read vector codes");
    }
    return codes;
}

template <typename QuantTmpl, typename IOTmpl>
void
SparseVectorDataCell<QuantTmpl, IOTmpl>::GetSparseVectorByInnerId(
    InnerIdType inner_id, SparseVector* data, Allocator* specified_allocator) const {
    Allocator* allocator = specified_allocator != nullptr ? specified_allocator : allocator_;

    std::shared_lock lock(mutex_);

    bool need_release{false};
    const auto* codes = this->get_codes_by_id_no_lock(inner_id, need_release);
    data->len_ = *reinterpret_cast<const uint32_t*>(codes);
    const auto* entries = reinterpret_cast<const BufferEntry*>(codes + sizeof(uint32_t));
    if (data->len_ == 0) {
        data->ids_ = nullptr;
        data->vals_ = nullptr;
        if (need_release) {
            this->Release(codes);
        }
        return;
    }
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
    layout_.Release(data);
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
        codes1 = this->get_codes_by_id_no_lock(id1, release1);
        codes2 = this->get_codes_by_id_no_lock(id2, release2);
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
    this->backend_ =
        QuantizerDistanceBackend<QuantTmpl>::Get(static_cast<const QuantTmpl&>(*this->quantizer_));
    auto io = std::make_shared<IOTmpl>(io_param, common_param);
    const auto& io_type = io_param->GetTypeName();
    if (io_type == IO_TYPE_VALUE_MEMORY_IO || io_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        this->query_io_strategy_ = QueryIOStrategy::DIRECT_READ;
    } else if (io_type == IO_TYPE_VALUE_MMAP_IO) {
        this->query_io_strategy_ = QueryIOStrategy::SORTED_DIRECT_READ;
    } else {
        this->query_io_strategy_ = QueryIOStrategy::MULTI_READ;
    }
    auto offset_io =
        std::make_shared<MemoryBlockIO>(Options::Instance().block_size_limit(), allocator_);
    layout_.SetIO(std::move(offset_io), std::move(io));
    this->max_code_size_ = std::max<uint64_t>(
        sizeof(uint32_t), (static_cast<uint64_t>(common_param.dim_) * 2 + 1) * sizeof(uint32_t));
    this->max_capacity_ = 0;
    this->code_size_ = this->quantizer_->GetCodeSize();
}

template <typename QuantTmpl, typename IOTmpl>
uint64_t
SparseVectorDataCell<QuantTmpl, IOTmpl>::GetMemoryUsage() const {
    uint64_t memory = sizeof(SparseVectorDataCell<QuantTmpl, IOTmpl>);
    memory += layout_.GetMemoryUsage();
    memory += sizeof(QuantTmpl);
    return memory;
}
}  // namespace vsag
