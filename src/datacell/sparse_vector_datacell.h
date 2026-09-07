
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

#include <limits>
#include <type_traits>

#include "flatten_datacell.h"
#include "flatten_interface.h"
#include "inner_string_params.h"
#include "io/memory_block_io/memory_block_io.h"
#include "io/mmap_io/mmap_io.h"
#include "layout/variable_record_layout.h"
#include "quantization/sparse_quantization/sparse_quantizer.h"
#include "vsag/dataset.h"

namespace vsag {

template <typename QuantTmpl, typename IOTmpl>
class SparseVectorDataCell : public FlattenInterface {
public:
    using ReadLease = typename IOTmpl::Lease;

    SparseVectorDataCell() = default;

    SparseVectorDataCell(const QuantizerParamPtr& quantization_param,
                         const IOParamPtr& io_param,
                         const IndexCommonParam& common_param);

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) override {
        auto comp = std::static_pointer_cast<Computer<QuantTmpl>>(computer);
        this->query(result_dists, comp, idx, id_count, ctx);
        if (ctx != nullptr and ctx->stats != nullptr and ctx->track_distance_evaluations and
            id_count > 0) {
            ctx->stats->AddDistance(ctx->distance_phase, backend_, id_count);
        }
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        return this->factory_computer(static_cast<const float*>(query));
    }

    bool
    Decode(const uint8_t* codes, float* vector) override {
        // TODO(inabao): Implement the decode function
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Decode function is not implemented for SparseVectorDataCell");
    }

    bool
    Encode(const float* vector, uint8_t* codes) override {
        // TODO(inabao): Implement the decode function
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "Encode function is not implemented for SparseVectorDataCell");
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
        std::lock_guard lock(mutex_);
        const InnerIdType effective_capacity = std::max(new_capacity, total_count_);
        if (effective_capacity <= this->max_capacity_) {
            return;
        }
        const uint64_t additional_count = static_cast<uint64_t>(effective_capacity - total_count_);
        const uint64_t current_size = layout_.GetNextOffset();
        if (max_code_size_ != 0 &&
            additional_count >
                (std::numeric_limits<uint64_t>::max() - current_size) / max_code_size_) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "SparseVectorDataCell resize payload size overflow");
        }
        const uint64_t io_size = additional_count * max_code_size_ + current_size;
        layout_.ReservePayload(io_size);
        layout_.ResizeLocations(effective_capacity);
        this->max_capacity_ = effective_capacity;
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
        auto ptr =
            std::dynamic_pointer_cast<FlattenDataCell<QuantTmpl, FixedLayout<IOTmpl>>>(other);
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

    void
    GetSparseVectorByInnerId(InnerIdType inner_id,
                             SparseVector* data,
                             Allocator* specified_allocator) const override;

    void
    Release(const uint8_t* data) const override;

    [[nodiscard]] bool
    InMemory() const override;

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(LvalueOrRvalue<StreamReader> reader) override;

    inline void
    SetQuantizer(std::shared_ptr<Quantizer<QuantTmpl>> quantizer) {
        this->quantizer_ = quantizer;
    }

    inline void
    SetIO(std::shared_ptr<IOTmpl> io) {
        layout_.Payload().SetIO(std::move(io));
    }

    void
    InitIO(const IOParamPtr& io_param) override {
        layout_.Payload().InitIO(io_param);
    }

    uint64_t
    GetMemoryUsage() const override;

private:
    enum class QueryIOStrategy : uint8_t {
        DIRECT_READ,
        SORTED_DIRECT_READ,
        MULTI_READ,
    };

    inline void
    query(float* result_dists,
          const std::shared_ptr<Computer<QuantTmpl>>& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx);

    ComputerInterfacePtr
    factory_computer(const float* query) {
        auto computer = this->quantizer_->FactoryComputer();
        computer->SetQuery(query);
        return computer;
    }

    const uint8_t*
    get_codes_by_id_no_lock(InnerIdType id, bool& need_release) const;

    [[nodiscard]] ReadLease
    acquire_codes_by_id_no_lock(InnerIdType id) const;

private:
    // Packed so each entry is exactly 12 bytes on disk and in the layout location table. The
    // unpacked layout would round sizeof up to 16 due to the uint64 alignment requirement,
    // wasting 33% of the location table.
    using DocLocation = OffsetAndLengthLocationPolicy::Entry;
    static_assert(sizeof(DocLocation) == 12, "DocLocation must be 12 bytes on disk");

    // Legacy on-disk layout: kept for backward-compatible deserialization of indexes
    // produced before the 64-bit offset upgrade. Offsets larger than 4 GiB used to
    // silently overflow and corrupt the stored payload (see PR #2056).
    struct LegacyDocLocation {
        uint32_t offset{0};
        uint32_t size{0};
    };
    static_assert(sizeof(LegacyDocLocation) == 8, "LegacyDocLocation must be 8 bytes");

    // Sentinel written in place of the legacy uint32 current_offset_ to mark the
    // new 64-bit format on disk. A legacy file can never produce this value at
    // that byte offset: legacy current_offset_ is the sum of code_size values,
    // each of which is `(len*2+1)*sizeof(uint32_t)` and therefore a multiple of
    // 4 bytes (see sparse_vector_datacell.inl::InsertVector). 0xFFFFFFFF is not
    // a multiple of 4, so the sentinel cannot collide with any well-formed
    // legacy serialization — including legacy files whose offsets had silently
    // overflowed, since the modulo-2^32 value remains 4-byte aligned.
    static constexpr uint32_t SERIALIZE_FORMAT_SENTINEL = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t SERIALIZE_FORMAT_VERSION_V2 = 2;

    std::shared_ptr<Quantizer<QuantTmpl>> quantizer_{nullptr};
    QueryIOStrategy query_io_strategy_{QueryIOStrategy::MULTI_READ};

    Allocator* const allocator_{nullptr};
    VariableRecordLayout<OffsetAndLengthLocationPolicy, MemoryBlockIO, IOTmpl> layout_{};
    uint64_t max_code_size_{0};
};

}  // namespace vsag

#include "sparse_vector_datacell.inl"
