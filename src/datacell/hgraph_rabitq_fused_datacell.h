// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <cstring>
#include <shared_mutex>
#include <string>
#include <utility>

#include "dense_duplicate_tracker.h"
#include "graph_datacell_parameter.h"
#include "graph_interface.h"
#include "index_common_param.h"
#include "rabitq_fused_code_storage.h"

namespace vsag {

class HGraphRaBitQFusedDataCell final : public GraphInterface, public RabitQFusedInterface {
public:
    struct NodeView {
        const uint8_t* record;
        const InnerIdType* neighbors;
        const uint8_t* one_bit_code;
        const uint8_t* supplement_code;
        uint32_t neighbor_count;
        uint32_t cluster_id;
    };

    struct CodeView {
        const uint8_t* one_bit_code;
        const uint8_t* supplement_code;
        uint32_t cluster_id;
    };

    HGraphRaBitQFusedDataCell(const GraphDataCellParamPtr& graph_param,
                              uint64_t one_bit_code_size,
                              uint64_t supplement_code_size,
                              const IndexCommonParam& common_param);

    void
    InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbor_ids) override;

    [[nodiscard]] uint32_t
    GetNeighborSize(InnerIdType id) const override;

    void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const override;

    [[nodiscard]] bool
    CheckIdExists(InnerIdType id) const override;

    void
    Resize(InnerIdType new_size) override;

    void
    Prefetch(InnerIdType id, uint32_t neighbor_i) override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(StreamReader& reader) override;

    [[nodiscard]] uint64_t
    GetMemoryUsage() const override;

    DuplicateTrackerPtr
    CreateDuplicateTracker() override;

    void
    SetNodeCodes(InnerIdType id,
                 LabelType label,
                 uint32_t cluster_id,
                 const uint8_t* one_bit_code,
                 const uint8_t* supplement_code);

    void
    SetLabel(InnerIdType id, LabelType label);

    [[nodiscard]] bool
    GetFusedCodeView(InnerIdType id, RaBitQFusedCodeView& view) const override;

    void
    SetFusedCodes(InnerIdType id,
                  uint32_t cluster_id,
                  const uint8_t* one_bit_code,
                  const uint8_t* supplement_code) override;

    void
    PrefetchFusedCodes(InnerIdType id, bool include_supplement) const override;

    void
    PrefetchNodeHeader(InnerIdType id) const {
        constexpr uint64_t cache_line_size = 64;
        const auto* record = GetNodeRecord(id);
        for (uint64_t offset = 0; offset < one_bit_offset_; offset += cache_line_size) {
            __builtin_prefetch(record + offset, 0, 2);
        }
    }

    void
    PrefetchFusedFilter(InnerIdType id) const {
        const auto* record = GetNodeRecord(id);
        PrefetchRange(record + cluster_id_offset_,
                      one_bit_offset_ + one_bit_code_size_ - cluster_id_offset_,
                      3);
    }

    void
    PrefetchFusedSupplement(InnerIdType id) const {
        PrefetchRange(GetNodeRecord(id) + supplement_offset_, supplement_code_size_, 2);
    }

    [[nodiscard]] uint64_t
    FusedOneBitCodeSize() const override {
        return one_bit_code_size_;
    }

    [[nodiscard]] uint64_t
    FusedSupplementCodeSize() const override {
        return supplement_code_size_;
    }

    void
    SetCodecModel(std::string codec_model) {
        codec_model_ = std::move(codec_model);
    }

    [[nodiscard]] const std::string&
    CodecModel() const {
        return codec_model_;
    }

    [[nodiscard]] const uint8_t*
    GetNodeRecord(InnerIdType id) const {
        return storage_.data() + aligned_offset_ + static_cast<uint64_t>(id) * record_size_;
    }

    [[nodiscard]] const InnerIdType*
    GetNeighborData(const uint8_t* record) const;

    [[nodiscard]] bool
    ResolveNeighbor(InnerIdType stored_neighbor, InnerIdType& neighbor) const {
        neighbor = stored_neighbor;
        return neighbor < total_count_;
    }

    [[nodiscard]] const uint8_t*
    GetOneBitCode(const uint8_t* record) const;

    [[nodiscard]] const uint8_t*
    GetSupplementCode(const uint8_t* record) const;

    [[nodiscard]] LabelType
    GetLabel(const uint8_t* record) const;

    [[nodiscard]] uint32_t
    GetClusterId(const uint8_t* record) const;

    [[nodiscard]] NodeView
    GetNodeView(InnerIdType id) const {
        const auto* record =
            storage_.data() + aligned_offset_ + static_cast<uint64_t>(id) * record_size_;
        uint32_t neighbor_count = 0;
        uint32_t cluster_id = 0;
        std::memcpy(&neighbor_count, record, sizeof(neighbor_count));
        std::memcpy(&cluster_id, record + cluster_id_offset_, sizeof(cluster_id));
        return {record,
                reinterpret_cast<const InnerIdType*>(record + neighbors_offset_),
                record + one_bit_offset_,
                record + supplement_offset_,
                neighbor_count,
                cluster_id};
    }

    [[nodiscard]] CodeView
    GetCodeView(InnerIdType id) const {
        const auto* record =
            storage_.data() + aligned_offset_ + static_cast<uint64_t>(id) * record_size_;
        uint32_t cluster_id = 0;
        std::memcpy(&cluster_id, record + cluster_id_offset_, sizeof(cluster_id));
        return {record + one_bit_offset_, record + supplement_offset_, cluster_id};
    }

    [[nodiscard]] uint64_t
    RecordSize() const {
        return record_size_;
    }

    [[nodiscard]] InnerIdType
    Capacity() const {
        return max_capacity_;
    }

    [[nodiscard]] uint64_t
    OneBitOffset() const {
        return one_bit_offset_;
    }

    [[nodiscard]] uint64_t
    NeighborsOffset() const {
        return neighbors_offset_;
    }

    [[nodiscard]] uint64_t
    ClusterIdOffset() const {
        return cluster_id_offset_;
    }

    [[nodiscard]] uint64_t
    LabelOffset() const {
        return label_offset_;
    }

    [[nodiscard]] uint64_t
    SupplementOffset() const {
        return supplement_offset_;
    }

    [[nodiscard]] uint64_t
    OneBitCodeSize() const {
        return one_bit_code_size_;
    }

private:
    static void
    PrefetchRange(const uint8_t* begin, uint64_t size, int locality) {
        constexpr uintptr_t cache_line_size = 64;
        const auto begin_address = reinterpret_cast<uintptr_t>(begin);
        const auto first = begin_address & ~(cache_line_size - 1U);
        const auto end = (begin_address + size + cache_line_size - 1U) & ~(cache_line_size - 1U);
        for (auto address = first; address < end; address += cache_line_size) {
            if (locality == 3) {
                __builtin_prefetch(reinterpret_cast<const void*>(address), 0, 3);
            } else {
                __builtin_prefetch(reinterpret_cast<const void*>(address), 0, 2);
            }
        }
    }

    static uint64_t
    AlignUp(uint64_t value, uint64_t alignment);

    [[nodiscard]] uint8_t*
    MutableNodeRecord(InnerIdType id);

    void
    Reallocate(InnerIdType new_capacity);

private:
    Vector<uint8_t> storage_;
    uint64_t aligned_offset_{0};
    uint64_t record_size_{0};
    uint64_t neighbors_offset_{0};
    uint64_t cluster_id_offset_{0};
    uint64_t label_offset_{0};
    uint64_t one_bit_offset_{0};
    uint64_t supplement_offset_{0};
    uint64_t one_bit_code_size_{0};
    uint64_t supplement_code_size_{0};
    int64_t dim_{0};
    std::string codec_model_;
    // Guards build-time slab resize/reallocation. Mutable HGraph searches hold the owning
    // HGraph's global read lock while resize takes its global write lock; after SetImmutable,
    // the slab cannot be resized and read-path access is lock-free.
    mutable std::shared_mutex storage_mutex_;
};

DEFINE_POINTER(HGraphRaBitQFusedDataCell);

}  // namespace vsag
