
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
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include "basic_types.h"
#include "container_types.h"
#include "flatten_datacell_parameter.h"
#include "flatten_interface_parameter.h"
#include "hash_types.h"
#include "impl/runtime_parameter.h"
#include "index_common_param_fwd.h"
#include "io/reader_io.h"
#include "quantization/computer.h"
#include "query_context.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "type_helpers.h"
#include "utils/pointer_define.h"
#include "vsag/constants.h"

namespace vsag {

DEFINE_POINTER(FlattenInterface);

class FlattenInterface {
public:
    FlattenInterface() = default;

    static FlattenInterfacePtr
    MakeInstance(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param);

public:
    virtual void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) = 0;

    virtual void
    QueryWithDistanceFilter(float* result_dists,
                            const ComputerInterfacePtr& computer,
                            const InnerIdType* idx,
                            InnerIdType id_count,
                            float threshold,
                            QueryContext* ctx = nullptr) {
        this->Query(result_dists, computer, idx, id_count, ctx);
    }

    virtual void
    QueryWithDistanceLowerBound(float* result_dists,
                                float* lower_bounds,
                                const ComputerInterfacePtr& computer,
                                const InnerIdType* idx,
                                InnerIdType id_count,
                                QueryContext* ctx = nullptr) {
        this->Query(result_dists, computer, idx, id_count, ctx);
        if (lower_bounds != nullptr) {
            std::fill(lower_bounds, lower_bounds + id_count, std::numeric_limits<float>::max());
        }
    }

    virtual ComputerInterfacePtr
    FactoryComputer(const void* query) = 0;

    virtual void
    Train(const void* data, uint64_t count) = 0;

    virtual void
    InsertVector(const void* vector, InnerIdType idx = std::numeric_limits<InnerIdType>::max()) = 0;

    virtual bool
    UpdateVector(const void* vector, InnerIdType idx = std::numeric_limits<InnerIdType>::max()) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "UpdateVector not implemented in FlattenInterface");
    };

    virtual void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec = nullptr) = 0;

    virtual float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) = 0;

    bool
    CompareVectors(InnerIdType id1, InnerIdType id2) {
        bool release1, release2;
        const auto* codes1 = this->GetCodesById(id1, release1);
        const auto* codes2 = this->GetCodesById(id2, release2);
        bool result = (std::memcmp(codes1, codes2, this->code_size_) == 0);
        if (release1) {
            this->Release(codes1);
        }
        if (release2) {
            this->Release(codes2);
        }
        return result;
    }

    virtual void
    Prefetch(InnerIdType id) = 0;

    [[nodiscard]] virtual std::string
    GetQuantizerName() = 0;

    [[nodiscard]] virtual MetricType
    GetMetricType() = 0;

    virtual void
    Resize(InnerIdType capacity) = 0;

    virtual void
    ExportModel(const FlattenInterfacePtr& other) const = 0;

    virtual void
    InitIO(const IOParamPtr& io_param) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "InitIO not implemented in FlattenInterface");
    }
    virtual int64_t
    GetMemoryUsage() const {
        return 0;
    }

    virtual IndexCommonParam
    ExportCommonParam();

public:
    void
    EnableSlotRedirect(Allocator* allocator) {
        redirect_allocator_ = allocator;
        use_slot_redirect_ = true;
        slot_redirect_.clear();
        next_physical_slot_ = 0;
    }

    void
    SetDuplicateSlot(InnerIdType dup_id, InnerIdType rep_id) {
        if (!use_slot_redirect_) {
            return;
        }
        std::lock_guard lock(redirect_mutex_);
        if (dup_id >= slot_redirect_.size()) {
            slot_redirect_.resize(dup_id + 1, std::numeric_limits<InnerIdType>::max());
        }
        slot_redirect_[dup_id] = GetPhysicalSlotLocked(rep_id);
    }

    InnerIdType
    AllocatePhysicalSlot(InnerIdType logical_id) {
        if (!use_slot_redirect_) {
            return logical_id;
        }
        std::lock_guard lock(redirect_mutex_);
        if (logical_id >= slot_redirect_.size()) {
            slot_redirect_.resize(logical_id + 1, std::numeric_limits<InnerIdType>::max());
        }
        auto slot = next_physical_slot_++;
        slot_redirect_[logical_id] = slot;
        return slot;
    }

    [[nodiscard]] InnerIdType
    GetPhysicalSlotCount() const {
        return use_slot_redirect_ ? next_physical_slot_ : total_count_;
    }

    void
    SerializeSlotRedirect(StreamWriter& writer) const {
        StreamWriter::WriteObj(writer, use_slot_redirect_);
        if (use_slot_redirect_) {
            StreamWriter::WriteObj(writer, next_physical_slot_);
            uint64_t redirect_size = slot_redirect_.size();
            StreamWriter::WriteObj(writer, redirect_size);
            for (uint64_t i = 0; i < redirect_size; ++i) {
                StreamWriter::WriteObj(writer, slot_redirect_[i]);
            }
        }
    }

    void
    DeserializeSlotRedirect(StreamReader& reader) {
        bool has_redirect = false;
        StreamReader::ReadObj(reader, has_redirect);
        if (has_redirect) {
            use_slot_redirect_ = true;
            StreamReader::ReadObj(reader, next_physical_slot_);
            uint64_t redirect_size = 0;
            StreamReader::ReadObj(reader, redirect_size);
            slot_redirect_.resize(redirect_size);
            for (uint64_t i = 0; i < redirect_size; ++i) {
                StreamReader::ReadObj(reader, slot_redirect_[i]);
            }
        }
    }

public:
    virtual bool
    SetRuntimeParameters(const UnorderedMap<std::string, float>& new_params) {
        bool ret = false;
        auto iter = new_params.find(PREFETCH_STRIDE_CODE);
        if (iter != new_params.end()) {
            prefetch_stride_code_ = static_cast<uint32_t>(iter->second);
            ret = true;
        }

        iter = new_params.find(PREFETCH_DEPTH_CODE);
        if (iter != new_params.end()) {
            prefetch_depth_code_ = static_cast<uint32_t>(iter->second);
            ret = true;
        }

        return ret;
    }

    virtual bool
    Decode(const uint8_t* codes, float* vector) = 0;

    virtual bool
    Encode(const float* vector, uint8_t* codes) = 0;

    [[nodiscard]] virtual const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const = 0;

    virtual void
    Release(const uint8_t* data) const = 0;

    virtual bool
    GetCodesById(InnerIdType id, uint8_t* codes) const = 0;

    [[nodiscard]] virtual InnerIdType
    TotalCount() const {
        std::shared_lock lock(mutex_);
        return this->total_count_;
    }

    virtual void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->total_count_);
        StreamWriter::WriteObj(writer, this->max_capacity_);
        StreamWriter::WriteObj(writer, this->code_size_);
    }

    virtual void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) {
        StreamReader::ReadObj(reader, this->total_count_);
        StreamReader::ReadObj(reader, this->max_capacity_);
        StreamReader::ReadObj(reader, this->code_size_);
    }

    uint64_t
    CalcSerializeSize() {
        auto calSizeFunc = [](uint64_t cursor, uint64_t size, void* buf) { return; };
        WriteFuncStreamWriter writer(calSizeFunc, 0);
        this->Serialize(writer);
        return writer.cursor_;
    }

    [[nodiscard]] virtual bool
    InMemory() const {
        return true;
    }

    [[nodiscard]] virtual bool
    HoldMolds() const {
        return false;
    }

    virtual void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "MergeOther not implemented");
    }

    virtual void
    Move(InnerIdType from, InnerIdType to) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Move not implemented in FlattenInterface");
    }

    virtual void
    ShrinkToFit(InnerIdType capacity) {
    }

public:
    mutable std::shared_mutex mutex_;

    InnerIdType total_count_{0};
    InnerIdType max_capacity_{800};
    uint32_t code_size_{0};
    uint32_t prefetch_stride_code_{1};
    uint32_t prefetch_depth_code_{1};

protected:
    std::vector<InnerIdType> slot_redirect_;
    InnerIdType next_physical_slot_{0};
    bool use_slot_redirect_{false};
    Allocator* redirect_allocator_{nullptr};
    mutable std::mutex redirect_mutex_;

    InnerIdType
    GetPhysicalSlotLocked(InnerIdType id) const {
        if (id >= slot_redirect_.size()) {
            return id;
        }
        auto slot = slot_redirect_[id];
        return (slot == std::numeric_limits<InnerIdType>::max()) ? id : slot;
    }

    InnerIdType
    GetPhysicalSlot(InnerIdType id) const {
        if (!use_slot_redirect_) {
            return id;
        }
        std::lock_guard<std::mutex> lock(redirect_mutex_);
        return GetPhysicalSlotLocked(id);
    }
};

}  // namespace vsag
