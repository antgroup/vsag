
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
#include <shared_mutex>
#include <string>
#include <vector>

#include "basic_types.h"
#include "flatten_datacell_parameter.h"
#include "flatten_interface_parameter.h"
#include "hash_types.h"
#include "impl/runtime_parameter.h"
#include "index_common_param_fwd.h"
#include "io/reader_io/reader_io.h"
#include "quantization/computer.h"
#include "query_context.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "type_helpers.h"
#include "utils/pointer_define.h"
#include "vsag/allocator.h"
#include "vsag/constants.h"
#include "vsag/dataset.h"

namespace vsag {

DEFINE_POINTER(FlattenInterface);

class FlattenCodesLease {
public:
    FlattenCodesLease() = default;

    ~FlattenCodesLease();

    FlattenCodesLease(const FlattenCodesLease&) = delete;
    FlattenCodesLease&
    operator=(const FlattenCodesLease&) = delete;

    FlattenCodesLease(FlattenCodesLease&& other) noexcept;
    FlattenCodesLease&
    operator=(FlattenCodesLease&& other) noexcept;

    [[nodiscard]] explicit operator bool() const {
        return data_ != nullptr;
    }

    [[nodiscard]] const uint8_t*
    Data() const {
        return data_;
    }

private:
    friend class FlattenInterface;

    FlattenCodesLease(const FlattenInterface* source,
                      const uint8_t* data,
                      bool needs_release) noexcept
        : source_(source), data_(data), needs_release_(needs_release) {
    }

    void
    Reset() noexcept;

    const FlattenInterface* source_{nullptr};
    const uint8_t* data_{nullptr};
    bool needs_release_{false};
};

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
    QueryById(float* result_dists,
              InnerIdType query_id,
              const InnerIdType* idx,
              InnerIdType id_count,
              QueryContext* /*ctx*/ = nullptr) {
        // This fallback performs one pairwise call per ID. Datacells with a batched or
        // storage-aware implementation should override it.
        for (InnerIdType i = 0; i < id_count; ++i) {
            result_dists[i] = this->ComputePairVectors(query_id, idx[i]);
        }
    }

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

    virtual void
    QueryWithDistanceHint(float* result_dists,
                          const float* /*hint_dists*/,
                          const ComputerInterfacePtr& computer,
                          const InnerIdType* idx,
                          InnerIdType id_count,
                          QueryContext* ctx = nullptr) {
        this->Query(result_dists, computer, idx, id_count, ctx);
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
        auto codes1 = this->AcquireCodesById(id1);
        auto codes2 = this->AcquireCodesById(id2);
        return codes1 and codes2 and
               std::memcmp(codes1.Data(), codes2.Data(), this->code_size_) == 0;
    }

    virtual bool
    CompareRawVectorWithId(const void* vector, InnerIdType id) {
        if (vector == nullptr) {
            return false;
        }
        std::vector<uint8_t> encoded(this->code_size_);
        if (not this->Encode(static_cast<const float*>(vector), encoded.data())) {
            return false;
        }

        auto codes = this->AcquireCodesById(id);
        if (not codes) {
            return false;
        }
        return std::memcmp(encoded.data(), codes.Data(), this->code_size_) == 0;
    }

    virtual void
    Prefetch(InnerIdType id) = 0;

    [[nodiscard]] virtual std::string
    GetQuantizerName() = 0;

    [[nodiscard]] virtual uint64_t
    GetQuantizerCodeSize() const {
        return this->code_size_;
    }

    [[nodiscard]] virtual bool
    SupportSplitCodeStorage() const {
        return false;
    }

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
    virtual uint64_t
    GetMemoryUsage() const {
        return 0;
    }

    virtual IndexCommonParam
    ExportCommonParam();

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

    [[nodiscard]] FlattenCodesLease
    AcquireCodesById(InnerIdType id) const {
        bool needs_release = false;
        const uint8_t* data = this->GetCodesById(id, needs_release);
        return FlattenCodesLease(this, data, needs_release);
    }

    virtual void
    GetSparseVectorByInnerId(InnerIdType inner_id,
                             SparseVector* data,
                             Allocator* specified_allocator) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "GetSparseVectorByInnerId is not implemented in FlattenInterface");
    }

    virtual void
    Release(const uint8_t* data) const = 0;

    virtual bool
    GetCodesById(InnerIdType id, uint8_t* codes) const = 0;

    // Optional fast path for algorithms that can consume a contiguous FP32 matrix directly.
    // Implementations return nullptr when data is quantized, external, block-backed, or otherwise
    // unavailable as a single stable buffer; callers must fall back to Query/GetCodesById.
    [[nodiscard]] virtual const float*
    TryGetContiguousRawFloatData(uint64_t* row_stride = nullptr) {
        if (row_stride != nullptr) {
            *row_stride = 0;
        }
        return nullptr;
    }

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
    Deserialize(LvalueOrRvalue<StreamReader> reader) {
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
    DistanceEvaluationBackend backend_{DistanceEvaluationBackend::UNKNOWN};
};

inline FlattenCodesLease::~FlattenCodesLease() {
    Reset();
}

inline FlattenCodesLease::FlattenCodesLease(FlattenCodesLease&& other) noexcept
    : source_(other.source_), data_(other.data_), needs_release_(other.needs_release_) {
    other.source_ = nullptr;
    other.data_ = nullptr;
    other.needs_release_ = false;
}

inline FlattenCodesLease&
FlattenCodesLease::operator=(FlattenCodesLease&& other) noexcept {
    if (this != &other) {
        Reset();
        source_ = other.source_;
        data_ = other.data_;
        needs_release_ = other.needs_release_;
        other.source_ = nullptr;
        other.data_ = nullptr;
        other.needs_release_ = false;
    }
    return *this;
}

inline void
FlattenCodesLease::Reset() noexcept {
    if (needs_release_ and source_ != nullptr and data_ != nullptr) {
        source_->Release(data_);
    }
    source_ = nullptr;
    data_ = nullptr;
    needs_release_ = false;
}

}  // namespace vsag
