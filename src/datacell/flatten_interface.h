
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

    virtual void
    QueryWithFilterInnerProduct(float* result_dists,
                                const float* /*filter_inner_products*/,
                                const ComputerInterfacePtr& computer,
                                const InnerIdType* idx,
                                InnerIdType id_count,
                                QueryContext* ctx = nullptr) {
        this->Query(result_dists, computer, idx, id_count, ctx);
    }

    virtual ComputerInterfacePtr
    FactoryComputer(const void* query) = 0;

    [[nodiscard]] virtual bool
    SupportResidualQueryTransform() const {
        return false;
    }

    [[nodiscard]] virtual uint64_t
    GetResidualQueryTransformSize() const {
        return 0;
    }

    virtual void
    TransformResidualQuery(const float* /*query*/, float* /*transformed_query*/) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "residual query transform is not supported");
    }

    virtual ComputerInterfacePtr
    FactoryComputerFromResidualQuery(const float* /*transformed_query*/) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "transformed residual query is not supported");
    }

    // Reinitializes a regular query computer from transformed residual data. Implementations may
    // reuse the existing computer allocation. Any query-derived computer created from its previous
    // state, such as a FastScan computer, must not be reused after this call.
    virtual void
    ResetComputerFromResidualQuery(const float* transformed_query, ComputerInterfacePtr& computer) {
        computer = this->FactoryComputerFromResidualQuery(transformed_query);
    }

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

    [[nodiscard]] virtual bool
    SupportFastScan32() const {
        return false;
    }

    [[nodiscard]] virtual uint64_t
    GetFastScan32BlockSize() const {
        return 0;
    }

    virtual ComputerInterfacePtr
    FactoryFastScan32Computer(const ComputerInterfacePtr& /*computer*/) const {
        return nullptr;
    }

    virtual void
    PackageFastScan32(const InnerIdType* /*ids*/,
                      InnerIdType /*valid_size*/,
                      uint8_t* /*block*/) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "32-vector FastScan packaging is not supported");
    }

    virtual void
    PackageFastScan32Residual(const InnerIdType* /*ids*/,
                              InnerIdType /*valid_size*/,
                              const float* /*transformed_centroid*/,
                              uint8_t* /*block*/) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "residual 32-vector FastScan packaging is not supported");
    }

    virtual void
    QueryFastScan32Batch(float* /*result_dists*/,
                         uint32_t* /*computed_masks*/,
                         const ComputerInterfacePtr& /*computer*/,
                         const ComputerInterfacePtr& /*fastscan_computer*/,
                         const uint8_t* /*blocks*/,
                         InnerIdType /*total_size*/,
                         float* /*filter_inner_products*/ = nullptr,
                         QueryContext* /*ctx*/ = nullptr) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "batched 32-vector FastScan query is not supported");
    }

    virtual void
    QueryFastScan32BatchSharedResidual(float* /*result_dists*/,
                                       uint32_t* /*computed_masks*/,
                                       const ComputerInterfacePtr& /*computer*/,
                                       const ComputerInterfacePtr& /*fastscan_computer*/,
                                       const uint8_t* /*blocks*/,
                                       InnerIdType /*total_size*/,
                                       float /*query_bucket_norm_sqr*/,
                                       float* /*filter_inner_products*/ = nullptr,
                                       QueryContext* /*ctx*/ = nullptr) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "shared residual FastScan query is not supported");
    }

    [[nodiscard]] virtual float
    ComputeTransformedResidualQueryNormSqr(const float* /*transformed_query*/) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "transformed residual query norm is not supported");
    }
    virtual void
    EnableExternalFilterCodeStorage() {
    }

    [[nodiscard]] virtual uint64_t
    GetFilterCodeSize() const {
        return 0;
    }

    virtual void
    InsertVectorWithFilterCode(const void* /*vector*/,
                               InnerIdType /*idx*/,
                               uint8_t* /*filter_code*/) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "external filter-code insertion is not supported");
    }

    virtual void
    SetFastScan32Code(const uint8_t* /*filter_code*/,
                      InnerIdType /*index_in_block*/,
                      uint8_t* /*block*/) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "32-vector FastScan code update is not supported");
    }

    virtual void
    UnpackFastScan32Code(const uint8_t* /*block*/,
                         InnerIdType /*index_in_block*/,
                         uint8_t* /*filter_code*/) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "32-vector FastScan code unpacking is not supported");
    }

    virtual void
    SetFastScan32ResidualCode(const uint8_t* /*filter_code*/,
                              const float* /*transformed_centroid*/,
                              InnerIdType /*index_in_block*/,
                              uint8_t* /*block*/) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "residual 32-vector FastScan code update is not supported");
    }

    virtual void
    QueryWithFilterCodes(float* /*result_dists*/,
                         const float* /*hint_dists*/,
                         const float* /*filter_inner_products*/,
                         const ComputerInterfacePtr& /*computer*/,
                         const InnerIdType* /*idx*/,
                         const uint8_t* /*filter_codes*/,
                         InnerIdType /*id_count*/,
                         QueryContext* /*ctx*/ = nullptr) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "query with external filter codes is not supported");
    }

    virtual void
    QueryWithFilterInnerProducts(float* /*result_dists*/,
                                 uint8_t* /*computed*/,
                                 const float* /*filter_inner_products*/,
                                 const ComputerInterfacePtr& /*computer*/,
                                 const InnerIdType* /*idx*/,
                                 InnerIdType /*id_count*/,
                                 QueryContext* /*ctx*/ = nullptr) const {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "supplement-only query is not supported");
    }

    [[nodiscard]] virtual bool
    GetCodesByIdWithFilterCode(InnerIdType /*id*/,
                               const uint8_t* /*filter_code*/,
                               uint8_t* /*codes*/) const {
        return false;
    }

    virtual float
    ComputePairVectorsWithFilterCodes(InnerIdType /*id1*/,
                                      const uint8_t* /*filter_code1*/,
                                      InnerIdType /*id2*/,
                                      const uint8_t* /*filter_code2*/) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "pair distance with external filter codes is not supported");
    }

    virtual void
    PrefetchSupplement(InnerIdType id) {
        this->Prefetch(id);
    }

    virtual void
    DiscardFilterCodes() {
    }

    [[nodiscard]] virtual bool
    HasFilterCodes() const {
        return true;
    }

    virtual void
    MergeSupplementCodes(const FlattenInterfacePtr& /*other*/, InnerIdType /*bias*/) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "supplement-only merge is not supported");
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

    /**
     * @brief Byte size of the underlying io data (io_->size_) written by
     * Serialize between the interface fields and the tail.
     *
     * A positive value also signals that this implementation serializes in
     * the strict head / io data / tail form, so chunked serialization can
     * precompute the boundaries; the default 0 opts out.
     */
    [[nodiscard]] virtual uint64_t
    GetIOSize() const {
        return 0;
    }

    /**
     * @brief Chunked-restore hooks: ReserveIO, WriteRaw and DeserializeTail.
     *
     * Implement all three or none. The parallel load probes only ReserveIO and
     * treats UNSUPPORTED_INDEX_OPERATION as "this io keeps no deserialized
     * bytes", falling back to consuming the component in place; it then assumes
     * the other two are available. An implementation that provided two of the
     * three would fail in the tail phase, after the chunk writes already landed.
     * Both current implementations gate the three on one `if constexpr`, so they
     * cannot come apart; keep it that way rather than relying on a runtime check.
     */
    /**
     * @brief Read the head (interface fields + io size) and pre-allocate
     * the io data extent, so that WriteRaw can fill it concurrently.
     *
     * The extent is overwritten in full right after this call, so the io
     * backing it should use a backend declaring
     * Capabilities::CanResizeForOverwrite; without it the pre-allocation
     * falls back to a zero-filling Resize, which is a measurable cost for
     * large components.
     *
     * @return the io data size in bytes
     */
    virtual uint64_t
    ReserveIO(StreamReader& reader) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "FlattenInterface does not support ReserveIO");
    }

    /**
     * @brief Write raw io data bytes into [offset, offset + size); the
     * extent must be pre-allocated by ReserveIO. Callers may invoke this
     * concurrently on non-overlapping ranges.
     */
    virtual void
    WriteRaw(const uint8_t* data, uint64_t size, uint64_t offset) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "FlattenInterface does not support WriteRaw");
    }

    /// Read the tail (e.g. the quantizer) following the io data.
    virtual void
    DeserializeTail(StreamReader& reader) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "FlattenInterface does not support DeserializeTail");
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
