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

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "datacell/flatten_interface.h"
#include "datacell/flatten_optimized_build_interface.h"
#include "inner_string_params.h"
#include "io/common/io_parameter.h"
#include "io/memory_io/memory_io.h"
#include "io/memory_io/memory_io_parameter.h"
#include "layout/fixed_layout.h"
#include "quantization/scalar_quantization/sq_local_split_quantizer.h"
#include "storage/stream_writer.h"

namespace vsag {

/**
 * Split storage for the rotated local scalar quantizer.
 *
 *   filter record     : [v_min f32][delta f32][|d_hat|^2 f32][sum(c) u32][high x bits]
 *   supplement record : [low y bits]
 *   build record      : [same metadata][one code byte per dimension]   (optimized build only)
 *
 * The two final records live in their own IO layouts, so the cheap high-bit filter can be kept
 * in memory while the low bits can be placed on a slower backend (same intent as the RaBitQ
 * x + y split). Query() reads both and evaluates the full (x + y)-bit distance.
 *
 * During construction the datacell keeps a temporary in-memory "scalar" layout with one code
 * byte per dimension. Graph construction is dominated by id-to-id distances; the byte layout
 * lets those run through a branch-free vectorizable kernel, and the packed split records are
 * only produced once by FinalizeOptimizedBuild().
 */
template <MetricType metric, typename FilterIOTmpl, typename SupplementIOTmpl = FilterIOTmpl>
class SQLocalSplitDataCell : public FlattenInterface, public FlattenOptimizedBuildInterface {
public:
    using QuantizerT = SQLocalSplitQuantizer<metric>;
    using ComputerT = typename QuantizerT::Computer;
    using FilterLayoutT = FixedLayout<FilterIOTmpl>;
    using SupplementLayoutT = FixedLayout<SupplementIOTmpl>;
    using BuildLayoutT = FixedLayout<MemoryIO>;

    SQLocalSplitDataCell(const QuantizerParamPtr& quantization_param,
                         const IOParamPtr& io_param,
                         const IOParamPtr& supplement_io_param,
                         const IndexCommonParam& common_param)
        : common_param_(common_param), allocator_(common_param.allocator_.get()) {
        CHECK_ARGUMENT(quantization_param != nullptr,
                       "sq_local_split requires a quantizer parameter");
        CHECK_ARGUMENT(io_param != nullptr, "sq_local_split requires an io parameter");
        this->quantizer_ = std::make_shared<QuantizerT>(quantization_param, common_param);
        this->filter_layout_ = std::make_shared<FilterLayoutT>(
            this->quantizer_->GetFilterCodeSize(), io_param, common_param);
        this->supplement_layout_ = std::make_shared<SupplementLayoutT>(
            this->quantizer_->GetSupplementCodeSize(),
            supplement_io_param == nullptr ? io_param : supplement_io_param,
            common_param);
        this->code_size_ = static_cast<uint32_t>(this->quantizer_->GetFilterCodeSize() +
                                                 this->quantizer_->GetSupplementCodeSize());
        this->metric_ = common_param.metric_;
        this->max_capacity_ = 0;
        this->scalar_record_size_ = this->quantizer_->GetScalarCodeSize();
    }

    // ------------------------------------------------------------------ FlattenInterface

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* /*ctx*/ = nullptr) override {
        auto* comp = static_cast<ComputerT*>(computer.get());
        CHECK_ARGUMENT(comp != nullptr, "sq_local_split query requires a valid computer");
        auto& scratch = this->scratch();
        if (this->optimized_build_active_) {
            for (InnerIdType i = 0; i < id_count; ++i) {
                this->prefetch(idx, i, id_count);
                auto lease = this->build_layout_->Acquire(idx[i]);
                CHECK_ARGUMENT(lease, "sq_local_split failed to read build codes");
                result_dists[i] = this->quantizer_->ComputeScalarDistance(*comp, lease.Data());
            }
            return;
        }
        const auto filter_size = this->quantizer_->GetFilterCodeSize();
        const auto supplement_size = this->quantizer_->GetSupplementCodeSize();
        scratch.resize(filter_size + supplement_size);
        auto* filter_buf = scratch.data();
        auto* supplement_buf = scratch.data() + filter_size;
        for (InnerIdType i = 0; i < id_count; ++i) {
            this->prefetch(idx, i, id_count);
            CHECK_ARGUMENT(this->filter_layout_->Read(idx[i], filter_buf) and
                               this->supplement_layout_->Read(idx[i], supplement_buf),
                           "sq_local_split failed to read codes");
            result_dists[i] =
                this->quantizer_->ComputeFullDistance(*comp, filter_buf, supplement_buf);
        }
    }

    void
    QueryWithDistanceLowerBound(float* result_dists,
                                float* lower_bounds,
                                const ComputerInterfacePtr& computer,
                                const InnerIdType* idx,
                                InnerIdType id_count,
                                QueryContext* ctx = nullptr) override {
        this->Query(result_dists, computer, idx, id_count, ctx);
        if (lower_bounds == nullptr or this->optimized_build_active_) {
            if (lower_bounds != nullptr) {
                std::copy(result_dists, result_dists + id_count, lower_bounds);
            }
            return;
        }
        auto* comp = static_cast<ComputerT*>(computer.get());
        const std::size_t filter_size = this->quantizer_->GetFilterCodeSize();
        std::vector<uint8_t> filter_buf(filter_size);
        for (InnerIdType i = 0; i < id_count; ++i) {
            if (comp == nullptr or not this->filter_layout_->Read(idx[i], filter_buf.data())) {
                lower_bounds[i] = result_dists[i];
                continue;
            }
            lower_bounds[i] = std::min(
                result_dists[i],
                this->quantizer_->ComputeFilterDistanceLowerBound(*comp, filter_buf.data()));
        }
    }

    [[nodiscard]] ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        return this->quantizer_->FactoryComputer(query);
    }

    /** Build-time computer: the query id's temporary scalar code is the query. */
    [[nodiscard]] ComputerInterfacePtr
    FactoryComputerForBuild(const void* /*query*/, InnerIdType id) override {
        CHECK_ARGUMENT(this->optimized_build_active_,
                       "sq_local_split optimized-build computer requested while inactive");
        auto computer = std::make_shared<ComputerT>(
            this->quantizer_->GetDim(), this->quantizer_->QueryBits(), this->allocator_);
        auto& buffer = this->scratch();
        buffer.resize(this->scalar_record_size_);
        CHECK_ARGUMENT(this->build_layout_->Read(id, buffer.data()),
                       "sq_local_split failed to read the build query code");
        this->quantizer_->RestoreComputerFromScalar(*computer, buffer.data());
        return computer;
    }

    bool
    BeginOptimizedBuild(const FlattenOptimizedBuildContext& /*context*/) override {
        if (this->optimized_build_active_) {
            return false;
        }
        auto io_param = std::make_shared<MemoryIOParameter>();
        this->build_layout_ = std::make_shared<BuildLayoutT>(io_param, this->common_param_);
        this->build_layout_->SetCodeSize(this->scalar_record_size_);
        if (this->max_capacity_ > 0) {
            this->build_layout_->Resize(this->max_capacity_);
        }
        this->optimized_build_active_ = true;
        return true;
    }

    void
    FinalizeOptimizedBuild() override {
        if (not this->optimized_build_active_) {
            return;
        }
        const auto count = this->total_count_.load(std::memory_order_acquire);
        const auto start = std::chrono::steady_clock::now();
        std::vector<uint8_t> build_code(this->scalar_record_size_);
        std::vector<uint8_t> filter_code(this->quantizer_->GetFilterCodeSize());
        std::vector<uint8_t> supplement_code(this->quantizer_->GetSupplementCodeSize());
        for (InnerIdType id = 0; id < count; ++id) {
            CHECK_ARGUMENT(this->build_layout_->Read(id, build_code.data()),
                           "sq_local_split failed to read build codes while finalizing");
            this->quantizer_->PackScalarToSplit(
                build_code.data(), filter_code.data(), supplement_code.data());
            this->filter_layout_->Write(id, filter_code.data());
            this->supplement_layout_->Write(id, supplement_code.data());
        }
        this->build_layout_.reset();
        this->optimized_build_active_ = false;
    }

    void
    AbortOptimizedBuild() noexcept override {
        this->build_layout_.reset();
        this->optimized_build_active_ = false;
    }

    [[nodiscard]] bool
    IsOptimizedBuildActive() const override {
        return this->optimized_build_active_;
    }

    void
    Train(const void* data, uint64_t count) override {
        this->quantizer_->Train(static_cast<const float*>(data), count);
    }

    void
    InsertVector(const void* vector,
                 InnerIdType idx = std::numeric_limits<InnerIdType>::max()) override {
        const std::lock_guard lock(this->mutex_);
        if (idx == std::numeric_limits<InnerIdType>::max()) {
            idx = this->total_count_.load(std::memory_order_relaxed);
        }
        this->encode_and_write(idx, static_cast<const float*>(vector));
        this->total_count_.store(std::max(this->total_count_.load(std::memory_order_relaxed),
                                          static_cast<InnerIdType>(idx + 1)),
                                 std::memory_order_release);
    }

    void
    BatchInsertVector(const void* vectors,
                      InnerIdType count,
                      InnerIdType* idx_vec = nullptr) override {
        const auto* data = static_cast<const float*>(vectors);
        const std::lock_guard lock(this->mutex_);
        InnerIdType start = this->total_count_.load(std::memory_order_relaxed);
        for (InnerIdType i = 0; i < count; ++i) {
            const InnerIdType idx = idx_vec == nullptr ? start + i : idx_vec[i];
            this->encode_and_write(idx,
                                   data + static_cast<int64_t>(i) * this->quantizer_->GetDim());
            start = std::max(start, static_cast<InnerIdType>(idx + 1));
        }
        this->total_count_.store(start, std::memory_order_release);
    }

    /**
     * Id-to-id distances dominate graph construction (edge selection, reverse-edge pruning).
     * The packed path uses the packed-vs-packed SIMD kernel; the optimized-build path uses the
     * byte-per-dimension kernel, so no decode happens per pair.
     */
    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override {
        auto& buffer = this->scratch();
        if (this->optimized_build_active_) {
            auto lease1 = this->build_layout_->Acquire(id1);
            auto lease2 = this->build_layout_->Acquire(id2);
            CHECK_ARGUMENT(lease1 and lease2, "sq_local_split failed to read build codes");
            return this->quantizer_->ComputeScalarPairDistance(lease1.Data(), lease2.Data());
        }
        const auto filter_size = this->quantizer_->GetFilterCodeSize();
        const auto supplement_size = this->quantizer_->GetSupplementCodeSize();
        buffer.resize(2 * (filter_size + supplement_size));
        auto* filter1 = buffer.data();
        auto* supplement1 = buffer.data() + filter_size;
        auto* filter2 = buffer.data() + filter_size + supplement_size;
        auto* supplement2 = filter2 + filter_size;
        CHECK_ARGUMENT(this->filter_layout_->Read(id1, filter1) and
                           this->supplement_layout_->Read(id1, supplement1) and
                           this->filter_layout_->Read(id2, filter2) and
                           this->supplement_layout_->Read(id2, supplement2),
                       "sq_local_split failed to read codes");
        return this->quantizer_->ComputePairDistance(filter1, supplement1, filter2, supplement2);
    }

    /** Batched id-to-id distances: merge the query code once and reuse it per candidate. */
    void
    QueryById(float* result_dists,
              InnerIdType query_id,
              const InnerIdType* idx,
              InnerIdType id_count,
              QueryContext* /*ctx*/ = nullptr) override {
        auto computer =
            std::make_shared<ComputerT>(this->quantizer_->GetDim(), 0, this->allocator_);
        auto& scratch = this->scratch();
        if (this->optimized_build_active_) {
            auto query_lease = this->build_layout_->Acquire(query_id);
            CHECK_ARGUMENT(query_lease, "sq_local_split failed to read the build query code");
            this->quantizer_->RestoreComputerFromScalar(*computer, query_lease.Data());
            for (InnerIdType i = 0; i < id_count; ++i) {
                this->prefetch(idx, i, id_count);
                auto lease = this->build_layout_->Acquire(idx[i]);
                CHECK_ARGUMENT(lease, "sq_local_split failed to read build codes");
                result_dists[i] = this->quantizer_->ComputeScalarDistance(*computer, lease.Data());
            }
            return;
        }
        const auto filter_size = this->quantizer_->GetFilterCodeSize();
        const auto supplement_size = this->quantizer_->GetSupplementCodeSize();
        scratch.resize(filter_size + supplement_size);
        auto* filter_buf = scratch.data();
        auto* supplement_buf = scratch.data() + filter_size;
        CHECK_ARGUMENT(this->filter_layout_->Read(query_id, filter_buf) and
                           this->supplement_layout_->Read(query_id, supplement_buf),
                       "sq_local_split failed to read the query code");
        this->quantizer_->RestoreComputer(*computer, filter_buf, supplement_buf);
        for (InnerIdType i = 0; i < id_count; ++i) {
            this->prefetch(idx, i, id_count);
            CHECK_ARGUMENT(this->filter_layout_->Read(idx[i], filter_buf) and
                               this->supplement_layout_->Read(idx[i], supplement_buf),
                           "sq_local_split failed to read codes");
            result_dists[i] =
                this->quantizer_->ComputeFullDistance(*computer, filter_buf, supplement_buf);
        }
    }

    void
    Prefetch(InnerIdType id) override {
        if (this->optimized_build_active_) {
            this->build_layout_->Prefetch(id, this->scalar_record_size_);
            return;
        }
        this->filter_layout_->Prefetch(id, this->quantizer_->GetFilterCodeSize());
        this->supplement_layout_->Prefetch(id, this->quantizer_->GetSupplementCodeSize());
    }

    [[nodiscard]] std::string
    GetQuantizerName() override {
        return QUANTIZATION_TYPE_VALUE_SQ_LOCAL_SPLIT;
    }

    /** Filter and supplement records together form one quantized vector. */
    [[nodiscard]] bool
    SupportSplitCodeStorage() const override {
        return true;
    }

    [[nodiscard]] MetricType
    GetMetricType() override {
        return this->metric_;
    }

    void
    Resize(InnerIdType capacity) override {
        if (capacity <= this->max_capacity_) {
            return;
        }
        // Grow geometrically: callers may insert one vector at a time, and a resize per vector
        // would copy the whole io extent every time (quadratic build cost on 1M-scale data).
        const uint64_t doubled = static_cast<uint64_t>(this->max_capacity_) * 2ULL;
        const uint64_t target = std::max<uint64_t>(doubled, capacity);
        const auto grown = static_cast<InnerIdType>(
            std::min<uint64_t>(target, std::numeric_limits<InnerIdType>::max()));
        this->filter_layout_->Resize(grown);
        this->supplement_layout_->Resize(grown);
        if (this->build_layout_ != nullptr) {
            this->build_layout_->Resize(grown);
        }
        this->max_capacity_ = grown;
    }

    void
    ExportModel(const FlattenInterfacePtr& other) const override {
        auto ptr = std::dynamic_pointer_cast<SQLocalSplitDataCell>(other);
        if (ptr == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "sq_local_split export model target type mismatch");
        }
        std::stringstream stream;
        IOStreamWriter writer(stream);
        this->quantizer_->Serialize(writer);
        stream.seekg(0, std::ios::beg);
        IOStreamReader reader(stream);
        ptr->quantizer_->Deserialize(reader);
    }

    bool
    Decode(const uint8_t* codes, float* vector) override {
        return this->quantizer_->Decode(
            codes, codes + this->quantizer_->GetFilterCodeSize(), vector);
    }

    bool
    Encode(const float* vector, uint8_t* codes) override {
        return this->quantizer_->Encode(
            vector, codes, codes + this->quantizer_->GetFilterCodeSize());
    }

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override {
        auto* buffer = static_cast<uint8_t*>(this->allocator_->Allocate(this->code_size_));
        if (buffer == nullptr) {
            need_release = false;
            return nullptr;
        }
        if (not this->GetCodesById(id, buffer)) {
            this->allocator_->Deallocate(buffer);
            need_release = false;
            return nullptr;
        }
        need_release = true;
        return buffer;
    }

    void
    Release(const uint8_t* data) const override {
        if (data != nullptr) {
            this->allocator_->Deallocate(const_cast<uint8_t*>(data));
        }
    }

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override {
        if (codes == nullptr) {
            return false;
        }
        const auto filter_size = this->quantizer_->GetFilterCodeSize();
        const auto supplement_size = this->quantizer_->GetSupplementCodeSize();
        if (this->optimized_build_active_) {
            std::vector<uint8_t> build_code(this->scalar_record_size_);
            if (not this->build_layout_->Read(id, build_code.data())) {
                return false;
            }
            this->quantizer_->PackScalarToSplit(build_code.data(), codes, codes + filter_size);
            return supplement_size > 0;
        }
        if (not this->filter_layout_->Read(id, codes)) {
            return false;
        }
        return this->supplement_layout_->Read(id, codes + filter_size) and supplement_size > 0;
    }

    void
    Move(InnerIdType from, InnerIdType to) override {
        if (this->optimized_build_active_) {
            auto lease = this->build_layout_->Acquire(from);
            CHECK_ARGUMENT(lease, "sq_local_split move failed to read source build code");
            this->build_layout_->Write(to, lease.Data());
            return;
        }
        auto filter_code = this->filter_layout_->Acquire(from);
        auto supplement_code = this->supplement_layout_->Acquire(from);
        CHECK_ARGUMENT(filter_code and supplement_code,
                       "sq_local_split move failed to read source codes");
        this->filter_layout_->Write(to, filter_code.Data());
        this->supplement_layout_->Write(to, supplement_code.Data());
    }

    void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) override {
        auto ptr = std::dynamic_pointer_cast<SQLocalSplitDataCell>(other);
        if (ptr == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "sq_local_split merge type mismatch");
        }
        const auto count = ptr->total_count_.load(std::memory_order_acquire);
        for (InnerIdType i = 0; i < count; ++i) {
            auto filter_code = ptr->filter_layout_->Acquire(i);
            auto supplement_code = ptr->supplement_layout_->Acquire(i);
            if (not filter_code or not supplement_code) {
                break;
            }
            this->filter_layout_->Write(bias + i, filter_code.Data());
            this->supplement_layout_->Write(bias + i, supplement_code.Data());
        }
        this->total_count_.store(std::max(this->total_count_.load(std::memory_order_relaxed),
                                          static_cast<InnerIdType>(bias + count)),
                                 std::memory_order_release);
    }

    void
    Serialize(StreamWriter& writer) override {
        FlattenInterface::Serialize(writer);
        this->filter_layout_->Serialize(writer);
        this->supplement_layout_->Serialize(writer);
        this->quantizer_->Serialize(writer);
    }

    void
    Deserialize(LvalueOrRvalue<StreamReader> reader) override {
        FlattenInterface::Deserialize(reader);
        const auto expected_code_size =
            this->quantizer_->GetFilterCodeSize() + this->quantizer_->GetSupplementCodeSize();
        CHECK_ARGUMENT(this->code_size_ == expected_code_size,
                       "sq_local_split code size mismatch while deserializing");
        this->filter_layout_->Deserialize(reader);
        this->supplement_layout_->Deserialize(reader);
        this->quantizer_->Deserialize(reader);
    }

    [[nodiscard]] uint64_t
    GetMemoryUsage() const override {
        return this->filter_layout_->GetMemoryUsage() + this->supplement_layout_->GetMemoryUsage() +
               (this->build_layout_ == nullptr ? 0 : this->build_layout_->GetMemoryUsage());
    }

    [[nodiscard]] bool
    InMemory() const override {
        return FilterLayoutT::InMemory and SupplementLayoutT::InMemory;
    }

    [[nodiscard]] bool
    HoldMolds() const override {
        return false;
    }

    void
    InitIO(const IOParamPtr& io_param) override {
        this->filter_layout_->InitIO(io_param);
    }

    void
    ShrinkToFit(InnerIdType capacity) override {
        const std::unique_lock lock(this->mutex_);
        const auto total_count = this->total_count_.load(std::memory_order_relaxed);
        this->total_count_.store(std::min(total_count, capacity), std::memory_order_release);
        this->filter_layout_->Shrink(capacity);
        this->supplement_layout_->Shrink(capacity);
        if (this->build_layout_ != nullptr) {
            this->build_layout_->Shrink(capacity);
        }
        this->max_capacity_ = capacity;
    }

    [[nodiscard]] const QuantizerT&
    GetQuantizer() const {
        return *this->quantizer_;
    }

private:
    struct PairCache {
        InnerIdType id{std::numeric_limits<InnerIdType>::max()};
        int build_mode{-1};
        std::unique_ptr<ComputerT> computer{nullptr};
    };

    static PairCache&
    pair_cache() {
        thread_local PairCache cache;
        return cache;
    }

    static std::vector<uint8_t>&
    scratch() {
        thread_local std::vector<uint8_t> buffer;
        return buffer;
    }

    void
    prefetch(const InnerIdType* idx, InnerIdType i, InnerIdType id_count) {
        const auto stride = static_cast<InnerIdType>(this->prefetch_stride_code_);
        if (stride > 0 and i + stride < id_count) {
            this->Prefetch(idx[i + stride]);
        }
    }

    void
    encode_and_write(InnerIdType idx, const float* vector) {
        if (idx >= this->max_capacity_) {
            this->Resize(idx + 1);
        }
        if (this->optimized_build_active_) {
            std::vector<uint8_t> build_code(this->scalar_record_size_);
            CHECK_ARGUMENT(this->quantizer_->EncodeScalar(vector, build_code.data()),
                           "sq_local_split scalar encode failed");
            this->build_layout_->Write(idx, build_code.data());
            return;
        }
        std::vector<uint8_t> filter_code(this->quantizer_->GetFilterCodeSize());
        std::vector<uint8_t> supplement_code(this->quantizer_->GetSupplementCodeSize());
        CHECK_ARGUMENT(this->quantizer_->Encode(vector, filter_code.data(), supplement_code.data()),
                       "sq_local_split encode failed");
        this->filter_layout_->Write(idx, filter_code.data());
        this->supplement_layout_->Write(idx, supplement_code.data());
    }

    IndexCommonParam common_param_;
    Allocator* allocator_{nullptr};
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
    uint64_t scalar_record_size_{0};
    bool optimized_build_active_{false};
    std::shared_ptr<QuantizerT> quantizer_{nullptr};
    std::shared_ptr<FilterLayoutT> filter_layout_{nullptr};
    std::shared_ptr<SupplementLayoutT> supplement_layout_{nullptr};
    std::shared_ptr<BuildLayoutT> build_layout_{nullptr};
};

}  // namespace vsag
