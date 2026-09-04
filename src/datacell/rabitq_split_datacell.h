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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <type_traits>

#include "common.h"
#include "flatten_interface.h"
#include "flatten_optimized_build_interface.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "inner_string_params.h"
#include "io/async_io/async_io_parameter.h"
#include "io/buffer_io/buffer_io_parameter.h"
#include "io/common/io_parameter.h"
#include "io/memory_io/memory_io.h"
#include "io/memory_io/memory_io_parameter.h"
#include "io/mmap_io/mmap_io.h"
#include "io/mmap_io/mmap_io_parameter.h"
#include "layout/fixed_layout.h"
#include "quantization/bottom_quantizer_accessor.h"
#include "quantization/rabitq_quantization/rabitq_quantizer.h"
#include "query_context.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "type_helpers.h"
#include "utils/byte_buffer.h"
#include "utils/timer.h"

namespace vsag {

template <MetricType metric,
          typename OneBitIOTmpl,
          typename SupplementIOTmpl = OneBitIOTmpl,
          typename QuantizerT = RaBitQuantizer<metric>>
class RaBitQSplitDataCell : public FlattenInterface, public FlattenOptimizedBuildInterface {
public:
    using Accessor = BottomQuantizerAccessor<QuantizerT>;
    using BottomQuantizer = typename Accessor::BottomQuantizerType;
    using BottomComputer = typename Accessor::BottomComputerType;

    static_assert(std::is_same_v<BottomQuantizer, RaBitQuantizer<metric>>,
                  "RaBitQSplitDataCell requires RaBitQuantizer as bottom quantizer");

    class OptimizedBuildComputer final : public ComputerInterface {
    public:
        OptimizedBuildComputer(uint64_t record_size, Allocator* allocator)
            : scalar_code_(record_size, allocator) {
        }

        ByteBuffer scalar_code_;
        uint64_t code_sum_{0};
    };

    RaBitQSplitDataCell() = default;

    explicit RaBitQSplitDataCell(const QuantizerParamPtr& quantization_param,
                                 const IOParamPtr& io_param,
                                 const IndexCommonParam& common_param)
        : RaBitQSplitDataCell(quantization_param, io_param, nullptr, common_param) {
    }

    explicit RaBitQSplitDataCell(const QuantizerParamPtr& quantization_param,
                                 const IOParamPtr& io_param,
                                 const IOParamPtr& supplement_io_param,
                                 const IndexCommonParam& common_param)
        : common_param_(common_param), allocator_(common_param.allocator_.get()) {
        this->quantizer_ = std::make_shared<QuantizerT>(quantization_param, common_param);
        if (not this->bottom_quantizer().SupportSplitCodeStorage()) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "rabitq split data cell requires rabitq_version=split, "
                                "rabitq_bits_per_dim_query=32, and "
                                "rabitq_bits_per_dim_base in [1, 8]");
        }
        // When a supplement-specific IO param is supplied, use it directly so
        // the caller can pick an entirely different IO type (e.g. x-bit in
        // memory + supplement on disk). Otherwise fall back to the shared
        // io_param with the legacy file-path suffix to keep the two backing
        // files separate for file-backed IO.
        const bool shares_io_param = supplement_io_param == nullptr;
        const IOParamPtr one_bit_io_param = SuffixIOParam(io_param, "_onebit", shares_io_param);
        const IOParamPtr supp_io_param = (supplement_io_param != nullptr)
                                             ? supplement_io_param
                                             : SuffixIOParam(io_param, "_supplement", true);
        if (supplement_io_param != nullptr) {
            this->supplement_io_type_ = supplement_io_param->GetTypeName();
        }
        this->x_bit_layout_ =
            std::make_shared<FixedLayout<OneBitIOTmpl>>(one_bit_io_param, common_param);
        this->supplement_layout_ =
            std::make_shared<FixedLayout<SupplementIOTmpl>>(supp_io_param, common_param);
        this->refresh_code_sizes();
    }

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) override {
        if (this->optimized_build_active_) {
            this->query_optimized_build_codes(result_dists, computer, idx, id_count);
            this->add_distance_evaluations(ctx, id_count);
            return;
        }
        auto* comp = this->get_bottom_computer(computer);
        if constexpr (not OneBitIOTmpl::InMemory or not SupplementIOTmpl::InMemory) {
            if (id_count > 1) {
                if constexpr (OneBitIOTmpl::InMemory and not SupplementIOTmpl::InMemory) {
                    this->query_full_dist_by_supplement_multiread(
                        result_dists, comp, idx, id_count, ctx);
                    this->add_distance_evaluations(ctx, id_count);
                    return;
                }
                this->query_full_dist_by_multiread(result_dists, comp, idx, id_count, ctx);
                this->add_distance_evaluations(ctx, id_count);
                return;
            }
        }

        for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; ++i) {
            this->prefetch_full_code(idx[i]);
        }

        for (InnerIdType i = 0; i < id_count; ++i) {
            if (i + this->prefetch_stride_code_ < id_count) {
                this->prefetch_full_code(idx[i + this->prefetch_stride_code_]);
            }
            this->compute_full_dist(idx[i], comp, result_dists + i, ctx);
        }
        this->add_distance_evaluations(ctx, id_count);
    }

    void
    QueryById(float* result_dists,
              InnerIdType query_id,
              const InnerIdType* idx,
              InnerIdType id_count,
              QueryContext* ctx = nullptr) override {
        if (not this->optimized_build_active_) {
            // Persisted split storage has no temporary scalar codes. Merge the query once and
            // reuse its full code; optimized builds use the scalar-code path below.
            ByteBuffer query_code(this->code_size_, allocator_);
            ByteBuffer base_code(this->code_size_, allocator_);
            if (not this->GetCodesById(query_id, query_code.data)) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to read split RaBitQ query code");
            }
            for (InnerIdType i = 0; i < id_count; ++i) {
                if (i + this->prefetch_stride_code_ < id_count) {
                    this->prefetch_full_code(idx[i + this->prefetch_stride_code_]);
                }
                if (not this->GetCodesById(idx[i], base_code.data)) {
                    throw VsagException(ErrorType::INTERNAL_ERROR,
                                        "failed to read split RaBitQ base code");
                }
                result_dists[i] = this->bottom_quantizer().Compute(query_code.data, base_code.data);
            }
            this->add_distance_evaluations(ctx, id_count);
            return;
        }

        auto query_lease = this->optimized_build_scalar_layout_->Acquire(query_id);
        if (not query_lease) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "failed to read temporary scalar RaBitQ query code");
        }
        this->query_optimized_build_code_pairs(result_dists,
                                               query_lease.Data(),
                                               (*this->optimized_build_code_sums_)[query_id],
                                               idx,
                                               id_count);
        this->add_distance_evaluations(ctx, id_count);
    }

    void
    QueryWithDistanceHint(float* result_dists,
                          const float* hint_dists,
                          const ComputerInterfacePtr& computer,
                          const InnerIdType* idx,
                          InnerIdType id_count,
                          QueryContext* ctx = nullptr) override {
        if (this->optimized_build_active_) {
            this->query_optimized_build_codes(result_dists, computer, idx, id_count);
            this->add_distance_evaluations(ctx, id_count);
            return;
        }
        auto* comp = this->get_bottom_computer(computer);
        if constexpr (not OneBitIOTmpl::InMemory or not SupplementIOTmpl::InMemory) {
            if (id_count > 1) {
                if constexpr (OneBitIOTmpl::InMemory and not SupplementIOTmpl::InMemory) {
                    this->query_full_dist_by_supplement_multiread(
                        result_dists, comp, idx, id_count, ctx, hint_dists);
                    this->add_distance_evaluations(ctx, id_count);
                    return;
                }
                this->query_full_dist_by_multiread(
                    result_dists, comp, idx, id_count, ctx, hint_dists);
                this->add_distance_evaluations(ctx, id_count);
                return;
            }
        }

        for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; ++i) {
            this->prefetch_full_code(idx[i]);
        }

        for (InnerIdType i = 0; i < id_count; ++i) {
            if (i + this->prefetch_stride_code_ < id_count) {
                this->prefetch_full_code(idx[i + this->prefetch_stride_code_]);
            }
            const float hint =
                hint_dists == nullptr ? std::numeric_limits<float>::max() : hint_dists[i];
            this->compute_full_dist(idx[i], comp, result_dists + i, ctx, hint);
        }
        this->add_distance_evaluations(ctx, id_count);
    }

    void
    QueryWithDistanceFilter(float* result_dists,
                            const ComputerInterfacePtr& computer,
                            const InnerIdType* idx,
                            InnerIdType id_count,
                            float threshold,
                            QueryContext* ctx = nullptr) override {
        if (this->optimized_build_active_) {
            this->query_optimized_build_codes(result_dists, computer, idx, id_count);
            this->add_distance_evaluations(ctx, id_count);
            return;
        }
        auto* comp = this->get_bottom_computer(computer);
        for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; ++i) {
            this->prefetch_full_code(idx[i]);
        }

        for (InnerIdType i = 0; i < id_count; ++i) {
            if (i + this->prefetch_stride_code_ < id_count) {
                this->prefetch_full_code(idx[i + this->prefetch_stride_code_]);
            }

            auto one_bit_lease = this->x_bit_layout_->Acquire(idx[i]);
            const auto* one_bit_code = one_bit_lease.Data();
            float one_bit_dist = 0.0F;
            float lower_bound = std::numeric_limits<float>::max();
            bool computed = false;
            computed = this->bottom_quantizer().ComputeDistWithOneBitLowerBound(
                *comp,
                one_bit_code,
                &one_bit_dist,
                &lower_bound,
                this->query_rabitq_error_rate(ctx));

            if (computed and std::isfinite(lower_bound) and lower_bound >= threshold) {
                result_dists[i] = threshold;
                continue;
            }

            auto supplement_lease = this->supplement_layout_->Acquire(idx[i]);
            this->compute_full_dist(
                one_bit_code, supplement_lease.Data(), comp, result_dists + i, ctx);
        }
        this->add_distance_evaluations(ctx, id_count);
    }

    void
    QueryWithDistanceLowerBound(float* result_dists,
                                float* lower_bounds,
                                const ComputerInterfacePtr& computer,
                                const InnerIdType* idx,
                                InnerIdType id_count,
                                QueryContext* ctx = nullptr) override {
        if (this->optimized_build_active_) {
            this->query_optimized_build_codes(result_dists, computer, idx, id_count);
            if (lower_bounds != nullptr) {
                std::fill(lower_bounds, lower_bounds + id_count, std::numeric_limits<float>::max());
            }
            this->add_distance_evaluations(ctx, id_count);
            return;
        }
        auto* comp = this->get_bottom_computer(computer);
        this->add_filter_count(ctx, id_count);
        if constexpr (not OneBitIOTmpl::InMemory) {
            if (id_count > 1) {
                this->query_one_bit_lower_bound_by_multiread(
                    result_dists, lower_bounds, comp, idx, id_count, ctx);
                this->add_distance_evaluations(ctx, id_count);
                return;
            }
        }

        for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; ++i) {
            this->prefetch_one_bit(idx[i]);
        }

        InnerIdType i = 0;
        for (; i + 3 < id_count; i += 4) {
            for (int64_t j = 0; j < 4; ++j) {
                if (i + j + this->prefetch_stride_code_ < id_count) {
                    this->prefetch_one_bit(idx[i + j + this->prefetch_stride_code_]);
                }
            }

            auto lease1 = this->x_bit_layout_->Acquire(idx[i]);
            auto lease2 = this->x_bit_layout_->Acquire(idx[i + 1]);
            auto lease3 = this->x_bit_layout_->Acquire(idx[i + 2]);
            auto lease4 = this->x_bit_layout_->Acquire(idx[i + 3]);
            const auto* code1 = lease1.Data();
            const auto* code2 = lease2.Data();
            const auto* code3 = lease3.Data();
            const auto* code4 = lease4.Data();
            {
                bool computed1 = false, computed2 = false, computed3 = false, computed4 = false;
                auto* lower_bound1 = lower_bounds == nullptr ? nullptr : lower_bounds + i;
                auto* lower_bound2 = lower_bounds == nullptr ? nullptr : lower_bounds + i + 1;
                auto* lower_bound3 = lower_bounds == nullptr ? nullptr : lower_bounds + i + 2;
                auto* lower_bound4 = lower_bounds == nullptr ? nullptr : lower_bounds + i + 3;
                this->bottom_quantizer().ComputeDistsWithOneBitLowerBoundBatch4(
                    *comp,
                    code1,
                    code2,
                    code3,
                    code4,
                    result_dists[i],
                    result_dists[i + 1],
                    result_dists[i + 2],
                    result_dists[i + 3],
                    lower_bound1,
                    lower_bound2,
                    lower_bound3,
                    lower_bound4,
                    computed1,
                    computed2,
                    computed3,
                    computed4,
                    this->query_rabitq_error_rate(ctx));
                if (not computed1) {
                    this->add_filter_fallback_full_count(ctx, 1);
                    this->compute_full_dist_after_one_bit_failure(
                        idx[i], code1, comp, result_dists + i, lower_bound1, ctx);
                }
                if (not computed2) {
                    this->add_filter_fallback_full_count(ctx, 1);
                    this->compute_full_dist_after_one_bit_failure(
                        idx[i + 1], code2, comp, result_dists + i + 1, lower_bound2, ctx);
                }
                if (not computed3) {
                    this->add_filter_fallback_full_count(ctx, 1);
                    this->compute_full_dist_after_one_bit_failure(
                        idx[i + 2], code3, comp, result_dists + i + 2, lower_bound3, ctx);
                }
                if (not computed4) {
                    this->add_filter_fallback_full_count(ctx, 1);
                    this->compute_full_dist_after_one_bit_failure(
                        idx[i + 3], code4, comp, result_dists + i + 3, lower_bound4, ctx);
                }
            }
        }

        for (; i < id_count; ++i) {
            if (i + this->prefetch_stride_code_ < id_count) {
                this->prefetch_one_bit(idx[i + this->prefetch_stride_code_]);
            }

            auto one_bit_lease = this->x_bit_layout_->Acquire(idx[i]);
            const auto* one_bit_code = one_bit_lease.Data();
            auto* lower_bound = lower_bounds == nullptr ? nullptr : lower_bounds + i;
            bool computed = false;
            computed = this->bottom_quantizer().ComputeDistWithOneBitLowerBound(
                *comp,
                one_bit_code,
                result_dists + i,
                lower_bound,
                this->query_rabitq_error_rate(ctx));
            if (not computed) {
                this->add_filter_fallback_full_count(ctx, 1);
                this->compute_full_dist_after_one_bit_failure(
                    idx[i], one_bit_code, comp, result_dists + i, lower_bound, ctx);
            }
        }
        this->add_distance_evaluations(ctx, id_count);
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        auto computer = this->quantizer_->FactoryComputer();
        computer->SetQuery(static_cast<const float*>(query));
        return computer;
    }

    ComputerInterfacePtr
    FactoryComputerForBuild(const void* query, InnerIdType id) override {
        if (this->optimized_build_active_) {
            auto computer = std::make_shared<OptimizedBuildComputer>(
                this->optimized_build_record_size_, this->allocator_);
            if (not this->optimized_build_scalar_layout_->Read(id, computer->scalar_code_.data)) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to read scalar RaBitQ build query code");
            }
            computer->code_sum_ = (*this->optimized_build_code_sums_)[id];
            return computer;
        }
        return this->FactoryComputer(query);
    }

    void
    Train(const void* data, uint64_t count) override {
        if (this->quantizer_) {
            this->quantizer_->Train(static_cast<const float*>(data), count);
        }
    }

    bool
    BeginOptimizedBuild(const FlattenOptimizedBuildContext& context) override {
        if (this->optimized_build_active_ or
            not this->bottom_quantizer().SupportScalarCodeBuild()) {
            return false;
        }
        auto io_param = std::make_shared<MemoryIOParameter>();
        auto build_codes = std::make_shared<FixedLayout<MemoryIO>>(io_param, this->common_param_);
        build_codes->SetCodeSize(this->bottom_quantizer().GetScalarCodeSize());
        auto code_sums = std::make_unique<Vector<uint64_t>>(this->allocator_);
        if (this->max_capacity_ > 0) {
            build_codes->Resize(this->max_capacity_);
            code_sums->resize(this->max_capacity_, 0);
        }
        this->optimized_build_scalar_layout_ = build_codes;
        this->optimized_build_code_sums_ = std::move(code_sums);
        this->optimized_build_record_size_ = this->bottom_quantizer().GetScalarCodeSize();
        this->refresh_prefetch_bytes();
        this->optimized_build_context_ = context;
        this->optimized_build_active_ = true;
        return true;
    }

    void
    FinalizeOptimizedBuild() override {
        if (not this->optimized_build_active_) {
            return;
        }

        // Finalize workers write disjoint IDs, but the backing IO must already be fully sized so
        // no worker enters a concurrent reallocation path.
        const InnerIdType final_capacity = std::max(this->max_capacity_, this->total_count_);
        this->x_bit_layout_->Resize(final_capacity);
        this->supplement_layout_->Resize(final_capacity);
        this->max_capacity_ = final_capacity;

        auto finalize_range = [this](InnerIdType begin, InnerIdType end) {
            ByteBuffer one_bit_code(this->one_bit_code_size_, allocator_);
            ByteBuffer supplement_code(this->supplement_code_size_, allocator_);
            for (InnerIdType id = begin; id < end; ++id) {
                auto scalar_lease = this->optimized_build_scalar_layout_->Acquire(id);
                if (not scalar_lease) {
                    throw VsagException(ErrorType::INTERNAL_ERROR,
                                        "failed to read temporary scalar RaBitQ build code");
                }
                this->bottom_quantizer().PackScalarCodeToSplitCode(
                    scalar_lease.Data(), one_bit_code.data, supplement_code.data);
                this->x_bit_layout_->Write(id, one_bit_code.data);
                this->supplement_layout_->Write(id, supplement_code.data);
            }
        };

        const auto& thread_pool = this->optimized_build_context_.thread_pool;
        const uint64_t worker_count = std::min<uint64_t>(
            this->optimized_build_context_.thread_count, static_cast<uint64_t>(this->total_count_));
        constexpr bool supports_parallel_finalize = not std::is_same_v<OneBitIOTmpl, MMapIO> and
                                                    not std::is_same_v<SupplementIOTmpl, MMapIO>;
        // MMapIO writes update shared logical size state even after Resize, so disjoint writes are
        // not thread-safe for that backend.
        if (thread_pool != nullptr and worker_count > 1 and supports_parallel_finalize) {
            const uint64_t block_size =
                (static_cast<uint64_t>(this->total_count_) + worker_count - 1) / worker_count;
            std::vector<std::future<void>> futures;
            futures.reserve(worker_count);
            auto wait_futures = [&futures]() {
                std::exception_ptr first_exception = nullptr;
                for (auto& future : futures) {
                    if (not future.valid()) {
                        continue;
                    }
                    try {
                        future.get();
                    } catch (...) {
                        if (not first_exception) {
                            first_exception = std::current_exception();
                        }
                    }
                }
                if (first_exception) {
                    std::rethrow_exception(first_exception);
                }
            };
            try {
                for (uint64_t begin = 0; begin < this->total_count_; begin += block_size) {
                    const uint64_t end = std::min<uint64_t>(begin + block_size, this->total_count_);
                    futures.emplace_back(
                        thread_pool->GeneralEnqueue(finalize_range,
                                                    static_cast<InnerIdType>(begin),
                                                    static_cast<InnerIdType>(end)));
                }
            } catch (...) {
                const auto enqueue_exception = std::current_exception();
                try {
                    wait_futures();
                } catch (...) {
                }
                std::rethrow_exception(enqueue_exception);
            }
            wait_futures();
        } else {
            finalize_range(0, this->total_count_);
        }

        this->optimized_build_active_ = false;
        this->optimized_build_scalar_layout_.reset();
        this->optimized_build_code_sums_.reset();
        this->optimized_build_record_size_ = 0;
        this->refresh_prefetch_bytes();
        this->optimized_build_context_ = {};
    }

    void
    AbortOptimizedBuild() noexcept override {
        this->optimized_build_active_ = false;
        this->optimized_build_scalar_layout_.reset();
        this->optimized_build_code_sums_.reset();
        this->optimized_build_record_size_ = 0;
        this->refresh_prefetch_bytes();
        this->optimized_build_context_ = {};
    }

    [[nodiscard]] bool
    IsOptimizedBuildActive() const override {
        return this->optimized_build_active_;
    }

    void
    InsertVector(const void* vector,
                 InnerIdType idx = std::numeric_limits<InnerIdType>::max()) override {
        {
            std::lock_guard lock(this->mutex_);
            if (idx == std::numeric_limits<InnerIdType>::max()) {
                idx = this->total_count_;
            }
            // Optimized-build workers write disjoint IDs without locking, so both temporary
            // arrays must be fully sized before the workers start.
            CHECK_ARGUMENT(
                not this->optimized_build_active_ or
                    static_cast<uint64_t>(idx) < this->optimized_build_code_sums_->size(),
                "optimized RaBitQ build storage must be resized before inserting vectors");
            this->total_count_ = std::max(this->total_count_, idx + 1);
        }
        this->write_encoded_vector(static_cast<const float*>(vector), idx);
    }

    bool
    UpdateVector(const void* vector,
                 InnerIdType idx = std::numeric_limits<InnerIdType>::max()) override {
        if (idx >= this->total_count_) {
            return false;
        }
        std::lock_guard lock(this->mutex_);
        this->write_encoded_vector(static_cast<const float*>(vector), idx);
        return true;
    }

    void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec) override {
        auto dim = quantizer_->GetDim();
        for (InnerIdType i = 0; i < count; ++i) {
            auto idx = idx_vec == nullptr ? std::numeric_limits<InnerIdType>::max() : idx_vec[i];
            this->InsertVector(static_cast<const float*>(vectors) + dim * i, idx);
        }
    }

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override {
        if (this->optimized_build_active_) {
            auto lease1 = this->optimized_build_scalar_layout_->Acquire(id1);
            auto lease2 = this->optimized_build_scalar_layout_->Acquire(id2);
            if (not lease1 or not lease2) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to read temporary scalar RaBitQ build codes");
            }
            return this->bottom_quantizer().ComputeScalarCodesDistance(
                lease1.Data(),
                (*optimized_build_code_sums_)[id1],
                lease2.Data(),
                (*optimized_build_code_sums_)[id2]);
        }
        ByteBuffer codes1(this->code_size_, allocator_);
        ByteBuffer codes2(this->code_size_, allocator_);
        this->GetCodesById(id1, codes1.data);
        this->GetCodesById(id2, codes2.data);
        return this->bottom_quantizer().Compute(codes1.data, codes2.data);
    }

    void
    Resize(InnerIdType new_capacity) override {
        if (new_capacity <= this->max_capacity_) {
            return;
        }
        this->x_bit_layout_->Resize(new_capacity);
        this->supplement_layout_->Resize(new_capacity);
        if (this->optimized_build_active_) {
            this->optimized_build_scalar_layout_->Resize(new_capacity);
            this->optimized_build_code_sums_->resize(new_capacity, 0);
        }
        this->max_capacity_ = new_capacity;
    }

    void
    Prefetch(InnerIdType id) override {
        if (this->optimized_build_active_) {
            this->optimized_build_scalar_layout_->Prefetch(id,
                                                           this->optimized_build_prefetch_bytes_);
            return;
        }
        this->prefetch_one_bit(id);
    }

    bool
    SetRuntimeParameters(const UnorderedMap<std::string, float>& new_params) override {
        const bool changed = FlattenInterface::SetRuntimeParameters(new_params);
        if (new_params.find(PREFETCH_DEPTH_CODE) != new_params.end()) {
            this->refresh_prefetch_bytes();
        }
        return changed;
    }

    [[nodiscard]] uint64_t
    GetOneBitPrefetchBytes() const {
        return this->one_bit_prefetch_bytes_;
    }

    [[nodiscard]] uint64_t
    GetSupplementPrefetchBytes() const {
        return this->supplement_prefetch_bytes_;
    }

    void
    ExportModel(const FlattenInterfacePtr& other) const override {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        this->quantizer_->Serialize(writer);
        ss.seekg(0, std::ios::beg);
        IOStreamReader reader(ss);
        auto ptr = std::dynamic_pointer_cast<
            RaBitQSplitDataCell<metric, OneBitIOTmpl, SupplementIOTmpl, QuantizerT>>(other);
        if (ptr == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Export model's rabitq split datacell failed");
        }
        ptr->quantizer_->Deserialize(reader);
        ptr->refresh_code_sizes();
    }

    void
    InitIO(const IOParamPtr& io_param) override {
        const bool shares_io_param = this->supplement_io_type_.empty();
        this->x_bit_layout_->InitIO(SuffixIOParam(io_param, "_onebit", shares_io_param));
        // In hybrid mode (one-bit and supplement use different IO backends)
        // the caller-facing `io_param` is the one-bit IO parameter type and
        // cannot be passed directly to `supplement_layout_`. Rebuild a fresh
        // IOParameter of the recorded supplement type so the underlying IO
        // implementation receives the correct parameter subclass.
        this->supplement_layout_->InitIO(RebuildSupplementIOParam(io_param));
    }

    void
    InitIO(const IOParamPtr& one_bit_io_param, const IOParamPtr& supplement_io_param) {
        const bool shares_io_param = supplement_io_param == nullptr;
        this->x_bit_layout_->InitIO(SuffixIOParam(one_bit_io_param, "_onebit", shares_io_param));
        if (supplement_io_param != nullptr) {
            // Refresh the recorded supplement type so subsequent
            // single-parameter InitIO calls (e.g. from Deserialize) can
            // reconstruct the same IO subtype.
            this->supplement_io_type_ = supplement_io_param->GetTypeName();
            this->supplement_layout_->InitIO(supplement_io_param);
        } else {
            this->supplement_layout_->InitIO(RebuildSupplementIOParam(one_bit_io_param));
        }
    }

    IndexCommonParam
    ExportCommonParam() override {
        return common_param_;
    }

    [[nodiscard]] std::string
    GetQuantizerName() override {
        return this->quantizer_->Name();
    }

    [[nodiscard]] bool
    SupportSplitCodeStorage() const override {
        return true;
    }

    [[nodiscard]] MetricType
    GetMetricType() override {
        return this->quantizer_->Metric();
    }

    bool
    Decode(const uint8_t* codes, float* data) override {
        return this->quantizer_->DecodeOne(codes, data);
    }

    bool
    Encode(const float* data, uint8_t* codes) override {
        return this->quantizer_->EncodeOne(data, codes);
    }

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override {
        if (this->optimized_build_active_) {
            auto* codes = static_cast<uint8_t*>(allocator_->Allocate(this->code_size_));
            if (not this->GetCodesById(id, codes)) {
                allocator_->Deallocate(codes);
                need_release = false;
                return nullptr;
            }
            need_release = true;
            return codes;
        }
        auto* codes = static_cast<uint8_t*>(allocator_->Allocate(this->code_size_));
        this->GetCodesById(id, codes);
        need_release = true;
        return codes;
    }

    void
    Release(const uint8_t* data) const override {
        allocator_->Deallocate(const_cast<uint8_t*>(data));
    }

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override {
        if (this->optimized_build_active_) {
            auto scalar_lease = this->optimized_build_scalar_layout_->Acquire(id);
            if (not scalar_lease) {
                return false;
            }
            memset(codes, 0, this->code_size_);
            this->bottom_quantizer().PackScalarCode(scalar_lease.Data(), codes);
            return true;
        }
        ByteBuffer one_bit(one_bit_code_size_, allocator_);
        ByteBuffer supplement(supplement_code_size_, allocator_);
        bool one_bit_ok = this->x_bit_layout_->Read(id, one_bit.data);
        bool supplement_ok = this->supplement_layout_->Read(id, supplement.data);
        if (not one_bit_ok or not supplement_ok) {
            return false;
        }
        memset(codes, 0, this->code_size_);
        this->bottom_quantizer().MergeSplitCode(one_bit.data, supplement.data, codes);
        return true;
    }

    [[nodiscard]] bool
    InMemory() const override {
        return OneBitIOTmpl::InMemory and SupplementIOTmpl::InMemory;
    }

    bool
    HoldMolds() const override {
        return this->quantizer_->HoldMolds();
    }

    void
    Serialize(StreamWriter& writer) override {
        CHECK_ARGUMENT(not this->optimized_build_active_,
                       "cannot serialize RaBitQ split codes during optimized build");
        FlattenInterface::Serialize(writer);
        StreamWriter::WriteString(writer, this->supplement_io_type_);
        this->x_bit_layout_->Serialize(writer);
        this->supplement_layout_->Serialize(writer);
        this->quantizer_->Serialize(writer);
    }

    void
    Deserialize(LvalueOrRvalue<StreamReader> reader) override {
        FlattenInterface::Deserialize(reader);
        this->DeserializeSupplementIOType(reader);
        this->x_bit_layout_->Deserialize(reader);
        this->supplement_layout_->Deserialize(reader);
        this->quantizer_->Deserialize(reader);
        this->refresh_code_sizes();
    }

    void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) override {
        auto ptr = std::dynamic_pointer_cast<
            RaBitQSplitDataCell<metric, OneBitIOTmpl, SupplementIOTmpl, QuantizerT>>(other);
        if (ptr == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Merge rabitq split datacell failed: not match type");
        }

        for (InnerIdType i = 0; i < ptr->total_count_; ++i) {
            ByteBuffer one_bit(one_bit_code_size_, allocator_);
            ByteBuffer supplement(supplement_code_size_, allocator_);
            ptr->x_bit_layout_->Read(i, one_bit.data);
            ptr->supplement_layout_->Read(i, supplement.data);
            auto target_id = static_cast<InnerIdType>(bias + i);
            this->x_bit_layout_->Write(target_id, one_bit.data);
            this->supplement_layout_->Write(target_id, supplement.data);
        }
        this->total_count_ = std::max(this->total_count_, bias + ptr->total_count_);
    }

    void
    Move(InnerIdType from, InnerIdType to) override {
        if (this->optimized_build_active_) {
            ByteBuffer build_record(this->optimized_build_record_size_, allocator_);
            this->optimized_build_scalar_layout_->Read(from, build_record.data);
            this->optimized_build_scalar_layout_->Write(to, build_record.data);
            (*this->optimized_build_code_sums_)[to] = (*this->optimized_build_code_sums_)[from];
            return;
        }
        ByteBuffer one_bit(one_bit_code_size_, allocator_);
        ByteBuffer supplement(supplement_code_size_, allocator_);
        this->x_bit_layout_->Read(from, one_bit.data);
        this->supplement_layout_->Read(from, supplement.data);
        this->x_bit_layout_->Write(to, one_bit.data);
        this->supplement_layout_->Write(to, supplement.data);
    }

    void
    ShrinkToFit(InnerIdType capacity) override {
        this->x_bit_layout_->Shrink(capacity);
        this->supplement_layout_->Shrink(capacity);
        if (this->optimized_build_active_) {
            this->optimized_build_scalar_layout_->Shrink(capacity);
            this->optimized_build_code_sums_->resize(capacity);
            this->optimized_build_code_sums_->shrink_to_fit();
        }
        this->max_capacity_ = capacity;
    }

    uint64_t
    GetMemoryUsage() const override {
        uint64_t memory =
            sizeof(RaBitQSplitDataCell<metric, OneBitIOTmpl, SupplementIOTmpl, QuantizerT>);
        memory += this->x_bit_layout_->GetMemoryUsage();
        memory += this->supplement_layout_->GetMemoryUsage();
        if (this->optimized_build_scalar_layout_ != nullptr) {
            memory += this->optimized_build_scalar_layout_->GetMemoryUsage();
        }
        if (this->optimized_build_code_sums_ != nullptr) {
            memory += this->optimized_build_code_sums_->capacity() * sizeof(uint64_t);
        }
        memory += sizeof(QuantizerT);
        return memory;
    }

public:
    IndexCommonParam common_param_;
    std::shared_ptr<QuantizerT> quantizer_{nullptr};
    std::shared_ptr<FixedLayout<OneBitIOTmpl>> x_bit_layout_{nullptr};
    std::shared_ptr<FixedLayout<SupplementIOTmpl>> supplement_layout_{nullptr};
    std::shared_ptr<FixedLayout<MemoryIO>> optimized_build_scalar_layout_{nullptr};
    std::unique_ptr<Vector<uint64_t>> optimized_build_code_sums_{nullptr};
    FlattenOptimizedBuildContext optimized_build_context_{};

    Allocator* allocator_{nullptr};
    uint64_t one_bit_code_size_{0};
    uint64_t supplement_code_size_{0};
    // Type name (e.g. "async_io") of the dedicated supplement IO when the
    // caller supplies a separate `supplement_io_param` at construction time.
    // Empty string means "supplement shares the same IO type as the x-bit
    // storage" (the legacy single-IO behaviour). Recorded so that the
    // single-parameter `InitIO(const IOParamPtr&)` overload (e.g. invoked
    // from Deserialize) can rebuild a parameter of the correct concrete
    // IOParameter subclass for `supplement_layout_` instead of feeding it the
    // mismatched one-bit IO parameter type.
    std::string supplement_io_type_{};
    bool optimized_build_active_{false};
    uint64_t optimized_build_record_size_{0};
    uint64_t one_bit_prefetch_bytes_{0};
    uint64_t supplement_prefetch_bytes_{0};
    uint64_t optimized_build_prefetch_bytes_{0};

private:
    BottomQuantizer&
    bottom_quantizer() {
        return Accessor::GetQuantizer(*this->quantizer_);
    }

    const BottomQuantizer&
    bottom_quantizer() const {
        return Accessor::GetQuantizer(*this->quantizer_);
    }

    BottomComputer*
    get_bottom_computer(const ComputerInterfacePtr& computer) const {
        auto* outer_computer = static_cast<Computer<QuantizerT>*>(computer.get());
        return &Accessor::GetComputer(*outer_computer);
    }

    static IOParamPtr
    SuffixIOParam(const IOParamPtr& io_param, const std::string& suffix, bool split_cache = false) {
        if (io_param == nullptr) {
            return nullptr;
        }
        auto json = io_param->ToJson();
        if (json.Contains(IO_FILE_PATH_KEY)) {
            std::string path = json[IO_FILE_PATH_KEY].GetString();
            json[IO_FILE_PATH_KEY].SetString(path + suffix);
        }
        if (split_cache and io_param->enable_read_cache_) {
            json[READ_CACHE_TOTAL_CACHE_SIZE_KEY].SetUint64(io_param->read_cache_total_size_ / 2);
        }
        return IOParameter::GetIOParameterByJson(json);
    }

    // Builds the IO parameter that should be handed to `supplement_layout_`
    // given the caller-supplied `io_param` (which is always typed for the
    // one-bit storage). If `supplement_io_type_` is empty the two storages
    // share the same IO type and we fall back to the legacy file-path-suffix
    // behaviour. Otherwise the JSON is cloned, its `type` field rewritten to
    // the recorded supplement type, the optional file path suffixed, and a
    // new IOParameter is constructed via the factory so `supplement_layout_`
    // receives the IOParameter subclass it actually expects.
    IOParamPtr
    RebuildSupplementIOParam(const IOParamPtr& io_param) const {
        if (io_param == nullptr) {
            return nullptr;
        }
        if (this->supplement_io_type_.empty()) {
            return SuffixIOParam(io_param, "_supplement", true);
        }
        auto json = io_param->ToJson();
        json[TYPE_KEY].SetString(this->supplement_io_type_);
        if (json.Contains(IO_FILE_PATH_KEY)) {
            std::string path = json[IO_FILE_PATH_KEY].GetString();
            json[IO_FILE_PATH_KEY].SetString(path + "_supplement");
        }
        return IOParameter::GetIOParameterByJson(json);
    }

    static bool
    IsKnownIOType(const std::string& io_type) {
        return IOParameter::KindFromName(io_type) != IOKind::UNKNOWN;
    }

    void
    DeserializeSupplementIOType(StreamReader& reader) {
        this->supplement_io_type_.clear();
        const uint64_t cursor = reader.GetCursor();
        uint64_t length = 0;
        StreamReader::ReadObj(reader, length);

        if (length == 0) {
            return;
        }

        constexpr uint64_t kMaxIOTypeLength = 64;
        if (length > kMaxIOTypeLength or reader.GetCursor() + length > reader.Length()) {
            reader.Seek(cursor);
            return;
        }

        std::string io_type(length, '\0');
        reader.Read(io_type.data(), length);
        if (IsKnownIOType(io_type)) {
            this->supplement_io_type_ = std::move(io_type);
            return;
        }

        reader.Seek(cursor);
    }

    void
    refresh_code_sizes() {
        this->code_size_ = static_cast<uint32_t>(quantizer_->GetCodeSize());
        this->one_bit_code_size_ = this->bottom_quantizer().GetOneBitCodeSize();
        this->supplement_code_size_ = this->bottom_quantizer().GetSupplementCodeSize();
        this->x_bit_layout_->SetCodeSize(one_bit_code_size_);
        this->supplement_layout_->SetCodeSize(supplement_code_size_);
        this->refresh_prefetch_bytes();
    }

    [[nodiscard]] static uint64_t
    calculate_prefetch_bytes(uint64_t record_size, uint32_t max_cache_lines) {
        constexpr uint64_t cache_line_size = 64;
        const uint64_t record_cache_lines =
            record_size / cache_line_size +
            static_cast<uint64_t>(record_size % cache_line_size != 0);
        return std::min<uint64_t>(record_cache_lines, max_cache_lines) * cache_line_size;
    }

    void
    refresh_prefetch_bytes() {
        this->one_bit_prefetch_bytes_ =
            calculate_prefetch_bytes(this->one_bit_code_size_, this->prefetch_depth_code_);
        this->supplement_prefetch_bytes_ =
            calculate_prefetch_bytes(this->supplement_code_size_, this->prefetch_depth_code_);
        this->optimized_build_prefetch_bytes_ = calculate_prefetch_bytes(
            this->optimized_build_record_size_, this->prefetch_depth_code_);
    }

    void
    write_encoded_vector(const float* vector, InnerIdType idx) {
        if (this->optimized_build_active_) {
            ByteBuffer scalar_code(this->optimized_build_record_size_, allocator_);
            Vector<float> transformed_input(this->allocator_);
            const float* bottom_input =
                Accessor::PrepareBottomInput(*this->quantizer_, vector, transformed_input);
            uint64_t code_sum = 0;
            if (not this->bottom_quantizer().EncodeOneToScalarCode(
                    bottom_input, scalar_code.data, code_sum)) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to encode temporary scalar RaBitQ build code");
            }
            (*this->optimized_build_code_sums_)[idx] = code_sum;
            this->optimized_build_scalar_layout_->Write(idx, scalar_code.data);
            return;
        }
        ByteBuffer full_code(this->code_size_, allocator_);
        this->quantizer_->EncodeOne(vector, full_code.data);
        ByteBuffer one_bit_code(one_bit_code_size_, allocator_);
        ByteBuffer supplement_code(supplement_code_size_, allocator_);
        this->bottom_quantizer().SplitCode(full_code.data, one_bit_code.data, supplement_code.data);
        this->x_bit_layout_->Write(idx, one_bit_code.data);
        this->supplement_layout_->Write(idx, supplement_code.data);
    }

    void
    query_optimized_build_codes(float* result_dists,
                                const ComputerInterfacePtr& computer,
                                const InnerIdType* idx,
                                InnerIdType id_count) const {
        if (const auto* build_computer =
                dynamic_cast<const OptimizedBuildComputer*>(computer.get());
            build_computer != nullptr) {
            this->query_optimized_build_code_pairs(result_dists,
                                                   build_computer->scalar_code_.data,
                                                   build_computer->code_sum_,
                                                   idx,
                                                   id_count);
            return;
        }
        auto* comp = this->get_bottom_computer(computer);
        for (InnerIdType i = 0; i < id_count; ++i) {
            auto scalar_lease = this->optimized_build_scalar_layout_->Acquire(idx[i]);
            if (not scalar_lease) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to read temporary scalar RaBitQ build code");
            }
            this->bottom_quantizer().ComputeDistWithScalarCode(
                *comp, scalar_lease.Data(), result_dists + i);
        }
    }

    void
    query_optimized_build_code_pairs(float* result_dists,
                                     const uint8_t* query_code,
                                     uint64_t query_sum,
                                     const InnerIdType* idx,
                                     InnerIdType id_count) const {
        for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; ++i) {
            this->optimized_build_scalar_layout_->Prefetch(idx[i],
                                                           this->optimized_build_prefetch_bytes_);
        }
        for (InnerIdType i = 0; i < id_count; ++i) {
            if (i + this->prefetch_stride_code_ < id_count) {
                this->optimized_build_scalar_layout_->Prefetch(
                    idx[i + this->prefetch_stride_code_], this->optimized_build_prefetch_bytes_);
            }
            auto base_lease = this->optimized_build_scalar_layout_->Acquire(idx[i]);
            if (not base_lease) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to read temporary scalar RaBitQ build code");
            }
            result_dists[i] = this->bottom_quantizer().ComputeScalarCodesDistance(
                query_code,
                query_sum,
                base_lease.Data(),
                (*this->optimized_build_code_sums_)[idx[i]]);
        }
    }

    void
    prefetch_one_bit(InnerIdType id) {
        this->x_bit_layout_->Prefetch(id, this->one_bit_prefetch_bytes_);
    }

    void
    prefetch_supplement(InnerIdType id) {
        this->supplement_layout_->Prefetch(id, this->supplement_prefetch_bytes_);
    }

    void
    prefetch_full_code(InnerIdType id) {
        this->prefetch_one_bit(id);
        this->prefetch_supplement(id);
    }

    [[nodiscard]] static float
    query_rabitq_error_rate(QueryContext* ctx) {
        return ctx == nullptr ? std::numeric_limits<float>::quiet_NaN() : ctx->rabitq_error_rate;
    }

    void
    add_distance_evaluations(QueryContext* ctx, uint64_t count) const {
        if (ctx != nullptr and ctx->stats != nullptr and ctx->track_distance_evaluations and
            count > 0)
            ctx->stats->AddDistance(ctx->distance_phase, DistanceEvaluationBackend::RABITQ, count);
    }

    void
    add_filter_count(QueryContext* ctx, uint64_t count) const {
        if (ctx != nullptr and ctx->stats != nullptr) {
            ctx->stats->rabitq_filter_count.fetch_add(static_cast<uint32_t>(count),
                                                      std::memory_order_relaxed);
        }
    }

    void
    add_full_count(QueryContext* ctx, uint64_t count) const {
        if (ctx != nullptr and ctx->stats != nullptr) {
            ctx->stats->rabitq_full_count.fetch_add(static_cast<uint32_t>(count),
                                                    std::memory_order_relaxed);
        }
    }

    void
    add_filter_fallback_full_count(QueryContext* ctx, uint64_t count) const {
        if (ctx != nullptr and ctx->stats != nullptr) {
            ctx->stats->rabitq_filter_fallback_full_count.fetch_add(static_cast<uint32_t>(count),
                                                                    std::memory_order_relaxed);
        }
    }

    void
    add_reorder_hint_full_count(QueryContext* ctx, uint64_t count) const {
        if (ctx != nullptr and ctx->stats != nullptr) {
            ctx->stats->rabitq_reorder_hint_full_count.fetch_add(static_cast<uint32_t>(count),
                                                                 std::memory_order_relaxed);
        }
    }

    void
    add_reorder_fallback_full_count(QueryContext* ctx, uint64_t count) const {
        if (ctx != nullptr and ctx->stats != nullptr) {
            ctx->stats->rabitq_reorder_fallback_full_count.fetch_add(static_cast<uint32_t>(count),
                                                                     std::memory_order_relaxed);
        }
    }

    void
    query_one_bit_lower_bound_by_multiread(float* result_dists,
                                           float* lower_bounds,
                                           BottomComputer* computer,
                                           const InnerIdType* idx,
                                           InnerIdType id_count,
                                           QueryContext* ctx) const {
        Allocator* search_alloc = select_query_allocator(ctx, allocator_);
        ByteBuffer one_bit_codes(id_count * one_bit_code_size_, search_alloc);
        double io_cost_ms = 0.0F;
        {
            Timer timer(io_cost_ms);
            this->x_bit_layout_->MultiRead(idx, id_count, one_bit_codes.data, search_alloc);
        }
        if (ctx != nullptr and ctx->stats != nullptr) {
            ctx->stats->io_cnt.fetch_add(id_count, std::memory_order_relaxed);
            ctx->stats->io_time_ms.fetch_add(static_cast<uint32_t>(io_cost_ms),
                                             std::memory_order_relaxed);
        }

        InnerIdType i = 0;
        for (; i + 3 < id_count; i += 4) {
            const auto* code1 = one_bit_codes.data + i * one_bit_code_size_;
            const auto* code2 = code1 + one_bit_code_size_;
            const auto* code3 = code2 + one_bit_code_size_;
            const auto* code4 = code3 + one_bit_code_size_;
            bool computed1 = false, computed2 = false, computed3 = false, computed4 = false;
            auto* lower_bound1 = lower_bounds == nullptr ? nullptr : lower_bounds + i;
            auto* lower_bound2 = lower_bounds == nullptr ? nullptr : lower_bounds + i + 1;
            auto* lower_bound3 = lower_bounds == nullptr ? nullptr : lower_bounds + i + 2;
            auto* lower_bound4 = lower_bounds == nullptr ? nullptr : lower_bounds + i + 3;
            this->bottom_quantizer().ComputeDistsWithOneBitLowerBoundBatch4(
                *computer,
                code1,
                code2,
                code3,
                code4,
                result_dists[i],
                result_dists[i + 1],
                result_dists[i + 2],
                result_dists[i + 3],
                lower_bound1,
                lower_bound2,
                lower_bound3,
                lower_bound4,
                computed1,
                computed2,
                computed3,
                computed4,
                this->query_rabitq_error_rate(ctx));
            if (not computed1) {
                this->add_filter_fallback_full_count(ctx, 1);
                this->compute_full_dist_after_one_bit_failure(
                    idx[i], code1, computer, result_dists + i, lower_bound1, ctx);
            }
            if (not computed2) {
                this->add_filter_fallback_full_count(ctx, 1);
                this->compute_full_dist_after_one_bit_failure(
                    idx[i + 1], code2, computer, result_dists + i + 1, lower_bound2, ctx);
            }
            if (not computed3) {
                this->add_filter_fallback_full_count(ctx, 1);
                this->compute_full_dist_after_one_bit_failure(
                    idx[i + 2], code3, computer, result_dists + i + 2, lower_bound3, ctx);
            }
            if (not computed4) {
                this->add_filter_fallback_full_count(ctx, 1);
                this->compute_full_dist_after_one_bit_failure(
                    idx[i + 3], code4, computer, result_dists + i + 3, lower_bound4, ctx);
            }
        }

        for (; i < id_count; ++i) {
            auto* lower_bound = lower_bounds == nullptr ? nullptr : lower_bounds + i;
            const auto* one_bit_code = one_bit_codes.data + i * one_bit_code_size_;
            bool computed = this->bottom_quantizer().ComputeDistWithOneBitLowerBound(
                *computer,
                one_bit_code,
                result_dists + i,
                lower_bound,
                this->query_rabitq_error_rate(ctx));
            if (not computed) {
                this->add_filter_fallback_full_count(ctx, 1);
                this->compute_full_dist_after_one_bit_failure(
                    idx[i], one_bit_code, computer, result_dists + i, lower_bound, ctx);
            }
        }
    }

    void
    query_full_dist_by_multiread(float* result_dists,
                                 BottomComputer* computer,
                                 const InnerIdType* idx,
                                 InnerIdType id_count,
                                 QueryContext* ctx,
                                 const float* hint_dists = nullptr) const {
        Allocator* search_alloc = select_query_allocator(ctx, allocator_);
        ByteBuffer one_bit_codes(id_count * one_bit_code_size_, search_alloc);
        ByteBuffer supplement_codes(id_count * supplement_code_size_, search_alloc);
        double io_cost_ms = 0.0F;
        {
            Timer timer(io_cost_ms);
            this->x_bit_layout_->MultiRead(idx, id_count, one_bit_codes.data, search_alloc);
            this->supplement_layout_->MultiRead(idx, id_count, supplement_codes.data, search_alloc);
        }
        if (ctx != nullptr and ctx->stats != nullptr) {
            ctx->stats->io_cnt.fetch_add(id_count * 2, std::memory_order_relaxed);
            ctx->stats->io_time_ms.fetch_add(static_cast<uint32_t>(io_cost_ms),
                                             std::memory_order_relaxed);
        }

        for (InnerIdType i = 0; i < id_count; ++i) {
            const auto* one_bit_code = one_bit_codes.data + i * one_bit_code_size_;
            const auto* supplement_code = supplement_codes.data + i * supplement_code_size_;
            const float hint =
                hint_dists == nullptr ? std::numeric_limits<float>::max() : hint_dists[i];
            this->compute_full_dist(
                one_bit_code, supplement_code, computer, result_dists + i, ctx, hint);
        }
    }

    void
    query_full_dist_by_supplement_multiread(float* result_dists,
                                            BottomComputer* computer,
                                            const InnerIdType* idx,
                                            InnerIdType id_count,
                                            QueryContext* ctx,
                                            const float* hint_dists = nullptr) const {
        Allocator* search_alloc = select_query_allocator(ctx, allocator_);
        ByteBuffer supplement_codes(id_count * supplement_code_size_, search_alloc);
        double io_cost_ms = 0.0F;
        {
            Timer timer(io_cost_ms);
            this->supplement_layout_->MultiRead(idx, id_count, supplement_codes.data, search_alloc);
        }
        if (ctx != nullptr and ctx->stats != nullptr) {
            ctx->stats->io_cnt.fetch_add(id_count, std::memory_order_relaxed);
            ctx->stats->io_time_ms.fetch_add(static_cast<uint32_t>(io_cost_ms),
                                             std::memory_order_relaxed);
        }

        for (InnerIdType i = 0; i < id_count; ++i) {
            auto one_bit_lease = this->x_bit_layout_->Acquire(idx[i]);
            const auto* supplement_code = supplement_codes.data + i * supplement_code_size_;
            const float hint =
                hint_dists == nullptr ? std::numeric_limits<float>::max() : hint_dists[i];
            this->compute_full_dist(
                one_bit_lease.Data(), supplement_code, computer, result_dists + i, ctx, hint);
        }
    }

    void
    compute_full_dist_after_one_bit_failure(InnerIdType id,
                                            const uint8_t* one_bit_code,
                                            BottomComputer* computer,
                                            float* result_dist,
                                            float* lower_bound,
                                            QueryContext* ctx) const {
        auto supplement_lease = this->supplement_layout_->Acquire(id);
        this->compute_full_dist(one_bit_code, supplement_lease.Data(), computer, result_dist, ctx);
        if (lower_bound != nullptr) {
            *lower_bound = std::numeric_limits<float>::max();
        }
    }

    void
    compute_full_dist(const uint8_t* one_bit_code,
                      const uint8_t* supplement_code,
                      BottomComputer* computer,
                      float* result_dist,
                      QueryContext* ctx = nullptr,
                      float hint_dist = std::numeric_limits<float>::max()) const {
        this->add_full_count(ctx, 1);
        bool computed = false;
        const bool has_hint =
            std::isfinite(hint_dist) and hint_dist < std::numeric_limits<float>::max();
        if (has_hint) {
            computed = this->bottom_quantizer().ComputeDistWithSplitCodeAndFilterDist(
                *computer, one_bit_code, supplement_code, hint_dist, result_dist);
        }
        if (computed) {
            this->add_reorder_hint_full_count(ctx, 1);
        } else if (has_hint) {
            this->add_reorder_fallback_full_count(ctx, 1);
        }
        if (not computed and not this->bottom_quantizer().ComputeDistWithSplitCode(
                                 *computer, one_bit_code, supplement_code, result_dist)) {
            ByteBuffer full_code(this->code_size_, allocator_);
            this->bottom_quantizer().MergeSplitCode(one_bit_code, supplement_code, full_code.data);
            computer->ComputeDist(full_code.data, result_dist);
        }
    }

    void
    compute_full_dist(InnerIdType id,
                      BottomComputer* computer,
                      float* result_dist,
                      QueryContext* ctx = nullptr,
                      float hint_dist = std::numeric_limits<float>::max()) const {
        auto one_bit_lease = this->x_bit_layout_->Acquire(id);
        auto supplement_lease = this->supplement_layout_->Acquire(id);
        this->compute_full_dist(
            one_bit_lease.Data(), supplement_lease.Data(), computer, result_dist, ctx, hint_dist);
    }
};

}  // namespace vsag
