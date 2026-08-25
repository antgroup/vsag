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
#include <array>
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
#include "io/common/basic_io.h"
#include "io/common/io_parameter.h"
#include "io/memory_io/memory_io.h"
#include "io/memory_io/memory_io_parameter.h"
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

class MMapIO;

class RaBitQSplitResidualOriginalQueryInterface {
public:
    virtual ~RaBitQSplitResidualOriginalQueryInterface() = default;

    virtual void
    InsertVectorWithFilterCodeAndResidualFullFactor(const void* vector,
                                                    InnerIdType idx,
                                                    const float* transformed_centroid,
                                                    uint8_t* filter_code,
                                                    float* full_add) = 0;

    virtual void
    PackageFastScan32ResidualWithFullFactors(const InnerIdType* ids,
                                             InnerIdType valid_size,
                                             const float* transformed_centroid,
                                             uint8_t* block,
                                             float* full_adds) const = 0;

    [[nodiscard]] virtual bool
    ComputeResidualFullFactorForId(const uint8_t* filter_code,
                                   InnerIdType id,
                                   const float* transformed_centroid,
                                   float* full_add) const = 0;

    [[nodiscard]] virtual bool
    RecoverFastScan32OriginalQueryFilterInnerProduct(
        const ComputerInterfacePtr& computer,
        const uint8_t* block,
        InnerIdType index_in_block,
        float shared_filter_inner_product,
        float* original_filter_inner_product) const = 0;

    virtual void
    QueryWithOriginalQueryFilterInnerProducts(float* result_dists,
                                              uint8_t* computed,
                                              const float* filter_inner_products,
                                              const float* full_adds,
                                              float query_bucket_norm_sqr,
                                              const ComputerInterfacePtr& computer,
                                              const InnerIdType* ids,
                                              InnerIdType id_count,
                                              QueryContext* ctx) const = 0;
};

template <MetricType metric,
          typename OneBitIOTmpl,
          typename SupplementIOTmpl = OneBitIOTmpl,
          typename QuantizerT = RaBitQuantizer<metric>>
class RaBitQSplitDataCell : public FlattenInterface,
                            public FlattenOptimizedBuildInterface,
                            public RaBitQSplitResidualOriginalQueryInterface {
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

    class FastScan32Computer final : public ComputerInterface {
    public:
        FastScan32Computer(uint64_t lookup_size,
                           uint64_t high_acc_lookup_size,
                           Allocator* allocator,
                           ComputerInterfacePtr bound_computer,
                           BottomComputer* bound_bottom_computer,
                           const uint8_t* query,
                           uint64_t query_size)
            : lookup_size_(lookup_size),
              high_acc_lookup_size_(high_acc_lookup_size),
              allocator_(allocator),
              bound_computer_(std::move(bound_computer)),
              bound_bottom_computer_(bound_bottom_computer) {
            const uint64_t query_seed = HashQuery(query, query_size);
            low_acc_random_state_ = MixSeed(query_seed ^ kLowAccSeedDomain);
            high_acc_random_state_ = MixSeed(query_seed ^ kHighAccSeedDomain);
        }

        const uint8_t*
        PrepareQuery(const BottomQuantizer& quantizer, BottomComputer& computer) {
            CHECK_ARGUMENT(&computer == bound_bottom_computer_,
                           "FastScan query computer does not match its factory computer");
            std::call_once(query_once_, [&]() {
                auto lookup_table = std::make_unique<ByteBuffer>(lookup_size_, allocator_);
                quantizer.PrepareFastScan32Query(computer,
                                                 lookup_table->data,
                                                 deltas_,
                                                 sum_vls_,
                                                 query_sum_,
                                                 &low_acc_random_state_);
                lookup_table_ = std::move(lookup_table);
            });
            return lookup_table_->data;
        }

        const uint8_t*
        PrepareHighAccQuery(const BottomQuantizer& quantizer, BottomComputer& computer) {
            CHECK_ARGUMENT(&computer == bound_bottom_computer_,
                           "FastScan query computer does not match its factory computer");
            std::call_once(high_acc_query_once_, [&]() {
                auto lookup_table = std::make_unique<ByteBuffer>(high_acc_lookup_size_, allocator_);
                quantizer.PrepareFastScan32HighAccQuery(computer,
                                                        lookup_table->data,
                                                        high_acc_deltas_,
                                                        high_acc_sum_vls_,
                                                        high_acc_query_sum_,
                                                        &high_acc_random_state_);
                high_acc_lookup_table_ = std::move(lookup_table);
            });
            return high_acc_lookup_table_->data;
        }

    private:
        static constexpr uint64_t kLowAccSeedDomain = 0x6C6F773846617374ULL;
        static constexpr uint64_t kHighAccSeedDomain = 0x686967684163634CULL;

        static uint64_t
        MixSeed(uint64_t value) {
            value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
            value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
            return value ^ (value >> 31U);
        }

        static uint64_t
        HashQuery(const uint8_t* query, uint64_t query_size) {
            CHECK_ARGUMENT(query != nullptr, "FastScan query is not initialized");
            uint64_t hash = 0xCBF29CE484222325ULL;
            for (uint64_t i = 0; i < query_size; ++i) {
                hash ^= query[i];
                hash *= 0x100000001B3ULL;
            }
            hash ^= query_size;
            return MixSeed(hash);
        }

    public:
        uint64_t lookup_size_;
        uint64_t high_acc_lookup_size_;
        Allocator* allocator_;
        ComputerInterfacePtr bound_computer_;
        BottomComputer* bound_bottom_computer_;
        std::unique_ptr<ByteBuffer> lookup_table_;
        std::unique_ptr<ByteBuffer> high_acc_lookup_table_;
        float deltas_[BottomQuantizer::FASTSCAN_MAX_FILTER_BITS]{};
        float sum_vls_[BottomQuantizer::FASTSCAN_MAX_FILTER_BITS]{};
        float query_sum_{0.0F};
        float high_acc_deltas_[BottomQuantizer::FASTSCAN_MAX_FILTER_BITS]{};
        float high_acc_sum_vls_[BottomQuantizer::FASTSCAN_MAX_FILTER_BITS]{};
        float high_acc_query_sum_{0.0F};
        uint64_t low_acc_random_state_{0};
        uint64_t high_acc_random_state_{0};
        std::once_flag query_once_;
        std::once_flag high_acc_query_once_;
    };

    [[nodiscard]] bool
    SupportFastScan32() const override {
        return this->bottom_quantizer().SupportFastScan32();
    }

    [[nodiscard]] uint64_t
    GetFastScan32BlockSize() const override {
        return this->bottom_quantizer().GetFastScan32BlockSize();
    }

    ComputerInterfacePtr
    FactoryFastScan32Computer(const ComputerInterfacePtr& computer) const override {
        if (not this->SupportFastScan32()) {
            return nullptr;
        }
        auto* bottom_computer = this->get_bottom_computer(computer);
        return std::make_shared<FastScan32Computer>(
            this->bottom_quantizer().GetFastScan32LookupSize(),
            this->bottom_quantizer().GetFastScan32HighAccLookupSize(),
            this->allocator_,
            computer,
            bottom_computer,
            bottom_computer->buf_,
            this->bottom_quantizer().GetQueryCodeSize());
    }

    void
    PackageFastScan32(const InnerIdType* ids,
                      InnerIdType valid_size,
                      uint8_t* block) const override {
        CHECK_ARGUMENT(valid_size <= BottomQuantizer::FASTSCAN_BATCH_SIZE,
                       "invalid FastScan batch size");
        ByteBuffer one_bit_codes(this->one_bit_code_size_ * BottomQuantizer::FASTSCAN_BATCH_SIZE,
                                 this->allocator_);
        ByteBuffer supplement_code(this->supplement_code_size_, this->allocator_);
        memset(
            one_bit_codes.data, 0, this->one_bit_code_size_ * BottomQuantizer::FASTSCAN_BATCH_SIZE);
        for (InnerIdType i = 0; i < valid_size; ++i) {
            if (ids[i] == std::numeric_limits<InnerIdType>::max()) {
                continue;
            }
            auto* one_bit_code = one_bit_codes.data + i * this->one_bit_code_size_;
            if (this->optimized_build_active_) {
                bool need_release = false;
                const auto* scalar_code =
                    this->optimized_build_scalar_layout_->Read(ids[i], need_release);
                CHECK_ARGUMENT(scalar_code != nullptr,
                               "failed to read scalar RaBitQ code for FastScan packaging");
                this->bottom_quantizer().PackScalarCodeToSplitCode(
                    scalar_code, one_bit_code, supplement_code.data);
                if (need_release) {
                    this->optimized_build_scalar_layout_->Release(scalar_code);
                }
            } else {
                CHECK_ARGUMENT(this->x_bit_layout_->Read(ids[i], one_bit_code),
                               "failed to read RaBitQ filter code for FastScan packaging");
            }
        }
        this->bottom_quantizer().PackageFastScan32(one_bit_codes.data, valid_size, block);
    }

    void
    PackageFastScan32Residual(const InnerIdType* ids,
                              InnerIdType valid_size,
                              const float* transformed_centroid,
                              uint8_t* block) const override {
        CHECK_ARGUMENT(valid_size <= BottomQuantizer::FASTSCAN_BATCH_SIZE,
                       "invalid FastScan batch size");
        ByteBuffer one_bit_codes(this->one_bit_code_size_ * BottomQuantizer::FASTSCAN_BATCH_SIZE,
                                 this->allocator_);
        ByteBuffer supplement_code(this->supplement_code_size_, this->allocator_);
        memset(
            one_bit_codes.data, 0, this->one_bit_code_size_ * BottomQuantizer::FASTSCAN_BATCH_SIZE);
        for (InnerIdType i = 0; i < valid_size; ++i) {
            if (ids[i] == std::numeric_limits<InnerIdType>::max()) {
                continue;
            }
            auto* one_bit_code = one_bit_codes.data + i * this->one_bit_code_size_;
            if (this->optimized_build_active_) {
                bool need_release = false;
                const auto* scalar_code =
                    this->optimized_build_scalar_layout_->Read(ids[i], need_release);
                CHECK_ARGUMENT(scalar_code != nullptr,
                               "failed to read scalar RaBitQ code for FastScan packaging");
                this->bottom_quantizer().PackScalarCodeToSplitCode(
                    scalar_code, one_bit_code, supplement_code.data);
                if (need_release) {
                    this->optimized_build_scalar_layout_->Release(scalar_code);
                }
            } else {
                CHECK_ARGUMENT(this->x_bit_layout_->Read(ids[i], one_bit_code),
                               "failed to read RaBitQ filter code for FastScan packaging");
            }
        }
        this->bottom_quantizer().PackageFastScan32Residual(
            one_bit_codes.data, transformed_centroid, valid_size, block);
    }
    void
    PackageFastScan32ResidualWithFullFactors(const InnerIdType* ids,
                                             InnerIdType valid_size,
                                             const float* transformed_centroid,
                                             uint8_t* block,
                                             float* full_adds) const override {
        CHECK_ARGUMENT(valid_size <= BottomQuantizer::FASTSCAN_BATCH_SIZE,
                       "invalid FastScan batch size");
        CHECK_ARGUMENT(full_adds != nullptr, "residual full factors are required");
        ByteBuffer one_bit_codes(this->one_bit_code_size_ * BottomQuantizer::FASTSCAN_BATCH_SIZE,
                                 this->allocator_);
        ByteBuffer supplement_code(this->supplement_code_size_, this->allocator_);
        memset(
            one_bit_codes.data, 0, this->one_bit_code_size_ * BottomQuantizer::FASTSCAN_BATCH_SIZE);
        std::fill_n(full_adds, valid_size, std::numeric_limits<float>::quiet_NaN());
        for (InnerIdType i = 0; i < valid_size; ++i) {
            if (ids[i] == std::numeric_limits<InnerIdType>::max()) {
                continue;
            }
            auto* one_bit_code = one_bit_codes.data + i * this->one_bit_code_size_;
            if (this->optimized_build_active_) {
                bool need_release = false;
                const auto* scalar_code =
                    this->optimized_build_scalar_layout_->Read(ids[i], need_release);
                CHECK_ARGUMENT(scalar_code != nullptr,
                               "failed to read scalar RaBitQ code for FastScan packaging");
                this->bottom_quantizer().PackScalarCodeToSplitCode(
                    scalar_code, one_bit_code, supplement_code.data);
                if (need_release) {
                    this->optimized_build_scalar_layout_->Release(scalar_code);
                }
            } else {
                CHECK_ARGUMENT(this->x_bit_layout_->Read(ids[i], one_bit_code),
                               "failed to read RaBitQ filter code for FastScan packaging");
                CHECK_ARGUMENT(this->supplement_layout_->Read(ids[i], supplement_code.data),
                               "failed to read RaBitQ supplement code for FastScan packaging");
            }
            this->bottom_quantizer().ComputeResidualFullFactor(
                one_bit_code, supplement_code.data, transformed_centroid, full_adds + i);
        }
        this->bottom_quantizer().PackageFastScan32Residual(
            one_bit_codes.data, transformed_centroid, valid_size, block);
    }

    void
    EnableExternalFilterCodeStorage() override {
        this->external_filter_code_storage_ = true;
    }

    [[nodiscard]] uint64_t
    GetFilterCodeSize() const override {
        return this->one_bit_code_size_;
    }

    void
    SetFastScan32Code(const uint8_t* filter_code,
                      InnerIdType index_in_block,
                      uint8_t* block) const override {
        this->bottom_quantizer().SetFastScan32Code(filter_code, index_in_block, block);
    }

    void
    SetFastScan32ResidualCode(const uint8_t* filter_code,
                              const float* transformed_centroid,
                              InnerIdType index_in_block,
                              uint8_t* block) const override {
        this->bottom_quantizer().SetFastScan32ResidualCode(
            filter_code, transformed_centroid, index_in_block, block);
    }

    void
    UnpackFastScan32Code(const uint8_t* block,
                         InnerIdType index_in_block,
                         uint8_t* filter_code) const override {
        this->bottom_quantizer().UnpackFastScan32Code(block, index_in_block, filter_code);
    }

    void
    InsertVectorWithFilterCode(const void* vector, InnerIdType idx, uint8_t* filter_code) override {
        CHECK_ARGUMENT(not this->optimized_build_active_,
                       "optimized build filter codes are packaged during finalization");
        {
            std::lock_guard lock(this->mutex_);
            if (idx == std::numeric_limits<InnerIdType>::max()) {
                idx = this->total_count_;
            }
            this->total_count_ = std::max(this->total_count_, idx + 1);
        }
        ByteBuffer full_code(this->code_size_, this->allocator_);
        ByteBuffer supplement_code(this->supplement_code_size_, this->allocator_);
        this->quantizer_->EncodeOne(static_cast<const float*>(vector), full_code.data);
        this->bottom_quantizer().SplitCode(full_code.data, filter_code, supplement_code.data);
        this->supplement_layout_->Write(idx, supplement_code.data);
    }
    void
    InsertVectorWithFilterCodeAndResidualFullFactor(const void* vector,
                                                    InnerIdType idx,
                                                    const float* transformed_centroid,
                                                    uint8_t* filter_code,
                                                    float* full_add) override {
        CHECK_ARGUMENT(not this->optimized_build_active_,
                       "optimized build filter codes are packaged during finalization");
        CHECK_ARGUMENT(full_add != nullptr, "residual full factor output is required");
        {
            std::lock_guard lock(this->mutex_);
            if (idx == std::numeric_limits<InnerIdType>::max()) {
                idx = this->total_count_;
            }
            this->total_count_ = std::max(this->total_count_, idx + 1);
        }
        ByteBuffer full_code(this->code_size_, this->allocator_);
        ByteBuffer supplement_code(this->supplement_code_size_, this->allocator_);
        this->quantizer_->EncodeOne(static_cast<const float*>(vector), full_code.data);
        this->bottom_quantizer().SplitCode(full_code.data, filter_code, supplement_code.data);
        *full_add = std::numeric_limits<float>::quiet_NaN();
        this->bottom_quantizer().ComputeResidualFullFactor(
            filter_code, supplement_code.data, transformed_centroid, full_add);
        this->supplement_layout_->Write(idx, supplement_code.data);
    }

    [[nodiscard]] bool
    ComputeResidualFullFactorForId(const uint8_t* filter_code,
                                   InnerIdType id,
                                   const float* transformed_centroid,
                                   float* full_add) const override {
        bool need_release = false;
        const uint8_t* supplement_code = nullptr;
        bool computed = false;
        try {
            supplement_code = this->get_supplement_code(id, need_release);
            if (supplement_code != nullptr) {
                computed = this->bottom_quantizer().ComputeResidualFullFactor(
                    filter_code, supplement_code, transformed_centroid, full_add);
            }
        } catch (...) {
            this->release_supplement_code(supplement_code, need_release);
            throw;
        }
        this->release_supplement_code(supplement_code, need_release);
        return computed;
    }

    void
    QueryWithFilterCodes(float* result_dists,
                         const float* hint_dists,
                         const float* filter_inner_products,
                         const ComputerInterfacePtr& computer,
                         const InnerIdType* idx,
                         const uint8_t* filter_codes,
                         InnerIdType id_count,
                         QueryContext* ctx = nullptr) const override {
        auto* comp = this->get_bottom_computer(computer);
        for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; ++i) {
            this->prefetch_supplement(idx[i]);
        }
        for (InnerIdType i = 0; i < id_count; ++i) {
            if (i + this->prefetch_stride_code_ < id_count) {
                this->prefetch_supplement(idx[i + this->prefetch_stride_code_]);
            }
            bool supplement_need_release = false;
            const uint8_t* supplement_code = nullptr;
            const auto* filter_code =
                filter_codes + static_cast<uint64_t>(i) * this->one_bit_code_size_;
            try {
                supplement_code = this->get_supplement_code(idx[i], supplement_need_release);
                bool computed = false;
                if (filter_inner_products != nullptr and
                    this->is_finite_float_bits(filter_inner_products[i])) {
                    computed =
                        this->bottom_quantizer().ComputeDistWithSplitCodeAndFilterInnerProduct(
                            *comp, supplement_code, filter_inner_products[i], result_dists + i);
                    if (computed) {
                        this->add_full_count(ctx, 1);
                        this->add_reorder_hint_full_count(ctx, 1);
                    } else {
                        this->add_reorder_fallback_full_count(ctx, 1);
                    }
                }
                if (not computed) {
                    const float hint =
                        hint_dists == nullptr ? std::numeric_limits<float>::max() : hint_dists[i];
                    this->compute_full_dist(
                        filter_code, supplement_code, comp, result_dists + i, ctx, hint);
                }
            } catch (...) {
                this->release_supplement_code(supplement_code, supplement_need_release);
                throw;
            }
            this->release_supplement_code(supplement_code, supplement_need_release);
        }
        this->add_distance_evaluations(ctx, id_count);
    }

    void
    QueryWithFilterInnerProducts(float* result_dists,
                                 uint8_t* computed,
                                 const float* filter_inner_products,
                                 const ComputerInterfacePtr& computer,
                                 const InnerIdType* idx,
                                 InnerIdType id_count,
                                 QueryContext* ctx = nullptr) const override {
        auto* comp = this->get_bottom_computer(computer);
        for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; ++i) {
            if (this->is_finite_float_bits(filter_inner_products[i])) {
                this->prefetch_supplement(idx[i]);
            }
        }

        uint64_t computed_count = 0;
        for (InnerIdType i = 0; i < id_count; ++i) {
            computed[i] = 0U;
            if (i + this->prefetch_stride_code_ < id_count and
                this->is_finite_float_bits(
                    filter_inner_products[i + this->prefetch_stride_code_])) {
                this->prefetch_supplement(idx[i + this->prefetch_stride_code_]);
            }
            if (not this->is_finite_float_bits(filter_inner_products[i])) {
                continue;
            }

            bool supplement_need_release = false;
            const uint8_t* supplement_code = nullptr;
            try {
                supplement_code = this->get_supplement_code(idx[i], supplement_need_release);
                computed[i] = static_cast<uint8_t>(
                    this->bottom_quantizer().ComputeDistWithSplitCodeAndFilterInnerProduct(
                        *comp, supplement_code, filter_inner_products[i], result_dists + i));
            } catch (...) {
                this->release_supplement_code(supplement_code, supplement_need_release);
                throw;
            }
            this->release_supplement_code(supplement_code, supplement_need_release);
            computed_count += computed[i];
        }
        this->add_full_count(ctx, computed_count);
        this->add_reorder_hint_full_count(ctx, computed_count);
        this->add_reorder_fallback_full_count(ctx,
                                              static_cast<uint64_t>(id_count) - computed_count);
        this->add_distance_evaluations(ctx, computed_count);
    }
    void
    QueryWithOriginalQueryFilterInnerProducts(float* result_dists,
                                              uint8_t* computed,
                                              const float* filter_inner_products,
                                              const float* full_adds,
                                              float query_bucket_norm_sqr,
                                              const ComputerInterfacePtr& computer,
                                              const InnerIdType* idx,
                                              InnerIdType id_count,
                                              QueryContext* ctx) const override {
        auto* comp = this->get_bottom_computer(computer);
        for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; ++i) {
            if (this->is_finite_float_bits(filter_inner_products[i]) and
                this->is_finite_float_bits(full_adds[i])) {
                this->prefetch_supplement(idx[i]);
            }
        }

        uint64_t computed_count = 0;
        for (InnerIdType i = 0; i < id_count; ++i) {
            computed[i] = 0U;
            if (i + this->prefetch_stride_code_ < id_count and
                this->is_finite_float_bits(
                    filter_inner_products[i + this->prefetch_stride_code_]) and
                this->is_finite_float_bits(full_adds[i + this->prefetch_stride_code_])) {
                this->prefetch_supplement(idx[i + this->prefetch_stride_code_]);
            }
            if (not this->is_finite_float_bits(filter_inner_products[i]) or
                not this->is_finite_float_bits(full_adds[i])) {
                continue;
            }

            bool need_release = false;
            const uint8_t* supplement_code = nullptr;
            try {
                supplement_code = this->get_supplement_code(idx[i], need_release);
                computed[i] = static_cast<uint8_t>(
                    this->bottom_quantizer()
                        .ComputeDistWithSplitCodeAndOriginalQueryFilterInnerProduct(
                            *comp,
                            supplement_code,
                            filter_inner_products[i],
                            full_adds[i],
                            query_bucket_norm_sqr,
                            result_dists + i));
            } catch (...) {
                this->release_supplement_code(supplement_code, need_release);
                throw;
            }
            this->release_supplement_code(supplement_code, need_release);
            computed_count += computed[i];
        }
        this->add_full_count(ctx, computed_count);
        this->add_reorder_hint_full_count(ctx, computed_count);
        this->add_reorder_fallback_full_count(ctx,
                                              static_cast<uint64_t>(id_count) - computed_count);
        this->add_distance_evaluations(ctx, computed_count);
    }

    [[nodiscard]] bool
    GetCodesByIdWithFilterCode(InnerIdType id,
                               const uint8_t* filter_code,
                               uint8_t* codes) const override {
        ByteBuffer supplement_code(this->supplement_code_size_, this->allocator_);
        if (not this->supplement_layout_->Read(id, supplement_code.data)) {
            return false;
        }
        memset(codes, 0, this->code_size_);
        this->bottom_quantizer().MergeSplitCode(filter_code, supplement_code.data, codes);
        return true;
    }

    float
    ComputePairVectorsWithFilterCodes(InnerIdType id1,
                                      const uint8_t* filter_code1,
                                      InnerIdType id2,
                                      const uint8_t* filter_code2) override {
        ByteBuffer codes1(this->code_size_, this->allocator_);
        ByteBuffer codes2(this->code_size_, this->allocator_);
        CHECK_ARGUMENT(this->GetCodesByIdWithFilterCode(id1, filter_code1, codes1.data),
                       "failed to read first split RaBitQ code");
        CHECK_ARGUMENT(this->GetCodesByIdWithFilterCode(id2, filter_code2, codes2.data),
                       "failed to read second split RaBitQ code");
        return this->bottom_quantizer().Compute(codes1.data, codes2.data);
    }

    void
    PrefetchSupplement(InnerIdType id) override {
        this->prefetch_supplement(id);
    }

    void
    DiscardFilterCodes() override {
        this->x_bit_layout_->Shrink(0);
    }

    [[nodiscard]] bool
    HasFilterCodes() const override {
        if (this->total_count_ == 0) {
            return false;
        }
        ByteBuffer filter_code(this->one_bit_code_size_, this->allocator_);
        return this->x_bit_layout_->Read(0, filter_code.data);
    }

    void
    MergeSupplementCodes(const FlattenInterfacePtr& other, InnerIdType bias) override {
        auto ptr = std::dynamic_pointer_cast<
            RaBitQSplitDataCell<metric, OneBitIOTmpl, SupplementIOTmpl, QuantizerT>>(other);
        CHECK_ARGUMENT(ptr != nullptr, "merge supplement codes datacell type mismatch");
        const InnerIdType final_count = bias + ptr->total_count_;
        const InnerIdType required_count = std::max(this->total_count_, final_count);
        this->supplement_layout_->Resize(required_count);
        ByteBuffer supplement_code(this->supplement_code_size_, this->allocator_);
        for (InnerIdType id = 0; id < ptr->total_count_; ++id) {
            CHECK_ARGUMENT(ptr->supplement_layout_->Read(id, supplement_code.data),
                           "failed to read source supplement code");
            this->supplement_layout_->Write(bias + id, supplement_code.data);
        }
        this->total_count_ = required_count;
        this->max_capacity_ = std::max(this->max_capacity_, required_count);
    }

    void
    QueryFastScan32(float* result_dists,
                    bool* computed,
                    const ComputerInterfacePtr& computer,
                    const ComputerInterfacePtr& fastscan_computer,
                    const uint8_t* block,
                    InnerIdType valid_size,
                    QueryContext* ctx = nullptr) const override {
        auto* fastscan = dynamic_cast<FastScan32Computer*>(fastscan_computer.get());
        CHECK_ARGUMENT(fastscan != nullptr, "invalid RaBitQ FastScan computer");
        auto* bottom_computer = this->get_bottom_computer(computer);
        const auto* lookup_table =
            fastscan->PrepareQuery(this->bottom_quantizer(), *bottom_computer);
        this->bottom_quantizer().ComputeDistsWithFastScan32(*bottom_computer,
                                                            block,
                                                            lookup_table,
                                                            fastscan->deltas_,
                                                            fastscan->sum_vls_,
                                                            fastscan->query_sum_,
                                                            result_dists,
                                                            computed,
                                                            valid_size,
                                                            this->query_rabitq_error_rate(ctx));
        this->add_filter_count(ctx, valid_size);
        this->add_distance_evaluations(ctx, valid_size);
    }

    void
    QueryFastScan32WithDistanceLowerBoundAndFilterInnerProduct(
        float* result_dists,
        float* lower_bounds,
        float* filter_inner_products,
        bool* computed,
        const ComputerInterfacePtr& computer,
        const ComputerInterfacePtr& fastscan_computer,
        const uint8_t* block,
        InnerIdType valid_size,
        QueryContext* ctx = nullptr) const override {
        auto* fastscan = dynamic_cast<FastScan32Computer*>(fastscan_computer.get());
        CHECK_ARGUMENT(fastscan != nullptr, "invalid RaBitQ FastScan computer");
        auto* bottom_computer = this->get_bottom_computer(computer);
        const auto* lookup_table =
            fastscan->PrepareQuery(this->bottom_quantizer(), *bottom_computer);
        this->bottom_quantizer().ComputeDistsWithFastScan32(*bottom_computer,
                                                            block,
                                                            lookup_table,
                                                            fastscan->deltas_,
                                                            fastscan->sum_vls_,
                                                            fastscan->query_sum_,
                                                            result_dists,
                                                            computed,
                                                            valid_size,
                                                            this->query_rabitq_error_rate(ctx),
                                                            lower_bounds,
                                                            filter_inner_products);
        this->add_filter_count(ctx, valid_size);
        this->add_distance_evaluations(ctx, valid_size);
    }

    void
    QueryFastScan32Batch(float* result_dists,
                         uint32_t* computed_masks,
                         const ComputerInterfacePtr& computer,
                         const ComputerInterfacePtr& fastscan_computer,
                         const uint8_t* blocks,
                         InnerIdType total_size,
                         float* filter_inner_products = nullptr,
                         QueryContext* ctx = nullptr) const override {
        auto* fastscan = dynamic_cast<FastScan32Computer*>(fastscan_computer.get());
        CHECK_ARGUMENT(fastscan != nullptr, "invalid RaBitQ FastScan computer");
        auto* bottom_computer = this->get_bottom_computer(computer);
        const auto* lookup_table =
            fastscan->PrepareHighAccQuery(this->bottom_quantizer(), *bottom_computer);
        this->bottom_quantizer().ComputeDistsWithFastScan32HighAccBatch(
            *bottom_computer,
            blocks,
            total_size,
            lookup_table,
            fastscan->high_acc_deltas_,
            fastscan->high_acc_sum_vls_,
            fastscan->high_acc_query_sum_,
            result_dists,
            computed_masks,
            filter_inner_products);
        this->add_filter_count(ctx, total_size);
        this->add_distance_evaluations(ctx, total_size);
    }

    void
    QueryFastScan32BatchWithDistanceLowerBoundAndFilterInnerProduct(
        float* result_dists,
        float* lower_bounds,
        float* filter_inner_products,
        uint32_t* computed_masks,
        const ComputerInterfacePtr& computer,
        const ComputerInterfacePtr& fastscan_computer,
        const uint8_t* blocks,
        InnerIdType total_size,
        QueryContext* ctx = nullptr) const override {
        auto* fastscan = dynamic_cast<FastScan32Computer*>(fastscan_computer.get());
        CHECK_ARGUMENT(fastscan != nullptr, "invalid RaBitQ FastScan computer");
        auto* bottom_computer = this->get_bottom_computer(computer);
        const auto* lookup_table =
            fastscan->PrepareQuery(this->bottom_quantizer(), *bottom_computer);
        const uint64_t block_size = this->GetFastScan32BlockSize();
        const uint64_t block_count =
            (static_cast<uint64_t>(total_size) + BottomQuantizer::FASTSCAN_BATCH_SIZE - 1) /
            BottomQuantizer::FASTSCAN_BATCH_SIZE;
        const float error_rate = this->query_rabitq_error_rate(ctx);
        for (uint64_t block_index = 0; block_index < block_count; ++block_index) {
            const auto begin =
                static_cast<InnerIdType>(block_index * BottomQuantizer::FASTSCAN_BATCH_SIZE);
            const auto valid_size =
                std::min<InnerIdType>(BottomQuantizer::FASTSCAN_BATCH_SIZE, total_size - begin);
            bool computed[BottomQuantizer::FASTSCAN_BATCH_SIZE] = {};
            this->bottom_quantizer().ComputeDistsWithFastScan32(*bottom_computer,
                                                                blocks + block_index * block_size,
                                                                lookup_table,
                                                                fastscan->deltas_,
                                                                fastscan->sum_vls_,
                                                                fastscan->query_sum_,
                                                                result_dists + begin,
                                                                computed,
                                                                valid_size,
                                                                error_rate,
                                                                lower_bounds + begin,
                                                                filter_inner_products + begin);
            uint32_t mask = 0;
            for (InnerIdType i = 0; i < valid_size; ++i) {
                mask |= static_cast<uint32_t>(computed[i]) << i;
            }
            computed_masks[block_index] = mask;
        }
        this->add_filter_count(ctx, total_size);
        this->add_distance_evaluations(ctx, total_size);
    }

    void
    QueryFastScan32BatchSharedResidual(float* result_dists,
                                       uint32_t* computed_masks,
                                       const ComputerInterfacePtr& computer,
                                       const ComputerInterfacePtr& fastscan_computer,
                                       const uint8_t* blocks,
                                       InnerIdType total_size,
                                       float query_bucket_norm_sqr,
                                       float* filter_inner_products = nullptr,
                                       QueryContext* ctx = nullptr) const override {
        auto* fastscan = dynamic_cast<FastScan32Computer*>(fastscan_computer.get());
        CHECK_ARGUMENT(fastscan != nullptr, "invalid RaBitQ FastScan computer");
        auto* bottom_computer = this->get_bottom_computer(computer);
        const auto* lookup_table =
            fastscan->PrepareHighAccQuery(this->bottom_quantizer(), *bottom_computer);
        this->bottom_quantizer().ComputeDistsWithFastScan32SharedResidualHighAccBatch(
            *bottom_computer,
            blocks,
            total_size,
            lookup_table,
            fastscan->high_acc_deltas_,
            fastscan->high_acc_sum_vls_,
            fastscan->high_acc_query_sum_,
            query_bucket_norm_sqr,
            result_dists,
            computed_masks,
            filter_inner_products);
        this->add_filter_count(ctx, total_size);
        this->add_distance_evaluations(ctx, total_size);
    }

    [[nodiscard]] bool
    ConvertFastScan32SharedResidualFilterInnerProduct(
        const ComputerInterfacePtr& shared_computer,
        const ComputerInterfacePtr& residual_computer,
        const uint8_t* block,
        InnerIdType index_in_block,
        float shared_filter_inner_product,
        float* residual_filter_inner_product) const override {
        return this->bottom_quantizer().ConvertFastScan32SharedResidualFilterInnerProduct(
            *this->get_bottom_computer(shared_computer),
            *this->get_bottom_computer(residual_computer),
            block,
            index_in_block,
            shared_filter_inner_product,
            residual_filter_inner_product);
    }
    [[nodiscard]] bool
    RecoverFastScan32OriginalQueryFilterInnerProduct(
        const ComputerInterfacePtr& computer,
        const uint8_t* block,
        InnerIdType index_in_block,
        float shared_filter_inner_product,
        float* original_filter_inner_product) const override {
        return this->bottom_quantizer().RecoverFastScan32OriginalQueryFilterInnerProduct(
            *this->get_bottom_computer(computer),
            block,
            index_in_block,
            shared_filter_inner_product,
            original_filter_inner_product);
    }

    [[nodiscard]] float
    ComputeTransformedResidualQueryNormSqr(const float* transformed_query) const override {
        if constexpr (std::is_same_v<QuantizerT, BottomQuantizer>) {
            return this->bottom_quantizer().ComputeTransformedResidualQueryNormSqr(
                transformed_query);
        }
        return FlattenInterface::ComputeTransformedResidualQueryNormSqr(transformed_query);
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

        bool need_release = false;
        const auto* query_code = this->optimized_build_scalar_layout_->Read(query_id, need_release);
        if (query_code == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "failed to read temporary scalar RaBitQ query code");
        }
        try {
            this->query_optimized_build_code_pairs(result_dists,
                                                   query_code,
                                                   (*this->optimized_build_code_sums_)[query_id],
                                                   idx,
                                                   id_count);
        } catch (...) {
            if (need_release) {
                this->optimized_build_scalar_layout_->Release(query_code);
            }
            throw;
        }
        if (need_release) {
            this->optimized_build_scalar_layout_->Release(query_code);
        }
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

            bool one_bit_need_release = false;
            const uint8_t* one_bit_code = this->get_one_bit_code(idx[i], one_bit_need_release);
            float one_bit_dist = 0.0F;
            float lower_bound = std::numeric_limits<float>::max();
            bool computed = false;
            try {
                computed = this->bottom_quantizer().ComputeDistWithOneBitLowerBound(
                    *comp,
                    one_bit_code,
                    &one_bit_dist,
                    &lower_bound,
                    this->query_rabitq_error_rate(ctx));
            } catch (...) {
                this->release_one_bit_code(one_bit_code, one_bit_need_release);
                throw;
            }

            if (computed and std::isfinite(lower_bound) and lower_bound >= threshold) {
                this->release_one_bit_code(one_bit_code, one_bit_need_release);
                result_dists[i] = threshold;
                continue;
            }

            bool supplement_need_release = false;
            const uint8_t* supplement_code = nullptr;
            try {
                supplement_code = this->get_supplement_code(idx[i], supplement_need_release);
                this->compute_full_dist(one_bit_code, supplement_code, comp, result_dists + i, ctx);
            } catch (...) {
                this->release_one_bit_code(one_bit_code, one_bit_need_release);
                this->release_supplement_code(supplement_code, supplement_need_release);
                throw;
            }
            this->release_one_bit_code(one_bit_code, one_bit_need_release);
            this->release_supplement_code(supplement_code, supplement_need_release);
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

            bool release1 = false, release2 = false, release3 = false, release4 = false;
            const uint8_t* code1 = nullptr;
            const uint8_t* code2 = nullptr;
            const uint8_t* code3 = nullptr;
            const uint8_t* code4 = nullptr;
            auto release_batch = [&]() {
                this->release_one_bit_code(code1, release1);
                this->release_one_bit_code(code2, release2);
                this->release_one_bit_code(code3, release3);
                this->release_one_bit_code(code4, release4);
            };

            try {
                code1 = this->get_one_bit_code(idx[i], release1);
                code2 = this->get_one_bit_code(idx[i + 1], release2);
                code3 = this->get_one_bit_code(idx[i + 2], release3);
                code4 = this->get_one_bit_code(idx[i + 3], release4);
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
            } catch (...) {
                release_batch();
                throw;
            }
            release_batch();
        }

        for (; i < id_count; ++i) {
            if (i + this->prefetch_stride_code_ < id_count) {
                this->prefetch_one_bit(idx[i + this->prefetch_stride_code_]);
            }

            bool one_bit_need_release = false;
            const uint8_t* one_bit_code = this->get_one_bit_code(idx[i], one_bit_need_release);
            auto* lower_bound = lower_bounds == nullptr ? nullptr : lower_bounds + i;
            bool computed = false;
            try {
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
            } catch (...) {
                this->release_one_bit_code(one_bit_code, one_bit_need_release);
                throw;
            }
            this->release_one_bit_code(one_bit_code, one_bit_need_release);
        }
        this->add_distance_evaluations(ctx, id_count);
    }

    void
    QueryWithDistanceLowerBoundAndFilterInnerProduct(float* result_dists,
                                                     float* lower_bounds,
                                                     float* filter_inner_products,
                                                     const ComputerInterfacePtr& computer,
                                                     const InnerIdType* idx,
                                                     InnerIdType id_count,
                                                     QueryContext* ctx = nullptr) override {
        if (this->optimized_build_active_) {
            this->query_optimized_build_codes(result_dists, computer, idx, id_count);
            std::fill(
                lower_bounds, lower_bounds + id_count, -std::numeric_limits<float>::infinity());
            std::fill(filter_inner_products,
                      filter_inner_products + id_count,
                      std::numeric_limits<float>::quiet_NaN());
            this->add_distance_evaluations(ctx, id_count);
            return;
        }

        auto* comp = this->get_bottom_computer(computer);
        this->add_filter_count(ctx, id_count);
        for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; ++i) {
            this->prefetch_one_bit(idx[i]);
        }

        for (InnerIdType i = 0; i < id_count; ++i) {
            if (i + this->prefetch_stride_code_ < id_count) {
                this->prefetch_one_bit(idx[i + this->prefetch_stride_code_]);
            }

            bool one_bit_need_release = false;
            const uint8_t* one_bit_code = this->get_one_bit_code(idx[i], one_bit_need_release);
            bool computed = false;
            try {
                computed = this->bottom_quantizer().ComputeDistWithOneBitLowerBound(
                    *comp,
                    one_bit_code,
                    result_dists + i,
                    lower_bounds + i,
                    this->query_rabitq_error_rate(ctx),
                    filter_inner_products + i);
                if (not computed) {
                    this->add_filter_fallback_full_count(ctx, 1);
                    this->compute_full_dist_after_one_bit_failure(
                        idx[i], one_bit_code, comp, result_dists + i, nullptr, ctx);
                    lower_bounds[i] = -std::numeric_limits<float>::infinity();
                    filter_inner_products[i] = std::numeric_limits<float>::quiet_NaN();
                }
            } catch (...) {
                this->release_one_bit_code(one_bit_code, one_bit_need_release);
                throw;
            }
            this->release_one_bit_code(one_bit_code, one_bit_need_release);
        }
        this->add_distance_evaluations(ctx, id_count);
    }

    void
    QueryWithFilterInnerProduct(float* result_dists,
                                const float* filter_inner_products,
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
        for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; ++i) {
            this->prefetch_supplement(idx[i]);
        }

        for (InnerIdType i = 0; i < id_count; ++i) {
            if (i + this->prefetch_stride_code_ < id_count) {
                this->prefetch_supplement(idx[i + this->prefetch_stride_code_]);
            }

            bool computed = false;
            const float filter_inner_product = filter_inner_products[i];
            if (this->is_finite_float_bits(filter_inner_product)) {
                bool supplement_need_release = false;
                const uint8_t* supplement_code = nullptr;
                try {
                    supplement_code = this->get_supplement_code(idx[i], supplement_need_release);
                    computed =
                        this->bottom_quantizer().ComputeDistWithSplitCodeAndFilterInnerProduct(
                            *comp, supplement_code, filter_inner_product, result_dists + i);
                } catch (...) {
                    this->release_supplement_code(supplement_code, supplement_need_release);
                    throw;
                }
                this->release_supplement_code(supplement_code, supplement_need_release);
            }

            if (computed) {
                this->add_full_count(ctx, 1);
                this->add_reorder_hint_full_count(ctx, 1);
            } else {
                if (this->is_finite_float_bits(filter_inner_product)) {
                    this->add_reorder_fallback_full_count(ctx, 1);
                }
                this->compute_full_dist(idx[i], comp, result_dists + i, ctx);
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

    [[nodiscard]] bool
    SupportResidualQueryTransform() const override {
        return metric == MetricType::METRIC_TYPE_L2SQR and
               std::is_same_v<QuantizerT, BottomQuantizer>;
    }

    [[nodiscard]] uint64_t
    GetResidualQueryTransformSize() const override {
        if constexpr (std::is_same_v<QuantizerT, BottomQuantizer>) {
            return this->bottom_quantizer().GetResidualQueryTransformSize();
        }
        return 0;
    }

    void
    TransformResidualQuery(const float* query, float* transformed_query) const override {
        if constexpr (std::is_same_v<QuantizerT, BottomQuantizer>) {
            this->bottom_quantizer().TransformResidualQuery(query, transformed_query);
            return;
        }
        FlattenInterface::TransformResidualQuery(query, transformed_query);
    }

    ComputerInterfacePtr
    FactoryComputerFromResidualQuery(const float* transformed_query) override {
        if constexpr (std::is_same_v<QuantizerT, BottomQuantizer>) {
            auto computer = this->quantizer_->FactoryComputer();
            this->bottom_quantizer().ProcessTransformedResidualQuery(
                transformed_query, *this->get_bottom_computer(computer));
            return computer;
        }
        return FlattenInterface::FactoryComputerFromResidualQuery(transformed_query);
    }

    void
    ResetComputerFromResidualQuery(const float* transformed_query,
                                   ComputerInterfacePtr& computer) override {
        if constexpr (std::is_same_v<QuantizerT, BottomQuantizer>) {
            if (computer == nullptr) {
                computer = this->quantizer_->FactoryComputer();
            }
            this->bottom_quantizer().ProcessTransformedResidualQuery(
                transformed_query, *this->get_bottom_computer(computer));
            return;
        }
        FlattenInterface::ResetComputerFromResidualQuery(transformed_query, computer);
    }

    void
    FactoryFastScan32ComputersFromResidualQueries(
        const float* transformed_queries,
        uint64_t query_count,
        ComputerInterfacePtr* computers,
        ComputerInterfacePtr* fastscan_computers) override {
        if constexpr (std::is_same_v<QuantizerT, BottomQuantizer>) {
            CHECK_ARGUMENT(query_count == 0 or transformed_queries != nullptr,
                           "transformed residual queries are required");
            CHECK_ARGUMENT(query_count == 0 or computers != nullptr,
                           "residual query computer output is required");
            CHECK_ARGUMENT(query_count == 0 or fastscan_computers != nullptr,
                           "FastScan computer output is required");
            const uint64_t transform_size = this->GetResidualQueryTransformSize();
            for (uint64_t i = 0; i < query_count; ++i) {
                auto computer = this->quantizer_->FactoryComputer();
                auto& bottom_computer = Accessor::GetComputer(*computer);
                this->bottom_quantizer().ProcessTransformedResidualQuery(
                    transformed_queries + i * transform_size, bottom_computer);
                computers[i] = std::move(computer);
                fastscan_computers[i] = this->FactoryFastScan32Computer(computers[i]);
            }
            return;
        }
        FlattenInterface::FactoryFastScan32ComputersFromResidualQueries(
            transformed_queries, query_count, computers, fastscan_computers);
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
        if (not this->external_filter_code_storage_) {
            this->x_bit_layout_->Resize(final_capacity);
        }
        this->supplement_layout_->Resize(final_capacity);
        this->max_capacity_ = final_capacity;

        auto finalize_range = [this](InnerIdType begin, InnerIdType end) {
            ByteBuffer one_bit_code(this->one_bit_code_size_, allocator_);
            ByteBuffer supplement_code(this->supplement_code_size_, allocator_);
            for (InnerIdType id = begin; id < end; ++id) {
                bool need_release = false;
                const auto* scalar_code =
                    this->optimized_build_scalar_layout_->Read(id, need_release);
                if (scalar_code == nullptr) {
                    throw VsagException(ErrorType::INTERNAL_ERROR,
                                        "failed to read temporary scalar RaBitQ build code");
                }
                try {
                    this->bottom_quantizer().PackScalarCodeToSplitCode(
                        scalar_code, one_bit_code.data, supplement_code.data);
                    if (not this->external_filter_code_storage_) {
                        this->x_bit_layout_->Write(id, one_bit_code.data);
                    }
                    this->supplement_layout_->Write(id, supplement_code.data);
                } catch (...) {
                    if (need_release) {
                        this->optimized_build_scalar_layout_->Release(scalar_code);
                    }
                    throw;
                }
                if (need_release) {
                    this->optimized_build_scalar_layout_->Release(scalar_code);
                }
            }
        };

        const auto& thread_pool = this->optimized_build_context_.thread_pool;
        const uint64_t worker_count = std::min<uint64_t>(
            this->optimized_build_context_.thread_count, static_cast<uint64_t>(this->total_count_));
        constexpr bool supports_parallel_finalize = not std::is_same_v<OneBitIOTmpl, MMapIO> and
                                                    not std::is_same_v<SupplementIOTmpl, MMapIO>;
        // MMapIO::WriteImpl updates its shared size_ even after Resize, so disjoint writes are not
        // thread-safe for that backend.
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
        this->optimized_build_context_ = {};
    }

    void
    AbortOptimizedBuild() noexcept override {
        this->optimized_build_active_ = false;
        this->optimized_build_scalar_layout_.reset();
        this->optimized_build_code_sums_.reset();
        this->optimized_build_record_size_ = 0;
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
            bool release1 = false;
            bool release2 = false;
            const auto* codes1 = this->optimized_build_scalar_layout_->Read(id1, release1);
            const auto* codes2 = this->optimized_build_scalar_layout_->Read(id2, release2);
            if (codes1 == nullptr or codes2 == nullptr) {
                if (release1) {
                    this->optimized_build_scalar_layout_->Release(codes1);
                }
                if (release2) {
                    this->optimized_build_scalar_layout_->Release(codes2);
                }
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to read temporary scalar RaBitQ build codes");
            }
            float distance = 0.0F;
            try {
                distance = this->bottom_quantizer().ComputeScalarCodesDistance(
                    codes1,
                    (*optimized_build_code_sums_)[id1],
                    codes2,
                    (*optimized_build_code_sums_)[id2]);
            } catch (...) {
                if (release1) {
                    this->optimized_build_scalar_layout_->Release(codes1);
                }
                if (release2) {
                    this->optimized_build_scalar_layout_->Release(codes2);
                }
                throw;
            }
            if (release1) {
                this->optimized_build_scalar_layout_->Release(codes1);
            }
            if (release2) {
                this->optimized_build_scalar_layout_->Release(codes2);
            }
            return distance;
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
        if (not this->external_filter_code_storage_) {
            this->x_bit_layout_->Resize(new_capacity);
        }
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
            this->optimized_build_scalar_layout_->Prefetch(id, this->optimized_build_record_size_);
            return;
        }
        this->prefetch_one_bit(id);
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
            bool need_release = false;
            const auto* scalar_code = this->optimized_build_scalar_layout_->Read(id, need_release);
            if (scalar_code == nullptr) {
                return false;
            }
            memset(codes, 0, this->code_size_);
            this->bottom_quantizer().PackScalarCode(scalar_code, codes);
            if (need_release) {
                this->optimized_build_scalar_layout_->Release(scalar_code);
            }
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
        this->serialize_supplement_layout(writer);
        this->x_bit_layout_->Serialize(writer);
        this->supplement_layout_->Serialize(writer);
        this->quantizer_->Serialize(writer);
    }

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override {
        FlattenInterface::Deserialize(reader);
        this->DeserializeSupplementIOType(reader);
        const auto supplement_layout = this->deserialize_supplement_layout(reader);
        this->validate_supplement_layout(supplement_layout);
        this->x_bit_layout_->Deserialize(reader);
        this->supplement_layout_->Deserialize(reader);
        this->quantizer_->Deserialize(reader);
        this->refresh_code_sizes();
        this->validate_supplement_layout(supplement_layout);
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
    bool external_filter_code_storage_{false};

private:
    static constexpr uint64_t SUPPLEMENT_STORAGE_MAGIC = 0x3150505553514252ULL;
    static constexpr uint32_t SUPPLEMENT_STORAGE_VERSION = 1;

    struct SupplementStorageLayout {
        uint64_t dim{0};
        uint32_t bits{0};
        uint64_t record_size{0};
    };

    void
    serialize_supplement_layout(StreamWriter& writer) const {
        StreamWriter::WriteObj(writer, SUPPLEMENT_STORAGE_MAGIC);
        StreamWriter::WriteObj(writer, SUPPLEMENT_STORAGE_VERSION);
        const uint64_t dim = static_cast<uint64_t>(this->bottom_quantizer().GetDim());
        StreamWriter::WriteObj(writer, dim);
        StreamWriter::WriteObj(writer, this->bottom_quantizer().ReorderBits());
        StreamWriter::WriteObj(writer, this->bottom_quantizer().GetSupplementCodeSize());
    }

    SupplementStorageLayout
    deserialize_supplement_layout(StreamReader& reader) const {
        uint64_t magic = 0;
        uint32_t version = 0;
        SupplementStorageLayout layout;
        StreamReader::ReadObj(reader, magic);
        StreamReader::ReadObj(reader, version);
        StreamReader::ReadObj(reader, layout.dim);
        StreamReader::ReadObj(reader, layout.bits);
        StreamReader::ReadObj(reader, layout.record_size);
        CHECK_ARGUMENT(magic == SUPPLEMENT_STORAGE_MAGIC,
                       "invalid packed RaBitQ supplement storage marker");
        CHECK_ARGUMENT(version == SUPPLEMENT_STORAGE_VERSION,
                       "unsupported packed RaBitQ supplement storage version");
        return layout;
    }

    void
    validate_supplement_layout(const SupplementStorageLayout& layout) const {
        CHECK_ARGUMENT(layout.dim == this->bottom_quantizer().GetDim(),
                       "packed RaBitQ supplement dimension mismatch");
        CHECK_ARGUMENT(layout.bits == this->bottom_quantizer().ReorderBits(),
                       "packed RaBitQ supplement bit width mismatch");
        CHECK_ARGUMENT(layout.record_size == this->bottom_quantizer().GetSupplementCodeSize(),
                       "packed RaBitQ supplement record size mismatch");
    }

    [[nodiscard]] static bool
    is_finite_float_bits(float value) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return (bits & 0x7F800000U) != 0x7F800000U;
    }

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
        return io_type == IO_TYPE_VALUE_MEMORY_IO or io_type == IO_TYPE_VALUE_BUFFER_IO or
               io_type == IO_TYPE_VALUE_MMAP_IO or io_type == IO_TYPE_VALUE_READER_IO or
               io_type == IO_TYPE_VALUE_ASYNC_IO or io_type == IO_TYPE_VALUE_URING_IO or
               io_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO;
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
            bool need_release = false;
            const auto* scalar_code =
                this->optimized_build_scalar_layout_->Read(idx[i], need_release);
            if (scalar_code == nullptr) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to read temporary scalar RaBitQ build code");
            }
            try {
                this->bottom_quantizer().ComputeDistWithScalarCode(
                    *comp, scalar_code, result_dists + i);
            } catch (...) {
                if (need_release) {
                    this->optimized_build_scalar_layout_->Release(scalar_code);
                }
                throw;
            }
            if (need_release) {
                this->optimized_build_scalar_layout_->Release(scalar_code);
            }
        }
    }

    void
    query_optimized_build_code_pairs(float* result_dists,
                                     const uint8_t* query_code,
                                     uint64_t query_sum,
                                     const InnerIdType* idx,
                                     InnerIdType id_count) const {
        for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; ++i) {
            this->optimized_build_scalar_layout_->Prefetch(idx[i], this->prefetch_depth_code_ * 64);
        }
        for (InnerIdType i = 0; i < id_count; ++i) {
            if (i + this->prefetch_stride_code_ < id_count) {
                this->optimized_build_scalar_layout_->Prefetch(idx[i + this->prefetch_stride_code_],
                                                               this->prefetch_depth_code_ * 64);
            }
            bool need_release = false;
            const auto* base_code =
                this->optimized_build_scalar_layout_->Read(idx[i], need_release);
            if (base_code == nullptr) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to read temporary scalar RaBitQ build code");
            }
            try {
                result_dists[i] = this->bottom_quantizer().ComputeScalarCodesDistance(
                    query_code, query_sum, base_code, (*this->optimized_build_code_sums_)[idx[i]]);
            } catch (...) {
                if (need_release) {
                    this->optimized_build_scalar_layout_->Release(base_code);
                }
                throw;
            }
            if (need_release) {
                this->optimized_build_scalar_layout_->Release(base_code);
            }
        }
    }

    void
    prefetch_one_bit(InnerIdType id) {
        this->x_bit_layout_->Prefetch(id, this->prefetch_depth_code_ * 64);
    }

    void
    prefetch_supplement(InnerIdType id) const {
        this->supplement_layout_->Prefetch(id, this->prefetch_depth_code_ * 64);
    }

    void
    prefetch_full_code(InnerIdType id) {
        this->prefetch_one_bit(id);
        this->prefetch_supplement(id);
    }

    const uint8_t*
    get_one_bit_code(InnerIdType id, bool& need_release) const {
        return this->x_bit_layout_->Read(id, need_release);
    }

    void
    release_one_bit_code(const uint8_t* code, bool need_release) const {
        if (need_release) {
            this->x_bit_layout_->Release(code);
        }
    }

    const uint8_t*
    get_supplement_code(InnerIdType id, bool& need_release) const {
        return this->supplement_layout_->Read(id, need_release);
    }

    void
    release_supplement_code(const uint8_t* code, bool need_release) const {
        if (need_release) {
            this->supplement_layout_->Release(code);
        }
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
            bool one_bit_need_release = false;
            const auto* one_bit_code = this->get_one_bit_code(idx[i], one_bit_need_release);
            const auto* supplement_code = supplement_codes.data + i * supplement_code_size_;
            const float hint =
                hint_dists == nullptr ? std::numeric_limits<float>::max() : hint_dists[i];
            try {
                this->compute_full_dist(
                    one_bit_code, supplement_code, computer, result_dists + i, ctx, hint);
            } catch (...) {
                this->release_one_bit_code(one_bit_code, one_bit_need_release);
                throw;
            }
            this->release_one_bit_code(one_bit_code, one_bit_need_release);
        }
    }

    void
    compute_full_dist_after_one_bit_failure(InnerIdType id,
                                            const uint8_t* one_bit_code,
                                            BottomComputer* computer,
                                            float* result_dist,
                                            float* lower_bound,
                                            QueryContext* ctx) const {
        bool supplement_need_release = false;
        const uint8_t* supplement_code = nullptr;
        try {
            supplement_code = this->get_supplement_code(id, supplement_need_release);
            this->compute_full_dist(one_bit_code, supplement_code, computer, result_dist, ctx);
            if (lower_bound != nullptr) {
                *lower_bound = std::numeric_limits<float>::max();
            }
        } catch (...) {
            this->release_supplement_code(supplement_code, supplement_need_release);
            throw;
        }
        this->release_supplement_code(supplement_code, supplement_need_release);
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
        bool one_bit_need_release = false;
        bool supplement_need_release = false;
        const auto* one_bit_code = this->get_one_bit_code(id, one_bit_need_release);
        const auto* supplement_code = this->get_supplement_code(id, supplement_need_release);
        try {
            this->compute_full_dist(
                one_bit_code, supplement_code, computer, result_dist, ctx, hint_dist);
        } catch (...) {
            this->release_one_bit_code(one_bit_code, one_bit_need_release);
            this->release_supplement_code(supplement_code, supplement_need_release);
            throw;
        }
        this->release_one_bit_code(one_bit_code, one_bit_need_release);
        this->release_supplement_code(supplement_code, supplement_need_release);
    }
};

}  // namespace vsag
