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
#include "impl/cluster/kmeans_cluster.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "inner_string_params.h"
#include "io/async_io/async_io_parameter.h"
#include "io/buffer_io/buffer_io_parameter.h"
#include "io/common/basic_io.h"
#include "io/common/io_parameter.h"
#include "io/memory_io/memory_io.h"
#include "io/memory_io/memory_io_parameter.h"
#include "io/mmap_io/mmap_io_parameter.h"
#include "quantization/rabitq_quantization/rabitq_quantizer.h"
#include "query_context.h"
#include "rabitq_fused_code_storage.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "type_helpers.h"
#include "utils/byte_buffer.h"
#include "utils/timer.h"

namespace vsag {

class MMapIO;

struct RaBitQFusedTraversalQuery {
    const uint8_t* query_planes{nullptr};
    const float* transformed_query{nullptr};
    const float* cluster_g_add{nullptr};
    const float* cluster_g_error{nullptr};
    uint64_t dim{0};
    uint64_t one_bit_metadata_offset{0};
    uint64_t supplement_metadata_offset{0};
    uint32_t cluster_count{0};
    uint32_t filter_bits{0};
    uint32_t supplement_bits{0};
    float query_delta{0.0F};
    float query_vl{0.0F};
    float query_sum{0.0F};
    float default_rabitq_error_rate{0.0F};
    bool affine{false};
    bool filter_inner_product_is_exact{false};
};

class RaBitQSplitDataCellInterface {
public:
    virtual ~RaBitQSplitDataCellInterface() = default;

    [[nodiscard]] virtual uint64_t
    OneBitCodeSize() const = 0;

    [[nodiscard]] virtual uint64_t
    SupplementCodeSize() const = 0;

    [[nodiscard]] virtual uint32_t
    FusedFilterBits() const = 0;

    [[nodiscard]] virtual uint32_t
    FusedSupplementBits() const = 0;

    [[nodiscard]] virtual bool
    UsesLegacyHnswFusedCodec() const = 0;

    virtual void
    AttachFusedCodeStorage(RaBitQFusedCodeStorageInterface* storage) = 0;

    [[nodiscard]] virtual bool
    UsesExternalFusedCodeStorage() const = 0;

    virtual bool
    CopySplitCodes(InnerIdType id, uint8_t* one_bit_code, uint8_t* supplement_code) const = 0;

    virtual bool
    DecodeFusedById(InnerIdType id, float* data) const = 0;

    virtual bool
    ComputeOneBitWithFilterIP(const ComputerInterfacePtr& computer,
                              const uint8_t* one_bit_code,
                              float* distance,
                              float* lower_bound,
                              float* filter_inner_product,
                              QueryContext* ctx) const = 0;

    virtual bool
    ComputeFullWithFilterIP(const ComputerInterfacePtr& computer,
                            const uint8_t* one_bit_code,
                            const uint8_t* supplement_code,
                            float filter_inner_product,
                            float* distance,
                            QueryContext* ctx) const = 0;

    virtual bool
    ComputeFull(const ComputerInterfacePtr& computer,
                const uint8_t* one_bit_code,
                const uint8_t* supplement_code,
                float* distance,
                QueryContext* ctx) const = 0;

    virtual void
    TrainFusedCodec(const float* data, uint64_t count, uint32_t cluster_count) = 0;

    virtual bool
    EncodeFused(const float* data,
                uint8_t* one_bit_code,
                uint8_t* supplement_code,
                uint32_t* cluster_id) const = 0;

    virtual ComputerInterfacePtr
    FactoryFusedComputer(const void* query) const = 0;

    virtual bool
    GetFusedTraversalQuery(const ComputerInterfacePtr& computer,
                           RaBitQFusedTraversalQuery* query) const = 0;

    virtual bool
    ComputeFusedOneBitWithFilterIP(const ComputerInterfacePtr& computer,
                                   uint32_t cluster_id,
                                   const uint8_t* one_bit_code,
                                   const uint8_t* supplement_code,
                                   float* distance,
                                   float* lower_bound,
                                   float* filter_inner_product,
                                   QueryContext* ctx) const = 0;

    virtual bool
    ComputeFusedFullWithFilterIP(const ComputerInterfacePtr& computer,
                                 uint32_t cluster_id,
                                 const uint8_t* one_bit_code,
                                 const uint8_t* supplement_code,
                                 float filter_inner_product,
                                 float* distance,
                                 QueryContext* ctx) const = 0;

    virtual bool
    ComputeFusedFull(const ComputerInterfacePtr& computer,
                     uint32_t cluster_id,
                     const uint8_t* one_bit_code,
                     const uint8_t* supplement_code,
                     float* distance,
                     QueryContext* ctx) const = 0;

    [[nodiscard]] virtual std::string
    ExportFusedCodec() const = 0;

    virtual void
    ImportFusedCodec(const std::string& serialized) = 0;
};

template <typename IOTmpl>
class RaBitQSplitCodeStorage {
public:
    static constexpr bool InMemory = IOTmpl::InMemory;

    RaBitQSplitCodeStorage(const IOParamPtr& io_param, const IndexCommonParam& common_param)
        : io_(std::make_shared<IOTmpl>(io_param, common_param)) {
    }

    void
    SetCodeSize(uint64_t code_size) {
        code_size_ = code_size;
    }

    [[nodiscard]] uint64_t
    GetCodeSize() const {
        return code_size_;
    }

    void
    Resize(uint64_t new_capacity) {
        io_->Resize(new_capacity * code_size_);
    }

    void
    Shrink(uint64_t new_capacity) {
        io_->Shrink(new_capacity * code_size_);
    }

    void
    Write(const uint8_t* code, InnerIdType id) {
        io_->Write(code, code_size_, static_cast<uint64_t>(id) * code_size_);
    }

    bool
    Read(InnerIdType id, uint8_t* dst) const {
        return io_->Read(code_size_, static_cast<uint64_t>(id) * code_size_, dst);
    }

    [[nodiscard]] const uint8_t*
    Read(InnerIdType id, bool& need_release) const {
        return io_->Read(code_size_, static_cast<uint64_t>(id) * code_size_, need_release);
    }

    void
    Release(const uint8_t* code) const {
        if (code != nullptr) {
            io_->Release(code);
        }
    }

    void
    Prefetch(InnerIdType id, uint64_t bytes) const {
        io_->Prefetch(static_cast<uint64_t>(id) * code_size_,
                      std::min<uint64_t>(bytes, code_size_));
    }

    void
    MultiRead(uint8_t* dst, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        io_->MultiRead(dst, sizes, offsets, count);
    }

    void
    InitIO(const IOParamPtr& io_param) {
        io_->InitIO(io_param);
    }

    void
    Serialize(StreamWriter& writer) {
        io_->Serialize(writer);
    }

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) {
        io_->Deserialize(reader);
    }

    void
    SkipSerialized(StreamReader& reader) {
        uint64_t size = 0;
        StreamReader::ReadObj(reader, size);
        const uint64_t cursor = reader.GetCursor();
        CHECK_ARGUMENT(cursor <= reader.Length() and size <= reader.Length() - cursor,
                       "serialized RaBitQ split code payload exceeds its stream boundary");
        reader.Seek(cursor + size);
    }

    [[nodiscard]] uint64_t
    GetMemoryUsage() const {
        if constexpr (IOTmpl::InMemory) {
            return io_->GetMemoryUsage();
        }
        return 0;
    }

private:
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};
    uint64_t code_size_{0};
};

template <MetricType metric, typename OneBitIOTmpl, typename SupplementIOTmpl = OneBitIOTmpl>
class RaBitQSplitDataCell : public FlattenInterface,
                            public FlattenOptimizedBuildInterface,
                            public RaBitQSplitDataCellInterface {
public:
    class OptimizedBuildComputer final : public ComputerInterface {
    public:
        OptimizedBuildComputer(uint64_t record_size, Allocator* allocator)
            : scalar_code_(record_size, allocator) {
        }

        ByteBuffer scalar_code_;
        uint64_t code_sum_{0};
    };

    class FusedComputer final : public ComputerInterface {
    public:
        explicit FusedComputer(Allocator* allocator)
            : transformed_query_(allocator),
              hnsw_query_planes_(allocator),
              hnsw_g_add_(allocator),
              hnsw_g_error_(allocator) {
        }

        Vector<float> transformed_query_;
        Vector<uint8_t> hnsw_query_planes_;
        float hnsw_query_delta_{0.0F};
        float hnsw_query_vl_{0.0F};
        float hnsw_query_sum_{0.0F};
        Vector<float> hnsw_g_add_;
        Vector<float> hnsw_g_error_;
        float query_raw_norm_{0.0F};
        typename RaBitQuantizer<metric>::norm_type mrq_norm_sqr_{0.0F};
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
        this->quantization_param_ =
            std::dynamic_pointer_cast<RaBitQuantizerParameter>(quantization_param);
        this->quantizer_ =
            std::make_shared<RaBitQuantizer<metric>>(quantization_param, common_param);
        if (not this->quantizer_->SupportSplitCodeStorage()) {
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
        this->x_bit_cell_ =
            std::make_shared<RaBitQSplitCodeStorage<OneBitIOTmpl>>(one_bit_io_param, common_param);
        this->supplement_cell_ =
            std::make_shared<RaBitQSplitCodeStorage<SupplementIOTmpl>>(supp_io_param, common_param);
        this->refresh_code_sizes();
    }

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) override {
        if (fused_code_storage_ != nullptr) {
            this->query_fused_full(result_dists, computer, idx, id_count, ctx);
            return;
        }
        if (this->optimized_build_active_) {
            this->query_optimized_build_codes(result_dists, computer, idx, id_count);
            return;
        }
        auto* comp = static_cast<Computer<RaBitQuantizer<metric>>*>(computer.get());
        if constexpr (not OneBitIOTmpl::InMemory or not SupplementIOTmpl::InMemory) {
            if (id_count > 1) {
                if constexpr (OneBitIOTmpl::InMemory and not SupplementIOTmpl::InMemory) {
                    this->query_full_dist_by_supplement_multiread(
                        result_dists, comp, idx, id_count, ctx);
                    return;
                }
                this->query_full_dist_by_multiread(result_dists, comp, idx, id_count, ctx);
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
    }

    void
    QueryWithDistanceHint(float* result_dists,
                          const float* hint_dists,
                          const ComputerInterfacePtr& computer,
                          const InnerIdType* idx,
                          InnerIdType id_count,
                          QueryContext* ctx = nullptr) override {
        if (fused_code_storage_ != nullptr) {
            this->query_fused_full(result_dists, computer, idx, id_count, ctx);
            return;
        }
        if (this->optimized_build_active_) {
            this->query_optimized_build_codes(result_dists, computer, idx, id_count);
            return;
        }
        auto* comp = static_cast<Computer<RaBitQuantizer<metric>>*>(computer.get());
        if constexpr (not OneBitIOTmpl::InMemory or not SupplementIOTmpl::InMemory) {
            if (id_count > 1) {
                if constexpr (OneBitIOTmpl::InMemory and not SupplementIOTmpl::InMemory) {
                    this->query_full_dist_by_supplement_multiread(
                        result_dists, comp, idx, id_count, ctx, hint_dists);
                    return;
                }
                this->query_full_dist_by_multiread(
                    result_dists, comp, idx, id_count, ctx, hint_dists);
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
    }

    void
    QueryWithDistanceFilter(float* result_dists,
                            const ComputerInterfacePtr& computer,
                            const InnerIdType* idx,
                            InnerIdType id_count,
                            float threshold,
                            QueryContext* ctx = nullptr) override {
        if (fused_code_storage_ != nullptr) {
            this->query_fused_with_distance_filter(
                result_dists, computer, idx, id_count, threshold, ctx);
            return;
        }
        if (this->optimized_build_active_) {
            this->query_optimized_build_codes(result_dists, computer, idx, id_count);
            return;
        }
        auto* comp = static_cast<Computer<RaBitQuantizer<metric>>*>(computer.get());
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
                computed = this->quantizer_->ComputeDistWithOneBitLowerBound(
                    *comp,
                    one_bit_code,
                    &one_bit_dist,
                    &lower_bound,
                    this->query_rabitq_error_rate(ctx));
            } catch (...) {
                this->release_one_bit_code(one_bit_code, one_bit_need_release);
                throw;
            }

            if (computed and IsFiniteRaBitQValue(lower_bound) and lower_bound >= threshold) {
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
    }

    void
    QueryWithDistanceLowerBoundAndFilterIP(float* result_dists,
                                           float* lower_bounds,
                                           float* filter_inner_products,
                                           const ComputerInterfacePtr& computer,
                                           const InnerIdType* idx,
                                           InnerIdType id_count,
                                           QueryContext* ctx = nullptr) override {
        if (fused_code_storage_ != nullptr) {
            this->query_fused_lower_bound(
                result_dists, lower_bounds, filter_inner_products, computer, idx, id_count, ctx);
            return;
        }
        auto* comp = static_cast<Computer<RaBitQuantizer<metric>>*>(computer.get());
        this->add_filter_count(ctx, id_count);
        for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; ++i) {
            this->prefetch_one_bit(idx[i]);
        }
        for (InnerIdType i = 0; i < id_count; ++i) {
            if (i + this->prefetch_stride_code_ < id_count) {
                this->prefetch_one_bit(idx[i + this->prefetch_stride_code_]);
            }
            bool need_release = false;
            const auto* one_bit_code = this->get_one_bit_code(idx[i], need_release);
            bool computed = false;
            try {
                computed = this->quantizer_->ComputeDistWithOneBitLowerBoundAndFilterIP(
                    *comp,
                    one_bit_code,
                    result_dists + i,
                    lower_bounds == nullptr ? nullptr : lower_bounds + i,
                    filter_inner_products == nullptr ? nullptr : filter_inner_products + i,
                    this->query_rabitq_error_rate(ctx));
            } catch (...) {
                this->release_one_bit_code(one_bit_code, need_release);
                throw;
            }
            if (not computed) {
                if (filter_inner_products != nullptr) {
                    filter_inner_products[i] = std::numeric_limits<float>::quiet_NaN();
                }
                this->compute_full_dist_after_one_bit_failure(
                    idx[i],
                    one_bit_code,
                    comp,
                    result_dists + i,
                    lower_bounds == nullptr ? nullptr : lower_bounds + i,
                    ctx);
            }
            this->release_one_bit_code(one_bit_code, need_release);
        }
    }

    void
    QueryWithDistanceLowerBound(float* result_dists,
                                float* lower_bounds,
                                const ComputerInterfacePtr& computer,
                                const InnerIdType* idx,
                                InnerIdType id_count,
                                QueryContext* ctx = nullptr) override {
        if (fused_code_storage_ != nullptr) {
            this->query_fused_lower_bound(
                result_dists, lower_bounds, nullptr, computer, idx, id_count, ctx);
            return;
        }
        if (this->optimized_build_active_) {
            this->query_optimized_build_codes(result_dists, computer, idx, id_count);
            if (lower_bounds != nullptr) {
                std::fill(lower_bounds, lower_bounds + id_count, std::numeric_limits<float>::max());
            }
            return;
        }
        auto* comp = static_cast<Computer<RaBitQuantizer<metric>>*>(computer.get());
        this->add_filter_count(ctx, id_count);
        if constexpr (not OneBitIOTmpl::InMemory) {
            if (id_count > 1) {
                this->query_one_bit_lower_bound_by_multiread(
                    result_dists, lower_bounds, comp, idx, id_count, ctx);
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
                this->quantizer_->ComputeDistsWithOneBitLowerBoundBatch4(
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
                computed = this->quantizer_->ComputeDistWithOneBitLowerBound(
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
    }

    void
    QueryWithFilterIPHint(float* result_dists,
                          const float* filter_inner_products,
                          const ComputerInterfacePtr& computer,
                          const InnerIdType* idx,
                          InnerIdType id_count,
                          QueryContext* ctx = nullptr) override {
        if (fused_code_storage_ != nullptr) {
            this->query_fused_full_with_filter_ip(
                result_dists, filter_inner_products, computer, idx, id_count, ctx);
            return;
        }
        auto* comp = static_cast<Computer<RaBitQuantizer<metric>>*>(computer.get());
        for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; ++i) {
            this->prefetch_full_code(idx[i]);
        }
        for (InnerIdType i = 0; i < id_count; ++i) {
            if (i + this->prefetch_stride_code_ < id_count) {
                this->prefetch_full_code(idx[i + this->prefetch_stride_code_]);
            }
            this->compute_full_dist_with_filter_ip(idx[i],
                                                   comp,
                                                   result_dists + i,
                                                   ctx,
                                                   filter_inner_products == nullptr
                                                       ? std::numeric_limits<float>::quiet_NaN()
                                                       : filter_inner_products[i]);
        }
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        if (fused_code_storage_ != nullptr) {
            return this->FactoryFusedComputer(query);
        }
        auto computer = this->quantizer_->FactoryComputer();
        computer->SetQuery(static_cast<const float*>(query));
        return computer;
    }

    ComputerInterfacePtr
    FactoryComputerForBuild(const void* query, InnerIdType id) override {
        if (this->optimized_build_active_) {
            auto computer = std::make_shared<OptimizedBuildComputer>(
                this->optimized_build_record_size_, this->allocator_);
            if (not this->optimized_build_scalar_codes_->Read(id, computer->scalar_code_.data)) {
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
        if (this->fused_code_storage_ != nullptr) {
            return false;
        }
        if (this->optimized_build_active_ or not this->quantizer_->SupportScalarCodeBuild()) {
            return false;
        }
        auto io_param = std::make_shared<MemoryIOParameter>();
        auto build_codes =
            std::make_shared<RaBitQSplitCodeStorage<MemoryIO>>(io_param, this->common_param_);
        build_codes->SetCodeSize(this->quantizer_->GetScalarCodeSize());
        auto code_sums = std::make_unique<Vector<uint64_t>>(this->allocator_);
        if (this->max_capacity_ > 0) {
            build_codes->Resize(this->max_capacity_);
            code_sums->resize(this->max_capacity_, 0);
        }
        this->optimized_build_scalar_codes_ = build_codes;
        this->optimized_build_code_sums_ = std::move(code_sums);
        this->optimized_build_record_size_ = this->quantizer_->GetScalarCodeSize();
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
        this->x_bit_cell_->Resize(final_capacity);
        this->supplement_cell_->Resize(final_capacity);
        this->max_capacity_ = final_capacity;

        auto finalize_range = [this](InnerIdType begin, InnerIdType end) {
            ByteBuffer one_bit_code(this->one_bit_code_size_, allocator_);
            ByteBuffer supplement_code(this->supplement_code_size_, allocator_);
            for (InnerIdType id = begin; id < end; ++id) {
                bool need_release = false;
                const auto* scalar_code =
                    this->optimized_build_scalar_codes_->Read(id, need_release);
                if (scalar_code == nullptr) {
                    throw VsagException(ErrorType::INTERNAL_ERROR,
                                        "failed to read temporary scalar RaBitQ build code");
                }
                try {
                    this->quantizer_->PackScalarCodeToSplitCode(
                        scalar_code, one_bit_code.data, supplement_code.data);
                    this->x_bit_cell_->Write(one_bit_code.data, id);
                    this->supplement_cell_->Write(supplement_code.data, id);
                } catch (...) {
                    if (need_release) {
                        this->optimized_build_scalar_codes_->Release(scalar_code);
                    }
                    throw;
                }
                if (need_release) {
                    this->optimized_build_scalar_codes_->Release(scalar_code);
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
        this->optimized_build_scalar_codes_.reset();
        this->optimized_build_code_sums_.reset();
        this->optimized_build_record_size_ = 0;
        this->optimized_build_context_ = {};
    }

    void
    AbortOptimizedBuild() noexcept override {
        this->optimized_build_active_ = false;
        this->optimized_build_scalar_codes_.reset();
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
        if (this->fused_code_storage_ != nullptr) {
            return;
        }
        this->write_encoded_vector(static_cast<const float*>(vector), idx);
    }

    bool
    UpdateVector(const void* vector,
                 InnerIdType idx = std::numeric_limits<InnerIdType>::max()) override {
        if (idx >= this->total_count_) {
            return false;
        }
        if (this->fused_code_storage_ != nullptr) {
            return true;
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
        if (this->fused_code_storage_ != nullptr) {
            throw VsagException(
                ErrorType::UNSUPPORTED_INDEX_OPERATION,
                "pairwise distance is unavailable for cluster-residual fused RaBitQ codes");
        }
        if (this->optimized_build_active_) {
            bool release1 = false;
            bool release2 = false;
            const auto* codes1 = this->optimized_build_scalar_codes_->Read(id1, release1);
            const auto* codes2 = this->optimized_build_scalar_codes_->Read(id2, release2);
            if (codes1 == nullptr or codes2 == nullptr) {
                if (release1) {
                    this->optimized_build_scalar_codes_->Release(codes1);
                }
                if (release2) {
                    this->optimized_build_scalar_codes_->Release(codes2);
                }
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to read temporary scalar RaBitQ build codes");
            }
            float distance = 0.0F;
            try {
                distance = this->quantizer_->ComputeScalarCodesDistance(
                    codes1,
                    (*optimized_build_code_sums_)[id1],
                    codes2,
                    (*optimized_build_code_sums_)[id2]);
            } catch (...) {
                if (release1) {
                    this->optimized_build_scalar_codes_->Release(codes1);
                }
                if (release2) {
                    this->optimized_build_scalar_codes_->Release(codes2);
                }
                throw;
            }
            if (release1) {
                this->optimized_build_scalar_codes_->Release(codes1);
            }
            if (release2) {
                this->optimized_build_scalar_codes_->Release(codes2);
            }
            return distance;
        }
        ByteBuffer codes1(this->code_size_, allocator_);
        ByteBuffer codes2(this->code_size_, allocator_);
        this->GetCodesById(id1, codes1.data);
        this->GetCodesById(id2, codes2.data);
        return this->quantizer_->Compute(codes1.data, codes2.data);
    }

    void
    Resize(InnerIdType new_capacity) override {
        if (new_capacity <= this->max_capacity_) {
            return;
        }
        if (this->fused_code_storage_ != nullptr) {
            this->max_capacity_ = new_capacity;
            return;
        }
        this->x_bit_cell_->Resize(new_capacity);
        this->supplement_cell_->Resize(new_capacity);
        if (this->optimized_build_active_) {
            this->optimized_build_scalar_codes_->Resize(new_capacity);
            this->optimized_build_code_sums_->resize(new_capacity, 0);
        }
        this->max_capacity_ = new_capacity;
    }

    void
    Prefetch(InnerIdType id) override {
        if (this->fused_code_storage_ != nullptr) {
            this->fused_code_storage_->PrefetchFusedCodes(id, false);
            return;
        }
        if (this->optimized_build_active_) {
            this->optimized_build_scalar_codes_->Prefetch(id, this->optimized_build_record_size_);
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
        auto ptr =
            std::dynamic_pointer_cast<RaBitQSplitDataCell<metric, OneBitIOTmpl, SupplementIOTmpl>>(
                other);
        if (ptr == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Export model's rabitq split datacell failed");
        }
        ptr->quantizer_->Deserialize(reader);
        ptr->refresh_code_sizes();
    }

    void
    InitIO(const IOParamPtr& io_param) override {
        if (this->fused_code_storage_ != nullptr) {
            CHECK_ARGUMENT(OneBitIOTmpl::InMemory and SupplementIOTmpl::InMemory,
                           "fused RaBitQ code storage requires memory IO");
            return;
        }
        const bool shares_io_param = this->supplement_io_type_.empty();
        this->x_bit_cell_->InitIO(SuffixIOParam(io_param, "_onebit", shares_io_param));
        // In hybrid mode (one-bit and supplement use different IO backends)
        // the caller-facing `io_param` is the one-bit IO parameter type and
        // cannot be passed directly to `supplement_cell_`. Rebuild a fresh
        // IOParameter of the recorded supplement type so the underlying IO
        // implementation receives the correct parameter subclass.
        this->supplement_cell_->InitIO(RebuildSupplementIOParam(io_param));
    }

    void
    InitIO(const IOParamPtr& one_bit_io_param, const IOParamPtr& supplement_io_param) {
        if (this->fused_code_storage_ != nullptr) {
            CHECK_ARGUMENT(OneBitIOTmpl::InMemory and SupplementIOTmpl::InMemory,
                           "fused RaBitQ code storage requires memory IO");
            return;
        }
        const bool shares_io_param = supplement_io_param == nullptr;
        this->x_bit_cell_->InitIO(SuffixIOParam(one_bit_io_param, "_onebit", shares_io_param));
        if (supplement_io_param != nullptr) {
            // Refresh the recorded supplement type so subsequent
            // single-parameter InitIO calls (e.g. from Deserialize) can
            // reconstruct the same IO subtype.
            this->supplement_io_type_ = supplement_io_param->GetTypeName();
            this->supplement_cell_->InitIO(supplement_io_param);
        } else {
            this->supplement_cell_->InitIO(RebuildSupplementIOParam(one_bit_io_param));
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

    [[nodiscard]] uint64_t
    OneBitCodeSize() const override {
        return one_bit_code_size_;
    }

    [[nodiscard]] uint64_t
    SupplementCodeSize() const override {
        return supplement_code_size_;
    }

    [[nodiscard]] uint32_t
    FusedFilterBits() const override {
        return quantizer_->FilterBits();
    }

    [[nodiscard]] uint32_t
    FusedSupplementBits() const override {
        return quantizer_->ReorderBits();
    }

    [[nodiscard]] bool
    UsesLegacyHnswFusedCodec() const override {
        return IsLegacyHnswFusedCodec();
    }

    void
    AttachFusedCodeStorage(RaBitQFusedCodeStorageInterface* storage) override {
        CHECK_ARGUMENT(storage != nullptr, "fused RaBitQ code storage must not be null");
        CHECK_ARGUMENT(not optimized_build_active_,
                       "cannot attach fused RaBitQ storage during optimized build");
        CHECK_ARGUMENT(storage->FusedOneBitCodeSize() == one_bit_code_size_ and
                           storage->FusedSupplementCodeSize() == supplement_code_size_,
                       "fused RaBitQ code sizes do not match the split model");
        fused_code_storage_ = storage;
        x_bit_cell_->Shrink(0);
        supplement_cell_->Shrink(0);
    }

    [[nodiscard]] bool
    UsesExternalFusedCodeStorage() const override {
        return fused_code_storage_ != nullptr;
    }

    bool
    CopySplitCodes(InnerIdType id, uint8_t* one_bit_code, uint8_t* supplement_code) const override {
        if (this->optimized_build_active_ or one_bit_code == nullptr or
            supplement_code == nullptr) {
            return false;
        }
        if (fused_code_storage_ != nullptr) {
            RaBitQFusedCodeView view;
            if (not fused_code_storage_->GetFusedCodeView(id, view)) {
                return false;
            }
            std::memcpy(one_bit_code, view.one_bit_code, one_bit_code_size_);
            std::memcpy(supplement_code, view.supplement_code, supplement_code_size_);
            return true;
        }
        return this->x_bit_cell_->Read(id, one_bit_code) and
               this->supplement_cell_->Read(id, supplement_code);
    }

    bool
    DecodeFusedById(InnerIdType id, float* data) const override {
        if (data == nullptr or fused_code_storage_ == nullptr or id >= this->TotalCount() or
            fused_quantizers_.empty()) {
            return false;
        }
        RaBitQFusedCodeView view;
        if (not fused_code_storage_->GetFusedCodeView(id, view) or
            view.cluster_id >= fused_quantizers_.size()) {
            return false;
        }
        return fused_quantizers_[view.cluster_id]->DecodeFusedSplitCode(
            view.one_bit_code, view.supplement_code, IsLegacyHnswFusedCodec(), data);
    }

    bool
    ComputeOneBitWithFilterIP(const ComputerInterfacePtr& computer,
                              const uint8_t* one_bit_code,
                              float* distance,
                              float* lower_bound,
                              float* filter_inner_product,
                              QueryContext* ctx) const override {
        auto* comp = static_cast<Computer<RaBitQuantizer<metric>>*>(computer.get());
        this->add_filter_count(ctx, 1);
        const bool computed = this->quantizer_->ComputeDistWithOneBitLowerBoundAndFilterIP(
            *comp,
            one_bit_code,
            distance,
            lower_bound,
            filter_inner_product,
            this->query_rabitq_error_rate(ctx));
        if (not computed) {
            this->add_filter_fallback_full_count(ctx, 1);
        }
        return computed;
    }

    bool
    ComputeFullWithFilterIP(const ComputerInterfacePtr& computer,
                            const uint8_t* one_bit_code,
                            const uint8_t* supplement_code,
                            float filter_inner_product,
                            float* distance,
                            QueryContext* ctx) const override {
        if (this->quantizer_->FilterBits() < 2) {
            return false;
        }
        auto* comp = static_cast<Computer<RaBitQuantizer<metric>>*>(computer.get());
        this->add_full_count(ctx, 1);
        const bool computed = this->quantizer_->ComputeDistWithSplitCodeAndFilterIP(
            *comp, one_bit_code, supplement_code, filter_inner_product, distance);
        if (computed) {
            this->add_reorder_hint_full_count(ctx, 1);
        } else {
            this->add_reorder_fallback_full_count(ctx, 1);
        }
        return computed;
    }

    bool
    ComputeFull(const ComputerInterfacePtr& computer,
                const uint8_t* one_bit_code,
                const uint8_t* supplement_code,
                float* distance,
                QueryContext* ctx) const override {
        auto* comp = static_cast<Computer<RaBitQuantizer<metric>>*>(computer.get());
        this->add_full_count(ctx, 1);
        return this->quantizer_->ComputeDistWithSplitCode(
            *comp, one_bit_code, supplement_code, distance);
    }

    void
    TrainFusedCodec(const float* data, uint64_t count, uint32_t cluster_count) override {
        CHECK_ARGUMENT(data != nullptr and count > 0,
                       "fused RaBitQ training data must not be empty");
        CHECK_ARGUMENT(cluster_count == 16, "fused RaBitQ requires exactly 16 clusters");
        CHECK_ARGUMENT(this->quantization_param_ != nullptr,
                       "fused RaBitQ quantizer parameter is unavailable");

        KMeansCluster kmeans(
            static_cast<int32_t>(common_param_.dim_), allocator_, common_param_.thread_pool_);
        const auto trained_cluster_count =
            static_cast<uint32_t>(std::min<uint64_t>(cluster_count, count));
        kmeans.Run(trained_cluster_count,
                   data,
                   count,
                   25,
                   nullptr,
                   false,
                   1e-6F,
                   KMeansInitMethod::KMEANS_PLUS_PLUS,
                   0x52425131U,
                   true);
        fused_centroids_.resize(static_cast<uint64_t>(cluster_count) * common_param_.dim_);
        for (uint32_t cluster_id = 0; cluster_id < cluster_count; ++cluster_id) {
            const auto source_cluster = cluster_id % trained_cluster_count;
            std::copy_n(
                kmeans.k_centroids_ + static_cast<uint64_t>(source_cluster) * common_param_.dim_,
                common_param_.dim_,
                fused_centroids_.data() + static_cast<uint64_t>(cluster_id) * common_param_.dim_);
        }

        std::stringstream model_stream;
        IOStreamWriter model_writer(model_stream);
        quantizer_->Serialize(model_writer);
        const auto serialized_model = model_stream.str();

        fused_quantizers_.clear();
        fused_quantizers_.reserve(cluster_count);
        for (uint32_t cluster_id = 0; cluster_id < cluster_count; ++cluster_id) {
            auto quantizer =
                std::make_shared<RaBitQuantizer<metric>>(quantization_param_, common_param_);
            std::stringstream input(serialized_model);
            IOStreamReader model_reader(input);
            quantizer->Deserialize(model_reader);
            quantizer->SetCentroid(fused_centroids_.data() +
                                   static_cast<uint64_t>(cluster_id) * common_param_.dim_);
            fused_quantizers_.push_back(std::move(quantizer));
        }
    }

    bool
    EncodeFused(const float* data,
                uint8_t* one_bit_code,
                uint8_t* supplement_code,
                uint32_t* cluster_id) const override {
        if (data == nullptr or one_bit_code == nullptr or supplement_code == nullptr or
            cluster_id == nullptr or fused_quantizers_.empty()) {
            return false;
        }
        *cluster_id = NearestFusedCluster(data);
        ByteBuffer full_code(code_size_, allocator_);
        auto& quantizer = fused_quantizers_[*cluster_id];
        if (not quantizer->EncodeOne(data, full_code.data)) {
            return false;
        }
        quantizer->SplitCode(full_code.data, one_bit_code, supplement_code);
        if (IsLegacyHnswFusedCodec()) {
            quantizer->EncodeHnswOneBitMetadata(data, one_bit_code);
            quantizer->EncodeHnswSupplement(data, supplement_code);
        } else if (not quantizer->EncodeFusedAffineMetadata(data, one_bit_code, supplement_code)) {
            return false;
        }
        return true;
    }

    ComputerInterfacePtr
    FactoryFusedComputer(const void* query) const override {
        auto result = std::make_shared<FusedComputer>(allocator_);
        if (fused_quantizers_.empty()) {
            return result;
        }
        result->transformed_query_.resize(fused_quantizers_.front()->GetDim());
        fused_quantizers_.front()->TransformFusedQuery(static_cast<const float*>(query),
                                                       result->transformed_query_,
                                                       result->query_raw_norm_,
                                                       result->mrq_norm_sqr_);
        if (quantizer_->FilterBits() == 1) {
            fused_quantizers_.front()->PrepareHnswFourBitQuery(result->transformed_query_.data(),
                                                               result->hnsw_query_planes_,
                                                               result->hnsw_query_delta_,
                                                               result->hnsw_query_vl_,
                                                               result->hnsw_query_sum_);
        } else {
            double query_sum = 0.0;
            for (const float value : result->transformed_query_) {
                query_sum += static_cast<double>(value);
            }
            result->hnsw_query_sum_ = static_cast<float>(query_sum);
        }
        result->hnsw_g_add_.resize(fused_quantizers_.size());
        result->hnsw_g_error_.resize(fused_quantizers_.size());
        for (uint64_t cluster_id = 0; cluster_id < fused_quantizers_.size(); ++cluster_id) {
            fused_quantizers_[cluster_id]->ComputeHnswCentroidTerms(
                result->transformed_query_.data(),
                result->hnsw_g_add_[cluster_id],
                result->hnsw_g_error_[cluster_id]);
        }
        return result;
    }

    bool
    GetFusedTraversalQuery(const ComputerInterfacePtr& computer,
                           RaBitQFusedTraversalQuery* query) const override {
        if (query == nullptr) {
            return false;
        }
        *query = {};
        auto* fused_computer = static_cast<FusedComputer*>(computer.get());
        if (fused_computer == nullptr or fused_quantizers_.empty()) {
            return false;
        }
        const auto filter_bits = quantizer_->FilterBits();
        query->query_planes =
            filter_bits == 1 ? fused_computer->hnsw_query_planes_.data() : nullptr;
        query->transformed_query = fused_computer->transformed_query_.data();
        query->cluster_g_add = fused_computer->hnsw_g_add_.data();
        query->cluster_g_error = fused_computer->hnsw_g_error_.data();
        query->dim = common_param_.dim_;
        query->one_bit_metadata_offset = IsLegacyHnswFusedCodec()
                                             ? quantizer_->PlaneBytes()
                                             : quantizer_->OneBitRecordNormOffset();
        query->supplement_metadata_offset = quantizer_->SupplementMetaOffset();
        query->cluster_count = static_cast<uint32_t>(fused_quantizers_.size());
        query->filter_bits = filter_bits;
        query->supplement_bits = quantizer_->ReorderBits();
        query->query_delta = fused_computer->hnsw_query_delta_;
        query->query_vl = fused_computer->hnsw_query_vl_;
        query->query_sum = fused_computer->hnsw_query_sum_;
        query->default_rabitq_error_rate = quantizer_->DefaultRaBitQErrorRate();
        query->affine = not IsLegacyHnswFusedCodec();
        query->filter_inner_product_is_exact = query->affine and filter_bits >= 2;
        return query->transformed_query != nullptr and query->cluster_g_add != nullptr and
               query->cluster_g_error != nullptr and
               (filter_bits != 1 or query->query_planes != nullptr);
    }

    bool
    ComputeFusedOneBitWithFilterIP(const ComputerInterfacePtr& computer,
                                   uint32_t cluster_id,
                                   const uint8_t* one_bit_code,
                                   const uint8_t* supplement_code,
                                   float* distance,
                                   float* lower_bound,
                                   float* filter_inner_product,
                                   QueryContext* ctx) const override {
        auto* fused_computer = static_cast<FusedComputer*>(computer.get());
        if (fused_computer == nullptr or cluster_id >= fused_quantizers_.size()) {
            return false;
        }
        this->add_filter_count(ctx, 1);
        if (filter_inner_product != nullptr) {
            *filter_inner_product = std::numeric_limits<float>::quiet_NaN();
        }
        float local_filter_inner_product = std::numeric_limits<float>::quiet_NaN();
        bool computed = false;
        if (IsLegacyHnswFusedCodec()) {
            computed = fused_quantizers_[cluster_id]->ComputeHnswOneBit(
                fused_computer->hnsw_query_planes_.data(),
                fused_computer->hnsw_query_delta_,
                fused_computer->hnsw_query_vl_,
                fused_computer->hnsw_query_sum_,
                fused_computer->hnsw_g_add_[cluster_id],
                fused_computer->hnsw_g_error_[cluster_id],
                one_bit_code,
                supplement_code,
                distance,
                lower_bound,
                &local_filter_inner_product,
                this->query_rabitq_error_rate(ctx));
        } else {
            RaBitQFusedIPPrecision precision = RaBitQFusedIPPrecision::INVALID;
            const auto* query_planes =
                quantizer_->FilterBits() == 1 ? fused_computer->hnsw_query_planes_.data() : nullptr;
            computed = fused_quantizers_[cluster_id]->ComputeFusedAffineFilter(
                fused_computer->transformed_query_.data(),
                query_planes,
                fused_computer->hnsw_query_delta_,
                fused_computer->hnsw_query_vl_,
                fused_computer->hnsw_query_sum_,
                fused_computer->hnsw_g_add_[cluster_id],
                fused_computer->hnsw_g_error_[cluster_id],
                one_bit_code,
                this->query_rabitq_error_rate(ctx),
                distance,
                lower_bound,
                &local_filter_inner_product,
                &precision);
            if (computed and quantizer_->FilterBits() >= 2 and
                precision == RaBitQFusedIPPrecision::EXACT and filter_inner_product != nullptr) {
                *filter_inner_product = local_filter_inner_product;
            }
        }
        if (not computed and (ctx == nullptr or ctx->enable_rabitq_reorder)) {
            this->add_filter_fallback_full_count(ctx, 1);
        }
        return computed;
    }

    bool
    ComputeFusedFullWithFilterIP(const ComputerInterfacePtr& computer,
                                 uint32_t cluster_id,
                                 const uint8_t* one_bit_code,
                                 const uint8_t* supplement_code,
                                 float filter_inner_product,
                                 float* distance,
                                 QueryContext* ctx) const override {
        auto* fused_computer = static_cast<FusedComputer*>(computer.get());
        if (fused_computer == nullptr or cluster_id >= fused_quantizers_.size()) {
            return false;
        }
        this->add_full_count(ctx, 1);
        bool computed = false;
        if (not IsLegacyHnswFusedCodec() and quantizer_->FilterBits() >= 2) {
            computed = fused_quantizers_[cluster_id]->ComputeFusedAffineFullWithFilterIP(
                fused_computer->transformed_query_.data(),
                fused_computer->hnsw_query_sum_,
                fused_computer->hnsw_g_add_[cluster_id],
                one_bit_code,
                supplement_code,
                filter_inner_product,
                distance);
        }
        if (computed) {
            this->add_reorder_hint_full_count(ctx, 1);
        } else {
            this->add_reorder_fallback_full_count(ctx, 1);
        }
        return computed;
    }

    bool
    ComputeFusedFull(const ComputerInterfacePtr& computer,
                     uint32_t cluster_id,
                     const uint8_t* one_bit_code,
                     const uint8_t* supplement_code,
                     float* distance,
                     QueryContext* ctx) const override {
        auto* fused_computer = static_cast<FusedComputer*>(computer.get());
        if (fused_computer == nullptr or cluster_id >= fused_quantizers_.size()) {
            return false;
        }
        this->add_full_count(ctx, 1);
        if (not IsLegacyHnswFusedCodec()) {
            return fused_quantizers_[cluster_id]->ComputeFusedAffineFullDirect(
                fused_computer->transformed_query_.data(),
                fused_computer->hnsw_query_sum_,
                fused_computer->hnsw_g_add_[cluster_id],
                one_bit_code,
                supplement_code,
                distance);
        }
        const float inv_sqrt_dim = 1.0F / std::sqrt(static_cast<float>(common_param_.dim_));
        const float signed_ip = RaBitQFloatBinaryIP(fused_computer->transformed_query_.data(),
                                                    one_bit_code,
                                                    common_param_.dim_,
                                                    inv_sqrt_dim);
        const float filter_inner_product =
            0.5F * (signed_ip / inv_sqrt_dim + fused_computer->hnsw_query_sum_);
        float lower_bound = 0.0F;
        return fused_quantizers_[cluster_id]->ComputeHnswFull(
            fused_computer->transformed_query_.data(),
            fused_computer->hnsw_query_sum_,
            fused_computer->hnsw_g_add_[cluster_id],
            fused_computer->hnsw_g_error_[cluster_id],
            one_bit_code,
            supplement_code,
            filter_inner_product,
            distance,
            &lower_bound);
    }

    [[nodiscard]] std::string
    ExportFusedCodec() const override {
        if (fused_quantizers_.empty()) {
            return {};
        }
        std::stringstream output;
        IOStreamWriter writer(output);
        constexpr uint32_t version = 1;
        StreamWriter::WriteObj(writer, version);
        StreamWriter::WriteVector(writer, fused_centroids_);
        const auto cluster_count = static_cast<uint32_t>(fused_quantizers_.size());
        StreamWriter::WriteObj(writer, cluster_count);
        return output.str();
    }

    void
    ImportFusedCodec(const std::string& serialized) override {
        CHECK_ARGUMENT(not serialized.empty(), "fused RaBitQ codec payload is empty");
        CHECK_ARGUMENT(this->quantization_param_ != nullptr,
                       "fused RaBitQ quantizer parameter is unavailable");
        constexpr uint64_t cluster_count = 16;
        constexpr uint64_t fixed_payload_size =
            sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t);
        constexpr uint64_t bytes_per_dimension = cluster_count * sizeof(float);
        CHECK_ARGUMENT(common_param_.dim_ > 0, "invalid fused RaBitQ dimension");
        const auto dim = static_cast<uint64_t>(common_param_.dim_);
        CHECK_ARGUMENT(dim <= (std::numeric_limits<uint64_t>::max() - fixed_payload_size) /
                                  bytes_per_dimension,
                       "fused RaBitQ codec size overflow");
        const uint64_t expected_centroid_count = cluster_count * dim;
        const uint64_t expected_payload_size =
            fixed_payload_size + expected_centroid_count * sizeof(float);
        CHECK_ARGUMENT(serialized.size() == expected_payload_size,
                       "invalid fused RaBitQ codec payload size");

        std::stringstream input(serialized);
        IOStreamReader reader(input);
        uint32_t version = 0;
        StreamReader::ReadObj(reader, version);
        CHECK_ARGUMENT(version == 1, "unsupported fused RaBitQ codec version");
        uint64_t centroid_count = 0;
        StreamReader::ReadObj(reader, centroid_count);
        CHECK_ARGUMENT(centroid_count == expected_centroid_count,
                       "invalid fused RaBitQ centroid payload");
        fused_centroids_.resize(expected_centroid_count);
        reader.Read(reinterpret_cast<char*>(fused_centroids_.data()),
                    expected_centroid_count * sizeof(float));
        uint32_t serialized_cluster_count = 0;
        StreamReader::ReadObj(reader, serialized_cluster_count);
        CHECK_ARGUMENT(serialized_cluster_count == cluster_count,
                       "invalid fused RaBitQ cluster count");
        CHECK_ARGUMENT(reader.GetCursor() == reader.Length(),
                       "trailing fused RaBitQ codec payload");

        std::stringstream model_output;
        IOStreamWriter model_writer(model_output);
        quantizer_->Serialize(model_writer);
        const auto serialized_model = model_output.str();
        fused_quantizers_.clear();
        fused_quantizers_.reserve(serialized_cluster_count);
        for (uint32_t cluster_id = 0; cluster_id < serialized_cluster_count; ++cluster_id) {
            auto quantizer =
                std::make_shared<RaBitQuantizer<metric>>(quantization_param_, common_param_);
            std::stringstream model_input(serialized_model);
            IOStreamReader model_reader(model_input);
            quantizer->Deserialize(model_reader);
            quantizer->SetCentroid(fused_centroids_.data() +
                                   static_cast<uint64_t>(cluster_id) * common_param_.dim_);
            fused_quantizers_.push_back(std::move(quantizer));
        }
    }

    bool
    Decode(const uint8_t* codes, float* data) override {
        return this->quantizer_->DecodeOne(codes, data);
    }

    bool
    Encode(const float* data, uint8_t* codes) override {
        return this->quantizer_->EncodeOne(data, codes);
    }

    bool
    CompareRawVectorWithId(const void* vector, InnerIdType id) override {
        if (this->fused_code_storage_ == nullptr) {
            return FlattenInterface::CompareRawVectorWithId(vector, id);
        }
        if (vector == nullptr) {
            return false;
        }
        ByteBuffer one_bit_code(one_bit_code_size_, allocator_);
        ByteBuffer supplement_code(supplement_code_size_, allocator_);
        uint32_t cluster_id = 0;
        if (not this->EncodeFused(static_cast<const float*>(vector),
                                  one_bit_code.data,
                                  supplement_code.data,
                                  &cluster_id)) {
            return false;
        }
        RaBitQFusedCodeView stored;
        if (not this->fused_code_storage_->GetFusedCodeView(id, stored)) {
            return false;
        }
        return stored.cluster_id == cluster_id and
               std::memcmp(stored.one_bit_code, one_bit_code.data, one_bit_code_size_) == 0 and
               std::memcmp(stored.supplement_code, supplement_code.data, supplement_code_size_) ==
                   0;
    }

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override {
        if (this->fused_code_storage_ != nullptr) {
            need_release = false;
            return nullptr;
        }
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
        if (this->fused_code_storage_ != nullptr) {
            return false;
        }
        if (this->optimized_build_active_) {
            bool need_release = false;
            const auto* scalar_code = this->optimized_build_scalar_codes_->Read(id, need_release);
            if (scalar_code == nullptr) {
                return false;
            }
            this->quantizer_->PackScalarCode(scalar_code, codes);
            if (need_release) {
                this->optimized_build_scalar_codes_->Release(scalar_code);
            }
            return true;
        }
        ByteBuffer one_bit(one_bit_code_size_, allocator_);
        ByteBuffer supplement(supplement_code_size_, allocator_);
        bool one_bit_ok = this->x_bit_cell_->Read(id, one_bit.data);
        bool supplement_ok = this->supplement_cell_->Read(id, supplement.data);
        if (not one_bit_ok or not supplement_ok) {
            return false;
        }
        this->quantizer_->MergeSplitCode(one_bit.data, supplement.data, codes);
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
        if (this->fused_code_storage_ != nullptr) {
            StreamWriter::WriteObj(writer, kFusedModelMagic);
            StreamWriter::WriteObj(writer, kFusedModelVersion);
            StreamWriter::WriteObj(writer, this->one_bit_code_size_);
            StreamWriter::WriteObj(writer, this->supplement_code_size_);
            this->quantizer_->Serialize(writer);
            return;
        }
        StreamWriter::WriteString(writer, this->supplement_io_type_);
        this->x_bit_cell_->Serialize(writer);
        this->supplement_cell_->Serialize(writer);
        this->quantizer_->Serialize(writer);
    }

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override {
        FlattenInterface::Deserialize(reader);
        if (this->fused_code_storage_ != nullptr) {
            const uint64_t payload_cursor = reader->GetCursor();
            uint64_t magic = 0;
            StreamReader::ReadObj(reader, magic);
            if (magic == kFusedModelMagic) {
                uint32_t version = 0;
                uint64_t serialized_one_bit_size = 0;
                uint64_t serialized_supplement_size = 0;
                StreamReader::ReadObj(reader, version);
                StreamReader::ReadObj(reader, serialized_one_bit_size);
                StreamReader::ReadObj(reader, serialized_supplement_size);
                CHECK_ARGUMENT(version == kFusedModelVersion,
                               "unsupported fused RaBitQ model-only serialization version");
                this->quantizer_->Deserialize(reader);
                this->refresh_code_sizes();
                CHECK_ARGUMENT(serialized_one_bit_size == this->one_bit_code_size_ and
                                   serialized_supplement_size == this->supplement_code_size_ and
                                   this->fused_code_storage_->FusedOneBitCodeSize() ==
                                       this->one_bit_code_size_ and
                                   this->fused_code_storage_->FusedSupplementCodeSize() ==
                                       this->supplement_code_size_,
                               "serialized fused RaBitQ code sizes do not match the node layout");
                return;
            }
            reader->Seek(payload_cursor);
            this->DeserializeSupplementIOType(reader);
            this->x_bit_cell_->SkipSerialized(*reader);
            this->supplement_cell_->SkipSerialized(*reader);
            this->quantizer_->Deserialize(reader);
            this->refresh_code_sizes();
            CHECK_ARGUMENT(
                this->fused_code_storage_->FusedOneBitCodeSize() == this->one_bit_code_size_ and
                    this->fused_code_storage_->FusedSupplementCodeSize() ==
                        this->supplement_code_size_,
                "legacy split RaBitQ code sizes do not match the fused node layout");
            return;
        }
        this->DeserializeSupplementIOType(reader);
        this->x_bit_cell_->Deserialize(reader);
        this->supplement_cell_->Deserialize(reader);
        this->quantizer_->Deserialize(reader);
        this->refresh_code_sizes();
    }

    void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) override {
        if (this->fused_code_storage_ != nullptr) {
            throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                "fused RaBitQ code storage does not support MergeOther");
        }
        auto ptr =
            std::dynamic_pointer_cast<RaBitQSplitDataCell<metric, OneBitIOTmpl, SupplementIOTmpl>>(
                other);
        if (ptr == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Merge rabitq split datacell failed: not match type");
        }

        for (InnerIdType i = 0; i < ptr->total_count_; ++i) {
            ByteBuffer one_bit(one_bit_code_size_, allocator_);
            ByteBuffer supplement(supplement_code_size_, allocator_);
            ptr->x_bit_cell_->Read(i, one_bit.data);
            ptr->supplement_cell_->Read(i, supplement.data);
            auto target_id = static_cast<InnerIdType>(bias + i);
            this->x_bit_cell_->Write(one_bit.data, target_id);
            this->supplement_cell_->Write(supplement.data, target_id);
        }
        this->total_count_ = std::max(this->total_count_, bias + ptr->total_count_);
    }

    void
    Move(InnerIdType from, InnerIdType to) override {
        if (this->fused_code_storage_ != nullptr) {
            return;
        }
        if (this->optimized_build_active_) {
            ByteBuffer build_record(this->optimized_build_record_size_, allocator_);
            this->optimized_build_scalar_codes_->Read(from, build_record.data);
            this->optimized_build_scalar_codes_->Write(build_record.data, to);
            (*this->optimized_build_code_sums_)[to] = (*this->optimized_build_code_sums_)[from];
            return;
        }
        ByteBuffer one_bit(one_bit_code_size_, allocator_);
        ByteBuffer supplement(supplement_code_size_, allocator_);
        this->x_bit_cell_->Read(from, one_bit.data);
        this->supplement_cell_->Read(from, supplement.data);
        this->x_bit_cell_->Write(one_bit.data, to);
        this->supplement_cell_->Write(supplement.data, to);
    }

    void
    ShrinkToFit(InnerIdType capacity) override {
        if (this->fused_code_storage_ != nullptr) {
            this->max_capacity_ = capacity;
            return;
        }
        this->x_bit_cell_->Shrink(capacity);
        this->supplement_cell_->Shrink(capacity);
        if (this->optimized_build_active_) {
            this->optimized_build_scalar_codes_->Shrink(capacity);
            this->optimized_build_code_sums_->resize(capacity);
            this->optimized_build_code_sums_->shrink_to_fit();
        }
        this->max_capacity_ = capacity;
    }

    uint64_t
    GetMemoryUsage() const override {
        uint64_t memory = sizeof(RaBitQSplitDataCell<metric, OneBitIOTmpl, SupplementIOTmpl>);
        memory += this->x_bit_cell_->GetMemoryUsage();
        memory += this->supplement_cell_->GetMemoryUsage();
        if (this->optimized_build_scalar_codes_ != nullptr) {
            memory += this->optimized_build_scalar_codes_->GetMemoryUsage();
        }
        if (this->optimized_build_code_sums_ != nullptr) {
            memory += this->optimized_build_code_sums_->capacity() * sizeof(uint64_t);
        }
        memory += sizeof(RaBitQuantizer<metric>);
        memory += fused_quantizers_.size() * sizeof(RaBitQuantizer<metric>);
        memory += fused_centroids_.capacity() * sizeof(float);
        return memory;
    }

public:
    IndexCommonParam common_param_;
    RaBitQuantizerParamPtr quantization_param_{nullptr};
    std::shared_ptr<RaBitQuantizer<metric>> quantizer_{nullptr};
    std::vector<std::shared_ptr<RaBitQuantizer<metric>>> fused_quantizers_;
    std::vector<float> fused_centroids_;
    std::shared_ptr<RaBitQSplitCodeStorage<OneBitIOTmpl>> x_bit_cell_{nullptr};
    std::shared_ptr<RaBitQSplitCodeStorage<SupplementIOTmpl>> supplement_cell_{nullptr};
    std::shared_ptr<RaBitQSplitCodeStorage<MemoryIO>> optimized_build_scalar_codes_{nullptr};
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
    // IOParameter subclass for `supplement_cell_` instead of feeding it the
    // mismatched one-bit IO parameter type.
    std::string supplement_io_type_{};
    bool optimized_build_active_{false};
    uint64_t optimized_build_record_size_{0};
    RaBitQFusedCodeStorageInterface* fused_code_storage_{nullptr};

private:
    static constexpr uint64_t kFusedModelMagic = 0x524246534D4F444CULL;
    static constexpr uint32_t kFusedModelVersion = 1;

    [[nodiscard]] bool
    IsLegacyHnswFusedCodec() const {
        return quantizer_->FilterBits() == 1 and quantizer_->ReorderBits() == 7;
    }

    [[nodiscard]] uint32_t
    NearestFusedCluster(const float* data) const {
        uint32_t nearest = 0;
        double nearest_distance = std::numeric_limits<double>::max();
        for (uint32_t cluster_id = 0; cluster_id < fused_quantizers_.size(); ++cluster_id) {
            const auto* centroid =
                fused_centroids_.data() + static_cast<uint64_t>(cluster_id) * common_param_.dim_;
            const double distance =
                FP32ComputeL2Sqr(data, centroid, static_cast<uint64_t>(common_param_.dim_));
            if (distance < nearest_distance) {
                nearest_distance = distance;
                nearest = cluster_id;
            }
        }
        return nearest;
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

    // Builds the IO parameter that should be handed to `supplement_cell_`
    // given the caller-supplied `io_param` (which is always typed for the
    // one-bit storage). If `supplement_io_type_` is empty the two storages
    // share the same IO type and we fall back to the legacy file-path-suffix
    // behaviour. Otherwise the JSON is cloned, its `type` field rewritten to
    // the recorded supplement type, the optional file path suffixed, and a
    // new IOParameter is constructed via the factory so `supplement_cell_`
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
        this->one_bit_code_size_ = quantizer_->GetOneBitCodeSize();
        this->supplement_code_size_ = quantizer_->GetSupplementCodeSize();
        this->x_bit_cell_->SetCodeSize(one_bit_code_size_);
        this->supplement_cell_->SetCodeSize(supplement_code_size_);
    }

    void
    write_encoded_vector(const float* vector, InnerIdType idx) {
        if (this->optimized_build_active_) {
            ByteBuffer scalar_code(this->optimized_build_record_size_, allocator_);
            uint64_t code_sum = 0;
            if (not this->quantizer_->EncodeOneToScalarCode(vector, scalar_code.data, code_sum)) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to encode temporary scalar RaBitQ build code");
            }
            (*this->optimized_build_code_sums_)[idx] = code_sum;
            this->optimized_build_scalar_codes_->Write(scalar_code.data, idx);
            return;
        }
        ByteBuffer full_code(this->code_size_, allocator_);
        this->quantizer_->EncodeOne(vector, full_code.data);
        ByteBuffer one_bit_code(one_bit_code_size_, allocator_);
        ByteBuffer supplement_code(supplement_code_size_, allocator_);
        this->quantizer_->SplitCode(full_code.data, one_bit_code.data, supplement_code.data);
        this->x_bit_cell_->Write(one_bit_code.data, idx);
        this->supplement_cell_->Write(supplement_code.data, idx);
    }

    [[nodiscard]] RaBitQFusedCodeView
    get_fused_code_view(InnerIdType id) const {
        RaBitQFusedCodeView view;
        if (not fused_code_storage_->GetFusedCodeView(id, view)) {
            throw VsagException(
                ErrorType::INTERNAL_ERROR, "failed to read fused RaBitQ code for id ", id);
        }
        return view;
    }

    void
    prefetch_fused_codes(const InnerIdType* ids,
                         InnerIdType id_count,
                         bool include_supplement) const {
        const auto count = std::min<InnerIdType>(prefetch_stride_code_, id_count);
        for (InnerIdType i = 0; i < count; ++i) {
            fused_code_storage_->PrefetchFusedCodes(ids[i], include_supplement);
        }
    }

    void
    query_fused_full(float* result_dists,
                     const ComputerInterfacePtr& computer,
                     const InnerIdType* ids,
                     InnerIdType id_count,
                     QueryContext* ctx) const {
        this->prefetch_fused_codes(ids, id_count, true);
        for (InnerIdType i = 0; i < id_count; ++i) {
            if (i + prefetch_stride_code_ < id_count) {
                fused_code_storage_->PrefetchFusedCodes(ids[i + prefetch_stride_code_], true);
            }
            const auto view = this->get_fused_code_view(ids[i]);
            CHECK_ARGUMENT(this->ComputeFusedFull(computer,
                                                  view.cluster_id,
                                                  view.one_bit_code,
                                                  view.supplement_code,
                                                  result_dists + i,
                                                  ctx),
                           "failed to compute fused RaBitQ distance");
        }
    }

    void
    query_fused_lower_bound(float* result_dists,
                            float* lower_bounds,
                            float* filter_inner_products,
                            const ComputerInterfacePtr& computer,
                            const InnerIdType* ids,
                            InnerIdType id_count,
                            QueryContext* ctx) const {
        const bool enable_reorder = ctx == nullptr or ctx->enable_rabitq_reorder;
        this->prefetch_fused_codes(ids, id_count, false);
        for (InnerIdType i = 0; i < id_count; ++i) {
            if (i + prefetch_stride_code_ < id_count) {
                fused_code_storage_->PrefetchFusedCodes(ids[i + prefetch_stride_code_], false);
            }
            const auto view = this->get_fused_code_view(ids[i]);
            float local_lower_bound = std::numeric_limits<float>::max();
            float local_filter_ip = std::numeric_limits<float>::quiet_NaN();
            const bool computed = this->ComputeFusedOneBitWithFilterIP(computer,
                                                                       view.cluster_id,
                                                                       view.one_bit_code,
                                                                       view.supplement_code,
                                                                       result_dists + i,
                                                                       &local_lower_bound,
                                                                       &local_filter_ip,
                                                                       ctx);
            if (not enable_reorder) {
                if (not IsFiniteRaBitQValue(result_dists[i])) {
                    result_dists[i] = std::numeric_limits<float>::max();
                }
                local_lower_bound = result_dists[i];
                local_filter_ip = std::numeric_limits<float>::quiet_NaN();
            } else if (not computed) {
                CHECK_ARGUMENT(this->ComputeFusedFull(computer,
                                                      view.cluster_id,
                                                      view.one_bit_code,
                                                      view.supplement_code,
                                                      result_dists + i,
                                                      ctx),
                               "failed to compute fused RaBitQ distance");
                local_lower_bound = std::numeric_limits<float>::max();
                local_filter_ip = std::numeric_limits<float>::quiet_NaN();
            }
            if (lower_bounds != nullptr) {
                lower_bounds[i] = local_lower_bound;
            }
            if (filter_inner_products != nullptr) {
                filter_inner_products[i] = local_filter_ip;
            }
        }
    }

    void
    query_fused_with_distance_filter(float* result_dists,
                                     const ComputerInterfacePtr& computer,
                                     const InnerIdType* ids,
                                     InnerIdType id_count,
                                     float threshold,
                                     QueryContext* ctx) const {
        const bool enable_reorder = ctx == nullptr or ctx->enable_rabitq_reorder;
        this->prefetch_fused_codes(ids, id_count, false);
        for (InnerIdType i = 0; i < id_count; ++i) {
            const auto view = this->get_fused_code_view(ids[i]);
            float lower_bound = std::numeric_limits<float>::max();
            float filter_inner_product = std::numeric_limits<float>::quiet_NaN();
            const bool computed = this->ComputeFusedOneBitWithFilterIP(computer,
                                                                       view.cluster_id,
                                                                       view.one_bit_code,
                                                                       view.supplement_code,
                                                                       result_dists + i,
                                                                       &lower_bound,
                                                                       &filter_inner_product,
                                                                       ctx);
            if (not enable_reorder) {
                if (not IsFiniteRaBitQValue(result_dists[i])) {
                    result_dists[i] = std::numeric_limits<float>::max();
                } else if (computed and IsFiniteRaBitQValue(lower_bound) and
                           lower_bound >= threshold) {
                    result_dists[i] = threshold;
                }
                continue;
            }
            if (computed and IsFiniteRaBitQValue(lower_bound) and lower_bound >= threshold) {
                result_dists[i] = threshold;
                continue;
            }
            CHECK_ARGUMENT(this->ComputeFusedFull(computer,
                                                  view.cluster_id,
                                                  view.one_bit_code,
                                                  view.supplement_code,
                                                  result_dists + i,
                                                  ctx),
                           "failed to compute fused RaBitQ distance");
        }
    }

    void
    query_fused_full_with_filter_ip(float* result_dists,
                                    const float* filter_inner_products,
                                    const ComputerInterfacePtr& computer,
                                    const InnerIdType* ids,
                                    InnerIdType id_count,
                                    QueryContext* ctx) const {
        this->prefetch_fused_codes(ids, id_count, true);
        const bool exact_filter_ip_hint =
            not IsLegacyHnswFusedCodec() and this->quantizer_->FilterBits() >= 2;
        for (InnerIdType i = 0; i < id_count; ++i) {
            const auto view = this->get_fused_code_view(ids[i]);
            const float filter_ip = filter_inner_products == nullptr
                                        ? std::numeric_limits<float>::quiet_NaN()
                                        : filter_inner_products[i];
            bool computed = false;
            if (exact_filter_ip_hint and IsFiniteRaBitQValue(filter_ip)) {
                computed = this->ComputeFusedFullWithFilterIP(computer,
                                                              view.cluster_id,
                                                              view.one_bit_code,
                                                              view.supplement_code,
                                                              filter_ip,
                                                              result_dists + i,
                                                              ctx);
            }
            if (not computed) {
                CHECK_ARGUMENT(this->ComputeFusedFull(computer,
                                                      view.cluster_id,
                                                      view.one_bit_code,
                                                      view.supplement_code,
                                                      result_dists + i,
                                                      ctx),
                               "failed to compute fused RaBitQ distance");
            }
        }
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
        auto* comp = static_cast<Computer<RaBitQuantizer<metric>>*>(computer.get());
        for (InnerIdType i = 0; i < id_count; ++i) {
            bool need_release = false;
            const auto* scalar_code =
                this->optimized_build_scalar_codes_->Read(idx[i], need_release);
            if (scalar_code == nullptr) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to read temporary scalar RaBitQ build code");
            }
            try {
                this->quantizer_->ComputeDistWithScalarCode(*comp, scalar_code, result_dists + i);
            } catch (...) {
                if (need_release) {
                    this->optimized_build_scalar_codes_->Release(scalar_code);
                }
                throw;
            }
            if (need_release) {
                this->optimized_build_scalar_codes_->Release(scalar_code);
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
            this->optimized_build_scalar_codes_->Prefetch(idx[i], this->prefetch_depth_code_ * 64);
        }
        for (InnerIdType i = 0; i < id_count; ++i) {
            if (i + this->prefetch_stride_code_ < id_count) {
                this->optimized_build_scalar_codes_->Prefetch(idx[i + this->prefetch_stride_code_],
                                                              this->prefetch_depth_code_ * 64);
            }
            bool need_release = false;
            const auto* base_code = this->optimized_build_scalar_codes_->Read(idx[i], need_release);
            if (base_code == nullptr) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "failed to read temporary scalar RaBitQ build code");
            }
            try {
                result_dists[i] = this->quantizer_->ComputeScalarCodesDistance(
                    query_code, query_sum, base_code, (*this->optimized_build_code_sums_)[idx[i]]);
            } catch (...) {
                if (need_release) {
                    this->optimized_build_scalar_codes_->Release(base_code);
                }
                throw;
            }
            if (need_release) {
                this->optimized_build_scalar_codes_->Release(base_code);
            }
        }
    }

    void
    prefetch_one_bit(InnerIdType id) {
        this->x_bit_cell_->Prefetch(id, this->prefetch_depth_code_ * 64);
    }

    void
    prefetch_supplement(InnerIdType id) {
        this->supplement_cell_->Prefetch(id, this->prefetch_depth_code_ * 64);
    }

    void
    prefetch_full_code(InnerIdType id) {
        this->prefetch_one_bit(id);
        this->prefetch_supplement(id);
    }

    const uint8_t*
    get_one_bit_code(InnerIdType id, bool& need_release) const {
        return this->x_bit_cell_->Read(id, need_release);
    }

    void
    release_one_bit_code(const uint8_t* code, bool need_release) const {
        if (need_release) {
            this->x_bit_cell_->Release(code);
        }
    }

    const uint8_t*
    get_supplement_code(InnerIdType id, bool& need_release) const {
        return this->supplement_cell_->Read(id, need_release);
    }

    void
    release_supplement_code(const uint8_t* code, bool need_release) const {
        if (need_release) {
            this->supplement_cell_->Release(code);
        }
    }

    [[nodiscard]] static float
    query_rabitq_error_rate(QueryContext* ctx) {
        return ctx == nullptr ? std::numeric_limits<float>::quiet_NaN() : ctx->rabitq_error_rate;
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
                                           Computer<RaBitQuantizer<metric>>* computer,
                                           const InnerIdType* idx,
                                           InnerIdType id_count,
                                           QueryContext* ctx) const {
        Allocator* search_alloc = select_query_allocator(ctx, allocator_);
        ByteBuffer one_bit_codes(id_count * one_bit_code_size_, search_alloc);
        Vector<uint64_t> sizes(id_count, one_bit_code_size_, search_alloc);
        Vector<uint64_t> offsets(id_count, one_bit_code_size_, search_alloc);
        for (InnerIdType i = 0; i < id_count; ++i) {
            offsets[i] = static_cast<uint64_t>(idx[i]) * one_bit_code_size_;
        }

        double io_cost_ms = 0.0F;
        {
            Timer timer(io_cost_ms);
            this->x_bit_cell_->MultiRead(
                one_bit_codes.data, sizes.data(), offsets.data(), id_count);
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
            this->quantizer_->ComputeDistsWithOneBitLowerBoundBatch4(
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
            bool computed = this->quantizer_->ComputeDistWithOneBitLowerBound(
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
                                 Computer<RaBitQuantizer<metric>>* computer,
                                 const InnerIdType* idx,
                                 InnerIdType id_count,
                                 QueryContext* ctx,
                                 const float* hint_dists = nullptr) const {
        Allocator* search_alloc = select_query_allocator(ctx, allocator_);
        ByteBuffer one_bit_codes(id_count * one_bit_code_size_, search_alloc);
        ByteBuffer supplement_codes(id_count * supplement_code_size_, search_alloc);
        Vector<uint64_t> one_bit_sizes(id_count, one_bit_code_size_, search_alloc);
        Vector<uint64_t> one_bit_offsets(id_count, 0, search_alloc);
        Vector<uint64_t> supp_sizes(id_count, supplement_code_size_, search_alloc);
        Vector<uint64_t> supp_offsets(id_count, 0, search_alloc);
        for (InnerIdType i = 0; i < id_count; ++i) {
            one_bit_offsets[i] = static_cast<uint64_t>(idx[i]) * one_bit_code_size_;
            supp_offsets[i] = static_cast<uint64_t>(idx[i]) * supplement_code_size_;
        }

        double io_cost_ms = 0.0F;
        {
            Timer timer(io_cost_ms);
            this->x_bit_cell_->MultiRead(
                one_bit_codes.data, one_bit_sizes.data(), one_bit_offsets.data(), id_count);
            this->supplement_cell_->MultiRead(
                supplement_codes.data, supp_sizes.data(), supp_offsets.data(), id_count);
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
                                            Computer<RaBitQuantizer<metric>>* computer,
                                            const InnerIdType* idx,
                                            InnerIdType id_count,
                                            QueryContext* ctx,
                                            const float* hint_dists = nullptr) const {
        Allocator* search_alloc = select_query_allocator(ctx, allocator_);
        ByteBuffer supplement_codes(id_count * supplement_code_size_, search_alloc);
        Vector<uint64_t> supp_sizes(id_count, supplement_code_size_, search_alloc);
        Vector<uint64_t> supp_offsets(id_count, 0, search_alloc);
        for (InnerIdType i = 0; i < id_count; ++i) {
            supp_offsets[i] = static_cast<uint64_t>(idx[i]) * supplement_code_size_;
        }

        double io_cost_ms = 0.0F;
        {
            Timer timer(io_cost_ms);
            this->supplement_cell_->MultiRead(
                supplement_codes.data, supp_sizes.data(), supp_offsets.data(), id_count);
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
                                            Computer<RaBitQuantizer<metric>>* computer,
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
                      Computer<RaBitQuantizer<metric>>* computer,
                      float* result_dist,
                      QueryContext* ctx = nullptr,
                      float hint_dist = std::numeric_limits<float>::max()) const {
        this->add_full_count(ctx, 1);
        bool computed = false;
        const bool has_hint = this->quantizer_->FilterBits() >= 2 and
                              IsFiniteRaBitQValue(hint_dist) and
                              hint_dist < std::numeric_limits<float>::max();
        if (has_hint) {
            computed = this->quantizer_->ComputeDistWithSplitCodeAndFilterDist(
                *computer, one_bit_code, supplement_code, hint_dist, result_dist);
        }
        if (computed) {
            this->add_reorder_hint_full_count(ctx, 1);
        } else if (has_hint) {
            this->add_reorder_fallback_full_count(ctx, 1);
        }
        if (not computed and not this->quantizer_->ComputeDistWithSplitCode(
                                 *computer, one_bit_code, supplement_code, result_dist)) {
            ByteBuffer full_code(this->code_size_, allocator_);
            this->quantizer_->MergeSplitCode(one_bit_code, supplement_code, full_code.data);
            computer->ComputeDist(full_code.data, result_dist);
        }
    }

    void
    compute_full_dist_with_filter_ip(const uint8_t* one_bit_code,
                                     const uint8_t* supplement_code,
                                     Computer<RaBitQuantizer<metric>>* computer,
                                     float* result_dist,
                                     QueryContext* ctx,
                                     float filter_inner_product) const {
        this->add_full_count(ctx, 1);
        const bool has_hint =
            this->quantizer_->FilterBits() >= 2 and IsFiniteRaBitQValue(filter_inner_product);
        bool computed = false;
        if (has_hint) {
            computed = this->quantizer_->ComputeDistWithSplitCodeAndFilterIP(
                *computer, one_bit_code, supplement_code, filter_inner_product, result_dist);
        }
        if (computed) {
            this->add_reorder_hint_full_count(ctx, 1);
            return;
        }
        if (has_hint) {
            this->add_reorder_fallback_full_count(ctx, 1);
        }
        if (not this->quantizer_->ComputeDistWithSplitCode(
                *computer, one_bit_code, supplement_code, result_dist)) {
            ByteBuffer full_code(this->code_size_, allocator_);
            this->quantizer_->MergeSplitCode(one_bit_code, supplement_code, full_code.data);
            computer->ComputeDist(full_code.data, result_dist);
        }
    }

    void
    compute_full_dist_with_filter_ip(InnerIdType id,
                                     Computer<RaBitQuantizer<metric>>* computer,
                                     float* result_dist,
                                     QueryContext* ctx,
                                     float filter_inner_product) const {
        bool one_bit_need_release = false;
        bool supplement_need_release = false;
        const auto* one_bit_code = this->get_one_bit_code(id, one_bit_need_release);
        const auto* supplement_code = this->get_supplement_code(id, supplement_need_release);
        try {
            this->compute_full_dist_with_filter_ip(
                one_bit_code, supplement_code, computer, result_dist, ctx, filter_inner_product);
        } catch (...) {
            this->release_one_bit_code(one_bit_code, one_bit_need_release);
            this->release_supplement_code(supplement_code, supplement_need_release);
            throw;
        }
        this->release_one_bit_code(one_bit_code, one_bit_need_release);
        this->release_supplement_code(supplement_code, supplement_need_release);
    }

    void
    compute_full_dist(InnerIdType id,
                      Computer<RaBitQuantizer<metric>>* computer,
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
