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
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>

#include "common.h"
#include "flatten_interface.h"
#include "io/basic_io.h"
#include "quantization/rabitq_quantization/rabitq_quantizer.h"
#include "query_context.h"
#include "utils/byte_buffer.h"
#include "utils/timer.h"

namespace vsag {

template <MetricType metric, typename IOTmpl>
class RaBitQSplitDataCell : public FlattenInterface {
public:
    RaBitQSplitDataCell() = default;

    explicit RaBitQSplitDataCell(const QuantizerParamPtr& quantization_param,
                                 const IOParamPtr& io_param,
                                 const IndexCommonParam& common_param)
        : common_param_(common_param), allocator_(common_param.allocator_.get()) {
        this->quantizer_ =
            std::make_shared<RaBitQuantizer<metric>>(quantization_param, common_param);
        if (not this->quantizer_->SupportSplitCodeStorage()) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "rabitq split data cell requires rabitq_version=split_1bit_7bit, "
                                "rabitq_bits_per_dim_query=32, and "
                                "rabitq_bits_per_dim_base in [1, 8]");
        }
        this->one_bit_io_ = std::make_shared<IOTmpl>(io_param, common_param);
        this->supplement_io_ = std::make_shared<IOTmpl>(io_param, common_param);
        this->refresh_code_sizes();
    }

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) override {
        auto* comp = static_cast<Computer<RaBitQuantizer<metric>>*>(computer.get());
        for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; ++i) {
            this->prefetch_full_code(idx[i]);
        }

        for (InnerIdType i = 0; i < id_count; ++i) {
            if (i + this->prefetch_stride_code_ < id_count) {
                this->prefetch_full_code(idx[i + this->prefetch_stride_code_]);
            }
            this->compute_full_dist(idx[i], comp, result_dists + i);
        }
    }

    void
    QueryWithDistanceFilter(float* result_dists,
                            const ComputerInterfacePtr& computer,
                            const InnerIdType* idx,
                            InnerIdType id_count,
                            float threshold,
                            QueryContext* ctx = nullptr) override {
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
                    *comp, one_bit_code, &one_bit_dist, &lower_bound);
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
                this->compute_full_dist(one_bit_code, supplement_code, comp, result_dists + i);
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
    QueryWithDistanceLowerBound(float* result_dists,
                                float* lower_bounds,
                                const ComputerInterfacePtr& computer,
                                const InnerIdType* idx,
                                InnerIdType id_count,
                                QueryContext* ctx = nullptr) override {
        auto* comp = static_cast<Computer<RaBitQuantizer<metric>>*>(computer.get());
        if constexpr (not IOTmpl::InMemory) {
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
                this->quantizer_->ComputeDistsWithOneBitLowerBoundBatch4(*comp,
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
                                                                         computed4);
                if (not computed1) {
                    this->compute_full_dist_after_one_bit_failure(
                        idx[i], code1, comp, result_dists + i, lower_bound1);
                }
                if (not computed2) {
                    this->compute_full_dist_after_one_bit_failure(
                        idx[i + 1], code2, comp, result_dists + i + 1, lower_bound2);
                }
                if (not computed3) {
                    this->compute_full_dist_after_one_bit_failure(
                        idx[i + 2], code3, comp, result_dists + i + 2, lower_bound3);
                }
                if (not computed4) {
                    this->compute_full_dist_after_one_bit_failure(
                        idx[i + 3], code4, comp, result_dists + i + 3, lower_bound4);
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
                    *comp, one_bit_code, result_dists + i, lower_bound);
                if (not computed) {
                    this->compute_full_dist_after_one_bit_failure(
                        idx[i], one_bit_code, comp, result_dists + i, lower_bound);
                }
            } catch (...) {
                this->release_one_bit_code(one_bit_code, one_bit_need_release);
                throw;
            }
            this->release_one_bit_code(one_bit_code, one_bit_need_release);
        }
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        auto computer = this->quantizer_->FactoryComputer();
        computer->SetQuery(static_cast<const float*>(query));
        return computer;
    }

    void
    Train(const void* data, uint64_t count) override {
        if (this->quantizer_) {
            this->quantizer_->Train(static_cast<const float*>(data), count);
        }
    }

    void
    InsertVector(const void* vector,
                 InnerIdType idx = std::numeric_limits<InnerIdType>::max()) override {
        {
            std::lock_guard lock(this->mutex_);
            if (idx == std::numeric_limits<InnerIdType>::max()) {
                idx = this->total_count_;
                ++this->total_count_;
            } else {
                this->total_count_ = std::max(this->total_count_, idx + 1);
            }
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
        this->one_bit_io_->Resize(static_cast<uint64_t>(new_capacity) * one_bit_code_size_);
        this->supplement_io_->Resize(static_cast<uint64_t>(new_capacity) * supplement_code_size_);
        this->max_capacity_ = new_capacity;
    }

    void
    Prefetch(InnerIdType id) override {
        this->prefetch_one_bit(id);
    }

    void
    ExportModel(const FlattenInterfacePtr& other) const override {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        this->quantizer_->Serialize(writer);
        ss.seekg(0, std::ios::beg);
        IOStreamReader reader(ss);
        auto ptr = std::dynamic_pointer_cast<RaBitQSplitDataCell<metric, IOTmpl>>(other);
        if (ptr == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Export model's rabitq split datacell failed");
        }
        ptr->quantizer_->Deserialize(reader);
        ptr->refresh_code_sizes();
    }

    void
    InitIO(const IOParamPtr& io_param) override {
        this->one_bit_io_->InitIO(io_param);
        this->supplement_io_->InitIO(io_param);
    }

    IndexCommonParam
    ExportCommonParam() override {
        return common_param_;
    }

    [[nodiscard]] std::string
    GetQuantizerName() override {
        return this->quantizer_->Name();
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
        ByteBuffer one_bit(one_bit_code_size_, allocator_);
        ByteBuffer supplement(supplement_code_size_, allocator_);
        bool one_bit_ok = this->one_bit_io_->Read(
            one_bit_code_size_, static_cast<uint64_t>(id) * one_bit_code_size_, one_bit.data);
        bool supplement_ok =
            this->supplement_io_->Read(supplement_code_size_,
                                       static_cast<uint64_t>(id) * supplement_code_size_,
                                       supplement.data);
        if (not one_bit_ok or not supplement_ok) {
            return false;
        }
        this->quantizer_->MergeSplitCode(one_bit.data, supplement.data, codes);
        return true;
    }

    [[nodiscard]] bool
    InMemory() const override {
        return IOTmpl::InMemory;
    }

    bool
    HoldMolds() const override {
        return this->quantizer_->HoldMolds();
    }

    void
    Serialize(StreamWriter& writer) override {
        FlattenInterface::Serialize(writer);
        this->one_bit_io_->Serialize(writer);
        this->supplement_io_->Serialize(writer);
        this->quantizer_->Serialize(writer);
    }

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override {
        FlattenInterface::Deserialize(reader);
        this->one_bit_io_->Deserialize(reader);
        this->supplement_io_->Deserialize(reader);
        this->quantizer_->Deserialize(reader);
        this->refresh_code_sizes();
    }

    void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) override {
        auto ptr = std::dynamic_pointer_cast<RaBitQSplitDataCell<metric, IOTmpl>>(other);
        if (ptr == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Merge rabitq split datacell failed: not match type");
        }

        for (InnerIdType i = 0; i < ptr->total_count_; ++i) {
            ByteBuffer one_bit(one_bit_code_size_, allocator_);
            ByteBuffer supplement(supplement_code_size_, allocator_);
            ptr->one_bit_io_->Read(
                one_bit_code_size_, static_cast<uint64_t>(i) * one_bit_code_size_, one_bit.data);
            ptr->supplement_io_->Read(supplement_code_size_,
                                      static_cast<uint64_t>(i) * supplement_code_size_,
                                      supplement.data);
            auto target = static_cast<uint64_t>(bias + i);
            this->one_bit_io_->Write(one_bit.data, one_bit_code_size_, target * one_bit_code_size_);
            this->supplement_io_->Write(
                supplement.data, supplement_code_size_, target * supplement_code_size_);
        }
        this->total_count_ = std::max(this->total_count_, bias + ptr->total_count_);
    }

    void
    Move(InnerIdType from, InnerIdType to) override {
        ByteBuffer one_bit(one_bit_code_size_, allocator_);
        ByteBuffer supplement(supplement_code_size_, allocator_);
        this->one_bit_io_->Read(
            one_bit_code_size_, static_cast<uint64_t>(from) * one_bit_code_size_, one_bit.data);
        this->supplement_io_->Read(supplement_code_size_,
                                   static_cast<uint64_t>(from) * supplement_code_size_,
                                   supplement.data);
        this->one_bit_io_->Write(
            one_bit.data, one_bit_code_size_, static_cast<uint64_t>(to) * one_bit_code_size_);
        this->supplement_io_->Write(supplement.data,
                                    supplement_code_size_,
                                    static_cast<uint64_t>(to) * supplement_code_size_);
    }

    void
    ShrinkToFit(InnerIdType capacity) override {
        this->one_bit_io_->Shrink(static_cast<uint64_t>(capacity) * one_bit_code_size_);
        this->supplement_io_->Shrink(static_cast<uint64_t>(capacity) * supplement_code_size_);
        this->max_capacity_ = capacity;
    }

    int64_t
    GetMemoryUsage() const override {
        int64_t memory = sizeof(RaBitQSplitDataCell<metric, IOTmpl>);
        if (IOTmpl::InMemory) {
            memory += this->one_bit_io_->GetMemoryUsage();
            memory += this->supplement_io_->GetMemoryUsage();
        }
        memory += sizeof(RaBitQuantizer<metric>);
        return memory;
    }

public:
    IndexCommonParam common_param_;
    std::shared_ptr<RaBitQuantizer<metric>> quantizer_{nullptr};
    std::shared_ptr<BasicIO<IOTmpl>> one_bit_io_{nullptr};
    std::shared_ptr<BasicIO<IOTmpl>> supplement_io_{nullptr};

    Allocator* allocator_{nullptr};
    uint64_t one_bit_code_size_{0};
    uint64_t supplement_code_size_{0};

private:
    void
    refresh_code_sizes() {
        this->code_size_ = static_cast<uint32_t>(quantizer_->GetCodeSize());
        this->one_bit_code_size_ = quantizer_->GetOneBitCodeSize();
        this->supplement_code_size_ = quantizer_->GetSupplementCodeSize();
    }

    void
    write_encoded_vector(const float* vector, InnerIdType idx) {
        ByteBuffer full_code(this->code_size_, allocator_);
        ByteBuffer one_bit_code(one_bit_code_size_, allocator_);
        ByteBuffer supplement_code(supplement_code_size_, allocator_);
        this->quantizer_->EncodeOne(vector, full_code.data);
        this->quantizer_->SplitCode(full_code.data, one_bit_code.data, supplement_code.data);
        this->one_bit_io_->Write(
            one_bit_code.data, one_bit_code_size_, static_cast<uint64_t>(idx) * one_bit_code_size_);
        this->supplement_io_->Write(supplement_code.data,
                                    supplement_code_size_,
                                    static_cast<uint64_t>(idx) * supplement_code_size_);
    }

    void
    prefetch_one_bit(InnerIdType id) {
        this->one_bit_io_->Prefetch(
            static_cast<uint64_t>(id) * one_bit_code_size_,
            std::min<uint64_t>(this->prefetch_depth_code_ * 64, one_bit_code_size_));
    }

    void
    prefetch_supplement(InnerIdType id) {
        this->supplement_io_->Prefetch(
            static_cast<uint64_t>(id) * supplement_code_size_,
            std::min<uint64_t>(this->prefetch_depth_code_ * 64, supplement_code_size_));
    }

    void
    prefetch_full_code(InnerIdType id) {
        this->prefetch_one_bit(id);
        this->prefetch_supplement(id);
    }

    const uint8_t*
    get_one_bit_code(InnerIdType id, bool& need_release) const {
        return this->one_bit_io_->Read(
            one_bit_code_size_, static_cast<uint64_t>(id) * one_bit_code_size_, need_release);
    }

    void
    release_one_bit_code(const uint8_t* code, bool need_release) const {
        if (need_release and code != nullptr) {
            this->one_bit_io_->Release(code);
        }
    }

    const uint8_t*
    get_supplement_code(InnerIdType id, bool& need_release) const {
        return this->supplement_io_->Read(
            supplement_code_size_, static_cast<uint64_t>(id) * supplement_code_size_, need_release);
    }

    void
    release_supplement_code(const uint8_t* code, bool need_release) const {
        if (need_release and code != nullptr) {
            this->supplement_io_->Release(code);
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
            this->one_bit_io_->MultiRead(
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
            this->quantizer_->ComputeDistsWithOneBitLowerBoundBatch4(*computer,
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
                                                                     computed4);
            if (not computed1) {
                this->compute_full_dist_after_one_bit_failure(
                    idx[i], code1, computer, result_dists + i, lower_bound1);
            }
            if (not computed2) {
                this->compute_full_dist_after_one_bit_failure(
                    idx[i + 1], code2, computer, result_dists + i + 1, lower_bound2);
            }
            if (not computed3) {
                this->compute_full_dist_after_one_bit_failure(
                    idx[i + 2], code3, computer, result_dists + i + 2, lower_bound3);
            }
            if (not computed4) {
                this->compute_full_dist_after_one_bit_failure(
                    idx[i + 3], code4, computer, result_dists + i + 3, lower_bound4);
            }
        }

        for (; i < id_count; ++i) {
            auto* lower_bound = lower_bounds == nullptr ? nullptr : lower_bounds + i;
            const auto* one_bit_code = one_bit_codes.data + i * one_bit_code_size_;
            bool computed = this->quantizer_->ComputeDistWithOneBitLowerBound(
                *computer, one_bit_code, result_dists + i, lower_bound);
            if (not computed) {
                this->compute_full_dist_after_one_bit_failure(
                    idx[i], one_bit_code, computer, result_dists + i, lower_bound);
            }
        }
    }

    void
    compute_full_dist_after_one_bit_failure(InnerIdType id,
                                            const uint8_t* one_bit_code,
                                            Computer<RaBitQuantizer<metric>>* computer,
                                            float* result_dist,
                                            float* lower_bound) const {
        bool supplement_need_release = false;
        const uint8_t* supplement_code = nullptr;
        try {
            supplement_code = this->get_supplement_code(id, supplement_need_release);
            this->compute_full_dist(one_bit_code, supplement_code, computer, result_dist);
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
                      float* result_dist) const {
        if (not this->quantizer_->ComputeDistWithSplitCode(
                *computer, one_bit_code, supplement_code, result_dist)) {
            ByteBuffer full_code(this->code_size_, allocator_);
            this->quantizer_->MergeSplitCode(one_bit_code, supplement_code, full_code.data);
            computer->ComputeDist(full_code.data, result_dist);
        }
    }

    void
    compute_full_dist(InnerIdType id,
                      Computer<RaBitQuantizer<metric>>* computer,
                      float* result_dist) const {
        bool one_bit_need_release = false;
        bool supplement_need_release = false;
        const auto* one_bit_code = this->get_one_bit_code(id, one_bit_need_release);
        const auto* supplement_code = this->get_supplement_code(id, supplement_need_release);
        try {
            this->compute_full_dist(one_bit_code, supplement_code, computer, result_dist);
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
