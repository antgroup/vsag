
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
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include "common.h"
#include "flatten_interface.h"
#include "index_common_param.h"
#include "layout/fixed_layout.h"
#include "quantization/quantizer.h"
#include "query_context.h"
#include "utils/byte_buffer.h"
#include "utils/timer.h"

namespace vsag {
/*
 * Most mutations are thread unsafe. Concurrent InsertVector is supported only for pre-resized
 * in-memory layouts when callers write distinct IDs.
*/
template <typename QuantTmpl, typename LayoutTmpl>
class FlattenDataCell : public FlattenInterface {
public:
    FlattenDataCell() : layout_(std::make_shared<LayoutTmpl>()) {
    }

    explicit FlattenDataCell(const QuantizerParamPtr& quantization_param,
                             const IOParamPtr& io_param,
                             const IndexCommonParam& common_param);

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) override {
        auto comp = static_cast<Computer<QuantTmpl>*>(computer.get());
        this->query(result_dists, comp, idx, id_count, ctx);
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        return this->factory_computer(static_cast<const float*>(query));
    }

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override;

    void
    Train(const void* data, uint64_t count) override;

    void
    InsertVector(const void* vector, InnerIdType idx) override;

    bool
    UpdateVector(const void* vector,
                 InnerIdType idx = std::numeric_limits<InnerIdType>::max()) override;

    void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec) override;

    bool
    Decode(const uint8_t* codes, float* data) override {
        return this->quantizer_->DecodeOne(codes, data);
    }

    bool
    Encode(const float* data, uint8_t* codes) override {
        return this->quantizer_->EncodeOne(data, codes);
    }

    void
    Resize(InnerIdType new_capacity) override {
        if (new_capacity <= this->max_capacity_) {
            return;
        }
        this->layout_->Resize(new_capacity);
        this->max_capacity_ = new_capacity;
    }

    void
    Prefetch(InnerIdType id) override {
        layout_->Prefetch(id, code_size_);
    };

    void
    ExportModel(const FlattenInterfacePtr& other) const override {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        this->quantizer_->Serialize(writer);
        ss.seekg(0, std::ios::beg);
        IOStreamReader reader(ss);
        auto ptr = std::dynamic_pointer_cast<FlattenDataCell<QuantTmpl, LayoutTmpl>>(other);
        if (ptr == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Export model's flatten datacell failed");
        }
        ptr->quantizer_->Deserialize(reader);
    }

    void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) override;

    void
    Move(InnerIdType from, InnerIdType to) override;

    void
    ShrinkToFit(InnerIdType capacity) override {
        this->layout_->Shrink(capacity);
        this->max_capacity_ = capacity;
    }

    [[nodiscard]] std::string
    GetQuantizerName() override;

    [[nodiscard]] MetricType
    GetMetricType() override;

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override;

    void
    Release(const uint8_t* data) const override;

    [[nodiscard]] bool
    InMemory() const override;

    [[nodiscard]] bool
    SupportConcurrentInsertAfterResize() const override {
        if constexpr (not LayoutTmpl::InMemory) {
            return false;
        }
        const auto quantizer_name = this->quantizer_->Name();
        return quantizer_name == QUANTIZATION_TYPE_VALUE_FP32 or
               quantizer_name == QUANTIZATION_TYPE_VALUE_RABITQ or
               quantizer_name == QUANTIZATION_TYPE_VALUE_SQ8;
    }

    bool
    HoldMolds() const override;

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override;

    [[nodiscard]] const float*
    TryGetContiguousRawFloatData(uint64_t* row_stride = nullptr) override {
        if (row_stride != nullptr) {
            *row_stride = 0;
        }
        if (this->GetQuantizerName() != QUANTIZATION_TYPE_VALUE_FP32) {
            return nullptr;
        }
        if constexpr (LayoutTmpl::InMemory) {
            const auto* data = this->layout_->TryGetContiguousData();
            if (data == nullptr) {
                return nullptr;
            }
            if (row_stride != nullptr) {
                *row_stride = this->code_size_ / sizeof(float);
            }
            return reinterpret_cast<const float*>(data);
        }
        return nullptr;
    }

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

    inline void
    SetQuantizer(std::shared_ptr<Quantizer<QuantTmpl>> quantizer) {
        this->quantizer_ = quantizer;
        this->code_size_ = quantizer_->GetCodeSize();
        this->layout_->SetCodeSize(this->code_size_);
        this->backend_ =
            QuantizerDistanceBackend<QuantTmpl>::Get(static_cast<const QuantTmpl&>(*quantizer_));
    }

    inline void
    SetIO(std::shared_ptr<BasicIO<typename LayoutTmpl::IOType>> io) {
        this->layout_->SetIO(std::move(io));
    }

    void
    InitIO(const IOParamPtr& io_param) override {
        this->layout_->InitIO(io_param);
    }

    IndexCommonParam
    ExportCommonParam() override {
        return common_param_;
    }

    uint64_t
    GetMemoryUsage() const override;

public:
    IndexCommonParam common_param_;

    std::shared_ptr<Quantizer<QuantTmpl>> quantizer_{nullptr};
    std::shared_ptr<LayoutTmpl> layout_{nullptr};

    Allocator* const allocator_{nullptr};

private:
    inline void
    query(float* result_dists,
          Computer<QuantTmpl>* computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx);

    ComputerInterfacePtr
    factory_computer(const float* query) {
        auto computer = this->quantizer_->FactoryComputer();
        computer->SetQuery(query);
        return computer;
    }
};

template <typename QuantTmpl, typename LayoutTmpl>
void
FlattenDataCell<QuantTmpl, LayoutTmpl>::Release(const uint8_t* data) const {
    this->layout_->Release(data);
}

template <typename QuantTmpl, typename LayoutTmpl>
bool
FlattenDataCell<QuantTmpl, LayoutTmpl>::HoldMolds() const {
    return this->quantizer_->HoldMolds();
}

template <typename QuantTmpl, typename LayoutTmpl>
FlattenDataCell<QuantTmpl, LayoutTmpl>::FlattenDataCell(const QuantizerParamPtr& quantization_param,
                                                        const IOParamPtr& io_param,
                                                        const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()) {
    this->common_param_ = common_param;
    this->quantizer_ = std::make_shared<QuantTmpl>(quantization_param, common_param);
    this->code_size_ = quantizer_->GetCodeSize();
    this->layout_ = std::make_shared<LayoutTmpl>(this->code_size_, io_param, common_param);
    this->backend_ =
        QuantizerDistanceBackend<QuantTmpl>::Get(static_cast<const QuantTmpl&>(*quantizer_));
}

template <typename QuantTmpl, typename LayoutTmpl>
void
FlattenDataCell<QuantTmpl, LayoutTmpl>::Train(const void* data, uint64_t count) {
    if (this->quantizer_) {
        this->quantizer_->Train(static_cast<const float*>(data), count);
    }
}

template <typename QuantTmpl, typename LayoutTmpl>
void
FlattenDataCell<QuantTmpl, LayoutTmpl>::InsertVector(const void* vector, InnerIdType idx) {
    {
        std::lock_guard lock(mutex_);
        if (idx == std::numeric_limits<InnerIdType>::max()) {
            idx = total_count_;
            ++total_count_;
        } else {
            total_count_ = std::max(total_count_, idx + 1);
        }
    }
    ByteBuffer codes(static_cast<uint64_t>(code_size_), allocator_);
    quantizer_->EncodeOne(static_cast<const float*>(vector), codes.data);
    layout_->Write(idx, codes.data);
}

template <typename QuantTmpl, typename LayoutTmpl>
bool
FlattenDataCell<QuantTmpl, LayoutTmpl>::UpdateVector(const void* vector, InnerIdType idx) {
    if (idx >= total_count_) {
        return false;
    }
    std::lock_guard lock(mutex_);
    ByteBuffer codes(static_cast<uint64_t>(code_size_), allocator_);
    quantizer_->EncodeOne(static_cast<const float*>(vector), codes.data);
    layout_->Write(idx, codes.data);
    return true;
}

template <typename QuantTmpl, typename LayoutTmpl>
void
FlattenDataCell<QuantTmpl, LayoutTmpl>::BatchInsertVector(const void* vectors,
                                                          InnerIdType count,
                                                          InnerIdType* idx_vec) {
    if (idx_vec == nullptr) {
        ByteBuffer codes(static_cast<uint64_t>(count) * static_cast<uint64_t>(code_size_),
                         allocator_);
        quantizer_->EncodeBatch(static_cast<const float*>(vectors), codes.data, count);
        uint64_t cur_count;
        {
            std::lock_guard lock(mutex_);
            cur_count = total_count_;
            total_count_ += count;
        }
        layout_->WriteRange(cur_count, codes.data, count);
    } else {
        auto dim = quantizer_->GetDim();
        for (int64_t i = 0; i < count; ++i) {
            this->InsertVector(static_cast<const float*>(vectors) + dim * i, idx_vec[i]);
        }
    }
}

template <typename QuantTmpl, typename LayoutTmpl>
std::string
FlattenDataCell<QuantTmpl, LayoutTmpl>::GetQuantizerName() {
    return this->quantizer_->Name();
}

template <typename QuantTmpl, typename LayoutTmpl>
MetricType
FlattenDataCell<QuantTmpl, LayoutTmpl>::GetMetricType() {
    return this->quantizer_->Metric();
}

template <typename QuantTmpl, typename LayoutTmpl>
bool
FlattenDataCell<QuantTmpl, LayoutTmpl>::InMemory() const {
    return LayoutTmpl::InMemory;
}

template <typename QuantTmpl, typename LayoutTmpl>
void
FlattenDataCell<QuantTmpl, LayoutTmpl>::query(float* result_dists,
                                              Computer<QuantTmpl>* computer,
                                              const InnerIdType* idx,
                                              InnerIdType id_count,
                                              QueryContext* ctx) {
    Allocator* search_alloc = select_query_allocator(ctx, allocator_);

    for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; i++) {
        this->layout_->Prefetch(idx[i], this->prefetch_depth_code_ * 64);
    }
    if constexpr (not LayoutTmpl::InMemory) {
        if (id_count > 1) {
            ByteBuffer codes(
                static_cast<uint64_t>(id_count) * static_cast<uint64_t>(this->code_size_),
                search_alloc);
            double io_cost_ms = 0.0F;
            {
                Timer timer(io_cost_ms);
                this->layout_->MultiRead(idx, id_count, codes.data, search_alloc);
            }

            if (ctx != nullptr and ctx->stats != nullptr) {
                ctx->stats->io_cnt.fetch_add(id_count, std::memory_order_relaxed);
                ctx->stats->io_time_ms.fetch_add(static_cast<uint32_t>(io_cost_ms),
                                                 std::memory_order_relaxed);
            }

            computer->ScanBatchDists(id_count, codes.data, result_dists);
            if (ctx != nullptr and ctx->stats != nullptr and ctx->track_distance_evaluations)
                ctx->stats->AddDistance(ctx->distance_phase, backend_, id_count);
            return;
        }

        if (ctx != nullptr and ctx->stats != nullptr and id_count > 0) {
            ctx->stats->io_cnt.fetch_add(static_cast<uint32_t>(id_count),
                                         std::memory_order_relaxed);
        }
    }

    memset(result_dists, 0, sizeof(float) * id_count);
    int64_t i = 0;
    for (; i + 3 < id_count; i += 4) {
        for (int64_t j = 0; j < 4; ++j) {
            if (i + j + this->prefetch_stride_code_ < id_count) {
                this->layout_->Prefetch(idx[i + j + this->prefetch_stride_code_],
                                        this->prefetch_depth_code_ * 64);
            }
        }
        bool release1 = false, release2 = false, release3 = false, release4 = false;
        const uint8_t* codes1 = nullptr;
        const uint8_t* codes2 = nullptr;
        const uint8_t* codes3 = nullptr;
        const uint8_t* codes4 = nullptr;
        auto release_batch = [&]() {
            if (release1 && codes1) {
                this->layout_->Release(codes1);
            }
            if (release2 && codes2) {
                this->layout_->Release(codes2);
            }
            if (release3 && codes3) {
                this->layout_->Release(codes3);
            }
            if (release4 && codes4) {
                this->layout_->Release(codes4);
            }
        };
        try {
            codes1 = this->GetCodesById(idx[i], release1);
            codes2 = this->GetCodesById(idx[i + 1], release2);
            codes3 = this->GetCodesById(idx[i + 2], release3);
            codes4 = this->GetCodesById(idx[i + 3], release4);
            computer->ComputeDistsBatch4(codes1,
                                         codes2,
                                         codes3,
                                         codes4,
                                         result_dists[i],
                                         result_dists[i + 1],
                                         result_dists[i + 2],
                                         result_dists[i + 3]);
        } catch (...) {
            release_batch();
            throw;
        }
        release_batch();
    }
    for (; i < id_count; ++i) {
        bool release = false;
        const uint8_t* codes = nullptr;
        try {
            codes = this->GetCodesById(idx[i], release);
            computer->ComputeDist(codes, result_dists + i);
        } catch (...) {
            if (release && codes) {
                this->layout_->Release(codes);
            }
            throw;
        }
        if (release && codes) {
            this->layout_->Release(codes);
        }
    }
    if (ctx != nullptr and ctx->stats != nullptr and ctx->track_distance_evaluations)
        ctx->stats->AddDistance(ctx->distance_phase, backend_, static_cast<uint64_t>(id_count));
}

template <typename QuantTmpl, typename LayoutTmpl>
float
FlattenDataCell<QuantTmpl, LayoutTmpl>::ComputePairVectors(InnerIdType id1, InnerIdType id2) {
    bool release1 = false, release2 = false;
    const uint8_t* codes1 = nullptr;
    const uint8_t* codes2 = nullptr;
    auto release_pair = [&]() {
        if (release1 && codes1) {
            this->layout_->Release(codes1);
        }
        if (release2 && codes2) {
            this->layout_->Release(codes2);
        }
    };
    try {
        codes1 = this->GetCodesById(id1, release1);
        codes2 = this->GetCodesById(id2, release2);
        auto result = this->quantizer_->Compute(codes1, codes2);
        release_pair();
        return result;
    } catch (...) {
        release_pair();
        throw;
    }
}

template <typename QuantTmpl, typename LayoutTmpl>
const uint8_t*
FlattenDataCell<QuantTmpl, LayoutTmpl>::GetCodesById(InnerIdType id, bool& need_release) const {
    return layout_->Read(id, need_release);
}

template <typename QuantTmpl, typename LayoutTmpl>
bool
FlattenDataCell<QuantTmpl, LayoutTmpl>::GetCodesById(InnerIdType id, uint8_t* codes) const {
    return layout_->Read(id, codes);
}

template <typename QuantTmpl, typename LayoutTmpl>
void
FlattenDataCell<QuantTmpl, LayoutTmpl>::Serialize(StreamWriter& writer) {
    FlattenInterface::Serialize(writer);
    this->layout_->Serialize(writer);
    this->quantizer_->Serialize(writer);
}

template <typename QuantTmpl, typename LayoutTmpl>
void
FlattenDataCell<QuantTmpl, LayoutTmpl>::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    FlattenInterface::Deserialize(reader);
    this->layout_->SetCodeSize(this->code_size_);
    this->layout_->Deserialize(reader);
    this->quantizer_->Deserialize(reader);
    this->backend_ =
        QuantizerDistanceBackend<QuantTmpl>::Get(static_cast<const QuantTmpl&>(*this->quantizer_));
}

template <typename QuantTmpl, typename LayoutTmpl>
void
FlattenDataCell<QuantTmpl, LayoutTmpl>::MergeOther(const FlattenInterfacePtr& other,
                                                   InnerIdType bias) {
    auto ptr = std::dynamic_pointer_cast<FlattenDataCell<QuantTmpl, LayoutTmpl>>(other);
    if (ptr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "Merge flatten datacell failed: not match type");
    }
    constexpr uint64_t BUFFER_SIZE = 1024 * 1024 * 10;
    uint64_t total_count = ptr->total_count_;
    uint64_t read_count = 0;
    while (read_count < total_count) {
        bool need_release = false;
        uint64_t count = std::min(BUFFER_SIZE / this->code_size_, total_count - read_count);
        auto* buffer = ptr->layout_->ReadRange(read_count, count, need_release);
        this->layout_->WriteRange(bias + read_count, buffer, count);
        if (need_release) {
            ptr->layout_->Release(buffer);
        }
        read_count += count;
    }
    this->total_count_ += total_count;
}

template <typename QuantTmpl, typename LayoutTmpl>
uint64_t
FlattenDataCell<QuantTmpl, LayoutTmpl>::GetMemoryUsage() const {
    uint64_t memory = sizeof(FlattenDataCell<QuantTmpl, LayoutTmpl>);
    if (LayoutTmpl::InMemory) {
        memory += this->layout_->GetMemoryUsage();
    }
    memory += sizeof(QuantTmpl);
    return memory;
}

template <typename QuantTmpl, typename LayoutTmpl>
void
FlattenDataCell<QuantTmpl, LayoutTmpl>::Move(InnerIdType from, InnerIdType to) {
    this->layout_->Move(from, to);
}

}  // namespace vsag
