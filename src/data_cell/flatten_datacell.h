
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
#include <memory>

#include "byte_buffer.h"
#include "common.h"
#include "flatten_interface.h"
#include "io/basic_io.h"
#include "io/memory_block_io.h"
#include "quantization/quantizer.h"

namespace vsag {
/*
* thread unsafe
*/
template <typename QuantTmpl, typename IOTmpl>
class FlattenDataCell : public FlattenInterface {
public:
    FlattenDataCell() = default;

    explicit FlattenDataCell(const QuantizerParamPtr& quantization_param,
                             const IOParamPtr& io_param,
                             const IndexCommonParam& common_param);

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          Allocator* allocator = nullptr) override {
        auto comp = std::static_pointer_cast<Computer<QuantTmpl>>(computer);
        this->query(result_dists, comp, idx, id_count, allocator);
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        return this->factory_computer((const float*)query);
    }

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override;

    void
    Train(const void* data, uint64_t count) override;

    void
    InsertVector(const void* vector, InnerIdType idx) override;

    void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec) override;

    bool
    Decode(const uint8_t* codes, DataType* data) override {
        return this->quantizer_->DecodeOne(codes, data);
    }

    void
    Resize(InnerIdType new_capacity) override {
        if (new_capacity <= this->max_capacity_) {
            return;
        }
        this->max_capacity_ = new_capacity;
        uint64_t io_size = static_cast<uint64_t>(new_capacity) * static_cast<uint64_t>(code_size_);
        uint8_t end_flag =
            127;  // the value is meaningless, only to occupy the position for io allocate
        this->io_->Write(&end_flag, 1, io_size);
    }

    void
    Prefetch(InnerIdType id) override {
        io_->Prefetch(id * code_size_, code_size_);
    };

    void
    ExportModel(const FlattenInterfacePtr& other) const override {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        this->quantizer_->Serialize(writer);
        ss.seekg(0, std::ios::beg);
        IOStreamReader reader(ss);
        auto ptr = std::dynamic_pointer_cast<FlattenDataCell<QuantTmpl, IOTmpl>>(other);
        if (ptr == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Export model's flatten datacell failed");
        }
        ptr->quantizer_->Deserialize(reader);
    }

    void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) override;

    [[nodiscard]] std::string
    GetQuantizerName() override;

    [[nodiscard]] MetricType
    GetMetricType() override;

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override;

    [[nodiscard]] bool
    InMemory() const override;

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

    inline void
    SetQuantizer(std::shared_ptr<Quantizer<QuantTmpl>> quantizer) {
        this->quantizer_ = quantizer;
        this->code_size_ = quantizer_->GetCodeSize();
    }

    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>> io) {
        this->io_ = io;
    }

    void
    InitIO(const IOParamPtr& io_param) override {
        this->io_->InitIO(io_param);
    }

public:
    std::shared_ptr<Quantizer<QuantTmpl>> quantizer_{nullptr};
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    Allocator* const allocator_{nullptr};

private:
    inline void
    query(float* result_dists,
          const std::shared_ptr<Computer<QuantTmpl>>& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          Allocator* allocator);

    ComputerInterfacePtr
    factory_computer(const float* query) {
        auto computer = this->quantizer_->FactoryComputer();
        computer->SetQuery(query);
        return computer;
    }
};

template <typename QuantTmpl, typename IOTmpl>
FlattenDataCell<QuantTmpl, IOTmpl>::FlattenDataCell(const QuantizerParamPtr& quantization_param,
                                                    const IOParamPtr& io_param,
                                                    const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()) {
    this->quantizer_ = std::make_shared<QuantTmpl>(quantization_param, common_param);
    this->io_ = std::make_shared<IOTmpl>(io_param, common_param);
    this->code_size_ = quantizer_->GetCodeSize();
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::Train(const void* data, uint64_t count) {
    if (this->quantizer_) {
        this->quantizer_->Train((const float*)data, count);
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::InsertVector(const void* vector, InnerIdType idx) {
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
    quantizer_->EncodeOne((const float*)vector, codes.data);
    io_->Write(
        codes.data, code_size_, static_cast<uint64_t>(idx) * static_cast<uint64_t>(code_size_));
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::BatchInsertVector(const void* vectors,
                                                      InnerIdType count,
                                                      InnerIdType* idx_vec) {
    if (idx_vec == nullptr) {
        ByteBuffer codes(static_cast<uint64_t>(count) * static_cast<uint64_t>(code_size_),
                         allocator_);
        quantizer_->EncodeBatch((const float*)vectors, codes.data, count);
        uint64_t cur_count;
        {
            std::lock_guard lock(mutex_);
            cur_count = total_count_;
            total_count_ += count;
        }
        io_->Write(codes.data,
                   static_cast<uint64_t>(count) * static_cast<uint64_t>(code_size_),
                   cur_count * static_cast<uint64_t>(code_size_));
    } else {
        auto dim = quantizer_->GetDim();
        for (int64_t i = 0; i < count; ++i) {
            this->InsertVector((const float*)vectors + dim * i, idx_vec[i]);
        }
    }
}

template <typename QuantTmpl, typename IOTmpl>
std::string
FlattenDataCell<QuantTmpl, IOTmpl>::GetQuantizerName() {
    return this->quantizer_->Name();
}

template <typename QuantTmpl, typename IOTmpl>
MetricType
FlattenDataCell<QuantTmpl, IOTmpl>::GetMetricType() {
    return this->quantizer_->Metric();
}

template <typename QuantTmpl, typename IOTmpl>
bool
FlattenDataCell<QuantTmpl, IOTmpl>::InMemory() const {
    return this->io_->InMemory();
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::query(float* result_dists,
                                          const std::shared_ptr<Computer<QuantTmpl>>& computer,
                                          const InnerIdType* idx,
                                          InnerIdType id_count,
                                          Allocator* allocator) {
    Allocator* search_alloc = allocator == nullptr ? allocator_ : allocator;
    for (uint32_t i = 0; i < this->prefetch_stride_code_ and i < id_count; i++) {
        this->io_->Prefetch(static_cast<uint64_t>(idx[i]) * static_cast<uint64_t>(code_size_),
                            this->prefetch_depth_code_ * 64);
    }
    if (not this->io_->InMemory() and id_count > 1) {
        ByteBuffer codes(id_count * this->code_size_, search_alloc);
        Vector<uint64_t> sizes(id_count, this->code_size_, search_alloc);
        Vector<uint64_t> offsets(id_count, this->code_size_, search_alloc);
        for (int64_t i = 0; i < id_count; ++i) {
            offsets[i] = static_cast<uint64_t>(idx[i]) * this->code_size_;
        }
        this->io_->MultiRead(codes.data, sizes.data(), offsets.data(), id_count);
        computer->ScanBatchDists(id_count, codes.data, result_dists);
        return;
    }

    memset(result_dists, 0, sizeof(float) * id_count);
    int64_t i = 0;
    for (; i + 3 < id_count; i += 4) {
        for (int64_t j = 0; j < 4; ++j) {
            if (i + j + this->prefetch_stride_code_ < id_count) {
                this->io_->Prefetch(
                    static_cast<uint64_t>(idx[i + j + this->prefetch_stride_code_]) *
                        static_cast<uint64_t>(code_size_),
                    this->prefetch_depth_code_ * 64);
            }
        }
        bool release1 = false;
        const auto* codes1 = this->GetCodesById(idx[i], release1);
        bool release2 = false;
        const auto* codes2 = this->GetCodesById(idx[i + 1], release2);
        bool release3 = false;
        const auto* codes3 = this->GetCodesById(idx[i + 2], release3);
        bool release4 = false;
        const auto* codes4 = this->GetCodesById(idx[i + 3], release4);
        computer->ComputeDistsBatch4(codes1,
                                     codes2,
                                     codes3,
                                     codes4,
                                     result_dists[i],
                                     result_dists[i + 1],
                                     result_dists[i + 2],
                                     result_dists[i + 3]);

        if (release1) {
            this->io_->Release(codes1);
        }
        if (release2) {
            this->io_->Release(codes2);
        }
        if (release3) {
            this->io_->Release(codes3);
        }
        if (release4) {
            this->io_->Release(codes4);
        }
    }
    for (; i < id_count; ++i) {
        bool release = false;
        const auto* codes = this->GetCodesById(idx[i], release);
        computer->ComputeDist(codes, result_dists + i);
        if (release) {
            this->io_->Release(codes);
        }
    }
}

template <typename QuantTmpl, typename IOTmpl>
float
FlattenDataCell<QuantTmpl, IOTmpl>::ComputePairVectors(InnerIdType id1, InnerIdType id2) {
    bool release1, release2;
    const auto* codes1 = this->GetCodesById(id1, release1);
    const auto* codes2 = this->GetCodesById(id2, release2);
    auto result = this->quantizer_->Compute(codes1, codes2);
    if (release1) {
        this->io_->Release(codes1);
    }
    if (release2) {
        this->io_->Release(codes2);
    }

    return result;
}

template <typename QuantTmpl, typename IOTmpl>
const uint8_t*
FlattenDataCell<QuantTmpl, IOTmpl>::GetCodesById(InnerIdType id, bool& need_release) const {
    return io_->Read(
        code_size_, static_cast<uint64_t>(id) * static_cast<uint64_t>(code_size_), need_release);
}

template <typename QuantTmpl, typename IOTmpl>
bool
FlattenDataCell<QuantTmpl, IOTmpl>::GetCodesById(InnerIdType id, uint8_t* codes) const {
    return io_->Read(
        code_size_, static_cast<uint64_t>(id) * static_cast<uint64_t>(code_size_), codes);
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::Serialize(StreamWriter& writer) {
    FlattenInterface::Serialize(writer);
    this->io_->Serialize(writer);
    this->quantizer_->Serialize(writer);
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    FlattenInterface::Deserialize(reader);
    this->io_->Deserialize(reader);
    this->quantizer_->Deserialize(reader);
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) {
    auto ptr = std::dynamic_pointer_cast<FlattenDataCell<QuantTmpl, IOTmpl>>(other);
    if (ptr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "Merge flatten datacell failed: not match type");
    }
    constexpr uint64_t BUFFER_SIZE = 1024 * 1024 * 10;
    uint64_t total_count = ptr->total_count_;
    uint64_t offset = bias * code_size_;
    uint64_t read_count = 0;
    while (read_count < total_count) {
        bool need_release = false;
        uint64_t count = std::min(BUFFER_SIZE / this->code_size_, total_count - read_count);
        uint64_t size = count * this->code_size_;
        auto* buffer = ptr->io_->Read(size, read_count * this->code_size_, need_release);
        this->io_->Write(buffer, size, offset);
        if (need_release) {
            ptr->io_->Release(buffer);
        }
        offset += size;
        read_count += count;
    }
    this->total_count_ += total_count;
}
}  // namespace vsag
