
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
          InnerIdType id_count) override {
        auto comp = std::static_pointer_cast<Computer<QuantTmpl>>(computer);
        this->query(result_dists, comp, idx, id_count);
    }

    ComputerInterfacePtr
    FactoryComputer(const float* query) override {
        return this->factory_computer(query);
    }

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override;

    void
    Train(const float* data, uint64_t count) override;

    void
    InsertVector(const float* vector, InnerIdType idx) override;

    void
    BatchInsertVector(const float* vectors, InnerIdType count, InnerIdType* idx) override;

    void
    SetMaxCapacity(InnerIdType capacity) override {
        this->max_capacity_ = std::max(capacity, this->total_count_);  // TODO(LHT): add warning
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
        if (force_in_memory_) {
            this->force_in_memory_io_->Write(&end_flag, 1, io_size);
        }
    }

    void
    Prefetch(InnerIdType id) override {
        if (this->force_in_memory_) {
            force_in_memory_io_->Prefetch(id * code_size_, code_size_);
        } else {
            io_->Prefetch(id * code_size_, code_size_);
        }
    };

    [[nodiscard]] std::string
    GetQuantizerName() override;

    [[nodiscard]] MetricType
    GetMetricType() override;

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override;

    [[nodiscard]] bool
    InMemory() const override;

    void
    EnableForceInMemory() override;

    void
    DisableForceInMemory() override;

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(StreamReader& reader) override;

    inline void
    SetQuantizer(std::shared_ptr<Quantizer<QuantTmpl>> quantizer) {
        this->quantizer_ = quantizer;
        this->code_size_ = quantizer_->GetCodeSize();
    }

    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>> io) {
        this->io_ = io;
    }

public:
    std::shared_ptr<Quantizer<QuantTmpl>> quantizer_{nullptr};
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    Allocator* const allocator_{nullptr};

private:
    inline void
    query(float* result_dists,
          const float* query_vector,
          const InnerIdType* idx,
          InnerIdType id_count);

    inline void
    query(float* result_dists,
          const std::shared_ptr<Computer<QuantTmpl>>& computer,
          const InnerIdType* idx,
          InnerIdType id_count);

    ComputerInterfacePtr
    factory_computer(const float* query) {
        auto computer = this->quantizer_->FactoryComputer();
        computer->SetQuery(query);
        return computer;
    }

    void
    trans_from_memory_io();

private:
    bool force_in_memory_{false};

    std::shared_ptr<MemoryBlockIO> force_in_memory_io_{};
};

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::trans_from_memory_io() {
    int64_t max_size = static_cast<int64_t>(this->code_size_) * this->total_count_;
    constexpr uint64_t block_size = 1024 * 1024;
    uint64_t offset = 0;
    bool need_release = false;
    while (max_size > block_size) {
        auto data = this->force_in_memory_io_->Read(block_size, offset, need_release);
        this->io_->Write(data, block_size, offset);
        max_size -= block_size;
        offset += block_size;
        if (need_release) {
            this->force_in_memory_io_->Release(data);
        }
    }
    if (max_size > 0) {
        auto data = this->force_in_memory_io_->Read(max_size, offset, need_release);
        this->io_->Write(data, max_size, offset);
        if (need_release) {
            this->force_in_memory_io_->Release(data);
        }
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::EnableForceInMemory() {
    if (this->TotalCount() != 0) {
        throw std::runtime_error("EnableForceInMemory must with empty flatten datacell");
    }
    this->force_in_memory_ = true;
}
template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::DisableForceInMemory() {
    this->force_in_memory_ = false;
    this->trans_from_memory_io();
}

template <typename QuantTmpl, typename IOTmpl>
FlattenDataCell<QuantTmpl, IOTmpl>::FlattenDataCell(const QuantizerParamPtr& quantization_param,
                                                    const IOParamPtr& io_param,
                                                    const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()) {
    this->quantizer_ = std::make_shared<QuantTmpl>(quantization_param, common_param);
    this->io_ = std::make_shared<IOTmpl>(io_param, common_param);
    this->code_size_ = quantizer_->GetCodeSize();
    this->force_in_memory_io_ =
        std::make_shared<MemoryBlockIO>(allocator_, Options::Instance().block_size_limit());
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::Train(const float* data, uint64_t count) {
    if (this->quantizer_) {
        this->quantizer_->Train(data, count);
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::InsertVector(const float* vector, InnerIdType idx) {
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
    quantizer_->EncodeOne(vector, codes.data);
    if (this->force_in_memory_) {
        force_in_memory_io_->Write(
            codes.data, code_size_, static_cast<uint64_t>(idx) * static_cast<uint64_t>(code_size_));
    } else {
        io_->Write(
            codes.data, code_size_, static_cast<uint64_t>(idx) * static_cast<uint64_t>(code_size_));
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::BatchInsertVector(const float* vectors,
                                                      InnerIdType count,
                                                      InnerIdType* idx) {
    if (idx == nullptr) {
        ByteBuffer codes(static_cast<uint64_t>(count) * static_cast<uint64_t>(code_size_),
                         allocator_);
        quantizer_->EncodeBatch(vectors, codes.data, count);
        uint64_t cur_count;
        {
            std::lock_guard lock(mutex_);
            cur_count = total_count_;
            total_count_ += count;
        }
        if (this->force_in_memory_) {
            force_in_memory_io_->Write(
                codes.data,
                static_cast<uint64_t>(count) * static_cast<uint64_t>(code_size_),
                cur_count * static_cast<uint64_t>(code_size_));
        } else {
            io_->Write(codes.data,
                       static_cast<uint64_t>(count) * static_cast<uint64_t>(code_size_),
                       cur_count * static_cast<uint64_t>(code_size_));
        }
    } else {
        auto dim = quantizer_->GetDim();
        for (int64_t i = 0; i < count; ++i) {
            this->InsertVector(vectors + dim * i, idx[i]);
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
                                          const float* query_vector,
                                          const InnerIdType* idx,
                                          InnerIdType id_count) {
    auto computer = quantizer_->FactoryComputer();
    computer->SetQuery(query_vector);
    this->Query(result_dists, computer, idx, id_count);
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::query(float* result_dists,
                                          const std::shared_ptr<Computer<QuantTmpl>>& computer,
                                          const InnerIdType* idx,
                                          InnerIdType id_count) {
    for (uint32_t i = 0; i < this->prefetch_jump_code_size_ and i < id_count; i++) {
        if (force_in_memory_) {
            this->force_in_memory_io_->Prefetch(
                static_cast<uint64_t>(idx[i]) * static_cast<uint64_t>(code_size_),
                this->code_size_);
        } else {
            this->io_->Prefetch(static_cast<uint64_t>(idx[i]) * static_cast<uint64_t>(code_size_),
                                this->code_size_);
        }
    }
    if (not force_in_memory_ and not this->io_->InMemory() and id_count > 1) {
        ByteBuffer codes(id_count * this->code_size_, allocator_);
        Vector<uint64_t> sizes(id_count, this->code_size_, allocator_);
        Vector<uint64_t> offsets(id_count, this->code_size_, allocator_);
        for (int64_t i = 0; i < id_count; ++i) {
            offsets[i] = idx[i] * code_size_;
        }
        this->io_->MultiRead(codes.data, sizes.data(), offsets.data(), id_count);
        computer->ComputeBatchDists(id_count, codes.data, result_dists);
        return;
    }

    for (int64_t i = 0; i < id_count; ++i) {
        if (i + this->prefetch_jump_code_size_ < id_count) {
            if (force_in_memory_) {
                this->force_in_memory_io_->Prefetch(
                    static_cast<uint64_t>(idx[i + this->prefetch_jump_code_size_]) *
                        static_cast<uint64_t>(code_size_),
                    this->code_size_);
            } else {
                this->io_->Prefetch(static_cast<uint64_t>(idx[i + this->prefetch_jump_code_size_]) *
                                        static_cast<uint64_t>(code_size_),
                                    this->code_size_);
            }
        }

        bool release = false;
        const auto* codes = this->GetCodesById(idx[i], release);
        computer->ComputeDist(codes, result_dists + i);
        if (release) {
            if (force_in_memory_) {
                this->force_in_memory_io_->Release(codes);
            } else {
                this->io_->Release(codes);
            }
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
        if (force_in_memory_) {
            this->force_in_memory_io_->Release(codes1);
        } else {
            this->io_->Release(codes1);
        }
    }
    if (release2) {
        if (force_in_memory_) {
            this->force_in_memory_io_->Release(codes2);
        } else {
            this->io_->Release(codes2);
        }
    }

    return result;
}

template <typename QuantTmpl, typename IOTmpl>
const uint8_t*
FlattenDataCell<QuantTmpl, IOTmpl>::GetCodesById(InnerIdType id, bool& need_release) const {
    if (force_in_memory_) {
        return force_in_memory_io_->Read(
            code_size_,
            static_cast<uint64_t>(id) * static_cast<uint64_t>(code_size_),
            need_release);
    } else {
        return io_->Read(code_size_,
                         static_cast<uint64_t>(id) * static_cast<uint64_t>(code_size_),
                         need_release);
    }
}

template <typename QuantTmpl, typename IOTmpl>
bool
FlattenDataCell<QuantTmpl, IOTmpl>::GetCodesById(InnerIdType id, uint8_t* codes) const {
    if (force_in_memory_) {
        return force_in_memory_io_->Read(
            code_size_, static_cast<uint64_t>(id) * static_cast<uint64_t>(code_size_), codes);
    } else {
        return io_->Read(
            code_size_, static_cast<uint64_t>(id) * static_cast<uint64_t>(code_size_), codes);
    }
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
FlattenDataCell<QuantTmpl, IOTmpl>::Deserialize(StreamReader& reader) {
    FlattenInterface::Deserialize(reader);
    this->io_->Deserialize(reader);
    this->quantizer_->Deserialize(reader);
}
}  // namespace vsag
