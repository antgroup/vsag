
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

#include "flatten_interface.h"
#include "io/basic_io.h"
#include "quantization/quantizer.h"
namespace vsag {
/*
* thread unsafe
*/
template <typename QuantTmpl, typename IOTmpl>
class FlattenDataCell : public FlattenInterface {
public:
    FlattenDataCell() = default;

    explicit FlattenDataCell(const JsonType& quantization_param,
                             const JsonType& io_param,
                             const IndexCommonParam& common_param);

    void
    Query(float* result_dists,
          ComputerInterfacePtr computer,
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
    Prefetch(InnerIdType id) override {
        io_->Prefetch(id * code_size_);
    };

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override;

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
          std::shared_ptr<Computer<QuantTmpl>> computer,
          const InnerIdType* idx,
          InnerIdType id_count);

    ComputerInterfacePtr
    factory_computer(const float* query) {
        auto computer = this->quantizer_->FactoryComputer();
        computer->SetQuery(query);
        return computer;
    }
};

template <typename QuantTmpl, typename IOTmpl>
FlattenDataCell<QuantTmpl, IOTmpl>::FlattenDataCell(const JsonType& quantization_param,
                                                    const JsonType& io_param,
                                                    const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_) {
    this->quantizer_ = std::make_shared<QuantTmpl>(quantization_param, common_param);
    this->io_ = std::make_shared<IOTmpl>(io_param, common_param);
    this->code_size_ = quantizer_->GetCodeSize();
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
    if (idx == std::numeric_limits<InnerIdType>::max()) {
        idx = total_count_;
        ++total_count_;
    } else {
        total_count_ = std::max(total_count_, idx + 1);
    }

    auto* codes = reinterpret_cast<uint8_t*>(allocator_->Allocate(code_size_));
    quantizer_->EncodeOne(vector, codes);
    io_->Write(codes, code_size_, static_cast<uint64_t>(idx) * static_cast<uint64_t>(code_size_));
    allocator_->Deallocate(codes);
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::BatchInsertVector(const float* vectors,
                                                      InnerIdType count,
                                                      InnerIdType* idx) {
    if (idx == nullptr) {
        auto* codes = reinterpret_cast<uint8_t*>(
            allocator_->Allocate(static_cast<uint64_t>(count) * static_cast<uint64_t>(code_size_)));
        quantizer_->EncodeBatch(vectors, codes, count);
        io_->Write(codes,
                   static_cast<uint64_t>(count) * static_cast<uint64_t>(code_size_),
                   static_cast<uint64_t>(total_count_) * static_cast<uint64_t>(code_size_));
        total_count_ += count;
        allocator_->Deallocate(codes);
    } else {
        auto dim = quantizer_->GetDim();
        for (int64_t i = 0; i < count; ++i) {
            this->InsertVector(vectors + dim * i, idx[i]);
        }
    }
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
                                          std::shared_ptr<Computer<QuantTmpl>> computer,
                                          const InnerIdType* idx,
                                          InnerIdType id_count) {
    for (int64_t i = 0; i < id_count; ++i) {
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
        io_->Release(codes1);
    }
    if (release2) {
        io_->Release(codes2);
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
FlattenDataCell<QuantTmpl, IOTmpl>::Deserialize(StreamReader& reader) {
    FlattenInterface::Deserialize(reader);
    this->io_->Deserialize(reader);
    this->quantizer_->Deserialize(reader);
}
}  // namespace vsag
