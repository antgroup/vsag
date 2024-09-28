
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

template <typename QuantTmpl, typename IOTmpl>
class FlattenDataCell : public FlattenInterface {
public:
    FlattenDataCell() = default;

    explicit FlattenDataCell(const std::string& initializeJson);

    void
    Query(float* result_dists,
          std::shared_ptr<ComputerInterface> computer,
          const uint64_t* idx,
          uint64_t id_count) override {
        auto comp = std::static_pointer_cast<Computer<QuantTmpl>>(computer);
        this->query(result_dists, comp, idx, id_count);
    }

    std::shared_ptr<ComputerInterface>
    FactoryComputer(const float* query) override {
        return this->factory_computer(query);
    }

    float
    ComputePairVectors(uint64_t id1, uint64_t id2) override;

    void
    Train(const float* data, uint64_t count) override;

    void
    InsertVector(const float* vector, uint64_t idx) override;

    void
    BatchInsertVector(const float* vectors, uint64_t count, uint64_t* idx) override;

    void
    SetMaxCapacity(uint64_t capacity) override {
        this->max_capacity_ = std::max(capacity, this->total_count_);  // TODO(LHT): add warning
    }

    [[nodiscard]] uint64_t
    TotalCount() const override {
        return this->total_count_;
    }

    [[nodiscard]] const uint8_t*
    GetCodesById(uint64_t id) const override;

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

private:
    inline void
    query(float* result_dists, const float* query_vector, const uint64_t* idx, uint64_t id_count);

    inline void
    query(float* result_dists,
          std::shared_ptr<Computer<QuantTmpl>> computer,
          const uint64_t* idx,
          uint64_t id_count);

    std::shared_ptr<ComputerInterface>
    factory_computer(const float* query) {
        auto computer = this->quantizer_->FactoryComputer();
        computer->SetQuery(query);
        return computer;
    }
};

template <typename QuantTmpl, typename IOTmpl>
FlattenDataCell<QuantTmpl, IOTmpl>::FlattenDataCell(const std::string& initializeJson) {
    // TODO(LHT): implement initial function
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
FlattenDataCell<QuantTmpl, IOTmpl>::InsertVector(const float* vector, uint64_t idx) {
    if (idx == std::numeric_limits<uint64_t>::max()) {
        idx = total_count_;
        ++total_count_;
    }
    auto* codes = new uint8_t[code_size_];
    quantizer_->EncodeOne(vector, codes);
    io_->Write(codes, code_size_, idx * code_size_);
    delete[] codes;
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::BatchInsertVector(const float* vectors,
                                                      uint64_t count,
                                                      uint64_t* idx) {
    if (idx == nullptr) {
        auto* codes = new uint8_t[code_size_ * count];
        quantizer_->EncodeBatch(vectors, codes, count);
        io_->Write(codes, code_size_ * count, total_count_ * code_size_);
        total_count_ += count;
        delete[] codes;
    } else {
        uint64_t dim = quantizer_->GetDim();
        for (uint64_t i = 0; i < count; ++i) {
            this->InsertVector(vectors + dim * i, idx[i]);
        }
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::query(float* result_dists,
                                          const float* query_vector,
                                          const uint64_t* idx,
                                          uint64_t id_count) {
    auto computer = quantizer_->FactoryComputer();
    computer->SetQuery(query_vector);
    this->Query(result_dists, computer, idx, id_count);
}

template <typename QuantTmpl, typename IOTmpl>
void
FlattenDataCell<QuantTmpl, IOTmpl>::query(float* result_dists,
                                          std::shared_ptr<Computer<QuantTmpl>> computer,
                                          const uint64_t* idx,
                                          uint64_t id_count) {
    for (uint64_t i = 0; i < id_count; ++i) {
        const auto* codes = this->GetCodesById(idx[i]);
        computer->ComputeDist(codes, result_dists + i);
    }
}

template <typename QuantTmpl, typename IOTmpl>
float
FlattenDataCell<QuantTmpl, IOTmpl>::ComputePairVectors(uint64_t id1, uint64_t id2) {
    const auto* codes1 = this->GetCodesById(id1);
    const auto* codes2 = this->GetCodesById(id1);
    return this->quantizer_->Compute(codes1, codes2);
}

template <typename QuantTmpl, typename IOTmpl>
const uint8_t*
FlattenDataCell<QuantTmpl, IOTmpl>::GetCodesById(uint64_t id) const {
    return io_->Read(code_size_, id * code_size_);
}

}  // namespace vsag
