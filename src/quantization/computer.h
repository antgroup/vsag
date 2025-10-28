
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

#include <cstdint>
#include <memory>

#include "metric_type.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "vsag/allocator.h"

namespace vsag {
DEFINE_POINTER(ComputerInterface);

using DataType = float;

class ComputerInterface {
public:
    ComputerInterface() = default;

    virtual ~ComputerInterface() = default;
};

template <typename T>
class Computer : public ComputerInterface {
public:
    explicit Computer(const T* quantizer, Allocator* allocator)
        : quantizer_(quantizer), allocator_(allocator), raw_query_(allocator){};

    ~Computer() override {
        quantizer_->ReleaseComputer(*this);
    }

    void
    SetQuery(const DataType* query) {
        quantizer_->ProcessQuery(query, *this);
    }

    inline void
    ComputeDist(const uint8_t* codes, float* dists) {
        quantizer_->ComputeDist(*this, codes, dists);
    }

    inline void
    ScanBatchDists(uint64_t count, const uint8_t* codes, float* dists) {
        quantizer_->ScanBatchDists(*this, count, codes, dists);
    }

    inline void
    ComputeDistsBatch4(const uint8_t* codes1,
                       const uint8_t* codes2,
                       const uint8_t* codes3,
                       const uint8_t* codes4,
                       float& dists1,
                       float& dists2,
                       float& dists3,
                       float& dists4) {
        quantizer_->ComputeDistsBatch4(
            *this, codes1, codes2, codes3, codes4, dists1, dists2, dists3, dists4);
    }

public:
    Allocator* const allocator_{nullptr};
    const T* quantizer_{nullptr};
    uint8_t* buf_{nullptr};
    Vector<float> raw_query_;
};

template <typename QuantImpl, MetricType metric>
class TransformQuantizer;

template <typename QuantImpl, MetricType metric>
class Computer<TransformQuantizer<QuantImpl, metric>> : public ComputerInterface {
public:
    explicit Computer(const TransformQuantizer<QuantImpl, metric>* quantizer, Allocator* allocator)
        : quantizer_(quantizer), allocator_(allocator) {
        inner_computer_ = new Computer<QuantImpl>(quantizer_->quantizer_.get(), allocator);
    }

    ~Computer() override {
        quantizer_->ReleaseComputer(*this);
        delete inner_computer_;
    }

    void
    SetQuery(const DataType* query) {
        quantizer_->ProcessQuery(query, *this);
    }

    inline void
    ComputeDist(const uint8_t* codes, float* dists) {
        quantizer_->ComputeDist(*this, codes, dists);
    }

    inline void
    ScanBatchDists(uint64_t count, const uint8_t* codes, float* dists) {
        quantizer_->ScanBatchDists(*this, count, codes, dists);
    }

    inline void
    ComputeDistsBatch4(const uint8_t* codes1,
                       const uint8_t* codes2,
                       const uint8_t* codes3,
                       const uint8_t* codes4,
                       float& dists1,
                       float& dists2,
                       float& dists3,
                       float& dists4) {
        quantizer_->ComputeDistsBatch4(
            *this, codes1, codes2, codes3, codes4, dists1, dists2, dists3, dists4);
    }

public:
    Allocator* const allocator_{nullptr};
    const TransformQuantizer<QuantImpl, metric>* quantizer_{nullptr};
    uint8_t* buf_{nullptr};
    Computer<QuantImpl>* inner_computer_{nullptr};
};

}  // namespace vsag
