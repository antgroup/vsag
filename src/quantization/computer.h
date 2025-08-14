
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

#include "computer_interface.h"
#include "metric_type.h"
#include "vsag/allocator.h"

namespace vsag {

using DataType = float;

template <typename T>
class Computer : public ComputerInterface, public std::enable_shared_from_this<Computer<T>> {
public:
    ~Computer() override {
        this->allocator_->Deallocate(buf_);
    }

    explicit Computer(const T* quantizer, Allocator* allocator)
        : quantizer_(quantizer), allocator_(allocator){};

    ComputerInterfacePtr
    GetComputerInterfacePtr() override {
        return this->shared_from_this();
    }

    void
    SetQuery(const DataType* query) override {
        quantizer_->ProcessQuery(query, this->shared_from_this());
    }

    inline void
    ComputeDist(const uint8_t* codes, float* dists) {
        quantizer_->ComputeDist(this->shared_from_this(), codes, dists);
    }

    inline void
    ScanBatchDists(uint64_t count, const uint8_t* codes, float* dists) {
        quantizer_->ScanBatchDists(this->shared_from_this(), count, codes, dists);
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
        quantizer_->ComputeDistsBatch4(this->shared_from_this(),
                                       codes1,
                                       codes2,
                                       codes3,
                                       codes4,
                                       dists1,
                                       dists2,
                                       dists3,
                                       dists4);
    }

public:
    Allocator* const allocator_{nullptr};
    const T* quantizer_{nullptr};
    uint8_t* buf_{nullptr};
    ComputerInterfacePtr inner_computer_{nullptr};
};

}  // namespace vsag
