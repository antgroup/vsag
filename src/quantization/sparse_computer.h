
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
#include "vsag/dataset.h"
#include "metric_type.h"

namespace vsag {
using SDataType = SparseVectors;

class SparseQuantizer;

class SparseComputerInterface {
protected:
    SparseComputerInterface() = default;
};

class SparseComputer : public SparseComputerInterface {
public:
    ~SparseComputer() {
        quantizer_->ReleaseComputer(*this);
    }

    explicit SparseComputer(const SparseQuantizer* quantizer) : quantizer_(quantizer){};

    void
    SetQuery(const SDataType* query) {
        quantizer_->ProcessQuery(query, *this);
    }

    inline void
    ComputeDist(const uint8_t* codes, float* dists) {
        quantizer_->ComputeDist(*this, codes, dists);
    }

public:
    const SparseQuantizer* quantizer_{nullptr};
    uint8_t* buf_{nullptr};
};

using SparseComputerInterfacePtr = std::shared_ptr<SparseComputerInterface>;

}  // namespace vsag
