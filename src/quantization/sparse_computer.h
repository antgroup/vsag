
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

namespace vsag {

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
    SetQuery(int32_t nnz, const uint32_t* ids, const float* vals) {
        quantizer_->ProcessQuery(nnz, ids, vals, *this);
    }

    inline void
    ComputeDist(uint32_t u, float* dists) {//u是quant中存储的第u个稀疏向量
        quantizer_->ComputeDist(*this, u, dists);
    }

public:
    const SparseQuantizer* quantizer_{nullptr};
    int32_t nnz_ = 0;
    const uint32_t* vals_{nullptr};
    const float* ids_{nullptr};
};

using SparseComputerInterfacePtr = std::shared_ptr<SparseComputerInterface>;

}  // namespace vsag
