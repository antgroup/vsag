
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
#include <random>
#include <unordered_map>
#include <vector>

#include "io/io_headers.h"
#include "quantization/computer.h"

namespace vsag {
class RedundantDataCellInterface {
public:
    virtual void
    MakeRedundant(double loading_factor = 1.0) = 0;

    virtual bool
    QueryLine(float* resultDists,
              const float* query,
              uint64_t id,
              std::vector<uint32_t>& to_be_visit,
              uint32_t count_no_visit) const = 0;

    virtual bool
    QueryLine(float* resultDists,
              std::shared_ptr<ComputerInterface>& computer,
              uint64_t id,
              std::vector<uint32_t>& to_be_visit,
              uint32_t count_no_visit) const = 0;

public:
    virtual void
    SetPrefetchParameters(uint32_t neighbor_codes_num, uint32_t cache_line) = 0;

    virtual uint64_t
    GetRedundantTotalCount() const = 0;

    virtual bool
    IsRedundant(uint64_t id) const = 0;
};
}