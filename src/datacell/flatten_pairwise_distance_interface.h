// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>

#include "basic_types.h"

namespace vsag {

struct QueryContext;

// A caller-owned distance computer. Codes must remain unchanged during its lifetime;
// separate graph operations use separate computers, including concurrent insertions.
class FlattenPairwiseDistanceComputer {
public:
    virtual ~FlattenPairwiseDistanceComputer() = default;

    virtual float
    Compute(InnerIdType id1, InnerIdType id2, QueryContext* ctx = nullptr) = 0;
};

class FlattenPairwiseDistanceInterface {
public:
    virtual ~FlattenPairwiseDistanceInterface() = default;

    // Optional acceleration for persisted codes. Returning nullptr retains the existing path.
    virtual std::unique_ptr<FlattenPairwiseDistanceComputer>
    FactoryPairwiseDistanceComputer() = 0;
};

}  // namespace vsag
