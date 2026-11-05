
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

#include "computable_bitset.h"

#include "fast_bitset.h"
#include "sparse_bitset.h"
#include "vsag_exception.h"

namespace vsag {
ComputableBitsetPtr
ComputableBitset::MakeInstance(ComputableBitsetType type, Allocator* allocator) {
    if (type == ComputableBitsetType::SparseBitset) {
        return std::make_shared<SparseBitset>();
    }
    if (type == ComputableBitsetType::FastBitset) {
        return std::make_shared<FastBitset>(allocator);
    }
    throw VsagException(ErrorType::INTERNAL_ERROR, "Unknown bitset type");
}

ComputableBitset*
ComputableBitset::MakeRawInstance(ComputableBitsetType type, Allocator* allocator) {
    if (type == ComputableBitsetType::SparseBitset) {
        return new SparseBitset();
    }
    if (type == ComputableBitsetType::FastBitset) {
        return new FastBitset(allocator);
    }
    throw VsagException(ErrorType::INTERNAL_ERROR, "Unknown bitset type");
}

void
ComputableBitset::And(const std::vector<const ComputableBitset*>& other_bitsets) {
    for (const auto& ptr : other_bitsets) {
        this->And(ptr);
    }
}

void
ComputableBitset::Or(const std::vector<const ComputableBitset*>& other_bitsets) {
    for (const auto& ptr : other_bitsets) {
        this->Or(ptr);
    }
}

}  // namespace vsag
