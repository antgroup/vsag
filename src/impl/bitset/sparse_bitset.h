
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
#include <mutex>
#include <roaring.hh>
#include <vector>

#include "computable_bitset.h"

namespace vsag {

class SparseBitset : public ComputableBitset {
public:
    explicit SparseBitset() : ComputableBitset() {
        this->type_ = ComputableBitsetType::SparseBitset;
    }
    ~SparseBitset() override = default;

    explicit SparseBitset(Allocator* allocator) : SparseBitset(){};

    SparseBitset(const SparseBitset&) = delete;
    SparseBitset&
    operator=(const SparseBitset&) = delete;
    SparseBitset(SparseBitset&&) = delete;

public:
    void
    Set(int64_t pos, bool value) override;

    bool
    Test(int64_t pos) const override;

    uint64_t
    Count() override;

    std::string
    Dump() override;

    void
    Or(const ComputableBitset& another) override;

    void
    And(const ComputableBitset& another) override;

    void
    Or(const ComputableBitset* another) override;

    void
    And(const ComputableBitset* another) override;

    void
    Not() override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    void
    Clear() override;

private:
    mutable std::mutex mutex_;
    roaring::Roaring r_;
};

}  //namespace vsag
