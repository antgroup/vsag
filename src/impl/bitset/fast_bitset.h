
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

#include <shared_mutex>

#include "computable_bitset.h"
#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {
class FastBitset : public ComputableBitset {
public:
    explicit FastBitset(Allocator* allocator)
        : ComputableBitset(), data_(allocator), fill_bit_(false) {
        this->type_ = ComputableBitsetType::FastBitset;
    };

    ~FastBitset() override = default;

    void
    Set(int64_t pos, bool value) override;

    bool
    Test(int64_t pos) const override;

    uint64_t
    Count() override;

    void
    Or(const ComputableBitset& another) override;

    void
    And(const ComputableBitset& another) override;

    void
    Or(const ComputableBitset* another) override;

    void
    And(const ComputableBitset* another) override;

    void
    And(const std::vector<const ComputableBitset*>& other_bitsets) override;

    void
    Or(const std::vector<const ComputableBitset*>& other_bitsets) override;

    void
    Not() override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    void
    Clear() override;

    std::string
    Dump() override;

private:
    bool fill_bit_{false};

    Vector<uint64_t> data_;
};

using FastBitsetPtr = std::shared_ptr<FastBitset>;
}  // namespace vsag
