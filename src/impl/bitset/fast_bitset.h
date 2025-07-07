
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
#include "safe_allocator.h"
#include "typing.h"

namespace vsag {
class FastBitset : public ComputableBitset {
public:
    explicit FastBitset(Allocator* allocator)
        : ComputableBitset(), allocator_(allocator), data_(allocator), fill_bit_(false) {
        this->type_ = ComputableBitsetType::FastBitset;
    };

    ~FastBitset() override = default;

    void
    Set(int64_t pos, bool value) override;

    bool
    Test(int64_t pos) override;

    uint64_t
    Count() override;

    void
    Or(const ComputableBitset& another) override;

    void
    And(const ComputableBitset& another) override;

    void
    Xor(const ComputableBitset& another) override;

    void
    Or(const ComputableBitsetPtr& another) override;

    void
    And(const ComputableBitsetPtr& another) override;

    void
    Xor(const ComputableBitsetPtr& another) override;

    void
    And(const std::vector<ComputableBitsetPtr>& other_bitsets) override;

    void
    Or(const std::vector<ComputableBitsetPtr>& other_bitsets) override;

    void
    Xor(const std::vector<ComputableBitsetPtr>& other_bitsets) override;

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
    Vector<uint64_t> data_;

    mutable std::shared_mutex mutex_;

    Allocator* const allocator_{nullptr};

    bool fill_bit_{false};

    const uint64_t FILL_ONE = 0xFFFFFFFFFFFFFFFF;
};

using FastBitsetPtr = std::shared_ptr<FastBitset>;
}  // namespace vsag
