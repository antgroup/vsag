
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
#include "utils/pointer_define.h"

namespace vsag {

class Allocator;
DEFINE_POINTER(FastBitset);
class FastBitset : public ComputableBitset {
public:
    explicit FastBitset(Allocator* allocator)
        : ComputableBitset(), data_(nullptr), size_(0), capacity_(0){};

    ~FastBitset() override {
        delete[] data_;
        data_ = nullptr;
    }

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
    void
    resize(uint32_t new_size, uint64_t fill = 0);

    constexpr bool
    get_fill_bit() const {
        return (capacity_ >> 31) & 1;
    }

    constexpr void
    set_fill_bit(bool value) {
        if (value) {
            capacity_ |= (1UL << 31);
        } else {
            capacity_ &= ~(1UL << 31);
        }
    }

    constexpr uint32_t
    get_capacity() const {
        return capacity_ & 0x7FFFFFFF;
    }

    constexpr void
    set_capacity(uint32_t cap) {
        capacity_ = (capacity_ & (1UL << 31)) | (cap & 0x7FFFFFFF);
    }

private:
    uint64_t* data_{nullptr};

    uint32_t size_{0};

    uint32_t capacity_{0};
};

}  // namespace vsag
