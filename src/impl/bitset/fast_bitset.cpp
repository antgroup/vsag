
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

#include "fast_bitset.h"

#include "simd/bit_simd.h"
#include "vsag_exception.h"

namespace vsag {

static constexpr uint64_t FILL_ONE = 0xFFFFFFFFFFFFFFFF;

void
FastBitset::Set(int64_t pos, bool value) {
    auto capacity = this->size_ * 64;
    if (pos >= capacity) {
        if (get_fill_bit()) {
            resize((pos / 64) + 1, FILL_ONE);
        } else {
            resize((pos / 64) + 1, 0);
        }
    }
    auto word_index = pos / 64;
    auto bit_index = pos % 64;
    if (value) {
        data_[word_index] |= (1ULL << bit_index);
    } else {
        data_[word_index] &= ~(1ULL << bit_index);
    }
}

bool
FastBitset::Test(int64_t pos) const {
    auto capacity = this->size_ * 64;
    if (pos >= capacity) {
        return get_fill_bit();
    }
    auto word_index = pos / 64;
    auto bit_index = pos % 64;
    return (data_[word_index] & (1ULL << bit_index)) != 0;
}

uint64_t
FastBitset::Count() {
    uint64_t count = 0;
    for (uint32_t i = 0; i < this->size_; i++) {
        count += __builtin_popcountll(data_[i]);
    }
    return count;
}

void
FastBitset::Or(const ComputableBitset& another) {
    const auto* fast_another = static_cast<const FastBitset*>(&another);
    if (fast_another == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "bitset not match");
    }
    if (fast_another->size_ == 0) {
        if (fast_another->get_fill_bit()) {
            this->Clear();
            this->set_fill_bit(true);
        }
        return;
    }
    if (this->size_ >= fast_another->size_) {
        auto min_size = fast_another->size_;
        BitOr(reinterpret_cast<const uint8_t*>(this->data_),
              reinterpret_cast<const uint8_t*>(fast_another->data_),
              min_size * sizeof(uint64_t),
              reinterpret_cast<uint8_t*>(this->data_));
        if (fast_another->get_fill_bit()) {
            resize(min_size);
            this->set_fill_bit(true);
        }
    } else {
        auto max_size = fast_another->size_;
        if (this->get_fill_bit()) {
            max_size = this->size_;
        } else {
            resize(max_size, 0);
            this->set_fill_bit(fast_another->get_fill_bit());
        }
        BitOr(reinterpret_cast<const uint8_t*>(this->data_),
              reinterpret_cast<const uint8_t*>(fast_another->data_),
              max_size * sizeof(uint64_t),
              reinterpret_cast<uint8_t*>(this->data_));
    }
}

void
FastBitset::And(const ComputableBitset& another) {
    const auto* fast_another = static_cast<const FastBitset*>(&another);
    if (fast_another == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "bitset not match");
    }
    if (fast_another->size_ == 0) {
        if (not fast_another->get_fill_bit()) {
            this->Clear();
        }
        return;
    }
    if (this->size_ >= fast_another->size_) {
        auto min_size = fast_another->size_;
        auto max_size = this->size_;
        BitAnd(reinterpret_cast<const uint8_t*>(this->data_),
               reinterpret_cast<const uint8_t*>(fast_another->data_),
               min_size * sizeof(uint64_t),
               reinterpret_cast<uint8_t*>(this->data_));
        if (max_size > min_size and not fast_another->get_fill_bit()) {
            std::fill(data_ + min_size, data_ + max_size, 0);
        }
    } else {
        auto max_size = fast_another->size_;
        if (this->get_fill_bit()) {
            resize(max_size, FILL_ONE);
        } else {
            resize(max_size, 0);
        }
        BitAnd(reinterpret_cast<const uint8_t*>(this->data_),
               reinterpret_cast<const uint8_t*>(fast_another->data_),
               max_size * sizeof(uint64_t),
               reinterpret_cast<uint8_t*>(this->data_));
    }
    this->set_fill_bit(this->get_fill_bit() && fast_another->get_fill_bit());
}

void
FastBitset::Or(const ComputableBitset* another) {
    if (another == nullptr) {
        return;
    }
    this->Or(*another);
}

void
FastBitset::And(const ComputableBitset* another) {
    if (another == nullptr) {
        this->Clear();
        return;
    }
    this->And(*another);
}

void
FastBitset::And(const std::vector<const ComputableBitset*>& other_bitsets) {
    for (const auto& ptr : other_bitsets) {
        if (ptr == nullptr) {
            this->Clear();
            return;
        }
        this->And(*ptr);
    }
}

void
FastBitset::Or(const std::vector<const ComputableBitset*>& other_bitsets) {
    for (const auto& ptr : other_bitsets) {
        if (ptr != nullptr) {
            this->Or(*ptr);
        }
    }
}

std::string
FastBitset::Dump() {
    std::string result = "{";
    auto capacity = this->size_ * 64;
    int count = 0;
    for (int64_t i = 0; i < capacity; ++i) {
        if (Test(i)) {
            if (count == 0) {
                result += std::to_string(i);
            } else {
                result += "," + std::to_string(i);
            }
            ++count;
        }
    }
    result += "}";
    return result;
}

void
FastBitset::Not() {
    BitNot(reinterpret_cast<const uint8_t*>(data_),
           this->size_ * sizeof(uint64_t),
           reinterpret_cast<uint8_t*>(data_));
    this->set_fill_bit(!this->get_fill_bit());
}

void
FastBitset::Serialize(StreamWriter& writer) const {
    bool fill_bit = this->get_fill_bit();
    StreamWriter::WriteObj(writer, fill_bit);
    uint64_t size = this->size_;
    StreamWriter::WriteObj(writer, size);
    if (size > 0) {
        writer.Write(reinterpret_cast<const char*>(data_), size * sizeof(uint64_t));
    }
}

void
FastBitset::Deserialize(StreamReader& reader) {
    bool fill_bit;
    StreamReader::ReadObj(reader, fill_bit);
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    data_ = new uint64_t[size];
    reader.Read(reinterpret_cast<char*>(data_), size * sizeof(uint64_t));
    this->size_ = size;
    this->set_capacity(size);
    this->set_fill_bit(fill_bit);
}

void
FastBitset::Clear() {
    this->size_ = 0;
    this->set_fill_bit(false);
}

void
FastBitset::resize(uint32_t new_size, uint64_t fill) {
    if (new_size > this->get_capacity()) {
        auto* tmp = new uint64_t[new_size];
        std::memcpy(tmp, data_, size_ * sizeof(uint64_t));
        delete[] data_;
        this->data_ = tmp;
        this->set_capacity(new_size);
    }
    if (new_size > size_) {
        std::fill(data_ + size_, data_ + new_size, fill);
    }
    size_ = new_size;
}

}  // namespace vsag
