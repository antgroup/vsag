
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
    auto capacity = data_.size() * 64;
    if (pos >= capacity) {
        if (fill_bit_) {
            data_.resize((pos / 64) + 1, FILL_ONE);
        } else {
            data_.resize((pos / 64) + 1, 0);
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
    auto capacity = data_.size() * 64;
    if (pos >= capacity) {
        return fill_bit_;
    }
    auto word_index = pos / 64;
    auto bit_index = pos % 64;
    return (data_[word_index] & (1ULL << bit_index)) != 0;
}

uint64_t
FastBitset::Count() {
    uint64_t count = 0;
    for (auto word : data_) {
        count += __builtin_popcountll(word);
    }
    return count;
}

void
FastBitset::Or(const ComputableBitset& another) {
    const auto* fast_another = static_cast<const FastBitset*>(&another);
    if (fast_another == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "bitset not match");
    }
    if (fast_another->data_.empty()) {
        if (fast_another->fill_bit_) {
            this->Clear();
            this->fill_bit_ = true;
        }
        return;
    }
    if (data_.size() >= fast_another->data_.size()) {
        auto min_size = fast_another->data_.size();
        BitOr(reinterpret_cast<const uint8_t*>(this->data_.data()),
              reinterpret_cast<const uint8_t*>(fast_another->data_.data()),
              min_size * sizeof(uint64_t),
              reinterpret_cast<uint8_t*>(this->data_.data()));
        if (fast_another->fill_bit_) {
            data_.resize(min_size);
            this->fill_bit_ = true;
        }
    } else {
        auto max_size = fast_another->data_.size();
        if (this->fill_bit_) {
            max_size = this->data_.size();
        } else {
            this->data_.resize(max_size, 0);
            this->fill_bit_ = fast_another->fill_bit_;
        }
        BitOr(reinterpret_cast<const uint8_t*>(this->data_.data()),
              reinterpret_cast<const uint8_t*>(fast_another->data_.data()),
              max_size * sizeof(uint64_t),
              reinterpret_cast<uint8_t*>(this->data_.data()));
    }
}

void
FastBitset::And(const ComputableBitset& another) {
    const auto* fast_another = static_cast<const FastBitset*>(&another);
    if (fast_another == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "bitset not match");
    }
    if (fast_another->data_.empty()) {
        if (not fast_another->fill_bit_) {
            this->Clear();
        }
        return;
    }
    if (data_.size() >= fast_another->data_.size()) {
        auto min_size = fast_another->data_.size();
        auto max_size = data_.size();
        BitAnd(reinterpret_cast<const uint8_t*>(this->data_.data()),
               reinterpret_cast<const uint8_t*>(fast_another->data_.data()),
               min_size * sizeof(uint64_t),
               reinterpret_cast<uint8_t*>(this->data_.data()));
        if (max_size > min_size and not fast_another->fill_bit_) {
            std::fill(data_.begin() + static_cast<int64_t>(min_size), data_.end(), 0);
        }
    } else {
        auto max_size = fast_another->data_.size();
        if (this->fill_bit_) {
            this->data_.resize(max_size, (uint64_t)(-1));
        } else {
            this->data_.resize(max_size, 0);
        }
        BitAnd(reinterpret_cast<const uint8_t*>(this->data_.data()),
               reinterpret_cast<const uint8_t*>(fast_another->data_.data()),
               max_size * sizeof(uint64_t),
               reinterpret_cast<uint8_t*>(this->data_.data()));
    }
    this->fill_bit_ = this->fill_bit_ && fast_another->fill_bit_;
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
    auto capacity = data_.size() * 64;
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
    BitNot(reinterpret_cast<const uint8_t*>(data_.data()),
           data_.size() * sizeof(uint64_t),
           reinterpret_cast<uint8_t*>(data_.data()));
    this->fill_bit_ = !this->fill_bit_;
}

void
FastBitset::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, fill_bit_);
    uint64_t size = data_.size();
    StreamWriter::WriteObj(writer, size);
    if (size > 0) {
        writer.Write(reinterpret_cast<const char*>(data_.data()), size * sizeof(uint64_t));
    }
}

void
FastBitset::Deserialize(StreamReader& reader) {
    StreamReader::ReadObj(reader, fill_bit_);
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    data_.resize(size);
    reader.Read(reinterpret_cast<char*>(data_.data()), size * sizeof(uint64_t));
}

void
FastBitset::Clear() {
    this->data_.clear();
    fill_bit_ = false;
}

}  // namespace vsag
