
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

void
FastBitset::Set(int64_t pos, bool value) {
    std::lock_guard<std::shared_mutex> lock(mutex_);
    auto capacity = data_.size() * 64;
    if (pos >= capacity) {
        data_.resize((pos / 64) + 1, 0);
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
FastBitset::Test(int64_t pos) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto capacity = data_.size() * 64;
    if (pos >= capacity) {
        return false;
    }
    auto word_index = pos / 64;
    auto bit_index = pos % 64;
    return (data_[word_index] & (1ULL << bit_index)) != 0;
}

uint64_t
FastBitset::Count() {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    uint64_t count = 0;
    for (auto word : data_) {
        count += __builtin_popcountll(word);
    }
    return count;
}
void
FastBitset::Or(const Bitset& another) {
    const auto* fast_another = dynamic_cast<const FastBitset*>(&another);
    if (fast_another == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "bitset not match");
    }
    std::lock(mutex_, fast_another->mutex_);
    std::lock_guard<std::shared_mutex> lock1(mutex_, std::adopt_lock);
    std::lock_guard<std::shared_mutex> lock2(fast_another->mutex_, std::adopt_lock);
    auto max_size = std::max(data_.size(), fast_another->data_.size());
    data_.resize(max_size, 0);
    BitOr(reinterpret_cast<const uint8_t*>(data_.data()),
          reinterpret_cast<const uint8_t*>(fast_another->data_.data()),
          max_size * sizeof(uint64_t),
          reinterpret_cast<uint8_t*>(data_.data()));
}

void
FastBitset::And(const Bitset& another) {
    const auto* fast_another = dynamic_cast<const FastBitset*>(&another);
    if (fast_another == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "bitset not match");
    }
    std::lock(mutex_, fast_another->mutex_);
    std::lock_guard<std::shared_mutex> lock1(mutex_, std::adopt_lock);
    std::lock_guard<std::shared_mutex> lock2(fast_another->mutex_, std::adopt_lock);
    auto max_size = std::max(data_.size(), fast_another->data_.size());
    data_.resize(max_size, 0);
    BitAnd(reinterpret_cast<const uint8_t*>(data_.data()),
           reinterpret_cast<const uint8_t*>(fast_another->data_.data()),
           max_size * sizeof(uint64_t),
           reinterpret_cast<uint8_t*>(data_.data()));
}

void
FastBitset::Xor(const Bitset& another) {
    const auto* fast_another = dynamic_cast<const FastBitset*>(&another);
    if (fast_another == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "bitset not match");
    }
    std::lock(mutex_, fast_another->mutex_);
    std::lock_guard<std::shared_mutex> lock1(mutex_, std::adopt_lock);
    std::lock_guard<std::shared_mutex> lock2(fast_another->mutex_, std::adopt_lock);
    auto max_size = std::max(data_.size(), fast_another->data_.size());
    data_.resize(max_size, 0);
    BitXor(reinterpret_cast<const uint8_t*>(data_.data()),
           reinterpret_cast<const uint8_t*>(fast_another->data_.data()),
           max_size * sizeof(uint64_t),
           reinterpret_cast<uint8_t*>(data_.data()));
}

std::string
FastBitset::Dump() {
    std::shared_lock<std::shared_mutex> lock(mutex_);
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
    std::lock_guard<std::shared_mutex> lock(mutex_);
    BitNot(reinterpret_cast<const uint8_t*>(data_.data()),
           data_.size() * sizeof(uint64_t),
           reinterpret_cast<uint8_t*>(data_.data()));
}

void
FastBitset::Serialize(StreamWriter& writer) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    uint64_t size = data_.size();
    StreamWriter::WriteObj(writer, size);
    if (size > 0) {
        writer.Write(reinterpret_cast<const char*>(data_.data()), size * sizeof(uint64_t));
    }
}

void
FastBitset::Deserialize(StreamReader& reader) {
    std::lock_guard<std::shared_mutex> lock(mutex_);
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    data_.resize(size);
    reader.Read(reinterpret_cast<char*>(data_.data()), size * sizeof(uint64_t));
}
}  // namespace vsag
