
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

#include "sparse_bitset.h"

#include <cstdint>
#include <mutex>

namespace vsag {

void
SparseBitset::Set(int64_t pos, bool value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (value) {
        r_.add(pos);
    } else {
        r_.remove(pos);
    }
}

bool
SparseBitset::Test(int64_t pos) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return r_.contains(pos);
}

uint64_t
SparseBitset::Count() {
    return r_.cardinality();
}

std::string
SparseBitset::Dump() {
    return r_.toString();
}

void
SparseBitset::Or(const ComputableBitset& another) {
    const auto* another_ptr = reinterpret_cast<const SparseBitset*>(&another);
    std::lock(mutex_, another_ptr->mutex_);
    std::lock_guard<std::mutex> lock(mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lock_other(another_ptr->mutex_, std::adopt_lock);
    r_ |= another_ptr->r_;
}

void
SparseBitset::And(const ComputableBitset& another) {
    const auto* another_ptr = reinterpret_cast<const SparseBitset*>(&another);
    std::lock(mutex_, another_ptr->mutex_);
    std::lock_guard<std::mutex> lock(mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lock_other(another_ptr->mutex_, std::adopt_lock);
    r_ &= another_ptr->r_;
}

void
SparseBitset::Or(const ComputableBitset* another) {
    if (another == nullptr) {
        return;
    }
    this->Or(*another);
}

void
SparseBitset::And(const ComputableBitset* another) {
    if (another == nullptr) {
        this->Clear();
        return;
    }
    this->And(*another);
}

void
SparseBitset::Not() {
    std::lock_guard<std::mutex> lock(mutex_);
    r_.flipClosed(r_.minimum(), r_.maximum());
}

void
SparseBitset::Serialize(StreamWriter& writer) const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t size = r_.getSizeInBytes();
    StreamWriter::WriteObj(writer, size);
    std::vector<char> buffer(size);
    r_.write(buffer.data());
    writer.Write(buffer.data(), size);
}

void
SparseBitset::Deserialize(StreamReader& reader) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    if (size == 0) {
        return;
    }
    std::vector<char> buffer(size);
    reader.Read(buffer.data(), size);
    r_ = roaring::Roaring::readSafe(buffer.data(), size);
}

void
SparseBitset::Clear() {
    this->r_ = std::move(roaring::Roaring());
}

}  // namespace vsag
