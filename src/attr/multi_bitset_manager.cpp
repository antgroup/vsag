
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

#include "multi_bitset_manager.h"

namespace vsag {

MultiBitsetManager::MultiBitsetManager(Allocator* allocator,
                                       uint64_t count,
                                       ComputableBitsetType bitset_type)
    : allocator_(allocator),
      count_(count),
      bitsets_(allocator),
      bitset_map_(count, -1, allocator),
      bitset_type_(bitset_type) {
}

MultiBitsetManager::MultiBitsetManager(Allocator* allocator, uint64_t count)
    : MultiBitsetManager(allocator, count, ComputableBitsetType::FastBitset) {
}

MultiBitsetManager::MultiBitsetManager(Allocator* allocator) : MultiBitsetManager(allocator, 1) {
}

MultiBitsetManager::~MultiBitsetManager() {
    for (auto* bitset : bitsets_) {
        delete bitset;
    }
}

void
MultiBitsetManager::SetNewCount(uint64_t new_count) {
    if (new_count <= count_) {
        return;
    }
    this->count_ = new_count;
    this->bitset_map_.resize(new_count, -1);
}

ComputableBitset*
MultiBitsetManager::GetOneBitset(uint64_t id) const {
    if (id >= count_) {
        return nullptr;
    }
    auto inner_id = bitset_map_[id];
    if (inner_id == -1) {
        return nullptr;
    }
    return this->bitsets_[inner_id];
}

void
MultiBitsetManager::InsertValue(uint64_t id, uint64_t offset, bool value) {
    if (id >= count_) {
        this->SetNewCount(id + 1);
    }
    auto inner_id = bitset_map_[id];
    if (inner_id == -1) {
        inner_id = static_cast<int16_t>(bitsets_.size());
        bitset_map_[id] = inner_id;
        bitsets_.emplace_back(
            ComputableBitset::MakeRawInstance(this->bitset_type_, this->allocator_));
    }
    this->bitsets_[inner_id]->Set(static_cast<int64_t>(offset), value);
}

void
MultiBitsetManager::Serialize(StreamWriter& writer) {
    StreamWriter::WriteObj(writer, count_);
    for (auto id : bitset_map_) {
        StreamWriter::WriteObj(writer, id);
    }
    uint64_t size = bitsets_.size();
    StreamWriter::WriteObj(writer, size);
    for (auto* bitset : bitsets_) {
        bitset->Serialize(writer);
    }
}

void
MultiBitsetManager::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    StreamReader::ReadObj(reader, count_);
    this->bitset_map_.resize(count_, -1);
    for (uint64_t i = 0; i < count_; i++) {
        StreamReader::ReadObj(reader, this->bitset_map_[i]);
    }
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    bitsets_.resize(size, nullptr);
    for (uint64_t i = 0; i < size; i++) {
        bitsets_[i] = ComputableBitset::MakeRawInstance(bitset_type_, allocator_);
        bitsets_[i]->Deserialize(reader);
    }
}

}  // namespace vsag
