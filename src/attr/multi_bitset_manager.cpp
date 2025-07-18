
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
    : allocator_(allocator), count_(count), bitsets_(allocator), bitset_type_(bitset_type) {
    bitsets_.resize(count, nullptr);
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
    this->bitsets_.resize(new_count, nullptr);
}

ComputableBitset*
MultiBitsetManager::GetOneBitset(uint64_t id) const {
    if (id >= count_) {
        return nullptr;
    }
    return this->bitsets_[id];
}

void
MultiBitsetManager::InsertValue(uint64_t id, uint64_t offset, bool value) {
    if (id >= count_) {
        this->SetNewCount(id + 1);
    }
    if (this->bitsets_[id] == nullptr) {
        this->bitsets_[id] =
            ComputableBitset::MakeRawInstance(this->bitset_type_, this->allocator_);
    }
    this->bitsets_[id]->Set(static_cast<int64_t>(offset), value);
}

void
MultiBitsetManager::Serialize(StreamWriter& writer) {
    StreamWriter::WriteObj(writer, count_);
    for (auto* bitset : bitsets_) {
        if (bitset == nullptr) {
            StreamWriter::WriteObj(writer, false);
        } else {
            StreamWriter::WriteObj(writer, true);
            bitset->Serialize(writer);
        }
    }
}

void
MultiBitsetManager::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    StreamReader::ReadObj(reader, count_);
    this->bitsets_.resize(count_, nullptr);
    for (uint64_t i = 0; i < count_; i++) {
        bool has_bitset;
        StreamReader::ReadObj(reader, has_bitset);
        if (has_bitset) {
            bitsets_[i] = ComputableBitset::MakeRawInstance(bitset_type_, allocator_);
            bitsets_[i]->Deserialize(reader);
        }
    }
}

}  // namespace vsag
