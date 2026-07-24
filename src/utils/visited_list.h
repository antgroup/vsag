
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

#include "resource_object.h"
#include "resource_object_pool.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "utils/prefetch.h"

namespace vsag {
class Allocator;

DEFINE_POINTER(VisitedList);
class VisitedList : public ResourceObject {
public:
    using WordType = uint64_t;
    using TagType = uint16_t;
    static constexpr uint64_t kBitsPerWord = sizeof(WordType) * 8;

public:
    explicit VisitedList(InnerIdType max_size, Allocator* allocator);
    ~VisitedList() override;

    void
    Set(const InnerIdType& id) {
        const auto word_id = static_cast<uint64_t>(id) / kBitsPerWord;
        const auto mask = WordType{1} << (static_cast<uint64_t>(id) % kBitsPerWord);
        if (this->tags_[word_id] != this->tag_) {
            this->tags_[word_id] = this->tag_;
            this->words_[word_id] = mask;
        } else {
            this->words_[word_id] |= mask;
        }
    }

    [[nodiscard]] bool
    Get(const InnerIdType& id) {
        const auto word_id = static_cast<uint64_t>(id) / kBitsPerWord;
        const auto mask = WordType{1} << (static_cast<uint64_t>(id) % kBitsPerWord);
        return this->tags_[word_id] == this->tag_ and (this->words_[word_id] & mask) != 0;
    }

    void
    Prefetch(const InnerIdType& id) {
        const auto word_id = static_cast<uint64_t>(id) / kBitsPerWord;
        PrefetchLines(this->tags_ + word_id, 64);
    }

    void
    Reset() override;

    uint64_t
    GetMemoryUsage() const override {
        return sizeof(VisitedList) + this->word_count_ * (sizeof(WordType) + sizeof(TagType));
    }

private:
    Allocator* const allocator_{nullptr};

    WordType* words_{nullptr};

    TagType* tags_{nullptr};

    TagType tag_{1};

    const uint64_t word_count_{0};
};

using VisitedListPool = ResourceObjectPool<VisitedList>;
}  // namespace vsag
