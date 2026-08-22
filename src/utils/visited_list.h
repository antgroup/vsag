
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

// Per-query visited-id bitmap. Words are indexed by id / 64; words dirtied
// since the last Reset are recorded in touched_words_ so Reset only clears
// what the query actually touched instead of scanning (or generation-stamping)
// the whole structure. This keeps Get() at a single load from one array,
// which sits on the graph-traversal hot path.
class VisitedList : public ResourceObject {
public:
    using WordType = uint64_t;
    using TouchedIndexType = uint32_t;
    static constexpr uint64_t kBitsPerWord = sizeof(WordType) * 8;

public:
    explicit VisitedList(InnerIdType max_size, Allocator* allocator);
    ~VisitedList() override;

    void
    Set(const InnerIdType& id) {
        const auto word_id = static_cast<uint64_t>(id) / kBitsPerWord;
        const auto mask = WordType{1} << (static_cast<uint64_t>(id) % kBitsPerWord);
        auto& word = this->words_[word_id];
        if (word == 0) {
            this->touched_words_[this->touched_count_++] =
                static_cast<TouchedIndexType>(word_id);
            word = mask;
        } else {
            word |= mask;
        }
    }

    [[nodiscard]] bool
    Get(const InnerIdType& id) {
        const auto word_id = static_cast<uint64_t>(id) / kBitsPerWord;
        const auto mask = WordType{1} << (static_cast<uint64_t>(id) % kBitsPerWord);
        return (this->words_[word_id] & mask) != 0;
    }

    // Marks id visited and returns whether it was previously unvisited.
    // Fuses the visitor-side Get-then-Set pair into a single word access.
    bool
    TestSet(const InnerIdType& id) {
        const auto word_id = static_cast<uint64_t>(id) / kBitsPerWord;
        const auto mask = WordType{1} << (static_cast<uint64_t>(id) % kBitsPerWord);
        auto& word = this->words_[word_id];
        if ((word & mask) != 0) {
            return false;
        }
        if (word == 0) {
            this->touched_words_[this->touched_count_++] =
                static_cast<TouchedIndexType>(word_id);
        }
        word |= mask;
        return true;
    }

    void
    Prefetch(const InnerIdType& id) {
        const auto word_id = static_cast<uint64_t>(id) / kBitsPerWord;
        PrefetchLines(this->words_ + word_id, 64);
    }

    void
    Reset() override {
        for (uint32_t i = 0; i < this->touched_count_; ++i) {
            this->words_[this->touched_words_[i]] = 0;
        }
        this->touched_count_ = 0;
    }

    uint64_t
    GetMemoryUsage() const override {
        return sizeof(VisitedList) +
               this->word_count_ * (sizeof(WordType) + sizeof(TouchedIndexType));
    }

private:
    Allocator* const allocator_{nullptr};

    WordType* words_{nullptr};

    TouchedIndexType* touched_words_{nullptr};

    uint32_t touched_count_{0};

    const uint64_t word_count_{0};
};

using VisitedListPool = ResourceObjectPool<VisitedList>;
}  // namespace vsag
