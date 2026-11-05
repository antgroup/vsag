
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

#include "visited_list.h"

#include <cstring>
#include <limits>

namespace vsag {
VisitedList::VisitedList(InnerIdType max_size, Allocator* allocator)
    : allocator_(allocator),
      word_count_((static_cast<uint64_t>(max_size) + kBitsPerWord - 1) / kBitsPerWord) {
    if (word_count_ == 0) {
        return;
    }
    const auto words_bytes = word_count_ * sizeof(WordType);
    const auto tags_bytes = word_count_ * sizeof(TagType);
    auto* buffer = static_cast<uint8_t*>(allocator_->Allocate(words_bytes + tags_bytes));
    this->words_ = reinterpret_cast<WordType*>(buffer);
    this->tags_ = reinterpret_cast<TagType*>(buffer + words_bytes);
    memset(tags_, 0, tags_bytes);
}

VisitedList::~VisitedList() {
    if (words_ != nullptr) {
        allocator_->Deallocate(words_);
    }
}

void
VisitedList::Reset() {
    if (tag_ == std::numeric_limits<TagType>::max()) {
        if (word_count_ > 0) {
            memset(tags_, 0, word_count_ * sizeof(TagType));
        }
        tag_ = 1;
    } else {
        ++tag_;
    }
}
}  // namespace vsag
