
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

#include "append_buffer.h"

#include <cstring>

const char*
AppendBuffer::Append(const char* data, uint64_t size) {
    if (cursor_ + size <= BUFFER_SIZE) {
        memcpy((void*)(buffer_[buffer_.size() - 1] + cursor_), data, size);
        cursor_ += size;
        return buffer_[buffer_.size() - 1] + cursor_ - size;
    }
    buffer_.emplace_back(new char[BUFFER_SIZE]);
    cursor_ = 0;
    memcpy((void*)(buffer_[buffer_.size() - 1]), data, size);
    cursor_ += size;
    return buffer_[buffer_.size() - 1];
}