
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

#include "stream_writer.h"

#include <cstring>
#include <utility>

BufferStreamWriter::BufferStreamWriter(char* buffer) : buffer_(buffer) {
}

void
BufferStreamWriter::Write(const char* data, uint64_t size) {
    memcpy(buffer_, data, size);
    buffer_ += size;
    bytes_written_ += size;
}

IOStreamWriter::IOStreamWriter(std::ostream& ostream) : ostream_(ostream) {
}

void
IOStreamWriter::Write(const char* data, uint64_t size) {
    ostream_.write(data, static_cast<int64_t>(size));
    bytes_written_ += size;
}

WriteFuncStreamWriter::WriteFuncStreamWriter(
    std::function<void(uint64_t, uint64_t, void*)> writeFunc, uint64_t cursor)
    : writeFunc_(std::move(writeFunc)), cursor_(cursor) {
}

void
WriteFuncStreamWriter::Write(const char* data, uint64_t size) {
    this->writeFunc_(cursor_, size, (void*)data);
    cursor_ += size;
    bytes_written_ += size;
}
