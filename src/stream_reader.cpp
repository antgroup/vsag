
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

#include "stream_reader.h"

#include <iostream>
ReadFuncStreamReader::ReadFuncStreamReader(
    const std::function<void(uint64_t, uint64_t, void*)>& read_func,
    uint64_t cursor,
    size_t max_size,
    vsag::Allocator* allocator,
    bool use_buffer)
    : readFunc_(read_func), StreamReader(allocator) {
    cursor_ = cursor;
    use_buffer_ = use_buffer;
    max_size_ = max_size;
    if (use_buffer_) {
        buffer_size_ = std::min(max_size_, vsag::Options::Instance().block_size_limit());
        buffer_.resize(buffer_size_);
        buffer_cursor_ = buffer_size_;
    }
}

void
ReadFuncStreamReader::ReadImpl(char* data, uint64_t size) {
    readFunc_(cursor_, size, data);
    cursor_ += size;
}

IOStreamReader::IOStreamReader(std::istream& istream, vsag::Allocator* allocator, bool use_buffer)
    : istream_(istream), StreamReader(allocator) {
    std::streampos current_position = istream.tellg();
    istream.seekg(0, std::ios::end);
    std::streamsize size = istream.tellg();
    istream.seekg(current_position);
    cursor_ = 0;
    max_size_ = size - current_position;

    if (use_buffer_) {
        buffer_size_ = std::min(max_size_, vsag::Options::Instance().block_size_limit());
        buffer_.resize(buffer_size_);
        buffer_cursor_ = buffer_size_;
    }
}

void
IOStreamReader::ReadImpl(char* data, uint64_t size) {
    this->istream_.read(data, static_cast<int64_t>(size));
    cursor_ += size;
}
