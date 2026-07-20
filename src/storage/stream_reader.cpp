
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

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>

#include "footer.h"
#include "impl/logger/logger.h"
#include "vsag/options.h"
#include "vsag_exception.h"

namespace vsag {

void
SkipForward(StreamReader& reader, uint64_t size) {
    constexpr uint64_t buffer_size = 8192;
    std::array<char, buffer_size> buffer{};
    uint64_t remaining = size;
    while (remaining > 0) {
        uint64_t read_size = std::min<uint64_t>(remaining, buffer_size);
        reader.Read(buffer.data(), read_size);
        remaining -= read_size;
    }
}

SliceStreamReader
StreamReader::Slice(uint64_t begin, uint64_t length) {
    return {this, begin, length};
}

SliceStreamReader
StreamReader::Slice(uint64_t length) {
    return {this, length};
}

void
ReadFuncStreamReader::Read(char* data, uint64_t size) {
    readFunc_(cursor_, size, data);
    cursor_ += size;
    io_count_++;
}

void
ReadFuncStreamReader::Seek(uint64_t cursor) {
    cursor_ = cursor;
}

uint64_t
ReadFuncStreamReader::GetCursor() const {
    return cursor_;
}

ReadFuncStreamReader::ReadFuncStreamReader(std::function<void(uint64_t, uint64_t, void*)> read_func,
                                           uint64_t cursor,
                                           uint64_t length)
    : StreamReader(length), readFunc_(std::move(read_func)), cursor_(cursor) {
}

ReadFuncStreamReader::~ReadFuncStreamReader() {
    vsag::logger::debug("ReadFuncStreamReader io count ({}) in deserialize process", io_count_);
}

void
IOStreamReader::Read(char* data, uint64_t size) {
    auto offset = std::to_string(istream_.tellg());
    // vsag::logger::trace("io read offset {} size {}", offset, size);
    this->istream_.read(data, static_cast<int64_t>(size));
    if (istream_.fail()) {
        auto remaining = std::streamsize(this->istream_.gcount());
        throw vsag::VsagException(
            vsag::ErrorType::READ_ERROR,
            fmt::format(
                "Attempted to read: {} bytes. Remaining content size: {} bytes.", size, remaining));
    }
    io_count_++;
}

void
IOStreamReader::Seek(uint64_t cursor) {
    // vsag::logger::trace("reader seek absolute::{}", cursor);
    istream_.seekg(static_cast<int64_t>(cursor), std::ios::beg);
}

uint64_t
IOStreamReader::GetCursor() const {
    uint64_t cursor = istream_.tellg();
    return cursor;
}

IOStreamReader::IOStreamReader(std::istream& istream) : istream_(istream) {
    auto cur_pos = istream.tellg();
    istream.seekg(0, std::ios::end);
    length_ = istream.tellg() - cur_pos;
    istream.seekg(cur_pos);
}

IOStreamReader::~IOStreamReader() {
    vsag::logger::info("IOStreamReader io count ({}) in deserialize process", io_count_);
}

void
ForwardStreamReader::Read(char* data, uint64_t size) {
    if (size == 0) {
        return;
    }
    if (size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw VsagException(ErrorType::READ_ERROR, "ForwardStreamReader read size is too large");
    }
    const auto stream_size = static_cast<std::streamsize>(size);
    istream_.read(data, stream_size);
    if (istream_.fail()) {
        auto bytes_read = istream_.gcount();
        throw vsag::VsagException(
            vsag::ErrorType::READ_ERROR,
            fmt::format(
                "Attempted to read: {} bytes. Bytes read before failure: {}.", size, bytes_read));
    }
    cursor_ += size;
    io_count_++;
}

void
ForwardStreamReader::Seek(uint64_t cursor) {
    (void)cursor;
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "ForwardStreamReader does not support seek");
}

uint64_t
ForwardStreamReader::GetCursor() const {
    return cursor_;
}

uint64_t
ForwardStreamReader::Length() {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "ForwardStreamReader does not support length");
}

ForwardStreamReader::ForwardStreamReader(std::istream& istream) : istream_(istream) {
}

uint64_t
BufferStreamReader::Length() {
    return reader_impl_->Length();
}

void
BufferStreamReader::Read(char* data, uint64_t size) {
    // Total bytes copied to dest
    uint64_t total_copied = 0;

    if (buffer_ == nullptr) {
        buffer_ = (char*)allocator_->Allocate(buffer_size_);
        if (buffer_ == nullptr) {
            throw vsag::VsagException(vsag::ErrorType::NO_ENOUGH_MEMORY,
                                      "fail to allocate buffer in BufferStreamReader");
        }
    }
    // Loop to read until read_size is satisfied
    while (total_copied < size) {
        // Calculate the available data in buffer_
        uint64_t available_in_src = valid_size_ - buffer_cursor_;

        // If there is available data in buffer_, copy it to dest
        if (available_in_src > 0) {
            uint64_t bytes_to_copy = std::min<uint64_t>(size - total_copied, available_in_src);
            memcpy(data + total_copied, buffer_ + buffer_cursor_, bytes_to_copy);
            total_copied += bytes_to_copy;
            buffer_cursor_ += bytes_to_copy;
        }
        // If we have copied enough data, we can exit
        if (total_copied >= size) {
            break;
        }

        // If buffer_ is full, reset cursor and read new data from reader
        buffer_cursor_ = 0;  // Reset cursor to overwrite buffer_'s content
        valid_size_ = std::min<uint64_t>(max_size_ - cursor_, buffer_size_);
        if (valid_size_ == 0) {
            throw vsag::VsagException(
                vsag::ErrorType::READ_ERROR,
                "BufferStreamReader: The file size is smaller than the memory you want to read.");
        }
        reader_impl_->Read(buffer_, valid_size_);
        cursor_ += valid_size_;
    }
}

void
BufferStreamReader::Seek(uint64_t cursor) {
    // vsag::logger::trace("reader seek absolute::{}", cursor);
    reader_impl_->Seek(cursor);
    buffer_cursor_ = valid_size_;  // record the invalidation of the buffer
    cursor_ = cursor;
}

uint64_t
BufferStreamReader::GetCursor() const {
    return reader_impl_->GetCursor() - (valid_size_ - buffer_cursor_);
}

BufferStreamReader::BufferStreamReader(StreamReader* reader,
                                       uint64_t max_size,
                                       vsag::Allocator* allocator)
    : reader_impl_(reader), max_size_(max_size), allocator_(allocator) {
    if (max_size == std::numeric_limits<uint64_t>::max()) {
        max_size_ = reader->Length() - reader->GetCursor();
    }
    buffer_size_ = std::min(max_size_, vsag::Options::Instance().block_size_limit());
    buffer_cursor_ = buffer_size_;
    valid_size_ = buffer_size_;
}

BufferStreamReader::~BufferStreamReader() {
    allocator_->Deallocate(buffer_);
}

uint64_t
SliceStreamReader::Length() {
    return length_;
}

void
SliceStreamReader::Read(char* data, uint64_t size) {
    if (cursor_ + size > length_) {
        throw vsag::VsagException(vsag::ErrorType::READ_ERROR,
                                  "SliceStreamReader: Read operation exceeds slice boundary");
    }
    reader_impl_->Read(data, size);
    cursor_ += size;
}

void
SliceStreamReader::Seek(uint64_t cursor) {
    if (cursor > length_) {
        throw vsag::VsagException(vsag::ErrorType::READ_ERROR,
                                  "SliceStreamReader: Seek operation exceeds slice boundary");
    }
    reader_impl_->Seek(begin_ + cursor);
    cursor_ = cursor;
}

uint64_t
SliceStreamReader::GetCursor() const {
    return cursor_;
}

SliceStreamReader::SliceStreamReader(StreamReader* reader, uint64_t begin, uint64_t length)
    : StreamReader(length), reader_impl_(reader), begin_(begin) {
    // vsag::logger::trace("SliceReader [{}, {})", begin_, begin_ + length_);
    reader_impl_->Seek(begin_);
}

SliceStreamReader::SliceStreamReader(StreamReader* reader, uint64_t length)
    : StreamReader(length), reader_impl_(reader) {
    begin_ = reader->GetCursor();
    // vsag::logger::trace("SliceReader [{}, {})", begin_, begin_ + length_);
}

void
BoundedForwardReader::Read(char* data, uint64_t size) {
    if (cursor_ > length_ || size > length_ - cursor_) {
        throw vsag::VsagException(vsag::ErrorType::READ_ERROR,
                                  "BoundedForwardReader: read exceeds block boundary");
    }
    reader_impl_->Read(data, size);
    cursor_ += size;
}

void
BoundedForwardReader::Seek(uint64_t cursor) {
    (void)cursor;
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "BoundedForwardReader does not support seek");
}

uint64_t
BoundedForwardReader::GetCursor() const {
    return cursor_;
}

uint64_t
BoundedForwardReader::Length() {
    return length_;
}

void
BoundedForwardReader::SkipRemaining() {
    if (cursor_ > length_) {
        throw vsag::VsagException(vsag::ErrorType::READ_ERROR,
                                  "BoundedForwardReader: cursor exceeds block boundary");
    }
    SkipForward(*reader_impl_, length_ - cursor_);
    cursor_ = length_;
}

BoundedForwardReader::BoundedForwardReader(StreamReader* reader, uint64_t length)
    : StreamReader(length), reader_impl_(reader) {
}

}  // namespace vsag
