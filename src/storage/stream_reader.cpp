
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

#include "storage/stream_reader.h"

#include <fmt/format-inl.h>

#include <cstdint>
#include <fstream>
#include <iostream>

#include "../logger.h"
#include "storage/footer.h"
#include "vsag/options.h"
#include "vsag_exception.h"

SliceReader
StreamReader::Slice(uint64_t begin, uint64_t length) {
    return {this, begin, length};
}

SliceReader
StreamReader::Slice(uint64_t length) {
    return {this, length};
}

ReadFuncStreamReader::ReadFuncStreamReader(std::function<void(uint64_t, uint64_t, void*)> read_func,
                                           uint64_t cursor,
                                           uint64_t length)
    : readFunc_(std::move(read_func)), cursor_(cursor), length_(length) {
}

uint64_t
ReadFuncStreamReader::Length() {
    return length_;
}

void
ReadFuncStreamReader::Read(char* data, uint64_t size) {
    readFunc_(cursor_, size, data);
    cursor_ += size;
}

void
ReadFuncStreamReader::Seek(uint64_t cursor) {
    cursor_ = cursor;
}

uint64_t
ReadFuncStreamReader::GetCursor() const {
    return cursor_;
}

IOStreamReader::IOStreamReader(std::istream& istream) : istream_(istream) {
    auto cur_pos = istream.tellg();
    istream.seekg(0, std::ios::end);
    length_ = istream.tellg() - cur_pos;
    istream.seekg(cur_pos);
}

uint64_t
IOStreamReader::Length() {
    return length_;
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

BufferStreamReader::BufferStreamReader(StreamReader* reader,
                                       size_t max_size,
                                       vsag::Allocator* allocator)
    : reader_impl_(reader), max_size_(max_size), allocator_(allocator) {
    buffer_size_ = std::min(max_size_, vsag::Options::Instance().block_size_limit());
    buffer_cursor_ = buffer_size_;
    valid_size_ = buffer_size_;
}

BufferStreamReader::~BufferStreamReader() {
    allocator_->Deallocate(buffer_);
}

uint64_t
BufferStreamReader::Length() {
    return reader_impl_->Length();
}

void
BufferStreamReader::Read(char* data, uint64_t size) {
    // Total bytes copied to dest
    size_t total_copied = 0;

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
        size_t available_in_src = valid_size_ - buffer_cursor_;

        // If there is available data in buffer_, copy it to dest
        if (available_in_src > 0) {
            size_t bytes_to_copy = std::min(size - total_copied, available_in_src);
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
        valid_size_ = std::min(max_size_ - cursor_, buffer_size_);
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

uint64_t
SliceReader::Length() {
    return length_;
}

void
SliceReader::Read(char* data, uint64_t size) {
    // TODO(wxyu): check if the read operation is out of bounds
    reader_impl_->Read(data, size);
    cursor_ += size;
}

void
SliceReader::Seek(uint64_t cursor) {
    // TODO(wxyu): check if the seek operation is out of bounds
    reader_impl_->Seek(begin_ + cursor);
    cursor_ = cursor;
}

uint64_t
SliceReader::GetCursor() const {
    return cursor_;
}
