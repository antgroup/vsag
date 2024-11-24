
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

#include <cstdint>
#include <functional>
#include <istream>

#include "typing.h"

class StreamReader {
public:
    StreamReader(vsag::Allocator* allocator) : buffer_(allocator) {
    }

    void
    Read(char* data, uint64_t size) {
        if (use_buffer_) {
            // Total bytes copied to dest
            size_t total_copied = 0;

            // Loop to read until read_size is satisfied
            while (total_copied < size) {
                // Calculate the available data in src
                size_t available_in_src = buffer_size_ - buffer_cursor_;

                // If there is available data in src, copy it to dest
                if (available_in_src > 0) {
                    size_t bytes_to_copy = std::min(size - total_copied, available_in_src);
                    memcpy(data + total_copied, buffer_.data() + buffer_cursor_, bytes_to_copy);
                    total_copied += bytes_to_copy;
                    buffer_cursor_ += bytes_to_copy;
                }

                // If we have copied enough data, we can exit
                if (total_copied >= size) {
                    break;
                }

                // If src is full, reset cursor and read new data from reader
                buffer_cursor_ = 0;  // Reset cursor to overwrite src's content
                auto read_size = std::min(max_size_ - cursor_, buffer_size_);
                ReadImpl(buffer_.data(), read_size);
            }
        } else {
            ReadImpl(data, size);
        }
    }

    template <typename T>
    static void
    ReadObj(StreamReader& reader, T& val) {
        reader.Read(reinterpret_cast<char*>(&val), sizeof(val));
    }

    template <typename T>
    static void
    ReadVector(StreamReader& reader, std::vector<T>& val) {
        uint64_t size;
        ReadObj(reader, size);
        val.resize(size);
        reader.Read(reinterpret_cast<char*>(val.data()), size * sizeof(T));
    }

    template <typename T>
    static void
    ReadVector(StreamReader& reader, vsag::Vector<T>& val) {
        uint64_t size;
        ReadObj(reader, size);
        val.resize(size);
        reader.Read(reinterpret_cast<char*>(val.data()), size * sizeof(T));
    }

private:
    virtual void
    ReadImpl(char* data, uint64_t size) = 0;

protected:
    vsag::Vector<char> buffer_;
    size_t buffer_cursor_{0};
    size_t buffer_size_{0};
    bool use_buffer_{false};
    size_t max_size_{0};
    size_t cursor_{0};
};

class ReadFuncStreamReader : public StreamReader {
public:
    ReadFuncStreamReader(const std::function<void(uint64_t, uint64_t, void*)>& read_func,
                         uint64_t cursor,
                         size_t max_size,
                         vsag::Allocator* allocator,
                         bool use_buffer = false);

    void
    ReadImpl(char* data, uint64_t size) override;

private:
    const std::function<void(uint64_t, uint64_t, void*)>& readFunc_;
};

class IOStreamReader : public StreamReader {
public:
    explicit IOStreamReader(std::istream& istream,
                            vsag::Allocator* allocator,
                            bool use_buffer = false);

    void
    ReadImpl(char* data, uint64_t size) override;

private:
    std::istream& istream_;
};
