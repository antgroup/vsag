
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
#include <stack>

#include "typing.h"

class StreamReader {
public:
    StreamReader() = default;

    [[nodiscard]] virtual uint64_t
    Length() = 0;

    virtual void
    Read(char* data, uint64_t size) = 0;

    virtual void
    Seek(uint64_t cursor) = 0;

    [[nodiscard]] virtual uint64_t
    GetCursor() const = 0;

public:
    void
    PushSeek(uint64_t cursor) {
        positions_.push(this->GetCursor());
        this->Seek(cursor);
    }

    void
    PopSeek() {
        this->Seek(positions_.top());
        positions_.pop();
    }

public:
    template <typename T>
    static void
    ReadObj(StreamReader& reader, T& val) {
        reader.Read(reinterpret_cast<char*>(&val), sizeof(val));
    }

    static std::string
    ReadString(StreamReader& reader) {
        size_t length = 0;
        StreamReader::ReadObj(reader, length);
        std::vector<char> buffer(length);
        reader.Read(buffer.data(), length);
        return {buffer.data(), length};
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
    std::stack<uint64_t> positions_;
};

class ReadFuncStreamReader : public StreamReader {
public:
    ReadFuncStreamReader(std::function<void(uint64_t, uint64_t, void*)> read_func,
                         uint64_t cursor,
                         uint64_t length);

    [[nodiscard]] uint64_t
    Length() override;

    void
    Read(char* data, uint64_t size) override;

    void
    Seek(uint64_t cursor) override;

    [[nodiscard]] uint64_t
    GetCursor() const override;

private:
    const std::function<void(uint64_t, uint64_t, void*)> readFunc_;
    uint64_t cursor_;
    uint64_t length_;
};

class IOStreamReader : public StreamReader {
public:
    explicit IOStreamReader(std::istream& istream);

    [[nodiscard]] uint64_t
    Length() override;

    void
    Read(char* data, uint64_t size) override;

    void
    Seek(uint64_t cursor) override;

    [[nodiscard]] uint64_t
    GetCursor() const override;

private:
    std::istream& istream_;
    uint64_t length_;
};

class BufferStreamReader : public StreamReader {
public:
    explicit BufferStreamReader(StreamReader* reader, size_t max_size, vsag::Allocator* allocator);

    ~BufferStreamReader();

    [[nodiscard]] uint64_t
    Length() override;

    void
    Read(char* data, uint64_t size) override;

    void
    Seek(uint64_t cursor) override;

    [[nodiscard]] uint64_t
    GetCursor() const override;

private:
    StreamReader* const reader_impl_{nullptr};
    vsag::Allocator* allocator_;
    char* buffer_{nullptr};    // Stores the cached content
    size_t buffer_cursor_{0};  // Current read position in the cache
    size_t valid_size_{0};     // Size of valid data in the cache
    size_t buffer_size_{0};    // Maximum capacity of the cache
    size_t max_size_{0};       // Maximum capacity of the actual data stream
    size_t cursor_{0};         // Current read position in the actual data stream
};
