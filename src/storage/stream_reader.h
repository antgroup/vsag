
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

#include "../typing.h"
#include "impl/logger/logger.h"

class SliceStreamReader;

class StreamReader {
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

public:
    virtual void
    Read(char* data, uint64_t size) = 0;

    virtual void
    Seek(uint64_t cursor) = 0;

    [[nodiscard]] virtual uint64_t
    GetCursor() const = 0;

    [[nodiscard]] virtual uint64_t
    Length() {
        return length_;
    }

public:
    [[nodiscard]] SliceStreamReader
    Slice(uint64_t begin, uint64_t length);

    [[nodiscard]] SliceStreamReader
    Slice(uint64_t length);

    void
    PushSeek(uint64_t cursor) {
        positions_.push(this->GetCursor());
        // vsag::logger::trace("reader goto relative::{}", cursor);
        this->Seek(cursor);
    }

    void
    PopSeek() {
        // vsag::logger::trace("reader goback relative::{}", positions_.top());
        this->Seek(positions_.top());
        positions_.pop();
    }

public:
    StreamReader() = default;
    StreamReader(uint64_t length) : length_(length) {
    }

protected:
    uint64_t length_{0};
    uint64_t io_count_{0};

private:
    std::stack<uint64_t> positions_;
};

class ReadFuncStreamReader : public StreamReader {
public:
    void
    Read(char* data, uint64_t size) override;

    void
    Seek(uint64_t cursor) override;

    [[nodiscard]] uint64_t
    GetCursor() const override;

    ~ReadFuncStreamReader();

public:
    ReadFuncStreamReader(std::function<void(uint64_t, uint64_t, void*)> read_func,
                         uint64_t cursor,
                         uint64_t length);

private:
    const std::function<void(uint64_t, uint64_t, void*)> readFunc_;
    uint64_t cursor_{0};
};

class IOStreamReader : public StreamReader {
public:
    void
    Read(char* data, uint64_t size) override;

    void
    Seek(uint64_t cursor) override;

    [[nodiscard]] uint64_t
    GetCursor() const override;

    ~IOStreamReader();

public:
    explicit IOStreamReader(std::istream& istream);

private:
    std::istream& istream_;
};

class BufferStreamReader : public StreamReader {
public:
    [[nodiscard]] uint64_t
    Length() override;

    void
    Read(char* data, uint64_t size) override;

    void
    Seek(uint64_t cursor) override;

    [[nodiscard]] uint64_t
    GetCursor() const override;

public:
    explicit BufferStreamReader(StreamReader* reader, size_t max_size, vsag::Allocator* allocator);

    ~BufferStreamReader();

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

class SliceStreamReader : public StreamReader {
public:
    [[nodiscard]] uint64_t
    Length() override;

    void
    Read(char* data, uint64_t size) override;

    void
    Seek(uint64_t cursor) override;

    [[nodiscard]] uint64_t
    GetCursor() const override;

public:
    // create a slice from specified position
    SliceStreamReader(StreamReader* reader, uint64_t begin, uint64_t length);
    // create a slice from current position
    SliceStreamReader(StreamReader* reader, uint64_t length);

private:
    StreamReader* const reader_impl_{nullptr};
    uint64_t begin_{0};
    uint64_t cursor_{0};
};
