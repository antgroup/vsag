
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
#include <ostream>

#include "utils.h"

class StreamWriter {
public:
    StreamWriter() = default;

    virtual void
    Write(const char* data, uint64_t size) = 0;

    template <typename T>
    static void
    WriteObj(StreamWriter& writer, const T& val) {
        writer.Write(reinterpret_cast<const char*>(&val), sizeof(val));
    }

    template <typename T>
    static void
    WriteVector(StreamWriter& writer, std::vector<T>& val) {
        uint64_t size = val.size();
        WriteObj(writer, size);
        writer.Write(reinterpret_cast<char*>(val.data()), size * sizeof(T));
    }

    template <typename T>
    static void
    WriteVector(StreamWriter& writer, vsag::Vector<T>& val) {
        uint64_t size = val.size();
        WriteObj(writer, size);
        writer.Write(reinterpret_cast<char*>(val.data()), size * sizeof(T));
    }
};

class BufferStreamWriter : public StreamWriter {
public:
    explicit BufferStreamWriter(char*& buffer);

    void
    Write(const char* data, uint64_t size) override;

    char*& buffer_;
};

class IOStreamWriter : public StreamWriter {
public:
    explicit IOStreamWriter(std::ostream& ostream);

    void
    Write(const char* data, uint64_t size) override;

    std::ostream& ostream_;
};

class WriteFuncStreamWriter : public StreamWriter {
public:
    explicit WriteFuncStreamWriter(std::function<void(uint64_t, uint64_t, void*)> writeFunc,
                                   uint64_t cursor);

    void
    Write(const char* data, uint64_t size) override;

    std::function<void(uint64_t, uint64_t, void*)> writeFunc_;

public:
    uint64_t cursor_{0};
};
