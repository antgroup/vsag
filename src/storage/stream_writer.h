
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

#include "typing.h"

class StreamWriter {
public:
    template <typename T>
    static void
    WriteObj(StreamWriter& writer, const T& val) {
        writer.Write(reinterpret_cast<const char*>(&val), sizeof(val));
    }

    static void
    WriteString(StreamWriter& writer, const std::string& str) {
        size_t length = str.size();
        StreamWriter::WriteObj(writer, length);
        writer.Write(str.c_str(), length);
    }

    template <typename T>
    static void
    WriteVector(StreamWriter& writer, const std::vector<T>& val) {
        uint64_t size = val.size();
        WriteObj(writer, size);
        if (size > 0) {
            writer.Write(reinterpret_cast<const char*>(val.data()), size * sizeof(T));
        }
    }

    template <typename T>
    static void
    WriteVector(StreamWriter& writer, const vsag::Vector<T>& val) {
        uint64_t size = val.size();
        WriteObj(writer, size);
        if (size > 0) {
            writer.Write(reinterpret_cast<const char*>(val.data()), size * sizeof(T));
        }
    }

public:
    virtual void
    Write(const char* data, uint64_t size) = 0;

    [[nodiscard]] uint64_t
    GetCursor() const {
        return bytes_written_;
    }

public:
    StreamWriter() = default;

    virtual ~StreamWriter() = default;

protected:
    uint64_t bytes_written_{0};
};

class BufferStreamWriter : public StreamWriter {
public:
    explicit BufferStreamWriter(char* buffer);

    void
    Write(const char* data, uint64_t size) override;

private:
    char* buffer_{nullptr};
};

class IOStreamWriter : public StreamWriter {
public:
    explicit IOStreamWriter(std::ostream& ostream);

    void
    Write(const char* data, uint64_t size) override;

private:
    std::ostream& ostream_;
    uint64_t written_bytes_{0};
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
    uint64_t written_bytes_{0};
};
