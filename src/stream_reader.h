
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

#include "utils.h"

class StreamReader {
public:
    StreamReader() = default;

    virtual void
    Read(char* data, uint64_t size) = 0;

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
};

class ReadFuncStreamReader : public StreamReader {
public:
    ReadFuncStreamReader(const std::function<void(uint64_t, uint64_t, void*)>& read_func,
                         uint64_t cursor);

    void
    Read(char* data, uint64_t size) override;

private:
    const std::function<void(uint64_t, uint64_t, void*)>& readFunc_;
    uint64_t cursor_;
};

class IOStreamReader : public StreamReader {
public:
    explicit IOStreamReader(std::istream& istream);

    void
    Read(char* data, uint64_t size) override;

private:
    std::istream& istream_;
};
