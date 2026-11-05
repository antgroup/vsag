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

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "serialization_tags.h"

namespace vsag::test {

constexpr uint64_t STREAM_MAGIC_SIZE = 8;
constexpr uint64_t STREAM_VERSION_OFFSET = STREAM_MAGIC_SIZE;
constexpr uint64_t STREAM_MAJOR_OFFSET = STREAM_VERSION_OFFSET;
constexpr uint64_t STREAM_MINOR_OFFSET = STREAM_VERSION_OFFSET + sizeof(uint16_t);
constexpr uint64_t STREAM_METADATA_LENGTH_OFFSET = STREAM_VERSION_OFFSET + 2 * sizeof(uint16_t);
constexpr uint64_t STREAM_BLOCK_HEADER_SIZE = 4 + 4 + 8 + 8 + 4;
constexpr uint64_t STREAM_BLOCK_CRITICAL_FLAG = 1;

struct StreamingBlockSlice {
    uint32_t tag{0};
    uint64_t header_offset{0};
    uint64_t payload_offset{0};
    uint64_t payload_size{0};

    [[nodiscard]] uint64_t
    end_offset() const {
        return payload_offset + payload_size;
    }
};

template <typename T>
T
ReadStreamingObj(const std::string& bytes, uint64_t offset) {
    REQUIRE(offset + sizeof(T) <= bytes.size());
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

template <typename T>
void
WriteStreamingObj(std::string& bytes, uint64_t offset, T value) {
    REQUIRE(offset + sizeof(T) <= bytes.size());
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

inline uint64_t
StreamingBodyOffset(const std::string& bytes) {
    REQUIRE(bytes.size() >= STREAM_METADATA_LENGTH_OFFSET + sizeof(uint64_t));
    const auto metadata_len = ReadStreamingObj<uint64_t>(bytes, STREAM_METADATA_LENGTH_OFFSET);
    return STREAM_METADATA_LENGTH_OFFSET + sizeof(uint64_t) + metadata_len + sizeof(uint32_t);
}

inline std::vector<StreamingBlockSlice>
ParseStreamingBlocks(const std::string& bytes) {
    std::vector<StreamingBlockSlice> blocks;
    uint64_t cursor = StreamingBodyOffset(bytes);
    while (cursor < bytes.size()) {
        REQUIRE(cursor + STREAM_BLOCK_HEADER_SIZE <= bytes.size());
        const auto tag = ReadStreamingObj<uint32_t>(bytes, cursor);
        const auto value_len = ReadStreamingObj<uint64_t>(bytes, cursor + 4 + 4 + 8);
        blocks.push_back(
            StreamingBlockSlice{tag, cursor, cursor + STREAM_BLOCK_HEADER_SIZE, value_len});
        cursor += STREAM_BLOCK_HEADER_SIZE;
        if (tag == static_cast<uint32_t>(StreamSerializationTag::SECTION_END)) {
            break;
        }
        REQUIRE(cursor + value_len <= bytes.size());
        cursor += value_len;
    }
    return blocks;
}

inline StreamingBlockSlice
FindStreamingBlock(const std::string& bytes, StreamSerializationTag tag) {
    const auto tag_value = static_cast<uint32_t>(tag);
    for (const auto& block : ParseStreamingBlocks(bytes)) {
        if (block.tag == tag_value) {
            return block;
        }
    }
    FAIL("streaming block not found");
    return StreamingBlockSlice{};
}

inline std::string
SetStreamingMinorVersion(std::string bytes, uint16_t minor) {
    WriteStreamingObj<uint16_t>(bytes, STREAM_MINOR_OFFSET, minor);
    return bytes;
}

inline std::string
SetStreamingMajorVersion(std::string bytes, uint16_t major) {
    WriteStreamingObj<uint16_t>(bytes, STREAM_MAJOR_OFFSET, major);
    return bytes;
}

inline std::string
SetStreamingBlockVersion(std::string bytes, StreamSerializationTag tag, uint32_t version) {
    const auto block = FindStreamingBlock(bytes, tag);
    WriteStreamingObj<uint32_t>(bytes, block.header_offset + sizeof(uint32_t), version);
    return bytes;
}

inline std::string
EraseStreamingBlock(std::string bytes, StreamSerializationTag tag) {
    const auto block = FindStreamingBlock(bytes, tag);
    bytes.erase(static_cast<size_t>(block.header_offset),
                static_cast<size_t>(STREAM_BLOCK_HEADER_SIZE + block.payload_size));
    return bytes;
}

inline std::string
InsertUnknownStreamingBlock(std::string bytes, bool critical, uint32_t version = 1) {
    const auto section_end = FindStreamingBlock(bytes, StreamSerializationTag::SECTION_END);
    std::string block(STREAM_BLOCK_HEADER_SIZE, '\0');
    constexpr uint32_t unknown_tag = 0x00ABCDEF;
    constexpr uint64_t payload_size = 5;
    WriteStreamingObj<uint32_t>(block, 0, unknown_tag);
    WriteStreamingObj<uint32_t>(block, 4, version);
    WriteStreamingObj<uint64_t>(block, 8, critical ? STREAM_BLOCK_CRITICAL_FLAG : 0);
    WriteStreamingObj<uint64_t>(block, 16, payload_size);
    WriteStreamingObj<uint32_t>(block, 24, 0);
    block.append("hello", payload_size);
    bytes.insert(static_cast<size_t>(section_end.header_offset), block);
    return bytes;
}

}  // namespace vsag::test
