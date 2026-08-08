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

#include <cstring>
#include <sstream>

#include "serialization.h"
#include "stream_reader.h"
#include "unittest.h"

namespace {
constexpr uint64_t FOOTER_MIN_SIZE = 36;

// Local mirror of the non-standard CRC32 written by old vsag versions:
// `crc ^= byte` sign-extends the (signed) char. Used here to emulate footers
// produced by those versions; must stay independent from the implementation
// under test so removing the read-side fallback makes the tests fail.
uint32_t
legacy_footer_checksum(const std::string& bytes) {
    constexpr uint32_t polynomial = 0xEDB88320;
    uint32_t crc = 0xFFFFFFFF;
    for (const char& byte : bytes) {
        crc ^= byte;
        for (uint64_t j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ ((crc & 1) == 1 ? polynomial : 0);
        }
    }
    return crc ^ 0xFFFFFFFF;
}

// Builds a stream containing only a footer:
// [magic (8B)] [len_of_metadata (8B)] [metadata (*B)] [checksum (4B)] [len_of_footer (8B)] [cigam (8B)]
std::string
build_footer_stream(const std::string& metadata_string, uint32_t checksum) {
    const uint64_t metadata_len = metadata_string.size();
    const uint64_t footer_len = metadata_len + FOOTER_MIN_SIZE;
    std::string data;
    data.reserve(footer_len);
    data.append("vsag0000", 8);
    data.append(reinterpret_cast<const char*>(&metadata_len), 8);
    data.append(metadata_string);
    data.append(reinterpret_cast<const char*>(&checksum), 4);
    data.append(reinterpret_cast<const char*>(&footer_len), 8);
    data.append("0000gasv", 8);
    return data;
}
}  // namespace

TEST_CASE("Footer Parse rejects streams shorter than minimum footer size", "[ut][footer]") {
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    ss.str(std::string(FOOTER_MIN_SIZE - 1, 'a'));
    vsag::IOStreamReader reader(ss);
    REQUIRE(vsag::Footer::Parse(reader) == nullptr);
}

TEST_CASE("Footer Parse accepts legacy checksum for metadata with non-ASCII bytes",
          "[ut][footer]") {
    // UTF-8 metadata (bytes >= 0x80) makes the legacy and canonical CRC32 diverge
    const std::string metadata_string = R"({"index_name":"索引-テスト"})";
    const uint32_t legacy_checksum = legacy_footer_checksum(metadata_string);
    const uint32_t canonical_checksum = vsag::StreamHeader::CalculateChecksum(metadata_string);
    REQUIRE(legacy_checksum != canonical_checksum);

    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    ss.str(build_footer_stream(metadata_string, legacy_checksum));
    vsag::IOStreamReader reader(ss);
    auto footer = vsag::Footer::Parse(reader);
    REQUIRE(footer != nullptr);
    REQUIRE(footer->Length() == metadata_string.size() + FOOTER_MIN_SIZE);
}

TEST_CASE("Footer Parse rejects checksum matching neither canonical nor legacy", "[ut][footer]") {
    const std::string metadata_string = R"({"index_name":"索引-テスト"})";
    const uint32_t legacy_checksum = legacy_footer_checksum(metadata_string);
    const uint32_t canonical_checksum = vsag::StreamHeader::CalculateChecksum(metadata_string);
    uint32_t wrong_checksum = legacy_checksum + 1;
    while (wrong_checksum == legacy_checksum || wrong_checksum == canonical_checksum) {
        ++wrong_checksum;
    }

    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    ss.str(build_footer_stream(metadata_string, wrong_checksum));
    vsag::IOStreamReader reader(ss);
    REQUIRE(vsag::Footer::Parse(reader) == nullptr);
}

TEST_CASE("Legacy and canonical CRC32 agree on pure ASCII input", "[ut][footer]") {
    const std::string metadata_string = R"({"index_name":"ascii_only_index"})";
    REQUIRE(legacy_footer_checksum(metadata_string) ==
            vsag::StreamHeader::CalculateChecksum(metadata_string));
}

TEST_CASE("Footer Parse rejects mismatched metadata_string_length", "[ut][footer]") {
    const uint64_t footer_length = 44;
    std::string data(footer_length, '\0');
    std::memcpy(&data[0], "vsag0000", 8);
    const uint64_t wrong_metadata_length = 16;
    std::memcpy(&data[8], &wrong_metadata_length, sizeof(wrong_metadata_length));
    std::memcpy(&data[footer_length - 16], &footer_length, sizeof(footer_length));
    std::memcpy(&data[footer_length - 8], "0000gasv", 8);
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    ss.str(data);
    vsag::IOStreamReader reader(ss);
    REQUIRE(vsag::Footer::Parse(reader) == nullptr);
}
