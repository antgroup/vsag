
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

#include "serialization.h"

#include <cstring>

#include "vsag_exception.h"

namespace vsag {

namespace {

constexpr uint64_t FOOTER_TRAILER_SIZE = 16;
constexpr uint64_t FOOTER_MIN_SIZE = 36;

}  // namespace

void
Metadata::make_sure_metadata_not_null() {
    metadata_["_update_time"].SetString("1970-01-01 00:00:00");
}

uint32_t
StreamHeader::InitialChecksum() {
    return 0xFFFFFFFF;
}

uint32_t
StreamHeader::UpdateChecksum(uint32_t crc, std::string_view bytes) {
    constexpr uint32_t polynomial = 0xEDB88320;
    for (const char& byte : bytes) {
        crc ^= static_cast<uint8_t>(byte);
        for (uint64_t j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ ((crc & 1) == 1 ? polynomial : 0);
        }
    }
    return crc;
}

uint32_t
StreamHeader::FinalizeChecksum(uint32_t crc) {
    return crc ^ 0xFFFFFFFF;
}

uint32_t
StreamHeader::CalculateChecksum(std::string_view bytes) {
    return StreamHeader::FinalizeChecksum(
        StreamHeader::UpdateChecksum(StreamHeader::InitialChecksum(), bytes));
}

void
StreamHeader::Write(StreamWriter& writer, const MetadataPtr& metadata) {
    writer.Write(SERIAL_STREAM_MAGIC, 8);
    StreamWriter::WriteObj(writer, SERIAL_STREAM_FORMAT_MAJOR);
    StreamWriter::WriteObj(writer, SERIAL_STREAM_FORMAT_MINOR);

    auto metadata_string = metadata->ToString();
    const uint64_t metadata_len = metadata_string.size();
    if (metadata_len > STREAM_HEADER_MAX_METADATA_LEN) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            "streaming serialization metadata is too large");
    }
    StreamWriter::WriteObj(writer, metadata_len);
    writer.Write(metadata_string.c_str(), metadata_len);
    const uint32_t checksum = StreamHeader::CalculateChecksum(metadata_string);
    StreamWriter::WriteObj(writer, checksum);
}

MetadataPtr
StreamHeader::Read(StreamReader& reader) {
    return StreamHeader::ReadRaw(reader).metadata;
}

StreamHeaderData
StreamHeader::ReadRaw(StreamReader& reader) {
    char magic[8] = {};
    reader.Read(magic, 8);
    if (std::memcmp(magic, SERIAL_STREAM_MAGIC, 8) != 0) {
        throw VsagException(ErrorType::INVALID_BINARY, "invalid streaming serialization magic");
    }

    uint16_t major = 0;
    uint16_t minor = 0;
    StreamReader::ReadObj(reader, major);
    StreamReader::ReadObj(reader, minor);
    if (major != SERIAL_STREAM_FORMAT_MAJOR) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "unsupported streaming serialization major version");
    }

    uint64_t metadata_len = 0;
    StreamReader::ReadObj(reader, metadata_len);
    if (metadata_len > STREAM_HEADER_MAX_METADATA_LEN) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            "streaming serialization metadata is too large");
    }
    std::string metadata_string(metadata_len, '\0');
    reader.Read(metadata_string.data(), metadata_len);

    uint32_t checksum = 0;
    StreamReader::ReadObj(reader, checksum);
    if (StreamHeader::CalculateChecksum(metadata_string) != checksum) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            "streaming serialization metadata checksum mismatch");
    }

    (void)minor;
    auto metadata_json = JsonType::Parse(metadata_string, false);
    if (metadata_json.IsDiscarded() || !metadata_json.IsObject()) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            "streaming serialization metadata must be a JSON object");
    }
    StreamHeaderData result;
    result.metadata = std::make_shared<Metadata>(std::move(metadata_json));
    result.metadata_string = std::move(metadata_string);
    return result;
}

FooterPtr
Footer::Parse(StreamReader& reader) {
    const auto stream_length = reader.Length();
    if (stream_length < FOOTER_MIN_SIZE) {
        return nullptr;
    }
    // check cigam
    reader.PushSeek(stream_length - FOOTER_TRAILER_SIZE);
    char buffer[FOOTER_TRAILER_SIZE] = {};
    reader.Read(buffer, FOOTER_TRAILER_SIZE);
    char cigam[9] = {};
    memcpy(cigam, buffer + 8, 8);
    logger::debug("deserial cigam: {}", cigam);
    if (strcmp(cigam, SERIAL_MAGIC_END) != 0) {
        reader.PopSeek();
        return nullptr;
    }

    uint64_t length;
    memcpy(&length, buffer, 8);
    logger::debug("deserial length: {}", length);
    if (length < FOOTER_MIN_SIZE || length > stream_length) {
        reader.PopSeek();
        return nullptr;
    }
    reader.PopSeek();

    // check magic
    reader.PushSeek(reader.Length() - length);
    std::vector<char> meta_buffer(length);
    reader.Read(meta_buffer.data(), length);
    char magic[9] = {};
    memcpy(magic, meta_buffer.data(), 8);
    logger::debug("deserial magic: {}", magic);
    if (strcmp(magic, SERIAL_MAGIC_BEGIN) != 0) {
        reader.PopSeek();
        return nullptr;
    }
    // no popseek, continue to parse

    uint64_t metadata_string_length = 0;
    memcpy(&metadata_string_length, meta_buffer.data() + 8, 8);
    if (metadata_string_length != length - FOOTER_MIN_SIZE) {
        reader.PopSeek();
        return nullptr;
    }
    std::string metadata_string(meta_buffer.data() + 16, metadata_string_length);
    uint32_t checksum;
    memcpy(&checksum, meta_buffer.data() + 16 + metadata_string_length, 4);
    logger::debug("deserial checksum: 0x{:x}", checksum);
    if (calculate_checksum(metadata_string) != checksum) {
        reader.PopSeek();
        return nullptr;
    }
    reader.PopSeek();

    auto metadata = std::make_shared<Metadata>(JsonType::Parse(metadata_string));
    auto footer = std::make_shared<Footer>(metadata);
    footer->length_ = metadata_string.length() + /* wrapper.length= */ FOOTER_MIN_SIZE;
    return footer;
}

/* [magic (8B)] [length_of_metadata (8B)] [metadata (*B)] [checksum (4B)] [length_of_footer (8B)] [cigam (8B)] */
void
Footer::Write(StreamWriter& writer) {
    uint64_t length = 0;

    std::string magic = SERIAL_MAGIC_BEGIN;
    logger::debug("serial magic: {}", magic);
    writer.Write(magic.c_str(), 8);
    length += 8;

    auto metadata_string = metadata_->ToString();
    logger::debug("serial metadata: {}", metadata_string);
    StreamWriter::WriteString(writer, metadata_string);
    length += (8 + metadata_string.length());

    const uint32_t checksum = Footer::calculate_checksum(metadata_string);
    logger::debug("serial checksum: 0x{:x}", checksum);
    StreamWriter::WriteObj(writer, checksum);
    length += 4;

    length += (8 + 8);
    logger::debug("serial length_of_footer: {}", length);
    StreamWriter::WriteObj(writer, length);

    std::string cigam = SERIAL_MAGIC_END;
    logger::debug("serial cigam: {}", cigam);
    writer.Write(cigam.c_str(), 8);
}

}  // namespace vsag
