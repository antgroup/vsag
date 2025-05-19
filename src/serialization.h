
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
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "logger.h"
#include "stream_reader.h"
#include "stream_writer.h"
#include "typing.h"
#include "utils/function_exists_check.h"
#include "vsag/constants.h"

namespace vsag {

// Metadata is using to describe how is the index create
class Metadata;
using MetadataPtr = std::shared_ptr<Metadata>;
class Metadata {
public:
    [[nodiscard]] JsonType
    Get(std::string_view name) const {
        return metadata_[name];
    }

    void
    Set(const std::string& name, JsonType jsonify_obj) {
        // name `_[0-9a-z_]*` is reserved
        if (name.empty() or name[0] == '_') {
            return;
        }
        metadata_[name] = std::move(jsonify_obj);
    }

public:
    [[nodiscard]] std::string
    Version() const {
        return metadata_["_version"];
    }

    void
    SetVersion(const std::string& version) {
        metadata_["_version"] = version;
    }

public:
    std::string
    Dump() {
        return metadata_.dump();
    }

public:
    Metadata(JsonType metadata) : metadata_(std::move(metadata)) {
    }
    Metadata() = default;
    ~Metadata() = default;

private:
    JsonType metadata_;
};

// Footer is a wrapper of metadata, only used in all-in-one serialize format
class Footer;
using FooterPtr = std::shared_ptr<Footer>;
class Footer {
public:
    static FooterPtr
    Parse(StreamReader& reader) {
        // check cigam
        reader.PushSeek(reader.Length() - 8);
        char cigam[9] = {};
        reader.Read(cigam, 8);
        logger::debug("deserial cigam: {}", cigam);
        if (strcmp(cigam, SERIAL_MAGIC_END) != 0) {
            reader.PopSeek();
            return nullptr;
        }
        reader.PopSeek();

        // get footer length
        reader.PushSeek(reader.Length() - 16);
        uint64_t length;
        StreamReader::ReadObj(reader, length);
        logger::debug("deserial length: {}", length);
        if (length > reader.Length()) {
            reader.PopSeek();
            return nullptr;
        }
        reader.PopSeek();

        // check magic
        reader.PushSeek(reader.Length() - length);
        char magic[9] = {};
        reader.Read(magic, 8);
        logger::debug("deserial magic: {}", magic);
        if (strcmp(magic, SERIAL_MAGIC_BEGIN) != 0) {
            reader.PopSeek();
            return nullptr;
        }
        // no popseek, continue to parse

        auto metadata_string = StreamReader::ReadString(reader);
        uint32_t checksum;
        StreamReader::ReadObj(reader, checksum);
        logger::debug("deserial checksum: 0x{:x}", checksum);
        if (calculate_checksum(metadata_string) != checksum) {
            reader.PopSeek();
            return nullptr;
        }
        reader.PopSeek();

        auto metadata = std::make_shared<Metadata>(JsonType::parse(metadata_string));
        auto footer = std::make_shared<Footer>(metadata);
        return footer;
    }

    /* [magic (8B)] [length_of_metadata (8B)] [metadata (*B)] [checksum (4B)] [length_of_footer (8B)] [cigam (8B)] */
    void
    Write(StreamWriter& writer) {
        uint64_t length = 0;

        std::string magic = SERIAL_MAGIC_BEGIN;
        logger::debug("serial magic: {}", magic);
        writer.Write(magic.c_str(), 8);
        length += 8;

        auto metadata_string = metadata_->Dump();
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

public:
    [[nodiscard]] MetadataPtr
    GetMetadata() const {
        return metadata_;
    }

public:
    Footer(MetadataPtr metadata) : metadata_(std::move(metadata)) {
    }
    ~Footer() = default;

private:
    static uint32_t
    calculate_checksum(std::string_view bytes) {
        const uint32_t polynomial = 0xEDB88320;
        uint32_t crc = 0xFFFFFFFF;

        for (const char& byte : bytes) {
            crc ^= byte;
            for (size_t j = 0; j < 8; ++j) {
                crc = (crc >> 1) ^ ((crc != 0U) ? polynomial : 0);
            }
        }

        return crc ^ 0xFFFFFFFF;
    }

private:
    MetadataPtr metadata_ = nullptr;
};

};  // namespace vsag

// namespace vsag
