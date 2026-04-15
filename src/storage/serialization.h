
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
#include <ctime>
#include <iostream>
#include <memory>
#include <string>

#include "../typing.h"
#include "impl/logger/logger.h"
#include "stream_reader.h"
#include "stream_writer.h"
#include "utils/function_exists_check.h"
#include "utils/pointer_define.h"
#include "vsag/binaryset.h"
#include "vsag/constants.h"

namespace vsag {

/// Metadata is using to describe how is the index create
DEFINE_POINTER(Metadata);

/**
 * @brief Metadata container for index serialization information.
 *
 * This class stores and manages metadata about the index structure,
 * including version information, creation parameters, and custom attributes.
 * Reserved keys starting with underscore (_) are used for internal metadata.
 */
class Metadata {
public:
    /**
     * @brief Gets a metadata value by name.
     *
     * @param name The metadata key to lookup.
     * @return JsonType containing the metadata value.
     */
    [[nodiscard]] JsonType
    Get(const std::string& name) const {
        return metadata_[name];
    }

    /**
     * @brief Sets a metadata value.
     *
     * Note: Keys starting with underscore (_) are reserved for internal use.
     *
     * @tparam T The type of the value (string, bool, int, float, or JsonType).
     * @param name The metadata key.
     * @param value The metadata value.
     */
    template <typename T>
    void
    Set(const std::string& name, T value) {
        // name `_[0-9a-z_]*` is reserved
        if (name.empty() or name[0] == '_') {
            return;
        }
        if constexpr (std::is_same_v<T, std::string>) {
            metadata_[name].SetString(value);
        } else if constexpr (std::is_same_v<T, bool>) {
            metadata_[name].SetBool(value);
        } else if constexpr (std::is_integral_v<T>) {
            metadata_[name].SetInt(value);
        } else if constexpr (std::is_same_v<T, float>) {
            metadata_[name].SetFloat(value);
        } else if constexpr (std::is_same_v<T, JsonType>) {
            metadata_[name].SetJson(value);
        }
    }

public:
    /**
     * @brief Gets the serialization format version.
     *
     * @return Version string, or empty if not set.
     */
    [[nodiscard]] std::string
    Version() const {
        if (metadata_.Contains("_version")) {
            return metadata_["_version"].GetString();
        }
        return "";
    }

    /**
     * @brief Sets the serialization format version.
     *
     * @param version The version string to set.
     */
    void
    SetVersion(const std::string& version) {
        metadata_["_version"].SetString(version);
    }

    /**
     * @brief Checks if this metadata represents an empty index.
     *
     * @return True if empty index flag is set, false otherwise.
     */
    [[nodiscard]] bool
    EmptyIndex() const {
        return metadata_.Contains("_empty") and metadata_["_empty"].GetBool();
    }

    /**
     * @brief Sets the empty index flag.
     *
     * @param empty True if index is empty, false otherwise.
     */
    void
    SetEmptyIndex(bool empty) {
        metadata_["_empty"].SetBool(empty);
    }

public:
    /**
     * @brief Converts metadata to a JSON string.
     *
     * @return JSON string representation of the metadata.
     */
    std::string
    ToString() {
        make_sure_metadata_not_null();
        return metadata_.Dump();
    }

    /**
     * @brief Converts metadata to a Binary object.
     *
     * @return Binary object containing the serialized metadata.
     */
    Binary
    ToBinary() {
        auto str = this->ToString();

        std::shared_ptr<int8_t[]> bin(new int8_t[str.length()]);
        Binary b{
            .data = bin,
            .size = str.length(),
        };
        memcpy(bin.get(), str.c_str(), str.length());

        return b;
    }

public:
    /**
     * @brief Constructs metadata from a JSON string.
     *
     * @param str JSON string to parse.
     */
    Metadata(std::string str) {
        metadata_ = JsonType::Parse(str);
    }
    /**
     * @brief Constructs metadata from a Binary object.
     *
     * @param binary Binary object containing serialized metadata.
     */
    Metadata(const Binary& binary) {
        auto str = std::string((char*)binary.data.get(), binary.size);
        metadata_ = JsonType::Parse(str);
    }
    /**
     * @brief Constructs metadata from a JsonType object.
     *
     * @param metadata JsonType object to use as metadata.
     */
    Metadata(JsonType metadata) : metadata_(std::move(metadata)) {
    }
    Metadata() = default;
    ~Metadata() = default;

private:
    void
    make_sure_metadata_not_null();

private:
    /// JSON object storing all metadata entries.
    JsonType metadata_;
};

/// Footer is a wrapper of metadata, only used in all-in-one serialize format
DEFINE_POINTER(Footer);

/**
 * @brief Footer wrapper for metadata in all-in-one serialization format.
 *
 * This class wraps a Metadata object and provides serialization/deserialization
 * with checksum validation for the footer section of index files.
 */
class Footer {
public:
    /**
     * @brief Parses and constructs a Footer from a stream.
     *
     * @param reader The stream reader containing the footer data.
     * @return FooterPtr pointing to the parsed Footer object.
     */
    static FooterPtr
    Parse(StreamReader& reader);

    /**
     * @brief Writes the footer to a stream.
     *
     * Format: [magic (8B)] [length_of_metadata (8B)] [metadata (*B)] [checksum (4B)] [length_of_footer (8B)] [magic (8B)]
     *
     * @param writer The stream writer to write to.
     */
    void
    Write(StreamWriter& writer);

public:
    /**
     * @brief Gets the metadata contained in this footer.
     *
     * @return MetadataPtr pointing to the metadata object.
     */
    [[nodiscard]] MetadataPtr
    GetMetadata() const {
        return metadata_;
    }

    /**
     * @brief Gets the total length of the footer section.
     *
     * @return Length in bytes.
     */
    [[nodiscard]] uint64_t
    Length() const {
        return length_;
    }

public:
    /**
     * @brief Constructs a Footer with the given metadata.
     *
     * @param metadata Pointer to the metadata object.
     */
    Footer(MetadataPtr metadata) : metadata_(std::move(metadata)) {
    }
    virtual ~Footer() = default;

private:
    /**
     * @brief Calculates CRC32 checksum for the given bytes.
     *
     * @param bytes The data to calculate checksum for.
     * @return CRC32 checksum value.
     */
    static uint32_t
    calculate_checksum(std::string_view bytes) {
        const uint32_t polynomial = 0xEDB88320;
        uint32_t crc = 0xFFFFFFFF;

        for (const char& byte : bytes) {
            crc ^= byte;
            for (uint64_t j = 0; j < 8; ++j) {
                crc = (crc >> 1) ^ ((crc & 1) == 1 ? polynomial : 0);
            }
        }

        return crc ^ 0xFFFFFFFF;
    }

private:
    /// Pointer to the metadata object.
    MetadataPtr metadata_{nullptr};
    /// Total length of the footer section in bytes.
    uint64_t length_{0};
};

};  // namespace vsag
