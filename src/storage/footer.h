
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
#include <fmt/format.h>

#include "stream_reader.h"
#include "typing.h"
#include "vsag/constants.h"

namespace vsag {

/// Magic number for identifying index files ("CGPH" = 0x43475048).
static const std::string MAGIC_NUM = "43475048";
/// Current serialization format version.
static const std::string VERSION = "1";
/// Fixed size of the footer section in bytes (4KB).
static const int FOOTER_SIZE = 4096;

/**
 * @brief Footer section for serialized index files containing metadata.
 *
 * This class manages the footer section that stores metadata about the index,
 * including format version, creation time, and custom key-value pairs.
 */
class SerializationFooter {
public:
    SerializationFooter();

    /**
     * @brief Clears all metadata entries.
     */
    void
    Clear();

    /**
     * @brief Sets a metadata key-value pair.
     *
     * @param key The metadata key.
     * @param value The metadata value.
     */
    void
    SetMetadata(const std::string& key, const std::string& value);

    /**
     * @brief Gets metadata value for a given key.
     *
     * @param key The metadata key to lookup.
     * @return The metadata value, or empty string if key not found.
     */
    std::string
    GetMetadata(const std::string& key) const;

    /**
     * @brief Serializes the footer to an output stream.
     *
     * @param out_stream The output stream to write to.
     */
    void
    Serialize(std::ostream& out_stream) const;

    /**
     * @brief Deserializes the footer from an input stream.
     *
     * @param in_stream The stream reader to read from.
     */
    void
    Deserialize(StreamReader& in_stream);

private:
    /// JSON object storing all metadata entries.
    JsonType json_;
};

}  // namespace vsag
