
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

// Metadata is using to describe how is the index create
DEFINE_POINTER(Metadata);
class Metadata {
public:
    [[nodiscard]] JsonType
    Get(const std::string& name) const {
        return metadata_[name];
    }

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
    [[nodiscard]] std::string
    Version() const {
        if (metadata_.Contains("_version")) {
            return metadata_["_version"].GetString();
        }
        return "";
    }

    void
    SetVersion(const std::string& version) {
        metadata_["_version"].SetString(version);
    }

    [[nodiscard]] bool
    EmptyIndex() const {
        return metadata_.Contains("_empty") and metadata_["_empty"].GetBool();
    }

    void
    SetEmptyIndex(bool empty) {
        metadata_["_empty"].SetBool(empty);
    }

public:
    std::string
    ToString() {
        make_sure_metadata_not_null();
        return metadata_.Dump();
    }

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
    Metadata(std::string str) {
        metadata_ = JsonType::Parse(str);
    }
    Metadata(const Binary& binary) {
        auto str = std::string((char*)binary.data.get(), binary.size);
        metadata_ = JsonType::Parse(str);
    }
    Metadata(JsonType metadata) : metadata_(std::move(metadata)) {
    }
    Metadata() = default;
    ~Metadata() = default;

private:
    void
    make_sure_metadata_not_null();

private:
    JsonType metadata_;
};

// Footer is a wrapper of metadata, only used in all-in-one serialize format
DEFINE_POINTER(Footer);
class Footer {
public:
    static FooterPtr
    Parse(StreamReader& reader);

    /* [magic (8B)] [length_of_metadata (8B)] [metadata (*B)] [checksum (4B)] [length_of_footer (8B)] [cigam (8B)] */
    void
    Write(StreamWriter& writer);

public:
    [[nodiscard]] MetadataPtr
    GetMetadata() const {
        return metadata_;
    }

    [[nodiscard]] uint64_t
    Length() const {
        return length_;
    }

public:
    Footer(MetadataPtr metadata) : metadata_(std::move(metadata)) {
    }
    virtual ~Footer() = default;

private:
    // TODO(wxyu): optimize performance
    static uint32_t
    calculate_checksum(std::string_view bytes) {
        const uint32_t polynomial = 0xEDB88320;
        uint32_t crc = 0xFFFFFFFF;

        for (const char& byte : bytes) {
            crc ^= byte;
            for (size_t j = 0; j < 8; ++j) {
                crc = (crc >> 1) ^ ((crc & 1) == 1 ? polynomial : 0);
            }
        }

        return crc ^ 0xFFFFFFFF;
    }

private:
    MetadataPtr metadata_{nullptr};
    uint64_t length_{0};
};

};  // namespace vsag
