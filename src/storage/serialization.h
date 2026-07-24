
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
#include <string_view>

#include "../typing.h"
#include "impl/logger/logger.h"
#include "stream_reader.h"
#include "stream_writer.h"
#include "utils/function_exists_check.h"
#include "utils/pointer_define.h"
#include "vsag/binaryset.h"
#include "vsag/constants.h"

namespace vsag {

constexpr uint64_t STREAM_HEADER_MAX_METADATA_LEN = 16ULL * 1024ULL * 1024ULL;

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

struct StreamHeaderData {
    MetadataPtr metadata;
    std::string metadata_string;
};

class StreamHeader {
public:
    static MetadataPtr
    Read(StreamReader& reader);

    static StreamHeaderData
    ReadRaw(StreamReader& reader);

    static void
    Write(StreamWriter& writer, const MetadataPtr& metadata);

    static uint32_t
    InitialChecksum();

    static uint32_t
    UpdateChecksum(uint32_t crc, std::string_view bytes);

    static uint32_t
    FinalizeChecksum(uint32_t crc);

    static uint32_t
    CalculateChecksum(std::string_view bytes);
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
    static const uint32_t*
    crc32_table() {
        static constexpr uint32_t table[256] = {
            0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F, 0xE963A535,
            0x9E6495A3, 0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988, 0x09B64C2B, 0x7EB17CBD,
            0xE7B82D07, 0x90BF1D91, 0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE, 0x1ADAD47D,
            0x6DDDE4EB, 0xF4D4B551, 0x83D385C7, 0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC,
            0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5, 0x3B6E20C8, 0x4C69105E, 0xD56041E4,
            0xA2677172, 0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B, 0x35B5A8FA, 0x42B2986C,
            0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59, 0x26D930AC,
            0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
            0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB,
            0xB6662D3D, 0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F,
            0x9FBFE4A5, 0xE8B8D433, 0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB,
            0x086D3D2D, 0x91646C97, 0xE6635C01, 0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
            0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457, 0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA,
            0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65, 0x4DB26158, 0x3AB551CE,
            0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A,
            0x346ED9FC, 0xAD678846, 0xDA60B8D0, 0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9,
            0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409,
            0xCE61E49F, 0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81,
            0xB7BD5C3B, 0xC0BA6CAD, 0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A, 0xEAD54739,
            0x9DD277AF, 0x04DB2615, 0x73DC1683, 0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8,
            0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1, 0xF00F9344, 0x8708A3D2, 0x1E01F268,
            0x6906C2FE, 0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7, 0xFED41B76, 0x89D32BE0,
            0x10DA7A5A, 0x67DD4ACC, 0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5, 0xD6D6A3E8,
            0xA1D1937E, 0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
            0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55, 0x316E8EEF,
            0x4669BE79, 0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236, 0xCC0C7795, 0xBB0B4703,
            0x220216B9, 0x5505262F, 0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7,
            0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D, 0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A,
            0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713, 0x95BF4A82, 0xE2B87A14, 0x7BB12BAE,
            0x0CB61B38, 0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21, 0x86D3D2D4, 0xF1D4E242,
            0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777, 0x88085AE6,
            0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45,
            0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2, 0xA7672661, 0xD06016F7, 0x4969474D,
            0x3E6E77DB, 0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5,
            0x47B2CF7F, 0x30B5FFE9, 0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605,
            0xCDD70693, 0x54DE5729, 0x23D967BF, 0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
            0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D,
        };
        return table;
    }

    // Keep this byte-for-byte compatible with the legacy footer checksum.
    static uint32_t
    calculate_checksum(std::string_view bytes) {
        const uint32_t* table = crc32_table();
        uint32_t crc = 0xFFFFFFFF;
        for (const char& byte : bytes) {
            crc = table[(crc ^ static_cast<uint8_t>(byte)) & 0xFF] ^ (crc >> 8);
        }
        return crc ^ 0xFFFFFFFF;
    }

private:
    MetadataPtr metadata_{nullptr};
    uint64_t length_{0};
};

};  // namespace vsag
