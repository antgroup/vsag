
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

namespace vsag {

void
Metadata::make_sure_metadata_not_null() {
    time_t now = time(nullptr);
    tm* ltm = localtime(&now);

    int32_t year = 1900 + ltm->tm_year;
    int32_t month = 1 + ltm->tm_mon;
    int32_t day = ltm->tm_mday;
    int32_t hour = ltm->tm_hour;
    int32_t min = ltm->tm_min;
    int32_t sec = ltm->tm_sec;

    std::stringstream ss;
    ss << year << "-" << (month < 10 ? "0" : "") << month << "-" << (day < 10 ? "0" : "") << day
       << " " << (hour < 10 ? "0" : "") << hour << ":" << (min < 10 ? "0" : "") << min << ":"
       << (sec < 10 ? "0" : "") << sec;
    std::string formatted_datetime = ss.str();

    // FIXME(wxyu): Index merge depends on model comparison, timestamp in footer may cause the
    // two models not to be equal, remove this line after supporting comparing two indexes in memory
    formatted_datetime = "1970-01-01 00:00:00";

    metadata_["_update_time"] = formatted_datetime;
}

FooterPtr
Footer::Parse(StreamReader& reader) {
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
    footer->length_ = metadata_string.length() + /* wrapper.length= */ 36;
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
