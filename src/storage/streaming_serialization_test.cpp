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

#include <limits>
#include <sstream>

#include "serialization.h"
#include "serialization_tags.h"
#include "tlv_section.h"
#include "unittest.h"
#include "vsag/options.h"

TEST_CASE("StreamHeader", "[ut][streaming_serialization]") {
    auto metadata = std::make_shared<vsag::Metadata>();
    metadata->Set("index_name", std::string("brute_force"));
    metadata->SetEmptyIndex(false);

    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    vsag::StreamHeader::Write(writer, metadata);

    auto bytes = stream.str();
    REQUIRE(bytes.substr(0, 8) == vsag::SERIAL_STREAM_MAGIC);

    vsag::ForwardStreamReader reader(stream);
    auto parsed = vsag::StreamHeader::Read(reader);
    REQUIRE(parsed->Get("index_name").GetString() == "brute_force");
    REQUIRE_FALSE(parsed->EmptyIndex());
}

TEST_CASE("StreamHeader rejects oversized metadata", "[ut][streaming_serialization]") {
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    writer.Write(vsag::SERIAL_STREAM_MAGIC, 8);
    vsag::StreamWriter::WriteObj(writer, vsag::SERIAL_STREAM_FORMAT_MAJOR);
    vsag::StreamWriter::WriteObj(writer, vsag::SERIAL_STREAM_FORMAT_MINOR);
    uint64_t oversized_metadata_len = 16ULL * 1024ULL * 1024ULL + 1ULL;
    vsag::StreamWriter::WriteObj(writer, oversized_metadata_len);

    vsag::ForwardStreamReader reader(stream);
    REQUIRE_THROWS(vsag::StreamHeader::Read(reader));
}

TEST_CASE("StreamHeader rejects non-object metadata", "[ut][streaming_serialization]") {
    const std::string metadata_string = "[]";
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    writer.Write(vsag::SERIAL_STREAM_MAGIC, 8);
    vsag::StreamWriter::WriteObj(writer, vsag::SERIAL_STREAM_FORMAT_MAJOR);
    vsag::StreamWriter::WriteObj(writer, vsag::SERIAL_STREAM_FORMAT_MINOR);
    vsag::StreamWriter::WriteObj(writer, static_cast<uint64_t>(metadata_string.size()));
    writer.Write(metadata_string.data(), metadata_string.size());
    vsag::StreamWriter::WriteObj(writer, vsag::StreamHeader::CalculateChecksum(metadata_string));

    vsag::ForwardStreamReader reader(stream);
    REQUIRE_THROWS(vsag::StreamHeader::Read(reader));
}

TEST_CASE("StreamBlockHeader rejects chunked payload sentinel", "[ut][streaming_serialization]") {
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    vsag::StreamBlockHeader header;
    header.tag = 42;
    header.block_version = vsag::kStreamSerializationBlockVersionV1;
    header.value_len = std::numeric_limits<uint64_t>::max();
    vsag::StreamBlockHeader::Write(writer, header);

    vsag::ForwardStreamReader reader(stream);
    REQUIRE_THROWS(vsag::StreamBlockHeader::Read(reader));
}

TEST_CASE("StreamBlockHeader accepts payload checksum", "[ut][streaming_serialization]") {
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    vsag::StreamBlockHeader header;
    header.tag = 42;
    header.block_version = vsag::kStreamSerializationBlockVersionV1;
    header.value_len = 3;
    header.payload_checksum = vsag::StreamHeader::CalculateChecksum("abc");
    vsag::StreamBlockHeader::Write(writer, header);

    vsag::ForwardStreamReader reader(stream);
    auto parsed = vsag::StreamBlockHeader::Read(reader);
    REQUIRE(parsed.payload_checksum == header.payload_checksum);
}

TEST_CASE("ReadSeekableBlockPayload rejects checksum mismatch", "[ut][streaming_serialization]") {
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    vsag::WriteStreamingBlock(
        writer, 42, true, [](vsag::StreamWriter& block_writer) { block_writer.Write("abc", 3); });

    auto bytes = stream.str();
    bytes.back() = 'd';
    std::stringstream corrupted(bytes);
    vsag::ForwardStreamReader reader(corrupted);
    auto header = vsag::StreamBlockHeader::Read(reader);
    REQUIRE_THROWS(
        vsag::ReadSeekableBlockPayload(reader, header, [](vsag::StreamReader& block_reader) {
            char buffer[3] = {};
            block_reader.Read(buffer, 3);
        }));
}

TEST_CASE("ReadSeekableBlockPayload rejects cursor past small payload",
          "[ut][streaming_serialization]") {
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    vsag::WriteStreamingBlock(
        writer, 42, true, [](vsag::StreamWriter& block_writer) { block_writer.Write("abc", 3); });

    vsag::ForwardStreamReader reader(stream);
    auto header = vsag::StreamBlockHeader::Read(reader);
    REQUIRE_THROWS(vsag::ReadSeekableBlockPayload(
        reader, header, [](vsag::StreamReader& block_reader) { block_reader.Seek(4); }));
}

TEST_CASE("ReadSeekableBlockPayload spills oversized payload", "[ut][streaming_serialization]") {
    const auto origin_size = vsag::Options::Instance().block_size_limit();
    vsag::Options::Instance().set_block_size_limit(256UL * 1024);

    std::string payload(512UL * 1024, 'x');
    payload[384UL * 1024] = 'y';
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    vsag::WriteStreamingBlock(writer, 42, true, [&payload](vsag::StreamWriter& block_writer) {
        block_writer.Write(payload.data(), payload.size());
    });

    vsag::ForwardStreamReader reader(stream);
    auto header = vsag::StreamBlockHeader::Read(reader);
    vsag::ReadSeekableBlockPayload(reader, header, [](vsag::StreamReader& block_reader) {
        block_reader.Seek(384UL * 1024);
        char value = 0;
        block_reader.Read(&value, 1);
        REQUIRE(value == 'y');
    });

    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE("ReadSeekableBlockPayload rejects cursor past spilled payload",
          "[ut][streaming_serialization]") {
    const auto origin_size = vsag::Options::Instance().block_size_limit();
    vsag::Options::Instance().set_block_size_limit(256UL * 1024);

    std::string payload(512UL * 1024, 'x');
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    vsag::WriteStreamingBlock(writer, 42, true, [&payload](vsag::StreamWriter& block_writer) {
        block_writer.Write(payload.data(), payload.size());
    });

    vsag::ForwardStreamReader reader(stream);
    auto header = vsag::StreamBlockHeader::Read(reader);
    REQUIRE_THROWS(vsag::ReadSeekableBlockPayload(
        reader, header, [&payload](vsag::StreamReader& block_reader) {
            block_reader.Seek(payload.size() + 1);
        }));

    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE("ValidateAndSkipBlockPayload", "[ut][streaming_serialization]") {
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    vsag::WriteStreamingBlock(
        writer, 42, true, [](vsag::StreamWriter& block_writer) { block_writer.Write("abc", 3); });
    vsag::StreamBlockHeader::WriteSectionEnd(writer);

    vsag::ForwardStreamReader reader(stream);
    auto header = vsag::StreamBlockHeader::Read(reader);
    vsag::ValidateAndSkipBlockPayload(reader, header);
    auto end = vsag::StreamBlockHeader::Read(reader);
    REQUIRE(end.IsSectionEnd());
}

TEST_CASE("ValidateAndSkipBlockPayload rejects checksum mismatch",
          "[ut][streaming_serialization]") {
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    vsag::WriteStreamingBlock(
        writer, 42, true, [](vsag::StreamWriter& block_writer) { block_writer.Write("abc", 3); });

    auto bytes = stream.str();
    bytes.back() = 'd';
    std::stringstream corrupted(bytes);
    vsag::ForwardStreamReader reader(corrupted);
    auto header = vsag::StreamBlockHeader::Read(reader);
    REQUIRE_THROWS(vsag::ValidateAndSkipBlockPayload(reader, header));
}

TEST_CASE("StreamBlockHeader", "[ut][streaming_serialization]") {
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    vsag::StreamBlockHeader header;
    header.tag = 42;
    header.block_version = vsag::kStreamSerializationBlockVersionV1;
    header.flags = vsag::StreamBlockHeader::kCriticalFlag;
    header.value_len = 3;
    vsag::StreamBlockHeader::Write(writer, header);
    writer.Write("abc", 3);
    vsag::StreamBlockHeader::WriteSectionEnd(writer);

    vsag::ForwardStreamReader reader(stream);
    auto parsed = vsag::StreamBlockHeader::Read(reader);
    REQUIRE(parsed.tag == 42);
    REQUIRE(parsed.IsCritical());
    vsag::SkipBlockPayload(reader, parsed);
    auto end = vsag::StreamBlockHeader::Read(reader);
    REQUIRE(end.IsSectionEnd());
}

TEST_CASE("SkipBlockPayload uses seekable reader", "[ut][streaming_serialization]") {
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    vsag::StreamBlockHeader header;
    header.tag = 42;
    header.block_version = vsag::kStreamSerializationBlockVersionV1;
    header.value_len = 3;
    vsag::StreamBlockHeader::Write(writer, header);
    writer.Write("abc", 3);
    vsag::StreamBlockHeader::WriteSectionEnd(writer);

    vsag::IOStreamReader reader(stream);
    auto parsed = vsag::StreamBlockHeader::Read(reader);
    vsag::SkipBlockPayload(reader, parsed);
    REQUIRE(reader.GetCursor() == vsag::StreamBlockHeader::kSerializedSize + parsed.value_len);
    auto end = vsag::StreamBlockHeader::Read(reader);
    REQUIRE(end.IsSectionEnd());
}

TEST_CASE("ForwardStreamReader forbids random access", "[ut][streaming_serialization]") {
    std::stringstream stream;
    stream << "abcdef";
    vsag::ForwardStreamReader reader(stream);
    char buffer[3] = {};
    reader.Read(buffer, 3);
    REQUIRE(std::string(buffer, 3) == "abc");
    REQUIRE(reader.GetCursor() == 3);
    REQUIRE_THROWS(reader.Seek(0));
    REQUIRE_THROWS(reader.Length());
}

TEST_CASE("BoundedForwardReader", "[ut][streaming_serialization]") {
    std::stringstream stream;
    stream << "abcdef";
    vsag::ForwardStreamReader reader(stream);
    vsag::BoundedForwardReader bounded(&reader, 4);
    char buffer[2] = {};
    bounded.Read(buffer, 2);
    REQUIRE(std::string(buffer, 2) == "ab");
    REQUIRE_THROWS(bounded.Read(buffer, 3));
    bounded.SkipRemaining();
    REQUIRE(reader.GetCursor() == 4);
}

TEST_CASE("BoundedForwardReader rejects overflow-sized read", "[ut][streaming_serialization]") {
    std::stringstream stream;
    stream << "abcdef";
    vsag::ForwardStreamReader reader(stream);
    vsag::BoundedForwardReader bounded(&reader, std::numeric_limits<uint64_t>::max());
    char buffer[1] = {};
    bounded.Read(buffer, 1);
    REQUIRE_THROWS(bounded.Read(buffer, std::numeric_limits<uint64_t>::max()));
}
