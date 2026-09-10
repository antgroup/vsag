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

#include <array>
#include <catch2/matchers/catch_matchers.hpp>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "serialization.h"
#include "serialization_tags.h"
#include "tlv_section.h"
#include "unittest.h"
#include "vsag/options.h"
#include "vsag_exception.h"

namespace {

class StringReader : public vsag::Reader {
public:
    explicit StringReader(std::string data) : data_(std::move(data)) {
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        if (offset > data_.size() or len > data_.size() - offset) {
            throw vsag::VsagException(vsag::ErrorType::READ_ERROR, "read exceeds string");
        }
        std::memcpy(dest, data_.data() + offset, len);
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, vsag::CallBack callback) override {
        try {
            Read(offset, len, dest);
            callback(vsag::IOErrorCode::IO_SUCCESS, "success");
        } catch (const std::exception& error) {
            callback(vsag::IOErrorCode::IO_ERROR, error.what());
        }
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return data_.size();
    }

private:
    std::string data_;
};

}  // namespace

TEST_CASE("Forward block reader validates consumed and drained bytes",
          "[ut][streaming_serialization]") {
    const uint64_t size = GENERATE(0, 3, 20000);
    const uint64_t consume = GENERATE(0, 1, 2);
    const std::string payload(size, 'x');
    std::istringstream input("prefix" + payload + "suffix");
    vsag::ForwardStreamReader reader(input);
    reader.Skip(6);
    vsag::StreamBlockHeader header;
    header.value_len = size;
    header.payload_checksum = vsag::StreamHeader::CalculateChecksum(payload);
    vsag::ReadForwardBlockPayload(reader, header, [&](vsag::StreamReader& block) {
        REQUIRE(block.GetCursor() == 0);
        REQUIRE(block.Length() == size);
        block.Read(nullptr, 0);
        if (consume == 1) {
            block.Skip(size / 2);
        } else if (consume == 2) {
            std::string actual(size, '\0');
            block.Read(actual.data(), size);
            REQUIRE(actual == payload);
        }
    });
    REQUIRE(reader.GetCursor() == 6 + size);
    char suffix[6];
    reader.Read(suffix, sizeof(suffix));
    REQUIRE(std::string(suffix, sizeof(suffix)) == "suffix");
}

TEST_CASE("Forward block reader rejects invalid input without unwinding drains",
          "[ut][streaming_serialization]") {
    std::istringstream input("abc");
    vsag::ForwardStreamReader reader(input);
    vsag::StreamBlockHeader header;
    header.value_len = 3;
    header.payload_checksum = vsag::StreamHeader::CalculateChecksum("abc");
    SECTION("seek is forbidden even at the current position") {
        REQUIRE_THROWS(vsag::ReadForwardBlockPayload(
            reader, header, [](vsag::StreamReader& block) { block.Seek(0); }));
        REQUIRE(reader.GetCursor() == 0);
    }
    SECTION("overflow-sized read is rejected before touching the source") {
        REQUIRE_THROWS(vsag::ReadForwardBlockPayload(reader, header, [](vsag::StreamReader& block) {
            char value;
            block.Read(&value, std::numeric_limits<uint64_t>::max());
        }));
        REQUIRE(reader.GetCursor() == 0);
    }
    SECTION("callback exception does not drain") {
        REQUIRE_THROWS_WITH(
            vsag::ReadForwardBlockPayload(reader,
                                          header,
                                          [](vsag::StreamReader& block) {
                                              block.Skip(1);
                                              throw std::runtime_error("callback failed");
                                          }),
            "callback failed");
        REQUIRE(reader.GetCursor() == 1);
    }
    SECTION("checksum failure occurs after component mutation") {
        ++header.payload_checksum;
        bool mutated = false;
        REQUIRE_THROWS(
            vsag::ReadForwardBlockPayload(reader, header, [&](vsag::StreamReader& block) {
                block.Skip(2);
                mutated = true;
            }));
        REQUIRE(mutated);  // The partially restored target must be discarded.
        REQUIRE(reader.GetCursor() == 3);
    }
    SECTION("truncated drain") {
        header.value_len = 4;
        REQUIRE_THROWS(vsag::ReadForwardBlockPayload(reader, header, [](vsag::StreamReader&) {}));
    }
    SECTION("truncated callback read") {
        header.value_len = 4;
        REQUIRE_THROWS(vsag::ReadForwardBlockPayload(
            reader, header, [](vsag::StreamReader& block) { block.Skip(4); }));
    }
    SECTION("chunked sentinel is unsupported") {
        header.value_len = std::numeric_limits<uint64_t>::max();
        REQUIRE_THROWS(vsag::ReadForwardBlockPayload(
            reader, header, [](vsag::StreamReader&) { FAIL("must not invoke callback"); }));
        REQUIRE(reader.GetCursor() == 0);
    }
}

TEST_CASE("Forward block reader drains a large generated payload with bounded reads",
          "[ut][streaming_serialization]") {
    const uint64_t size = vsag::Options::Instance().block_size_limit() + 1;
    std::array<char, 8192> bytes{};
    uint32_t checksum = vsag::StreamHeader::InitialChecksum();
    for (uint64_t offset = 0; offset < size;) {
        const uint64_t count = std::min<uint64_t>(bytes.size(), size - offset);
        checksum = vsag::StreamHeader::UpdateChecksum(checksum, {bytes.data(), count});
        offset += count;
    }
    uint64_t consumed = 0;
    vsag::ReadFuncStreamReader reader(
        [&](uint64_t offset, uint64_t count, void* destination) {
            REQUIRE(offset == consumed);
            REQUIRE(count <= bytes.size());
            std::memset(destination, 0, count);
            consumed += count;
        },
        0,
        size);
    vsag::StreamBlockHeader header;
    header.value_len = size;
    header.payload_checksum = vsag::StreamHeader::FinalizeChecksum(checksum);
    vsag::ReadForwardBlockPayload(reader, header, [&](vsag::StreamReader& block) {
        REQUIRE(consumed == 0);  // No payload materialization before the callback.
        block.Skip(size / 2);
    });
    REQUIRE(consumed == size);
}

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

TEST_CASE("ReadExternalBlockPayload validates checksum before deserializing",
          "[ut][streaming_serialization]") {
    const std::string payload = "abcdefgh";
    auto corrupted = payload;
    corrupted[3] = 'x';
    auto reader = std::make_shared<StringReader>(std::move(corrupted));
    vsag::StreamBlockHeader header;
    header.value_len = payload.size();
    header.payload_checksum = vsag::StreamHeader::CalculateChecksum(payload);
    bool deserialize_called = false;

    try {
        vsag::ReadExternalBlockPayload(reader, header, [&deserialize_called](vsag::StreamReader&) {
            deserialize_called = true;
        });
        FAIL("corrupted external payload should be rejected");
    } catch (const vsag::VsagException& error) {
        REQUIRE(error.error_.type == vsag::ErrorType::INVALID_BINARY);
    }
    REQUIRE_FALSE(deserialize_called);
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
