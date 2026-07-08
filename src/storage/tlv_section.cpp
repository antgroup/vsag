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

#include "tlv_section.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

#if !defined(_WIN32)
#include <sys/types.h>
#endif

#include "storage/serialization.h"
#include "storage/serialization_tags.h"
#include "vsag/constants.h"
#include "vsag/options.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

void
SeekTemporaryFile(std::FILE* file, uint64_t offset) {
#if defined(_WIN32)
    if (offset > static_cast<uint64_t>(std::numeric_limits<__int64>::max())) {
        throw VsagException(ErrorType::READ_ERROR,
                            "seekable streaming block reader offset is too large");
    }
    if (::_fseeki64(file, static_cast<__int64>(offset), SEEK_SET) != 0) {
        throw VsagException(ErrorType::READ_ERROR, "failed to seek streaming block temporary file");
    }
#else
    if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
        throw VsagException(ErrorType::READ_ERROR,
                            "seekable streaming block reader offset is too large");
    }
    if (::fseeko(file, static_cast<off_t>(offset), SEEK_SET) != 0) {
        throw VsagException(ErrorType::READ_ERROR, "failed to seek streaming block temporary file");
    }
#endif
}

class ChecksumStreamWriter : public StreamWriter {
public:
    explicit ChecksumStreamWriter(StreamWriter& writer) : writer_(writer) {
    }

    void
    Write(const char* data, uint64_t size) override {
        writer_.Write(data, size);
        bytes_written_ += size;
        crc_ = StreamHeader::UpdateChecksum(crc_, std::string_view(data, size));
    }

    [[nodiscard]] uint32_t
    GetChecksum() const {
        return StreamHeader::FinalizeChecksum(crc_);
    }

private:
    StreamWriter& writer_;
    uint32_t crc_{StreamHeader::InitialChecksum()};
};

class CountingChecksumStreamWriter : public StreamWriter {
public:
    void
    Write(const char* data, uint64_t size) override {
        bytes_written_ += size;
        crc_ = StreamHeader::UpdateChecksum(crc_, std::string_view(data, size));
    }

    [[nodiscard]] uint64_t
    GetCursor() const {
        return bytes_written_;
    }

    [[nodiscard]] uint32_t
    GetChecksum() const {
        return StreamHeader::FinalizeChecksum(crc_);
    }

private:
    uint32_t crc_{StreamHeader::InitialChecksum()};
};

}  // namespace

bool
StreamBlockHeader::IsSectionEnd() const {
    return tag == SERIAL_STREAM_SECTION_END;
}

StreamBlockHeader
StreamBlockHeader::Read(StreamReader& reader) {
    StreamBlockHeader header;
    StreamReader::ReadObj(reader, header.tag);
    StreamReader::ReadObj(reader, header.block_version);
    StreamReader::ReadObj(reader, header.flags);
    StreamReader::ReadObj(reader, header.value_len);
    StreamReader::ReadObj(reader, header.payload_checksum);
    if (header.IsSectionEnd() && (header.block_version != 0 || header.flags != 0 ||
                                  header.value_len != 0 || header.payload_checksum != 0)) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            "invalid streaming serialization section end block");
    }
    if (!header.IsSectionEnd() && header.value_len == std::numeric_limits<uint64_t>::max()) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "chunked streaming block payload is not implemented yet");
    }
    return header;
}

void
StreamBlockHeader::Write(StreamWriter& writer, const StreamBlockHeader& header) {
    StreamWriter::WriteObj(writer, header.tag);
    StreamWriter::WriteObj(writer, header.block_version);
    StreamWriter::WriteObj(writer, header.flags);
    StreamWriter::WriteObj(writer, header.value_len);
    StreamWriter::WriteObj(writer, header.payload_checksum);
}

void
StreamBlockHeader::WriteSectionEnd(StreamWriter& writer) {
    StreamBlockHeader header;
    header.tag = SERIAL_STREAM_SECTION_END;
    StreamBlockHeader::Write(writer, header);
}

uint64_t
CountStreamingBlockPayload(const std::function<void(StreamWriter&)>& serialize) {
    CountingStreamWriter counting_writer;
    serialize(counting_writer);
    return counting_writer.GetCursor();
}

uint32_t
CalculateStreamingBlockPayloadChecksum(const std::function<void(StreamWriter&)>& serialize) {
    CountingChecksumStreamWriter checksum_writer;
    serialize(checksum_writer);
    return checksum_writer.GetChecksum();
}

void
WriteStreamingBlock(StreamWriter& writer,
                    uint32_t tag,
                    bool critical,
                    const std::function<void(StreamWriter&)>& serialize) {
    CountingChecksumStreamWriter counting_checksum_writer;
    serialize(counting_checksum_writer);

    StreamBlockHeader header;
    header.tag = tag;
    header.block_version = StreamSerializationBlockCurrentVersion(tag);
    header.flags = critical ? StreamBlockHeader::kCriticalFlag : 0;
    header.value_len = counting_checksum_writer.GetCursor();
    header.payload_checksum = counting_checksum_writer.GetChecksum();
    StreamBlockHeader::Write(writer, header);
    ChecksumStreamWriter payload_writer(writer);
    serialize(payload_writer);
    if (payload_writer.GetCursor() != header.value_len ||
        payload_writer.GetChecksum() != header.payload_checksum) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            "streaming block serializer produced inconsistent payload");
    }
}

void
SkipBlockPayload(StreamReader& reader, const StreamBlockHeader& header) {
    if (header.value_len == std::numeric_limits<uint64_t>::max()) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "chunked streaming block payload is not implemented yet");
    }
    const auto current = reader.GetCursor();
    if (header.value_len > std::numeric_limits<uint64_t>::max() - current) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            "streaming block payload size overflows reader cursor");
    }

    try {
        reader.Seek(current + header.value_len);
    } catch (const VsagException& e) {
        if (e.error_.type != ErrorType::UNSUPPORTED_INDEX_OPERATION) {
            throw;
        }
        SkipForward(reader, header.value_len);
    }
}

void
ValidateSeekableBlockCursor(const StreamReader& reader, const StreamBlockHeader& header) {
    if (reader.GetCursor() > header.value_len) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            "seekable streaming block reader cursor exceeds payload boundary");
    }
}

void
ValidateAndSkipBlockPayload(StreamReader& reader, const StreamBlockHeader& header) {
    if (header.value_len == std::numeric_limits<uint64_t>::max()) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "chunked streaming block payload is not implemented yet");
    }

    constexpr uint64_t buffer_size = 8192;
    std::array<char, buffer_size> buffer{};
    uint64_t remaining = header.value_len;
    uint32_t crc = StreamHeader::InitialChecksum();
    while (remaining > 0) {
        const auto read_size = std::min<uint64_t>(remaining, buffer_size);
        reader.Read(buffer.data(), read_size);
        crc = StreamHeader::UpdateChecksum(crc, std::string_view(buffer.data(), read_size));
        remaining -= read_size;
    }
    if (StreamHeader::FinalizeChecksum(crc) != header.payload_checksum) {
        throw VsagException(ErrorType::INVALID_BINARY, "streaming block payload checksum mismatch");
    }
}

void
ReadSeekableBlockPayload(StreamReader& reader,
                         const StreamBlockHeader& header,
                         const std::function<void(StreamReader&)>& deserialize) {
    if (header.value_len > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                            "streaming block payload is too large to buffer");
    }

    const auto block_size_limit = Options::Instance().block_size_limit();
    if (header.value_len <= block_size_limit) {
        std::vector<char> payload(static_cast<size_t>(header.value_len));
        if (!payload.empty()) {
            reader.Read(payload.data(), header.value_len);
        }
        const auto payload_view =
            payload.empty() ? std::string_view{} : std::string_view(payload.data(), payload.size());
        if (StreamHeader::CalculateChecksum(payload_view) != header.payload_checksum) {
            throw VsagException(ErrorType::INVALID_BINARY,
                                "streaming block payload checksum mismatch");
        }

        auto read_func = [&payload, payload_size = static_cast<uint64_t>(payload.size())](
                             uint64_t offset, uint64_t size, void* dest) {
            if (offset > payload_size || size > payload_size - offset) {
                throw VsagException(ErrorType::READ_ERROR,
                                    "seekable streaming block reader exceeds payload boundary");
            }
            if (size > 0) {
                std::memcpy(
                    dest, payload.data() + static_cast<size_t>(offset), static_cast<size_t>(size));
            }
        };
        ReadFuncStreamReader block_reader(read_func, 0, header.value_len);
        deserialize(block_reader);
        ValidateSeekableBlockCursor(block_reader, header);
        return;
    }

    std::unique_ptr<std::FILE, decltype(&std::fclose)> temp_file(std::tmpfile(), std::fclose);
    if (temp_file == nullptr) {
        throw VsagException(ErrorType::READ_ERROR,
                            "failed to create temporary file for streaming block payload");
    }

    if (block_size_limit > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                            "streaming block size limit is too large to buffer");
    }
    std::vector<char> buffer(static_cast<size_t>(block_size_limit));
    uint64_t remaining = header.value_len;
    uint32_t crc = StreamHeader::InitialChecksum();
    while (remaining > 0) {
        const auto read_size = std::min<uint64_t>(remaining, block_size_limit);
        const auto read_size_bytes = static_cast<size_t>(read_size);
        reader.Read(buffer.data(), read_size);
        crc = StreamHeader::UpdateChecksum(crc, std::string_view(buffer.data(), read_size));
        if (std::fwrite(buffer.data(), 1, read_size_bytes, temp_file.get()) != read_size_bytes) {
            throw VsagException(ErrorType::READ_ERROR,
                                "failed to spill streaming block payload to temporary file");
        }
        remaining -= read_size;
    }
    if (StreamHeader::FinalizeChecksum(crc) != header.payload_checksum) {
        throw VsagException(ErrorType::INVALID_BINARY, "streaming block payload checksum mismatch");
    }
    std::fflush(temp_file.get());

    auto read_func = [&temp_file, payload_size = header.value_len](
                         uint64_t offset, uint64_t size, void* dest) {
        if (offset > payload_size || size > payload_size - offset) {
            throw VsagException(ErrorType::READ_ERROR,
                                "seekable streaming block reader exceeds payload boundary");
        }
        SeekTemporaryFile(temp_file.get(), offset);
        if (size > 0 && std::fread(dest, 1, static_cast<size_t>(size), temp_file.get()) != size) {
            throw VsagException(ErrorType::READ_ERROR,
                                "failed to read streaming block temporary file");
        }
    };
    ReadFuncStreamReader block_reader(read_func, 0, header.value_len);
    deserialize(block_reader);
    ValidateSeekableBlockCursor(block_reader, header);
}

}  // namespace vsag
