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

#include <limits>

#include "vsag/constants.h"
#include "vsag_exception.h"

namespace vsag {

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

void
WriteStreamingBlock(StreamWriter& writer,
                    uint32_t tag,
                    bool critical,
                    const std::function<void(StreamWriter&)>& serialize) {
    CountingStreamWriter counting_writer;
    serialize(counting_writer);

    StreamBlockHeader header;
    header.tag = tag;
    header.block_version = 1;
    header.flags = critical ? StreamBlockHeader::kCriticalFlag : 0;
    header.value_len = counting_writer.GetCursor();
    StreamBlockHeader::Write(writer, header);
    serialize(writer);
}

void
SkipBlockPayload(StreamReader& reader, const StreamBlockHeader& header) {
    if (header.value_len == std::numeric_limits<uint64_t>::max()) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "chunked streaming block payload is not implemented yet");
    }
    SkipForward(reader, header.value_len);
}

}  // namespace vsag
