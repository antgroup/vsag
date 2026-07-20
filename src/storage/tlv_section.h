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
#include <functional>

#include "stream_reader.h"
#include "stream_writer.h"

namespace vsag {

struct StreamBlockHeader {
    uint32_t tag{0};
    uint32_t block_version{0};
    uint64_t flags{0};
    uint64_t value_len{0};
    uint32_t payload_checksum{0};

    [[nodiscard]] bool
    IsCritical() const {
        return (flags & kCriticalFlag) != 0;
    }

    [[nodiscard]] bool
    IsSectionEnd() const;

    static StreamBlockHeader
    Read(StreamReader& reader);

    static void
    Write(StreamWriter& writer, const StreamBlockHeader& header);

    static void
    WriteSectionEnd(StreamWriter& writer);

    static constexpr uint64_t kCriticalFlag = 1;
    static constexpr uint64_t kSerializedSize = sizeof(uint32_t) + sizeof(uint32_t) +
                                                sizeof(uint64_t) + sizeof(uint64_t) +
                                                sizeof(uint32_t);
};

uint64_t
CountStreamingBlockPayload(const std::function<void(StreamWriter&)>& serialize);

uint32_t
CalculateStreamingBlockPayloadChecksum(const std::function<void(StreamWriter&)>& serialize);

void
SkipBlockPayload(StreamReader& reader, const StreamBlockHeader& header);

void
ValidateAndSkipBlockPayload(StreamReader& reader, const StreamBlockHeader& header);

void
ReadSeekableBlockPayload(StreamReader& reader,
                         const StreamBlockHeader& header,
                         const std::function<void(StreamReader&)>& deserialize);

// Invokes serialize twice: first to compute payload length/checksum, then to emit bytes.
// Serializers must be deterministic and side-effect free for a given index state; the second
// pass is checked against the payload length and checksum stored in the header.
void
WriteStreamingBlock(StreamWriter& writer,
                    uint32_t tag,
                    bool critical,
                    const std::function<void(StreamWriter&)>& serialize);

}  // namespace vsag
