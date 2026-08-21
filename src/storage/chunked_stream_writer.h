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

#include <string>

#include "chunked_layout.h"
#include "stream_writer.h"
#include "vsag/serialize_writer.h"

namespace vsag {

/**
 * @brief StreamWriter adapter over vsag::SerializeWriter that splits each
 * declared component into head (plain) / io data (one compressed frame per
 * chunk) / tail (plain) purely by byte counting, and records the physical
 * layout while writing.
 *
 * Component Serialize() implementations stay untouched: they keep writing
 * one continuous byte stream through this adapter, which switches between
 * plain passthrough and compressed frames at the precomputed boundaries.
 * Bytes written outside any component (e.g. the footer) pass through
 * verbatim.
 */
class ChunkedStreamWriter : public StreamWriter {
public:
    ChunkedStreamWriter(vsag::SerializeWriter& writer, uint64_t chunk_size);

    void
    Write(const char* data, uint64_t size) override;

    /// Declare a three-part component; head_size/io_size are the
    /// precomputed boundaries within the upcoming byte stream.
    void
    BeginChunkedComponent(const std::string& name, uint64_t head_size, uint64_t io_size);

    /// Declare a component serialized into one single frame.
    void
    BeginWholeComponent(const std::string& name);

    void
    EndComponent();

    [[nodiscard]] uint64_t
    GetPhysicalCursor() const {
        return physical_cursor_;
    }

    [[nodiscard]] const ChunkedLayout&
    GetLayout() const {
        return layout_;
    }

private:
    enum class Stage { OUTSIDE, HEAD, IO_DATA, TAIL, WHOLE };

    void
    write_plain(const char* data, uint64_t size);

    void
    write_framed(const char* data, uint64_t size);

    void
    open_frame();

    uint64_t
    close_frame();

    vsag::SerializeWriter& writer_;
    ChunkedLayout layout_;
    const bool compress_;

    Stage stage_{Stage::OUTSIDE};
    ComponentLayout* cur_{nullptr};
    uint64_t head_remaining_{0};
    uint64_t io_remaining_{0};
    bool frame_open_{false};
    uint64_t frame_start_{0};      // physical offset of the open frame
    uint64_t frame_remaining_{0};  // logical bytes until the open frame is full
    uint64_t frame_logical_{0};    // logical bytes fed into the open frame
    uint64_t physical_cursor_{0};  // bytes emitted to the underlying writer
};

}  // namespace vsag
