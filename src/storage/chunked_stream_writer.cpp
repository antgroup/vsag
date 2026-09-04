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

#include "chunked_stream_writer.h"

#include <fmt/format.h>

#include <algorithm>

#include "vsag_exception.h"

namespace vsag {

ChunkedStreamWriter::ChunkedStreamWriter(vsag::SerializeWriter& writer, uint64_t chunk_size)
    : writer_(writer), compress_(writer.GetCompressorName() != "none") {
    if (chunk_size == 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "chunk_size must be positive");
    }
    // the codec name is persisted verbatim into the footer and read back by
    // ChunkedManifest::FromJson, so a SerializeWriter must return a stable
    // string that the matching reader recognizes. Any name other than "none"
    // marks every frame as compressed, which makes the body loadable only
    // through ParallelDeserialize.
    manifest_.codec_ = writer.GetCompressorName();
    manifest_.chunk_size_ = chunk_size;
}

void
ChunkedStreamWriter::write_plain(const char* data, uint64_t size) {
    writer_.Write(data, size);
    physical_cursor_ += size;
}

void
ChunkedStreamWriter::write_framed(const char* data, uint64_t size) {
    writer_.Write(data, size);
    if (not compress_) {
        // plain writer: frame bytes hit the sink verbatim right away
        physical_cursor_ += size;
    }
    frame_logical_ += size;
}

void
ChunkedStreamWriter::open_frame() {
    frame_start_ = physical_cursor_;
    frame_logical_ = 0;
    frame_remaining_ = std::min(manifest_.chunk_size_, io_remaining_);
    if (compress_) {
        writer_.BeginCompressedFrame();
    }
    frame_open_ = true;
}

uint64_t
ChunkedStreamWriter::close_frame() {
    uint64_t csize = 0;
    if (compress_) {
        csize = writer_.EndCompressedFrame();
        physical_cursor_ += csize;
    } else {
        // plain writer: bytes already hit the sink verbatim in Write
        csize = frame_logical_;
    }
    frame_open_ = false;
    return csize;
}

void
ChunkedStreamWriter::Write(const char* data, uint64_t size) {
    bytes_written_ += size;
    while (size > 0) {
        switch (stage_) {
            case Stage::OUTSIDE:
            case Stage::TAIL: {
                write_plain(data, size);
                return;
            }
            case Stage::WHOLE: {
                // a whole component is one frame of unknown length, so there is
                // no frame_remaining_ to decrement here; BeginWholeComponent
                // never sets it. A size-hinted BeginWholeComponent would have to
                // start tracking it in both places at once.
                write_framed(data, size);
                return;
            }
            case Stage::HEAD: {
                auto take = std::min(size, head_remaining_);
                write_plain(data, take);
                head_remaining_ -= take;
                data += take;
                size -= take;
                if (head_remaining_ == 0) {
                    stage_ = Stage::IO_DATA;
                }
                break;
            }
            case Stage::IO_DATA: {
                if (io_remaining_ == 0) {
                    stage_ = Stage::TAIL;
                    cur_->tail_offset = physical_cursor_;
                    break;
                }
                if (not frame_open_) {
                    open_frame();
                }
                auto take = std::min(size, frame_remaining_);
                write_framed(data, take);
                frame_remaining_ -= take;
                io_remaining_ -= take;
                data += take;
                size -= take;
                if (frame_remaining_ == 0) {
                    cur_->chunks.push_back(ChunkRecord{frame_start_, close_frame()});
                }
                break;
            }
        }
    }
}

void
ChunkedStreamWriter::BeginChunkedComponent(const std::string& name,
                                           uint64_t head_size,
                                           uint64_t io_size) {
    if (stage_ != Stage::OUTSIDE) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "previous component is not finished");
    }
    manifest_.components_.emplace_back();
    cur_ = &manifest_.components_.back();
    cur_->name = name;
    cur_->granularity = ComponentGranularity::Byte;
    cur_->head_offset = physical_cursor_;
    cur_->head_size = head_size;
    cur_->io_size = io_size;
    head_remaining_ = head_size;
    io_remaining_ = io_size;
    stage_ = (head_size > 0) ? Stage::HEAD : Stage::IO_DATA;
}

void
ChunkedStreamWriter::BeginWholeComponent(const std::string& name) {
    if (stage_ != Stage::OUTSIDE) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "previous component is not finished");
    }
    manifest_.components_.emplace_back();
    cur_ = &manifest_.components_.back();
    cur_->name = name;
    cur_->granularity = ComponentGranularity::Whole;
    cur_->offset = physical_cursor_;
    frame_start_ = physical_cursor_;
    frame_logical_ = 0;
    if (compress_) {
        writer_.BeginCompressedFrame();
    }
    frame_open_ = true;
    stage_ = Stage::WHOLE;
}

void
ChunkedStreamWriter::EndComponent() {
    if (cur_ == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "no component to end");
    }
    if (stage_ == Stage::WHOLE) {
        cur_->logical_size = frame_logical_;
        cur_->compressed_size = close_frame();
    } else {
        // an io_size == 0 component may still sit in IO_DATA here; the
        // TAIL transition then never ran inside Write
        if (stage_ == Stage::IO_DATA and io_remaining_ == 0 and not frame_open_) {
            stage_ = Stage::TAIL;
            cur_->tail_offset = physical_cursor_;
        }
        // cursor check: the declared head/io extents must be fully written,
        // otherwise the precomputed boundaries diverged from the actual
        // byte stream (e.g. upstream added a field)
        if (stage_ != Stage::TAIL or head_remaining_ != 0 or io_remaining_ != 0) {
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                fmt::format("component {} byte stream diverged from the declared layout "
                            "(head remaining: {}, io remaining: {})",
                            cur_->name,
                            head_remaining_,
                            io_remaining_));
        }
        // tail_size is definitional rather than declared: BeginChunkedComponent
        // takes head_size and io_size but no tail size, so the tail is whatever
        // the component writes after its io data and there is no expected value
        // to check it against here. A component whose DeserializeTail does not
        // read back what its Serialize wrote is caught on load, where the tail
        // pass requires the reader to land exactly on tail_offset + tail_size.
        cur_->tail_size = physical_cursor_ - cur_->tail_offset;
    }
    cur_ = nullptr;
    stage_ = Stage::OUTSIDE;
}

}  // namespace vsag
