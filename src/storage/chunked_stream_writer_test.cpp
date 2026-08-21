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

#include <string>

#include "unittest.h"
#include "vsag_exception.h"

namespace {

// plain writer: verbatim passthrough, codec "none"
class BufferSerializeWriter : public vsag::SerializeWriter {
public:
    void
    Write(const char* data, uint64_t size) override {
        buffer_.append(data, size);
    }

    std::string buffer_;
};

// mock compressing writer: a "compressed" frame is the payload prefixed
// with an 8-byte marker, so csize == payload_size + 8 (deterministic and
// distinguishable from the plain path)
class MockCompressSerializeWriter : public vsag::SerializeWriter {
public:
    static constexpr const char* FRAME_MARKER = "[FRAME!]";

    void
    Write(const char* data, uint64_t size) override {
        if (in_frame_) {
            frame_payload_.append(data, size);
        } else {
            buffer_.append(data, size);
        }
    }

    [[nodiscard]] std::string
    GetCompressorName() const override {
        return "mock";
    }

    void
    BeginCompressedFrame() override {
        in_frame_ = true;
        frame_payload_.clear();
    }

    uint64_t
    EndCompressedFrame() override {
        in_frame_ = false;
        buffer_.append(FRAME_MARKER, 8);
        buffer_.append(frame_payload_);
        return frame_payload_.size() + 8;
    }

    std::string buffer_;

private:
    bool in_frame_{false};
    std::string frame_payload_;
};

void
write_in_pieces(vsag::ChunkedStreamWriter& writer, const std::string& bytes, uint64_t piece) {
    uint64_t offset = 0;
    while (offset < bytes.size()) {
        auto size = std::min(piece, static_cast<uint64_t>(bytes.size()) - offset);
        writer.Write(bytes.data() + offset, size);
        offset += size;
    }
}

}  // namespace

TEST_CASE("ChunkedStreamWriter Plain Writer Passthrough", "[ut][chunked_stream_writer]") {
    BufferSerializeWriter sink;
    vsag::ChunkedStreamWriter writer(sink, /*chunk_size=*/4);

    const std::string head = "HEAD";           // 4B
    const std::string io_data = "0123456789";  // 10B -> chunks of 4/4/2
    const std::string tail = "TAILDATA";       // 8B
    const std::string outside = "FOOTER";

    writer.BeginChunkedComponent("comp", head.size(), io_data.size());
    // odd-sized writes crossing every boundary
    write_in_pieces(writer, head + io_data + tail, 3);
    writer.EndComponent();
    writer.Write(outside.data(), outside.size());

    // plain writer: output is byte-identical to the logical stream
    REQUIRE(sink.buffer_ == head + io_data + tail + outside);
    REQUIRE(writer.GetPhysicalCursor() == sink.buffer_.size());
    REQUIRE(writer.GetCursor() == head.size() + io_data.size() + tail.size() + outside.size());

    const auto& layout = writer.GetLayout();
    REQUIRE(layout.codec_ == "none");
    REQUIRE(layout.chunk_size_ == 4);
    const auto* comp = layout.FindComponent("comp");
    REQUIRE(comp != nullptr);
    REQUIRE(comp->granularity == vsag::ComponentGranularity::Byte);
    REQUIRE(comp->head_offset == 0);
    REQUIRE(comp->head_size == 4);
    REQUIRE(comp->io_size == 10);
    REQUIRE(comp->chunks.size() == 3);
    // csize == lsize for the plain writer
    REQUIRE(comp->chunks[0].offset == 4);
    REQUIRE(comp->chunks[0].compressed_size == 4);
    REQUIRE(comp->chunks[1].offset == 8);
    REQUIRE(comp->chunks[1].compressed_size == 4);
    REQUIRE(comp->chunks[2].offset == 12);
    REQUIRE(comp->chunks[2].compressed_size == 2);
    REQUIRE(comp->tail_offset == 14);
    REQUIRE(comp->tail_size == 8);
}

TEST_CASE("ChunkedStreamWriter Compressed Frames", "[ut][chunked_stream_writer]") {
    MockCompressSerializeWriter sink;
    vsag::ChunkedStreamWriter writer(sink, /*chunk_size=*/4);

    const std::string head = "HD";             // 2B
    const std::string io_data = "0123456789";  // 10B -> frames of 4/4/2
    const std::string tail = "T";              // 1B

    writer.BeginWholeComponent("whole");
    writer.Write("wholedata", 9);
    writer.EndComponent();

    writer.BeginChunkedComponent("comp", head.size(), io_data.size());
    write_in_pieces(writer, head + io_data + tail, 5);
    writer.EndComponent();

    const auto& layout = writer.GetLayout();
    REQUIRE(layout.codec_ == "mock");

    // whole component: single frame at offset 0, csize = 9 + 8
    const auto* whole = layout.FindComponent("whole");
    REQUIRE(whole != nullptr);
    REQUIRE(whole->granularity == vsag::ComponentGranularity::Whole);
    REQUIRE(whole->offset == 0);
    REQUIRE(whole->compressed_size == 17);
    REQUIRE(whole->logical_size == 9);

    // chunked component: head plain, then 3 frames, then tail plain
    const auto* comp = layout.FindComponent("comp");
    REQUIRE(comp != nullptr);
    REQUIRE(comp->head_offset == 17);
    REQUIRE(comp->head_size == 2);
    REQUIRE(comp->chunks.size() == 3);
    REQUIRE(comp->chunks[0].offset == 19);
    REQUIRE(comp->chunks[0].compressed_size == 12);  // 4 + 8
    REQUIRE(comp->chunks[1].offset == 31);
    REQUIRE(comp->chunks[1].compressed_size == 12);
    REQUIRE(comp->chunks[2].offset == 43);
    REQUIRE(comp->chunks[2].compressed_size == 10);  // 2 + 8
    REQUIRE(comp->tail_offset == 53);
    REQUIRE(comp->tail_size == 1);
    REQUIRE(writer.GetPhysicalCursor() == 54);
    REQUIRE(writer.GetPhysicalCursor() == sink.buffer_.size());

    // every recorded frame extent really starts with the frame marker
    for (const auto& chunk : comp->chunks) {
        REQUIRE(sink.buffer_.compare(chunk.offset, 8, MockCompressSerializeWriter::FRAME_MARKER) ==
                0);
    }
    // head/tail bytes are stored verbatim at their recorded offsets
    REQUIRE(sink.buffer_.compare(comp->head_offset, head.size(), head) == 0);
    REQUIRE(sink.buffer_.compare(comp->tail_offset, tail.size(), tail) == 0);
}

TEST_CASE("ChunkedStreamWriter Boundary Cases", "[ut][chunked_stream_writer]") {
    SECTION("io size not divisible vs divisible by chunk size") {
        BufferSerializeWriter sink;
        vsag::ChunkedStreamWriter writer(sink, 5);
        writer.BeginChunkedComponent("comp", 1, 10);  // exactly 2 chunks
        std::string bytes = "H0123456789";
        writer.Write(bytes.data(), bytes.size());
        writer.EndComponent();
        REQUIRE(writer.GetLayout().FindComponent("comp")->chunks.size() == 2);
    }

    SECTION("empty io data") {
        BufferSerializeWriter sink;
        vsag::ChunkedStreamWriter writer(sink, 4);
        writer.BeginChunkedComponent("comp", 2, 0);
        writer.Write("HDTAIL", 6);
        writer.EndComponent();
        const auto* comp = writer.GetLayout().FindComponent("comp");
        REQUIRE(comp->chunks.empty());
        REQUIRE(comp->tail_offset == 2);
        REQUIRE(comp->tail_size == 4);
    }

    SECTION("zero chunk size is rejected") {
        BufferSerializeWriter sink;
        REQUIRE_THROWS_AS(vsag::ChunkedStreamWriter(sink, 0), vsag::VsagException);
    }

    SECTION("cursor check catches diverged byte stream") {
        BufferSerializeWriter sink;
        vsag::ChunkedStreamWriter writer(sink, 4);
        writer.BeginChunkedComponent("comp", 4, 8);
        writer.Write("ONLY", 4);  // io data never written
        REQUIRE_THROWS_AS(writer.EndComponent(), vsag::VsagException);
    }

    SECTION("unbalanced begin/end is rejected") {
        BufferSerializeWriter sink;
        vsag::ChunkedStreamWriter writer(sink, 4);
        REQUIRE_THROWS_AS(writer.EndComponent(), vsag::VsagException);
        writer.BeginWholeComponent("a");
        REQUIRE_THROWS_AS(writer.BeginWholeComponent("b"), vsag::VsagException);
    }
}
