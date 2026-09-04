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
#include <string>

namespace vsag {

/**
 * @brief Default logical chunk granularity (in bytes) for chunked
 * serialization, used as the default value of Index::Serialize(writer,
 * chunk_size).
 */
constexpr uint64_t DEFAULT_SERIALIZE_CHUNK_SIZE = 128ULL * 1024 * 1024;

/**
 * @class SerializeWriter
 *
 * @brief Abstract byte sink for index serialization, with optional
 * per-frame compression.
 *
 * A plain (non-compressing) writer only needs to implement Write(); the
 * compressed-frame methods have no-op defaults. A compressing writer
 * additionally overrides GetCompressorName() and the frame methods.
 *
 * The writer is a dumb pipe: it carries no knowledge about index
 * components or file layout. The index decides where compressed frames
 * begin and end, and records the physical layout (using the sizes
 * reported by EndCompressedFrame) into its own footer.
 */
class SerializeWriter {
public:
    SerializeWriter() = default;

    virtual ~SerializeWriter() = default;

    /**
     * @brief Write bytes to the output.
     *
     * Bytes written outside a compressed frame MUST be stored verbatim;
     * bytes written inside a frame (between BeginCompressedFrame and
     * EndCompressedFrame) are fed to the compressor.
     *
     * @param data Pointer to the bytes to write.
     * @param size Number of bytes to write.
     */
    virtual void
    Write(const char* data, uint64_t size) = 0;

    /**
     * @brief Name of the compressor applied to frames (e.g. "zstd").
     *
     * The name is recorded in the index footer so that a matching reader
     * can be chosen at load time. The default "none" declares a plain
     * writer; in that case the frame methods are never invoked and all
     * bytes are written verbatim.
     *
     * A subclass returning anything other than "none" MUST override
     * BeginCompressedFrame and EndCompressedFrame. Leaving them as the
     * default no-ops writes plain frames under a footer that claims
     * compression, which only surfaces as a mismatch at load time.
     *
     * @return The compressor name, or "none" for a plain writer.
     */
    [[nodiscard]] virtual std::string
    GetCompressorName() const {
        return "none";
    }

    /**
     * @brief Start a new compressed frame; subsequent Write calls are
     * compressed until EndCompressedFrame.
     */
    virtual void
    BeginCompressedFrame() {
    }

    /**
     * @brief Finish the current frame.
     *
     * The implementation MUST flush a complete, independently
     * decompressible frame to the output before returning.
     *
     * The returned count MUST equal the number of bytes the frame actually
     * occupies: every later frame offset recorded in the layout is derived
     * from it, so a misreporting writer produces a layout whose offsets do
     * not match the file. Such a file is rejected on load rather than
     * silently accepted, because the reader decompresses each frame from its
     * recorded offset, but the failure surfaces far from its cause.
     *
     * @return Number of compressed bytes the frame occupies in the output.
     */
    virtual uint64_t
    EndCompressedFrame() {
        return 0;
    }
};

}  // namespace vsag
