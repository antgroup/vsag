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
#include <istream>
#include <stdexcept>

namespace vsag {

/**
 * @brief Abstract data source for parallel index deserialization.
 *
 * A plain source only needs to implement Size() and Read(); a source over
 * compressed frames additionally overrides ReadDecompressed().
 *
 * Thread safety: Read() and ReadDecompressed() are called concurrently
 * from multiple threads on non-overlapping ranges; each decompressed
 * stream is consumed by a single thread within its consume callback.
 */
class DeserializeReader {
public:
    virtual ~DeserializeReader() = default;

    /// Total size of the underlying data in bytes.
    [[nodiscard]] virtual uint64_t
    Size() const = 0;

    /// Copy len bytes starting at offset into dest (positioned read).
    /// Implementations must fail (throw) on out-of-range reads instead of
    /// returning partial or undefined data; the requested range may come
    /// from an untrusted index file, so guard the bounds check against
    /// offset + len overflow (e.g. len > Size() || offset > Size() - len).
    virtual void
    Read(uint64_t offset, uint64_t len, void* dest) = 0;

    /**
     * @brief Decompress the frame at [offset, offset + compressed_size)
     * as a sequential std::istream and hand it to consume; the stream is
     * only valid within the callback (no seek, no known length required).
     *
     * The default implementation signals the unsupported case with
     * std::runtime_error rather than the library's internal exception type,
     * which is not part of the installed public headers. Every error raised
     * while deserializing derives from std::exception, so callers should
     * catch std::exception rather than a specific derived type.
     */
    virtual void
    ReadDecompressed(uint64_t /*offset*/,
                     uint64_t /*compressed_size*/,
                     const std::function<void(std::istream&)>& /*consume*/) {
        throw std::runtime_error("reader does not support compressed frames");
    }
};

}  // namespace vsag
