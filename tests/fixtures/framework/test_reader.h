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

/**
 * @file test_reader.h
 * @brief Test implementation of vsag::Reader interface.
 */

#pragma once

#include "vsag/binaryset.h"
#include "vsag/readerset.h"

namespace fixtures {

/**
 * @class TestReader
 * @brief Simple in-memory Reader implementation for testing serialization.
 * Wraps a Binary object and provides read access to its data.
 */
class TestReader : public vsag::Reader {
public:
    /**
     * @brief Constructs a TestReader from a Binary object.
     * @param binary The binary data to wrap for reading.
     */
    explicit TestReader(vsag::Binary binary);

    /**
     * @brief Reads data from the binary at the specified offset.
     * @param offset Byte offset to start reading from.
     * @param len Number of bytes to read.
     * @param dest Destination buffer to write the data.
     */
    void
    Read(uint64_t offset, uint64_t len, void* dest) override;

    /**
     * @brief Asynchronously reads data (synchronous implementation for testing).
     * @param offset Byte offset to start reading from.
     * @param len Number of bytes to read.
     * @param dest Destination buffer to write the data.
     * @param callback Callback function to invoke after read completes.
     */
    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, vsag::CallBack callback) override;

    /**
     * @brief Returns the total size of the binary data.
     * @return Size in bytes of the wrapped binary data.
     */
    [[nodiscard]] uint64_t
    Size() const override;

private:
    vsag::Binary binary_;  // The wrapped binary data.
};

}  // namespace fixtures
