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

#include <memory>
#include <mutex>
#include <string>

namespace vsag {

class Bitset;
using BitsetPtr = std::shared_ptr<Bitset>;

class Bitset {
public:
    /**
     * @brief Generate a random bitset with specified length, indices start from 0.
     *
     * @param length The number of bits in the bitset.
     * @return BitsetPtr A shared pointer to the generated random bitset.
     */
    static BitsetPtr
    Random(int64_t length);

    /**
     * @brief Create an empty bitset object.
     *
     * @return BitsetPtr A shared pointer to the created empty bitset.
     */
    static BitsetPtr
    Make();

    Bitset(const Bitset&) = delete;
    Bitset(Bitset&&) = delete;

protected:
    Bitset() = default;
    virtual ~Bitset() = default;

public:
    /**
     * @brief Set one bit to specified value.
     *
     * @param pos The position of the bit to set.
     * @param value The value to set the bit to (true or false).
     */
    virtual void
    Set(int64_t pos, bool value) = 0;

    /**
     * @brief Set one bit to true.
     *
     * @param pos The position of the bit to set.
     */
    void
    Set(int64_t pos) {
        return Set(pos, true);
    }

    /**
     * @brief Return the value of the bit at a specific position.
     *
     * @param pos The position of the bit.
     * @return true If the bit is set (true), false otherwise.
     */
    virtual bool
    Test(int64_t pos) const = 0;

    /**
     * @brief Returns the number of bits that are set to true.
     *
     * @return uint64_t The number of bits set to true.
     */
    virtual uint64_t
    Count() = 0;

public:
    /**
      * For debugging
      */
    virtual std::string
    Dump() = 0;
};

}  //namespace vsag
