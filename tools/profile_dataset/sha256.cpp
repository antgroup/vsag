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

#include "sha256.h"

#include <array>
#include <cstdint>

namespace vsag::dataset_profile {

namespace {

constexpr std::array<uint32_t, 64> ROUND_CONSTANTS = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

uint32_t
rotate_right(uint32_t value, uint32_t shift) {
    return (value >> shift) | (value << (32U - shift));
}

void
compress(std::array<uint32_t, 8>& state, const uint8_t* block) {
    std::array<uint32_t, 64> words{};
    for (uint64_t i = 0; i < 16; ++i) {
        words[i] = (static_cast<uint32_t>(block[i * 4]) << 24U) |
                   (static_cast<uint32_t>(block[i * 4 + 1]) << 16U) |
                   (static_cast<uint32_t>(block[i * 4 + 2]) << 8U) |
                   static_cast<uint32_t>(block[i * 4 + 3]);
    }
    for (uint64_t i = 16; i < 64; ++i) {
        const uint32_t s0 = rotate_right(words[i - 15], 7) ^ rotate_right(words[i - 15], 18) ^
                            (words[i - 15] >> 3U);
        const uint32_t s1 =
            rotate_right(words[i - 2], 17) ^ rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10U);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];
    for (uint32_t i = 0; i < 64; ++i) {
        const uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const uint32_t choice = (e & f) ^ (~e & g);
        const uint32_t temp1 = h + s1 + choice + ROUND_CONSTANTS[i] + words[i];
        const uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = s0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

}  // namespace

std::string
sha256_hex(const std::string& data) {
    std::array<uint32_t, 8> state = {0x6a09e667,
                                     0xbb67ae85,
                                     0x3c6ef372,
                                     0xa54ff53a,
                                     0x510e527f,
                                     0x9b05688c,
                                     0x1f83d9ab,
                                     0x5be0cd19};
    const auto* bytes = reinterpret_cast<const uint8_t*>(data.data());
    const uint64_t length = data.size();
    uint64_t offset = 0;
    for (; offset + 64 <= length; offset += 64) {
        compress(state, bytes + offset);
    }
    std::array<uint8_t, 128> tail{};
    const uint64_t remaining = length - offset;
    for (uint64_t i = 0; i < remaining; ++i) {
        tail[i] = bytes[offset + i];
    }
    tail[remaining] = 0x80;
    const uint64_t tail_size = remaining + 1 + 8 <= 64 ? 64 : 128;
    const uint64_t bit_length = length * 8;
    for (uint64_t i = 0; i < 8; ++i) {
        tail[tail_size - 1 - i] = static_cast<uint8_t>(bit_length >> (8 * i));
    }
    compress(state, tail.data());
    if (tail_size == 128) {
        compress(state, tail.data() + 64);
    }
    static constexpr char hex_digits[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(64);
    for (const uint32_t word : state) {
        for (int shift = 28; shift >= 0; shift -= 4) {
            hex.push_back(hex_digits[(word >> static_cast<uint32_t>(shift)) & 0xFU]);
        }
    }
    return hex;
}

}  // namespace vsag::dataset_profile
