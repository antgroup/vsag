
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

namespace vsag {

#define DECLARE_BIT_FUNCTIONS(ns)                                                         \
    namespace ns {                                                                        \
    void                                                                                  \
    BitAnd(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result); \
    void                                                                                  \
    BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result);  \
    void                                                                                  \
    BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result); \
    void                                                                                  \
    BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result);                   \
    }  // namespace ns
DECLARE_BIT_FUNCTIONS(generic)
DECLARE_BIT_FUNCTIONS(sse)
DECLARE_BIT_FUNCTIONS(avx)
DECLARE_BIT_FUNCTIONS(avx2)
DECLARE_BIT_FUNCTIONS(avx512)
DECLARE_BIT_FUNCTIONS(neon)
DECLARE_BIT_FUNCTIONS(sve)

#undef DECLARE_BIT_FUNCTIONS

using BitOperatorType = void (*)(const uint8_t* x,
                                 const uint8_t* y,
                                 const uint64_t num_byte,
                                 uint8_t* result);
extern BitOperatorType BitAnd;
extern BitOperatorType BitOr;
extern BitOperatorType BitXor;

using BitNotType = void (*)(const uint8_t* x, const uint64_t num_byte, uint8_t* result);
extern BitNotType BitNot;
}  // namespace vsag
