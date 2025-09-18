
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

#include "prefetch.h"
namespace vsag {

#define PREFETCH_LINE(X)       \
    case X:                    \
        PrefetchImpl<X>(data); \
        break;

template <>
void
PrefetchImpl<0>(const void* data){};

void
PrefetchLines(const void* data, uint64_t size) {
    uint64_t n = std::min(size / 64, 63UL);
    switch (n) {
        PREFETCH_LINE(0);
        PREFETCH_LINE(1);
        PREFETCH_LINE(2);
        PREFETCH_LINE(3);
        PREFETCH_LINE(4);
        PREFETCH_LINE(5);
        PREFETCH_LINE(6);
        PREFETCH_LINE(7);
        PREFETCH_LINE(8);
        PREFETCH_LINE(9);
        PREFETCH_LINE(10);
        PREFETCH_LINE(11);
        PREFETCH_LINE(12);
        PREFETCH_LINE(13);
        PREFETCH_LINE(14);
        PREFETCH_LINE(15);
        PREFETCH_LINE(16);
        PREFETCH_LINE(17);
        PREFETCH_LINE(18);
        PREFETCH_LINE(19);
        PREFETCH_LINE(20);
        PREFETCH_LINE(21);
        PREFETCH_LINE(22);
        PREFETCH_LINE(23);
        PREFETCH_LINE(24);
        PREFETCH_LINE(25);
        PREFETCH_LINE(26);
        PREFETCH_LINE(27);
        PREFETCH_LINE(28);
        PREFETCH_LINE(29);
        PREFETCH_LINE(30);
        PREFETCH_LINE(31);
        PREFETCH_LINE(32);
        PREFETCH_LINE(33);
        PREFETCH_LINE(34);
        PREFETCH_LINE(35);
        PREFETCH_LINE(36);
        PREFETCH_LINE(37);
        PREFETCH_LINE(38);
        PREFETCH_LINE(39);
        PREFETCH_LINE(40);
        PREFETCH_LINE(41);
        PREFETCH_LINE(42);
        PREFETCH_LINE(43);
        PREFETCH_LINE(44);
        PREFETCH_LINE(45);
        PREFETCH_LINE(46);
        PREFETCH_LINE(47);
        PREFETCH_LINE(48);
        PREFETCH_LINE(49);
        PREFETCH_LINE(50);
        PREFETCH_LINE(51);
        PREFETCH_LINE(52);
        PREFETCH_LINE(53);
        PREFETCH_LINE(54);
        PREFETCH_LINE(55);
        PREFETCH_LINE(56);
        PREFETCH_LINE(57);
        PREFETCH_LINE(58);
        PREFETCH_LINE(59);
        PREFETCH_LINE(60);
        PREFETCH_LINE(61);
        PREFETCH_LINE(62);
        default:
            PrefetchImpl<63>(data);
            break;
    }
}
}  // namespace vsag
