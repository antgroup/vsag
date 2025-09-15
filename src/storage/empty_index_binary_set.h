
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

#include <cstring>

#include "vsag/binaryset.h"
#include "vsag/constants.h"

namespace vsag {
class EmptyIndexBinarySet {
public:
    static BinarySet
    Make(const std::string& name = "EMPTY_INDEX") {
        const std::string empty_str = name;
        size_t num_bytes = empty_str.length();
        std::shared_ptr<int8_t[]> bin(new int8_t[num_bytes]);
        memcpy(bin.get(), empty_str.c_str(), empty_str.length());
        Binary b{
            .data = bin,
            .size = num_bytes,
        };
        BinarySet bs;
        bs.Set(BLANK_INDEX, b);

        return bs;
    }
};

}  // namespace vsag
